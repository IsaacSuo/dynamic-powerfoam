"""Weak acoustic boundary solve in the actual sphere-motion degrees of freedom.

At a spherical cap use B(n)=[I,n], u_b=B a, a=(site velocity,radius rate).
Solve integral B^T [sigma*n + Z*(B*a-v) - t(B*a)] dA = 0.
The same u_b enters the geometric boundary velocity, velocity gradient and
traction work. There is no post-solve kinematic projection. This is a Galerkin
acoustic boundary adaptation for sphere caps, NOT an existing published full
solid ALE scheme. Four local unknowns are solved, not the whole body.

Unilateral normal traction: max(0,k*penetration-eta*u_b,y).
Tangential law: -min(Z*|u_b,t|,mu*normal_pressure)*unit(u_b,t), a viscously
regularized Coulomb law (not exact static sticking). Dissipated tangential
work returns to the body as heat. Normal work is exchanged with the compliant
obstacle. Tiny almost-flat caps use translation only (radius rate zero),
removing the geometrically indistinguishable fourth degree of freedom.

Contact is solved by a local active-set / reweighted linear iteration. Only
converged boundary states are accepted. Quadrature is supplied by the exact
exposure topology in fv_ale_contact, with finite-order integration errors.
"""
import torch


def surface_traction(velocity,Z,penetration,k,friction,damping):
    pressure=(k*penetration-damping*Z*velocity[:,1]).clamp(min=0)
    tangent=velocity[:,[0,2]];speed=tangent.norm(dim=1)
    drag=torch.minimum(Z,friction*pressure/speed.clamp(min=1e-100))
    t=torch.zeros_like(velocity);t[:,1]=pressure;t[:,[0,2]]=-drag[:,None]*tangent
    return t,pressure,drag


def solve_boundary(mesh,v,sigma,Z,k,friction,damping,max_iter=60,tol=2e-9):
    ncell=len(v);area=mesh.cap_area.clamp(min=1e-300)
    N=mesh.cap_normal_integral;Q=mesh.cap_Q
    mean_n=N/area[:,None]
    mean_f=-(sigma@mean_n[:,:,None])[:,:,0]/Z[:,None]
    mean_b=-(sigma*Q).sum((1,2))/(Z*area)
    variance=1-(mean_n*mean_n).sum(1)
    use_radius=mesh.surface&(variance>1e-3)
    dr=torch.where(use_radius,(mean_b-(mean_f*mean_n).sum(1))/variance.clamp(min=1e-300),0.)
    w=v+mean_f-dr[:,None]*mean_n
    w=torch.where(mesh.surface[:,None],w,v)
    force=torch.zeros_like(v);power=torch.zeros_like(area);heat=torch.zeros_like(area)
    contact_area=torch.zeros_like(area);delta=torch.zeros_like(area)
    iterations=0;relative_residual=0.;point_velocity=None;point_traction=None
    quad=mesh.contact_quad
    if quad is not None and len(quad[0]):
        owner,n,weights,penetration=quad
        cells,local=torch.unique(owner,sorted=True,return_inverse=True)
        nc=len(cells);A=area[cells];normal=mean_n[cells];z=Z[cells]
        G=torch.zeros((nc,4,4),device=v.device,dtype=v.dtype)
        G[:,:3,:3]=torch.eye(3,device=v.device,dtype=v.dtype)
        G[:,:3,3]=normal;G[:,3,:3]=normal;G[:,3,3]=1
        b=torch.cat((v[cells]+mean_f[cells],((v[cells]*normal).sum(1)+mean_b[cells])[:,None]),1)
        B=torch.zeros((len(owner),3,4),device=v.device,dtype=v.dtype)
        B[:,:,:3]=torch.eye(3,device=v.device,dtype=v.dtype);B[:,:,3]=n
        measure=weights/(Z[owner]*area[owner])
        radial=use_radius[cells]
        a=torch.cat((w[cells],dr[cells,None]),1)
        scale=b.norm(dim=1).clamp(min=1e-3)
        for iterations in range(1,max_iter+1):
            u=a[local,:3]+a[local,3,None]*n
            _,pressure,drag=surface_traction(u,Z[owner],penetration,k,friction,damping)
            active=pressure>0
            resistance=torch.stack((drag,damping*Z[owner]*active,drag),1)
            matrix=G+torch.zeros_like(G).index_add(0,local,
                measure[:,None,None]*torch.einsum('pai,pa,paj->pij',B,resistance,B))
            rhs=b+torch.zeros_like(b).index_add(0,local,
                (measure*k*penetration*active)[:,None]*B[:,1,:])
            matrix[~radial,3,:]=0;matrix[~radial,:,3]=0;matrix[~radial,3,3]=1
            rhs[~radial,3]=0
            updated=torch.linalg.solve(matrix,rhs[:,:,None])[:,:,0]
            # Under-relax only after the first few active-set solves. Typical
            # viscous-stick contact solves exactly in one or two iterations.
            if iterations>4:updated=.5*(updated+a)
            a=updated
            u=a[local,:3]+a[local,3,None]*n
            t,_,_=surface_traction(u,Z[owner],penetration,k,friction,damping)
            residual=(G@a[:,:,None])[:,:,0]-b-torch.zeros_like(b).index_add(0,local,
                measure[:,None]*torch.einsum('pai,pa->pi',B,t))
            residual[~radial,3]=0
            relative_residual=float((residual.norm(dim=1)/scale).max())
            if relative_residual<tol:break
        else:
            raise RuntimeError('local acoustic contact solve not converged: residual=%g'%relative_residual)
        w[cells]=a[:,:3];dr[cells]=a[:,3]
        force.index_add_(0,owner,weights[:,None]*t)
        power.index_add_(0,owner,weights*t[:,1]*u[:,1])
        heat.index_add_(0,owner,-weights*(t[:,[0,2]]*u[:,[0,2]]).sum(1))
        contact_area.index_add_(0,owner,weights)
        delta.scatter_reduce_(0,owner,penetration,reduce='amax',include_self=True)
        point_velocity=u;point_traction=t
    # The full cap velocity integral; includes both free and contacted portions.
    grad_integral=(w-v)[:,:,None]*N[:,None,:]+dr[:,None,None]*Q
    cap_rate=((w-v)*N).sum(1)+dr*mesh.cap_area
    info={'iterations':iterations,'relative_residual':relative_residual,
          'translation_only_caps':int((mesh.surface&~use_radius).sum()),
          'point_velocity':point_velocity,'point_traction':point_traction}
    return w,dr,grad_integral,cap_rate,(force,power,heat,k*contact_area,delta,contact_area),info
