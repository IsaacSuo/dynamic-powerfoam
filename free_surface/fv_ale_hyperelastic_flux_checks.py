"""Independent local work identities plus full-foam affine velocity probes."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from free_surface.fv_ale_hyperelastic import HyperelasticALE,HyperelasticDeviceOps
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_hyperelastic_material import response,symmetric_deviator


def transport_checks():
    from free_surface.fv_ale_reconstruction import compatible_transport
    torch.manual_seed(491);n=500;dtype=torch.float64;mu,bulk,rho0=2e5,2e6,1000.
    H=symmetric_deviator(torch.randn((3*n,3,3),dtype=dtype)*.3)
    rho=rho0*torch.exp(.2*torch.randn(3*n,dtype=dtype))
    v=torch.randn((3*n,3),dtype=dtype);thermal=torch.rand(3*n,dtype=dtype)
    f=torch.randn(n,dtype=dtype)
    def data(velocity):
        pressure,_,_,elastic,_,B,_=response(H,rho,mu,bulk,rho0)
        T=mu/(2*rho0)*symmetric_deviator(B)
        U=torch.zeros((3*n,16),dtype=dtype);chi=torch.zeros_like(U)
        U[:,0]=rho;U[:,1:4]=rho[:,None]*velocity;U[:,5:14]=rho[:,None]*H.reshape(-1,9)
        U[:,4]=rho*(elastic+.5*velocity.square().sum(1)+thermal)
        chi[:,0]=elastic+pressure/rho-(T*H).sum((1,2))-.5*velocity.square().sum(1)
        chi[:,1:4]=velocity;chi[:,5:14]=T.reshape(-1,9)
        return U,chi,pressure
    U,chi,p=data(v);i=torch.arange(n);j=i+n
    low=torch.where(f[:,None]>=0,U[i],U[j]);up=torch.where(f>=0,thermal[i],thermal[j])
    flux,theta=compatible_transport(U[2*n:],low,chi[i],chi[j],p[i],p[j],f,up)
    heat_i=f*((chi[i]*flux).sum(1)-flux[:,4]-p[i])+f*flux[:,0]*up
    heat_j=f*(flux[:,4]-(chi[j]*flux).sum(1)+p[j])-f*flux[:,0]*up
    scale=float((f.abs()*(p[i].abs()+p[j].abs()+1)).max())
    error=float((heat_i-heat_j).abs().max())/scale
    minimum=float(torch.minimum(heat_i,heat_j).min())/scale
    assert error<1e-12 and minimum>-1e-12,(error,minimum)
    boost=torch.tensor([1.,-.4,.7],dtype=dtype)
    ub,chib,pb=data(v+boost);lob=torch.where(f[:,None]>=0,ub[i],ub[j])
    fb,tb=compatible_transport(ub[2*n:],lob,chib[i],chib[j],pb[i],pb[j],f,up)
    transform=flux[:,4]+(flux[:,1:4]*boost).sum(1)+.5*boost.square().sum()*flux[:,0]
    boost_error=float((fb[:,4]-transform).abs().max())/float(transform.abs().max())
    assert boost_error<1e-12,boost_error
    return {'equal_heating_scaled_error':error,'minimum_heating_scaled':minimum,
            'Galilean_energy_flux_relative_error':boost_error,
            'limited_faces':int((theta<1).sum()),'minimum_reconstructed_density':float(flux[:,0].min())}


def local_checks():
    torch.manual_seed(723);dtype=torch.float64;nface=300
    v=torch.randn((2*nface,3),dtype=dtype);Z=torch.exp(torch.randn(2*nface,dtype=dtype))
    ti=torch.randn((nface,3),dtype=dtype);tj=torch.randn_like(ti)
    i=torch.arange(nface);j=i+nface
    boost=torch.tensor([.3,-.4,.7],dtype=dtype)
    results={}
    for alpha in [0.,.01,.1,.5,1.]:
        ops=object.__new__(HyperelasticDeviceOps);ops.s=SimpleNamespace(interface_dissipation=alpha,cap_dissipation=alpha,mechanical_reconstruction='constant')
        u,t=ops.acoustic_interface(v,Z,ti,tj,i,j);work=ops.interface_energy_flux(u,t,v,ti,tj,i,j)
        hi=work-(v[i]*t).sum(1)-(ti*(u-v[i])).sum(1)
        hj=-work+(v[j]*t).sum(1)+(tj*(u-v[j])).sum(1)
        expected=alpha*((tj-ti).square().sum(1)+Z[i]*Z[j]*(v[j]-v[i]).square().sum(1))/(Z[i]+Z[j])
        error=float((hi+hj-expected).norm()/expected.norm().clamp(min=1.))
        ub,tb=ops.acoustic_interface(v+boost,Z,ti,tj,i,j)
        wb=ops.interface_energy_flux(ub,tb,v+boost,ti,tj,i,j)
        invariant=float((wb-work-(t*boost).sum(1)).abs().max())
        assert error<1e-12 and min(float(hi.min()),float(hj.min()))>-1e-12
        assert invariant<1e-12
        results[str(alpha)]={'pair_heating_relative_error':error,'minimum_side_heating':min(float(hi.min()),float(hj.min())),
                             'Galilean_work_transform_max_error':invariant}
    return results


def centroids(mesh):
    i,j,A,n,c=mesh.i,mesh.j,mesh.area,mesh.normal,mesh.centroid
    di=c-mesh.p[i];dj=c-mesh.p[j]
    hi=(di*n).sum(1);hj=-(dj*n).sum(1)
    first=torch.zeros_like(mesh.p).index_add(0,i,(A*hi)[:,None]*di).index_add(0,j,(A*hj)[:,None]*dj)
    first+=mesh.r[:,None].square()*mesh.cap_normal_integral
    df=mesh.ground_centroid-mesh.p;hf=(df*mesh.ground_normal).sum(1)
    first+=(mesh.ground_area*hf)[:,None]*df
    return mesh.p+first/(4*mesh.volume.clamp(min=1e-300)[:,None])


@torch.no_grad()
def main():
    root=Path('output/ale');result={'local_work_checks':local_checks(),'transport_work_checks':transport_checks()}
    data=torch.load(root/'fv_state_dt8us.pt',weights_only=False);m=data['material']
    sol=HyperelasticALE(data['p'],data['r'],m['E'],m['nu'],m['ys'],groups=data['groups'],
                        contact_model='plane',ground=-10.,gravity=(0.,0.,0.))
    s=sol.initial;V=s.mesh.volume
    centers=centroids(s.mesh)
    center_group=sol.group_sum(V[:,None]*centers)[sol.groups]/sol.group_sum(V)[sol.groups,None]
    origin=(V[:,None]*centers).sum(0)/V.sum()
    fields={'rotation':torch.tensor([[0.,-1.,.3],[1.,0.,-.2],[-.3,.2,0.]],device=s.p.device,dtype=s.p.dtype),
            'shear':torch.tensor([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]],device=s.p.device,dtype=s.p.dtype)}
    result['affine_velocity']={}
    from free_surface.fv_ale_reconstruction import group_centroids,slopes,face_values
    mesh=s.mesh
    b={'p':s.p,'r':s.r,'V':V,'groups':sol.groups,'i':mesh.i,'j':mesh.j,'A':mesh.area,
       'n':mesh.normal,'c':mesh.centroid,'N':mesh.cap_normal_integral,'floor':mesh.ground_area,
       'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
    reconstructed_centers=group_centroids(b)
    centroid_error=float((reconstructed_centers-center_group).abs().max())
    assert centroid_error<1e-12,centroid_error
    linear=torch.tensor([[.3,-.7,.2],[1.,.4,-.1]],dtype=s.p.dtype,device=s.p.device)
    value=reconstructed_centers@linear.T+.7
    gradient,good=slopes(b,value,reconstructed_centers,limited=False)
    gradient_error=float((gradient[good]-linear).abs().max())
    assert gradient_error<1e-9,gradient_error
    limited,_=slopes(b,value,reconstructed_centers)
    faces=face_values(b,value,limited,reconstructed_centers)
    lo=value.clone();hi=value.clone();g=sol.groups
    for dest,source in [(g[mesh.i],mesh.j),(g[mesh.j],mesh.i)]:
        ix=dest[:,None].expand(-1,value.shape[1])
        lo.scatter_reduce_(0,ix,value[source],reduce='amin',include_self=True)
        hi.scatter_reduce_(0,ix,value[source],reduce='amax',include_self=True)
    violation=0.
    for k,f in zip([mesh.i,mesh.j],faces):
        violation=max(violation,float(torch.maximum(f-hi[g[k]],lo[g[k]]-f).clamp(min=0).max()))
    assert violation<1e-12,violation
    result['reconstruction']={'group_centroid_max_error_m':centroid_error,
        'unlimited_affine_gradient_max_error':gradient_error,'limited_face_extrema_violation':violation,
        'rank_deficient_groups':int(torch.unique(g[~good]).numel())}
    for name,L in fields.items():
        v=(center_group-origin)@L.T
        q=s.q.clone();q[:,1:4]=q[:,:1]*v;q[:,4]=q[:,0]*.5*v.square().sum(1)
        state=State(s.p,s.r,q,s.mesh);rhs=sol.rhs(state)
        rate=sol.group_sum(rhs[2][:,5:14])[sol.groups]/sol.group_sum(q[:,0])[sol.groups,None]
        rate=rate.reshape(-1,3,3)
        exact=L+L.T-2*torch.trace(L)/3*torch.eye(3,dtype=L.dtype,device=L.device)
        error=(rate-exact).square().sum((1,2))
        interior=~s.mesh.surface
        result['affine_velocity'][name]={'weighted_rms_log_shape_rate_error':float((V*error).sum().div(V.sum()).sqrt()),
             'interior_weighted_rms_error':float((V[interior]*error[interior]).sum().div(V[interior].sum()).sqrt()),
             'surface_weighted_rms_error':float((V[~interior]*error[~interior]).sum().div(V[~interior].sum()).sqrt()),
             'max_rate_error':float(error.sqrt().max()),'expected_log_shape_rate_norm':float(exact.norm())}
    sol.mechanical_reconstruction='affine';sol._device_ops=HyperelasticDeviceOps(sol)
    result['affine_velocity_reconstructed']={}
    for name,L in fields.items():
        v=(center_group-origin)@L.T
        q=s.q.clone();q[:,1:4]=q[:,:1]*v;q[:,4]=q[:,0]*.5*v.square().sum(1)
        state=State(s.p,s.r,q,s.mesh);rhs=sol.rhs(state)
        rate=sol.group_sum(rhs[2][:,5:14])[sol.groups]/sol.group_sum(q[:,0])[sol.groups,None]
        exact=L+L.T-2*torch.trace(L)/3*torch.eye(3,dtype=L.dtype,device=L.device)
        error=(rate.reshape(-1,3,3)-exact).square().sum((1,2))
        from free_surface.fv_ale_energy_ledger import rate as energy_rate
        result['affine_velocity_reconstructed'][name]={'weighted_rms_log_shape_rate_error':float((V*error).sum().div(V.sum()).sqrt()),
                'max_rate_error':float(error.sqrt().max()),'energy_identity':energy_rate(sol,state,rhs,fd=True)}
    (root/'hyperelastic_flux_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
