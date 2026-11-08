"""Actual-foam power/adjoint checks independent of the force implementation."""
import json
import argparse
from pathlib import Path
import torch
from free_surface.fv_ale_hyperelastic import HyperelasticALE
from free_surface.fv_ale_reconstruction import group_sum,group_centroids
from free_surface.fv_ale_compatible_mechanics import affine_map
from free_surface.fv_ale_energy_ledger import rate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--transport',choices=['linear','entropy'],default='linear')
    parser.add_argument('--stencil',choices=['one_ring','extended','smooth'],default='one_ring')
    parser.add_argument('--cap-motion',choices=['isotropic','volume','balanced'],default='isotropic')
    parser.add_argument('--snapshot')
    args=parser.parse_args()
    root=Path('output/ale');data=torch.load(root/'fv_state_dt8us.pt',weights_only=False)
    snap=torch.load(args.snapshot,weights_only=False,map_location='cpu') if args.snapshot else None
    if snap:data={**snap['before'],'material':snap['material'],'groups':snap['groups']}
    m=data['material']
    sol=HyperelasticALE(data['p'],data['r'],m['E'],m['nu'],m['ys'],groups=data['groups'],
        contact_model='plane',ground=-10.,gravity=(0.,0.,0.),
        mechanical_reconstruction='compatible',transport_reconstruction=args.transport,affine_stencil=args.stencil,
        affine_cap_motion=args.cap_motion)
    s=sol.initial;s=sol.initialize_strain(s,torch.diag(s.p.new_tensor([1.5,1.5**-.5,1.5**-.5])))
    mesh=s.mesh
    b={'q':s.q,'p':s.p,'r':s.r,'V':mesh.volume,'groups':sol.groups,'i':mesh.i,'j':mesh.j,
       'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,'N':mesh.cap_normal_integral,'Q':mesh.cap_Q,
       'cap':mesh.cap_area,'surface':mesh.surface,'floor':mesh.ground_area,
       'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
    centers=group_centroids(b);x=centers-centers.mean(0)
    if args.stencil!='one_ring':
        from free_surface.fv_ale_affine_stencil import build
        extra,stencil_summary=build(b,transition_ratio=.01 if args.stencil=='smooth' else None);b.update(extra)
    variation=.1*torch.sin(20*x[:,0])
    s.q[:,5]+=s.q[:,0]*variation;s.q[:,9]-=s.q[:,0]*variation
    velocity=torch.stack((torch.sin(9*x[:,0])+x[:,1],torch.cos(7*x[:,1])-.2*x[:,2],torch.sin(5*x[:,2])+.4*x[:,0]),1)
    sol.set_velocity(s,velocity);ops=sol.device_ops()
    _,_,v,rho,_,_,sigma,sound,_=ops.primitive(s.q,mesh.volume,sol.groups);Z=rho*sound
    results={}
    for alpha in [0.,.1,1.]:
        sol.interface_dissipation=alpha;sol.cap_dissipation=alpha
        with torch.no_grad():
            kin=ops.mechanical_kinematics(b,v,sigma,Z)
            force=ops._compatible['force'].clone()
            _,inv,d,weight=affine_map(b,v,centers)
            compact_error=None
            if args.stencil=='smooth':
                from free_surface.fv_ale_affine_stencil import compact
                edges,_,_=compact(b)
                ops.mechanical_kinematics(dict(b,**edges),v,sigma,Z)
                compact_error=float((ops._compatible['force']-force).norm()/force.norm().clamp(min=1.))
                assert compact_error<1e-10,compact_error
        # Independent scalar elastic power plus quadratic velocity damping;
        # autograd differentiates only this ordinary Torch linear map.
        rootv=v.detach().clone().requires_grad_(True)
        vv=rootv[sol.groups];i,j,A,n,g=b['i'],b['j'],b['A'],b['n'],b['groups']
        if args.stencil!='one_ring':
            ids=b['affine_neighbors'];dd=centers[ids]-centers[:,None,:]
            summed=(b['affine_weights'][:,:,None,None]*(vv[ids]-vv[:,None,:])[:,:,:,None]*dd[:,:,None,:]).sum(1)
            G=(summed@b['affine_inverse'])[g]
        else:
            term=weight[:,None,None]*(vv[j]-vv[i])[:,:,None]*d[:,None,:]
            summed=torch.zeros_like(sigma).index_add(0,g[i],term).index_add(0,g[j],term)
            G=summed[g]@inv
        vl=vv[i]+(G[i]@(b['c']-centers[i])[:,:,None])[:,:,0]
        vr=vv[j]+(G[j]@(b['c']-centers[j])[:,:,None])[:,:,0]
        u=(Z[i,None]*vl+Z[j,None]*vr)/(Z[i]+Z[j])[:,None]
        grad=torch.zeros_like(sigma).index_add(0,i,A[:,None,None]*(u-vv[i])[:,:,None]*n[:,None,:])
        grad.index_add_(0,j,-A[:,None,None]*(u-vv[j])[:,:,None]*n[:,None,:])
        N,Q,cap=b['N'],b['Q'],b['cap'];off=b['p']-centers
        moment=off[:,:,None]*N[:,None,:]+b['r'][:,None,None]*Q
        cg=G@moment;dw=(G@off[:,:,None])[:,:,0]
        cr=(dw*N).sum(1)+b['r']*cap*G.diagonal(dim1=1,dim2=2).sum(1)/3
        if args.cap_motion!='isotropic':
            extra=torch.where(b['surface'],b['r']*(G*Q).sum((1,2)),
                              b['r']*cap*G.diagonal(dim1=1,dim2=2).sum(1)/3)
            cr=(dw*N).sum(1)+extra
        cg=cg+(cr-cg.diagonal(dim1=1,dim2=2).sum(1))[:,None,None]*torch.eye(3,device=v.device)/3
        power=(sigma*(grad+cg)).sum()
        diss=.5*(alpha*A*Z[i]*Z[j]/(Z[i]+Z[j])*(vr-vl).square().sum(1)).sum()
        ref=-torch.autograd.grad(power+diss,rootv)[0]
        measured=group_sum(force,g)
        err=float((measured-ref).norm()/ref.norm())
        net=float(force.sum(0).norm()/force.norm())
        torque=torch.cross(centers,force,dim=1).sum(0)
        torque_scaled=float(torque.norm()/(force.norm()*centers.norm()))
        assert err<1e-10 and net<1e-11 and torque_scaled<1e-10,(err,net,torque_scaled)
        # Fresh operator after parameter changes avoids reusing a graph built
        # under a different Python-level configuration.
        sol._device_ops=None;rhs=sol.rhs(s)
        ledger=rate(sol,s,rhs,fd=True)
        energy_rate=float(rhs[2][:,4].sum())
        assert abs(energy_rate)<1e-8*max(abs(ledger['mechanical_rate_chain_W']),1.),energy_rate
        results[str(alpha)]={'adjoint_relative_error':err,'net_force_scaled':net,
            'torque_scaled':torque_scaled,'total_energy_rate_W':energy_rate,'ledger':ledger,
            'compact_force_relative_error':compact_error}
        ops=sol.device_ops()
    if args.stencil!='one_ring':results['stencil_summary']=stencil_summary
    tag='hyperelastic_compatible_checks_'+args.transport+'_'+args.stencil+('_snapshot' if snap else '')
    if args.cap_motion!='isotropic':tag+='_'+args.cap_motion
    (root/(tag+'.json')).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
