"""Independent full-foam frictionless fixed-plane power and force checks.

Uses the production sphere/plane cut, actual groups and complete constitutive
RHS. This checks a compressive wall, not detachment or a full impact.
"""
import json
import importlib.util
import argparse
from pathlib import Path
import torch
import warp as wp
from free_surface.fv_ale_compatible_mesh_audit import restored
from free_surface.fv_ale_precise_geometry_check import cpu_mesh
from free_surface.fv_ale_hyperelastic import HyperelasticDeviceOps,HyperelasticALE
from free_surface.fv_ale_hyperelastic_material import response
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_reconstruction import group_centroids,group_sum
from free_surface.fv_ale_energy_ledger import rate,mechanical


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--cut-depth',type=float,default=.02)
    parser.add_argument('--wall-mode',choices=['compatible_slip','compatible_slip_acoustic'],default='compatible_slip')
    args=parser.parse_args()
    torch.set_num_threads(1);wp.init();root=Path('output/ale')
    data=torch.load(root/'hyperelastic_smooth_balanced001_guarded50ms.pt',map_location='cpu',weights_only=False)
    original=torch.load(root/'fv_state_dt8us.pt',map_location='cpu',weights_only=False)
    data.update({key:original[key] for key in ['p','r','groups']})
    data['q']=torch.zeros((len(data['r']),16),dtype=torch.float64)
    sol,_,_,_=restored(data);sol.wall_mode=args.wall_mode;sol.cap_dissipation=0.
    floor=float((data['p'][:,1]-data['r']).min())+args.cut_depth
    mesh=cpu_mesh(data['p'],data['r'],floor)
    HyperelasticALE.normalize_volume_roundoff(sol,mesh)
    assert bool((sol.group_sum(mesh.volume)[sol.groups]>0).all())
    if args.device=='cuda':
        for key,value in vars(mesh).items():
            if isinstance(value,torch.Tensor):setattr(mesh,key,value.cuda())
        sol.groups=sol.groups.cuda();sol.g=sol.g.cuda()
        sol.group_sum=lambda value:torch.zeros_like(value).index_add(0,sol.groups,value)
    q=torch.zeros_like(data['q'],device=args.device);q[:,0]=sol.rho0*mesh.volume/.9
    H=torch.diag(q.new_tensor([.2,-.1,-.1])).expand(len(q),3,3).clone()
    # Nonzero shear activates the tangential wall power and its transpose;
    # diagonal pressure alone would not test that part of the new operator.
    H[:,0,1]=H[:,1,0]=.06;H[:,1,2]=H[:,2,1]=-.035
    q[:,5:14]=(q[:,0,None,None]*H).reshape(-1,9)
    state=State(mesh.p,mesh.r,q,mesh);ops=HyperelasticDeviceOps(sol)
    sol._device_ops=ops
    b={'q':q,'V':mesh.volume,'groups':sol.groups,'p':mesh.p,'r':mesh.r,'i':mesh.i,'j':mesh.j,
       'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,'N':mesh.cap_normal_integral,'cap':mesh.cap_area,
       'floor':mesh.ground_area,'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
    mesh.cap_Q,mesh.surface,mesh.surface_area,_=ops.moments_gpu(b)
    b.update(Q=mesh.cap_Q,surface=mesh.surface);b.update(ops.extra_rhs_values(mesh))
    centers=group_centroids(b);x=centers-centers.mean(0)
    variation=.02*torch.sin(20*x[:,0])
    H[:,0,0]+=variation;H[:,1,1]-=variation
    q[:,5:14]=(q[:,0,None,None]*H).reshape(-1,9)
    v=torch.stack((torch.sin(9*x[:,0])+.2*x[:,1],-.3+.2*torch.cos(7*x[:,1]),.3*torch.sin(5*x[:,2])),1)
    q[:,1:4]=q[:,0,None]*v
    elastic=response(H,sol.rho0/torch.full_like(mesh.r,.9),sol.mu,sol.bulk,sol.rho0)[3]
    q[:,4]=q[:,0]*(elastic+.5*v.square().sum(1))
    _,_,v,rho,_,_,sigma,sound,_=ops.primitive(q,mesh.volume,sol.groups);Z=rho*sound
    rows=[]
    for alpha in [0.,.1,1.]:
        sol.interface_dissipation=alpha
        with torch.no_grad():
            kin=ops.mechanical_kinematics(b,v,sigma,Z);force=ops._compatible['force'].clone()
        rootv=v.detach().clone().requires_grad_(True);vv=rootv[sol.groups]
        # Independent ordinary-Torch evaluation of the reconstructed gradient.
        from free_surface.fv_ale_affine_stencil import gradient
        G=gradient(b,vv)
        i,j,A,n=b['i'],b['j'],b['A'],b['n']
        vl=vv[i]+(G[i]@(b['c']-centers[i])[:,:,None])[:,:,0]
        vr=vv[j]+(G[j]@(b['c']-centers[j])[:,:,None])[:,:,0]
        u=(Z[i,None]*vl+Z[j,None]*vr)/(Z[i]+Z[j])[:,None]
        grad=torch.zeros_like(sigma).index_add(0,i,A[:,None,None]*(u-vv[i])[:,:,None]*n[:,None,:])
        grad.index_add_(0,j,-A[:,None,None]*(u-vv[j])[:,:,None]*n[:,None,:])
        N,Q,cap=b['N'],b['Q'],b['cap'];off=b['p']-centers
        moment=off[:,:,None]*N[:,None,:]+b['r'][:,None,None]*Q
        cg=G@moment
        cr=((G@off[:,:,None])[:,:,0]*N).sum(1)+b['r']*(G*Q).sum((1,2))
        cg=cg+(cr-cg.diagonal(dim1=1,dim2=2).sum(1))[:,None,None]*torch.eye(3,dtype=q.dtype,device=q.device)/3
        uf=vv+(G@(b['floor_c']-centers)[:,:,None])[:,:,0]
        normal_face_velocity=uf[:,1]
        uf=uf*uf.new_tensor([1.,0.,1.])
        wall=-b['floor'][:,None,None]*(uf-vv)[:,:,None]*uf.new_tensor([0.,1.,0.])[None,None,:]
        power=(sigma*(grad+cg+wall)).sum()
        damping=.5*(alpha*A*Z[i]*Z[j]/(Z[i]+Z[j])*(vr-vl).square().sum(1)).sum()
        if args.wall_mode=='compatible_slip_acoustic':
            damping=damping+.5*(b['floor']*Z*normal_face_velocity.square()).sum()
        reference=-torch.autograd.grad(power+damping,rootv)[0]
        measured=group_sum(force,sol.groups)
        adjoint=float((measured-reference).norm()/reference.norm().clamp(min=1.))
        expected=q.new_tensor([0.,float((-mesh.ground_area*sigma[:,1,1]).sum()),0.])
        if args.wall_mode=='compatible_slip_acoustic':
            expected[1]-=(mesh.ground_area*Z*normal_face_velocity.detach()).sum()
        reaction=float((force.sum(0)-expected).norm()/expected.norm().clamp(min=1.))
        with torch.no_grad():
            w,dr,dq,packed=ops.rhs_gpu(b)
            diagnostics=dict(zip(ops.rhs_keys,packed.tolist()))
            account=rate(sol,state,(w,dr,dq,diagnostics),fd=True)
        energy=float(dq[:,4].sum())
        assert adjoint<1e-10 and reaction<1e-10,(adjoint,reaction)
        assert abs(energy)<1e-8*max(abs(account['mechanical_rate_chain_W']),1.),energy
        assert diagnostics['wall_tensile_prediction_cells']==0,diagnostics
        graph_error=None
        if args.device=='cuda':
            graphops=HyperelasticDeviceOps(sol)
            actual=graphops.call('rhs',graphops.rhs_gpu,b,('i','j','A','n','c'))
            graph_error=max(float((a-bb).norm()/bb.norm().clamp(min=1.)) for a,bb in zip(actual[:3],[w,dr,dq]))
            assert graph_error<1e-10,graph_error
        rows.append({'alpha':alpha,'adjoint_relative_error':adjoint,'net_reaction_relative_error':reaction,
                     'total_energy_rate_W':energy,'floor_area_m2':float(mesh.ground_area.sum()),'ledger':account,
                     'captured_graph_relative_error':graph_error})
    geometric=[]
    if args.device=='cpu':
        # This is a real re-cut, not V +/- h*Vdot fed back into the energy.
        # It independently verifies that the clipped geometric boundary has
        # the same volume motion used by the full material/force identity.
        predicted=mesh.volume_rate(w,dr);group_predicted=sol.group_sum(predicted)
        for h in [1e-7,1e-8]:
            plus=cpu_mesh(mesh.p+h*w,mesh.r+h*dr,floor)
            minus=cpu_mesh(mesh.p-h*w,mesh.r-h*dr,floor)
            measured=(plus.volume-minus.volume)/(2*h)
            err=float((sol.group_sum(measured)-group_predicted).norm()/group_predicted.norm().clamp(min=1e-30))
            finite=(mechanical(sol,q+h*dq,plus.volume)-mechanical(sol,q-h*dq,minus.volume))/(2*h)
            work_error=float(finite)-account['mechanical_rate_chain_W']
            geometric.append({'h_seconds':h,'group_volume_rate_relative_error':err,
                              'actual_recut_mechanical_power_W':float(finite),'power_difference_W':work_error})
        assert min(r['group_volume_rate_relative_error'] for r in geometric)<1e-4,geometric
    incoming_outgoing=[]
    if args.wall_mode=='compatible_slip_acoustic':
        # A stress-free cell approaching the wall must produce a compressive
        # impulse even though its constitutive pressure is initially zero.
        # The corresponding receding state must still request separation.
        for speed in [-1.,1.]:
            testq=torch.zeros_like(q);testq[:,0]=sol.rho0*mesh.volume
            testq[:,2]=speed*testq[:,0];testq[:,4]=.5*testq[:,0]*speed**2
            with torch.no_grad():
                _,_,tdq,packed=ops.rhs_gpu(dict(b,q=testq))
                diagnostics=dict(zip(ops.rhs_keys,packed.tolist()))
            active=int((mesh.ground_area>0).sum())
            assert diagnostics['wall_tensile_prediction_cells']==(active if speed>0 else 0)
            assert diagnostics['force_y']*speed<0
            assert abs(float(tdq[:,4].sum()))<1e-8
            incoming_outgoing.append({'normal_velocity':speed,'force_y_N':diagnostics['force_y'],
                                      'tensile_cells':diagnostics['wall_tensile_prediction_cells'],
                                      'total_energy_rate_W':float(tdq[:,4].sum()),
                                      'wall_acoustic_W':diagnostics['wall_acoustic_W']})
    # A real clipped-contact state also checks that factoring the original
    # wall into a method did not change the old Wilkins force/work path.
    from free_surface.fv_ale_device import PlaneDeviceOps
    oldfile=root/'hyperelastic_baseline_20261108/fv_ale_device.py'
    spec=importlib.util.spec_from_file_location('pre_hyperelastic_wall',oldfile)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sol.gamma=2.;sol.pinf=sol.bulk/2
    oldq=q.clone();oldq[:,4]=oldq[:,0]*(sol.bulk/sol.rho0+.5*v.square().sum(1))
    oldq[:,5:14]=(oldq[:,0,None,None]*torch.diag(q.new_tensor([1000.,-500.,-500.]))).reshape(-1,9)
    regressions=[]
    for friction in [0.,10.]:
        sol.friction=friction;bb=dict(b,q=oldq)
        old=module.PlaneDeviceOps(sol).rhs_gpu(bb)
        new=PlaneDeviceOps(sol).rhs_gpu(bb)
        errors=[float((a-bb).norm()/a.norm().clamp(min=1.)) for a,bb in zip(old[:3],new[:3])]
        assert max(errors)<1e-14,errors
        regressions.append({'friction':friction,'w_rdot_qdot_relative_errors':errors})
    result={'passed':True,'device':args.device,'wall_mode':args.wall_mode,'cells':len(q),'ground':floor,
            'contact_cells':int((mesh.ground_area>0).sum()),'rows':rows,
            'actual_recut_volume_checks':geometric,
            'stress_free_incoming_outgoing':incoming_outgoing,
            'original_clipped_wall_regression':regressions,
            'scope':'Full-foam compressive fixed-plane identity; does not establish impact stability or release.'}
    tag='hyperelastic_compatible_wall_check'+('_acoustic' if args.wall_mode=='compatible_slip_acoustic' else '')+('_cuda' if args.device=='cuda' else '')
    (root/(tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
