"""Rotate a real frozen geometry/state and test the full face/cap adaptation."""
import json
import argparse
from pathlib import Path
import torch
from free_surface.fv_ale_hyperelastic import HyperelasticALE
from free_surface.fv_ale_solid import State


@torch.no_grad()
def main():
    torch.set_num_threads(1);root=Path('output/ale')
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['affine','compatible'],default='affine')
    parser.add_argument('--stencil',choices=['one_ring','extended','smooth'],default='one_ring')
    parser.add_argument('--cap-motion',choices=['isotropic','volume','balanced'],default='isotropic')
    args=parser.parse_args()
    data=torch.load(root/'hyperelastic_release_blend01_compatible.pt',weights_only=False);m=data['material']
    sol=HyperelasticALE(data['p'],data['r'],m['E'],m['nu'],m['ys'],groups=data['groups'],ground=-10.,
                       gravity=(0.,0.,0.),contact_model='plane',mechanical_reconstruction=args.mode,transport_reconstruction='linear',affine_stencil=args.stencil,
                       affine_cap_motion=args.cap_motion)
    s=State(sol.initial.p,sol.initial.r,data['q'].cuda(),sol.initial.mesh);mesh=s.mesh
    total,groupV,v,rho,_,_,sigma,sound,_=sol.device_ops().primitive(s.q,mesh.volume,sol.groups)
    b={'p':s.p,'r':s.r,'q':s.q,'V':mesh.volume,'groups':sol.groups,'i':mesh.i,'j':mesh.j,'A':mesh.area,
       'n':mesh.normal,'c':mesh.centroid,'N':mesh.cap_normal_integral,'Q':mesh.cap_Q,'cap':mesh.cap_area,
       'surface':mesh.surface,'floor':mesh.ground_area,'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
    b={k:x.cpu() for k,x in b.items()};v=v.cpu();sigma=sigma.cpu();Z=(rho*sound).cpu()
    if args.stencil!='one_ring':
        from free_surface.fv_ale_affine_stencil import build
        b.update(build(b,transition_ratio=.01 if args.stencil=='smooth' else None)[0])
    ops=sol.device_ops();old=ops.mechanical_kinematics(b,v,sigma,Z)
    old_force=ops._compatible['force'].clone() if args.mode=='compatible' else None
    R=torch.matrix_exp(torch.tensor([[0.,-.7,.2],[.7,0.,-.4],[-.2,.4,0.]],dtype=v.dtype))
    shift=torch.tensor([.17,-.21,.11],dtype=v.dtype);boost=torch.tensor([.3,.4,-.2],dtype=v.dtype)
    rotated={k:x.clone() for k,x in b.items()}
    for key in ['p','c','floor_c']:rotated[key]=b[key]@R.T+shift
    for key in ['n','N','floor_n']:rotated[key]=b[key]@R.T
    rotated['Q']=R@b['Q']@R.T
    if args.stencil!='one_ring':
        # Rebuild independently in the transformed coordinates, including
        # conditioning decisions, rather than rotating a cached stencil.
        rotated.update(build(rotated,transition_ratio=.01 if args.stencil=='smooth' else None)[0])
    new=ops.mechanical_kinematics(rotated,v@R.T+boost,R@sigma@R.T,Z)
    expected=[old[0]@R.T+boost,old[1],old[2],R@old[3]@R.T,old[4],old[5]@R.T+boost,old[6]@R.T,old[7]@R.T,old[8]@R.T]
    labels=['mesh_velocity','radius_velocity','radial_mask','cap_gradient','cap_relative_volume_rate',
            'interface_velocity','interface_traction','left_traction','right_traction']
    error={}
    for k,name in enumerate(labels):
        if k==2:error[name]=int((new[k]!=expected[k]).sum())
        else:error[name]=float((new[k]-expected[k]).norm()/expected[k].norm().clamp(min=1.))
    if old_force is not None:
        error['compatible_force']=float((ops._compatible['force']-old_force@R.T).norm()/old_force.norm())
    assert max(error.values())<1e-10,error
    suffix='' if args.stencil=='one_ring' else '_'+args.stencil
    if args.cap_motion!='isotropic':suffix+='_'+args.cap_motion
    (root/('hyperelastic_objectivity_'+args.mode+suffix+'.json')).write_text(json.dumps(error,indent=2)+'\n')
    print(json.dumps(error,indent=2))


if __name__=='__main__':main()
