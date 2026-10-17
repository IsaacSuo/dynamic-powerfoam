"""Original PowerfoamScene.forward driven by the validated TLSPH trajectory.

The beam is hand authored, not reconstructed. Rotation/isotropic-radius transfer
is a PowerSim-style baseline. Quantifies bounded power SUPPORT volumes, not
dipole occupancy, texture transport accuracy, or a coupled radius solve.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


def bounded_volumes(p, radii, exponent=14):
    """Sobol integration of each ball intersected with all power halfspaces.

All overlapping spheres are included, even if they do not share a true face.
No soft ownership, voxel grid, or face-area proxy is used.
"""
    u=qmc.Sobol(3,scramble=True,seed=2026).random_base2(exponent)
    z=1-2*u[:,0]; phi=2*np.pi*u[:,1]; radial=u[:,2]**(1/3)
    sphere=radial[:,None]*np.column_stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])
    tree=cKDTree(p); output=np.empty(len(p))
    for i in range(len(p)):
        ids=np.asarray(tree.query_ball_point(p[i],radii[i]+radii.max()),dtype=int)
        ids=ids[(ids!=i)&(np.linalg.norm(p[ids]-p[i],axis=1)<radii[i]+radii[ids])]
        offset=p[i]-p[ids]
        rhs=np.sum(offset**2,axis=1)+radii[i]**2-radii[ids]**2
        occupied=np.ones(len(sphere),dtype=bool)
        points=sphere*radii[i]
        for k in range(0,len(ids),16):
            occupied &= np.all(2*points@offset[k:k+16].T+rhs[k:k+16]>=0,axis=1)
        output[i]=occupied.mean()*4*np.pi*radii[i]**3/3
    return output


def volume_checks():
    radius=.4
    whole=4*np.pi*radius**3/3
    isolated=bounded_volumes(np.zeros((1,3)),np.array([radius]),16)[0]
    centers=np.array([[-.2,0.,0.],[.2,0.,0.]])
    cap_height=radius-.2
    expected=whole-np.pi*cap_height**2*(radius-cap_height/3)
    measured=bounded_volumes(centers,np.full(2,radius),16)
    relative=float(np.max(abs(measured/expected-1)))
    assert abs(isolated/whole-1)<1e-12
    assert relative<.002
    return dict(isolated_sphere_relative_error=float(abs(isolated/whole-1)),
                two_equal_spheres_max_relative_error=relative)


def volume_diagnostic(data, r0):
    p0=data['reference']; base=bounded_volumes(p0,r0,16)
    lower=bounded_volumes(p0,r0,14)
    result=dict(method='Scrambled Sobol 3D per-sphere integration, exact power halfspace membership; fixed seed 2026.',
        analytic_checks=volume_checks(),
        rest_volume_14_vs_16_relative_l1=float(np.sum(abs(lower-base))/sum(base)),
        rest_support_volume=float(sum(base)),
        mechanical_reference_volume=float(sum(data['volumes'])),
        rest_support_vs_mechanical_volume_relative_l1=float(np.sum(abs(base-data['volumes']))/sum(data['volumes'])),
        note='Compare bounded support volume and assigned mechanical volume explicitly; the rectangular reference beam and a union of balls need not match at rest.',
        frames=[])
    for k in np.linspace(0,len(data['times'])-1,7).astype(int):
        J=np.linalg.det(data['deformation'][k]); target=base*J
        radii=r0*J**(1/3)
        vol=bounded_volumes(data['positions'][k],radii,16)
        fixedvol=bounded_volumes(data['positions'][k],r0,16)
        result['frames'].append(dict(time=float(data['times'][k]),
            radius_scaled_relative_l1=float(np.sum(abs(vol-target))/sum(target)),
            radius_fixed_relative_l1=float(np.sum(abs(fixedvol-target))/sum(target)),
            scaled_total_volume_error=float((sum(vol)-sum(target))/sum(target)),
            target_note='Each initial support-cell volume times its material detF, to isolate change from rest-domain mismatch.'))
    # The deformation is small: distinguish geometric mismatch from integration
    # error using a higher-resolution, matched rest/deformed quadrature.
    high_base=bounded_volumes(p0,r0,18)
    J=np.linalg.det(data['deformation'][0])
    high_vol=bounded_volumes(data['positions'][0],r0*J**(1/3),18)
    high_fixed=bounded_volumes(data['positions'][0],r0,18)
    target=high_base*J
    result['loaded_frame_refinement']=dict(
        samples_per_cell=2**18,
        rest_volume_16_vs_18_relative_l1=float(np.sum(abs(base-high_base))/sum(high_base)),
        radius_scaled_relative_l1=float(np.sum(abs(high_vol-target))/sum(target)),
        radius_fixed_relative_l1=float(np.sum(abs(high_fixed-target))/sum(target)),
        note='Volume agreement alone does not establish material-region or texture transport agreement.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('output/tlsph'))
    parser.add_argument('--volumes-only',action='store_true')
    args=parser.parse_args()
    data=np.load(args.out/'trajectory.npz')
    p0=data['reference']; n=len(p0)
    r0=data['radii0'] if 'radii0' in data else np.full(n,.125*1.1)
    metrics=volume_diagnostic(data,r0)
    (args.out/'cell_volumes.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2),flush=True)
    if args.volumes_only: return
    render_trajectory(data,r0,args.out)


def render_trajectory(data,r0,out,radii_sequence=None,label='PowerSim-style radius / rotation transfer',
                      frame_label='t',auto_frame=False,frame_duration_ms=50):
    """Render the same trajectory with optional externally solved radii."""
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    p0=data['reference']; n=len(p0)
    if radii_sequence is not None:
        if radii_sequence.shape!=(len(data['times']),n): raise ValueError('Radius trajectory shape mismatch')
        if not np.isfinite(radii_sequence).all() or np.any(radii_sequence<=0): raise ValueError('Invalid radius trajectory')
    import torch
    import warp as wp
    from PIL import Image,ImageDraw
    from experiments.single_cell_shear import make_model
    from experiments.full_model_rigid_probe import load_state,tensor,render
    from powerfoam.camera import TorchCamera
    wp.init(); model=make_model(pinhole=True)
    if auto_frame:
        bound_r=r0*np.linalg.det(data['deformation'])**(1/3) if radii_sequence is None else radii_sequence
        lower=np.min(data['positions']-bound_r[:,:,None],axis=(0,1))
        upper=np.max(data['positions']+bound_r[:,:,None],axis=(0,1))
        center=.5*(lower+upper);width=1.1*max(upper[0]-lower[0],1.6*(upper[1]-lower[1]))
        cam=TorchCamera(tensor([center[0],center[1],upper[2]+5]),tensor([width/10,0,0]),tensor([0,width/16,0]),512,320)
    else:
        cam=TorchCamera(tensor([1.,-.12,5.]),tensor([.25,0,0]),tensor([0,.15,0]),512,320)
    cam.ray_maps=cam._build_pinhole_ray_maps()
    frame0=Rotation.from_euler('y',90,degrees=True).as_matrix()
    uv=np.column_stack([np.tile([-.6,-.2,.2,.6],2),np.repeat([-.4,.4],4)])
    axes=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,1],[-1,1,1]],dtype=float)
    axes=3*axes/np.linalg.norm(axes,axis=1)[:,None]
    color=np.where((np.floor(p0[:,0]*4).astype(int)%2)[:,None],np.array([.9,.45,.12]),np.array([.12,.55,.8]))
    color[data['fixed']]=[.55,.55,.55]
    values=np.broadcast_to((color-.5)[:,None,None,:],(n,8,8,3)).copy()
    values+=.025*np.sin(5*uv[:,0])[None,:,None,None]
    base=dict(density=np.full(n,60.),texel_sites=np.tile(uv,(n,1,1)),
        texel_height=np.tile(.6+.025*np.sin(4*uv[:,0]),(n,1)),texel_sv_rgb=values.reshape(n,8,24))
    images=[]; alpha_sums=[]
    for k,(p,F) in enumerate(zip(data['positions'],data['deformation'])):
        U,s,Vt=np.linalg.svd(F); R=U@Vt
        if np.any(np.linalg.det(R)<0): raise ValueError('Improper polar rotation')
        radii=r0*np.linalg.det(F)**(1/3) if radii_sequence is None else radii_sequence[k]
        state=dict(base,points=p,radii=radii+np.log(-np.expm1(-100*radii))/100,
            quaternions=np.roll(Rotation.from_matrix(frame0@R.transpose(0,2,1)).as_quat(),1,axis=1),
            texel_sv_axis=np.broadcast_to(np.einsum('ndc,kc->nkd',R,axes)[:,None,:,:],(n,8,8,3)).reshape(n,8,24))
        load_state(model,state)
        result=render(model,cam)
        if not np.isfinite(result['rgb']).all(): raise ValueError('Nonfinite renderer result')
        if result['alpha'].max()<.9: raise ValueError('Empty render')
        rgb=result['rgb']+(1-result['alpha'][...,None])*np.array([.96,.97,.98])
        im=Image.fromarray((255*np.clip(rgb,0,1)).astype(np.uint8))
        draw=ImageDraw.Draw(im)
        draw.text((12,12),f'TLSPH -> original Power Foam | {frame_label}={data["times"][k]:.3f}',fill=(20,20,20))
        draw.text((12,30),'Synthetic beam; '+label,fill=(20,20,20))
        images.append(im); alpha_sums.append(float(result['alpha'].sum()))
        if k%16==0 or len(data['times'])<=16: im.save(out/f'foam_{k:03d}.png')
    images[0].save(out/'powerfoam.gif',save_all=True,append_images=images[1:],duration=frame_duration_ms,loop=0)
    sheet=Image.new('RGB',(512*3,320),'white')
    shown=[0,16,32] if len(images)>32 else np.linspace(0,len(images)-1,3).astype(int)
    for col,k in enumerate(shown): sheet.paste(images[k],(512*col,0))
    sheet.save(out/'powerfoam_frames.png')
    (out/'render.json').write_text(json.dumps(dict(frames=len(images),resolution=[512,320],
        renderer='PowerfoamScene.forward, original dipole/detail/SV/rasterizer, per-frame adjacency rebuild',
        optical_density=60, radius_method=label, frame_parameter=frame_label,auto_frame=auto_frame,
        alpha_sum_range=[min(alpha_sums),max(alpha_sums)],
        scope='Hand-authored foam, no image optimization, no claim of material texture transport accuracy.'),indent=2)+'\n')
    print('Saved original-renderer frames and animation.',flush=True)
    return images


if __name__=='__main__': main()
