"""Spatially varying F: exact quadratic ray events and analytic bend/twist probes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.affine_probe import GPURenderer, membership, save_image, scene, texture
from experiments.shared_affine_render import camera_rays, oracle
from powerfoam.quadratic_raytrace import QuadraticRayTracer
import matplotlib.pyplot as plt


def deformation(x, kind, strength):
    x=np.asarray(x); current=x.copy(); f=np.broadcast_to(np.eye(3),x.shape[:-1]+(3,3)).copy()
    if kind=='bend':
        frequency=np.pi/2
        current[...,2]+=strength*np.sin(frequency*x[...,0])
        f[...,2,0]=strength*frequency*np.cos(frequency*x[...,0])
    elif kind=='twist':
        angle=strength*x[...,0];c,s=np.cos(angle),np.sin(angle)
        current[...,1]=c*x[...,1]-s*x[...,2]
        current[...,2]=s*x[...,1]+c*x[...,2]
        f[...,1,0]=-strength*current[...,2];f[...,2,0]=strength*current[...,1]
        f[...,1,1]=c;f[...,1,2]=-s;f[...,2,1]=s;f[...,2,2]=c
    else: raise ValueError(kind)
    return current,f


def inverse_map(x,kind,strength):
    xx,yy,zz=x.unbind(-1)
    if kind=='bend':return torch.stack((xx,yy,zz-strength*torch.sin(np.pi/2*xx)),-1)
    angle=strength*xx;c,s=torch.cos(angle),torch.sin(angle)
    return torch.stack((xx,c*yy+s*zz,-s*yy+c*zz),-1)


def refined_scene(grid):
    if grid==11:return scene(7,False)
    rng=np.random.default_rng(7);h=2.2/(grid-1)
    x,y=np.meshgrid(np.linspace(-1.1,1.1,grid),np.linspace(-1.1,1.1,grid))
    p=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    p[:,:2]+=rng.uniform(-.025,.025,(len(p),2))*h/.22
    r=h*rng.uniform(.8,1.2,len(p))
    tx,ty=np.meshgrid(np.linspace(-.65,.65,4),[-.45,.45])
    offsets=np.column_stack((tx.ravel(),ty.ravel(),np.zeros(8)))
    sites=p[:,None]+r[:,None,None]*offsets
    parity=np.floor((sites[...,:2]+1.6)/.32).sum(-1).astype(int)%2
    return p,r,sites,np.array([[.95,.78,.20],[.07,.22,.65]])[parity]


def local_membership(query,p,r,f):
    inv=np.linalg.inv(f); owners=[]
    for start in range(0,len(query),2048):
        local=np.einsum('nij,bnj->bni',inv,query[start:start+2048,None]-p)
        value=(local*local).sum(2)-r*r
        owner=value.argmin(1);owner[value[np.arange(len(owner)),owner]>0]=-1
        owners.append(owner)
    return np.concatenate(owners)


def material_metrics(p0,r,sites,colors,kind,strength,points):
    p,f=deformation(p0,kind,strength);x,_=deformation(points,kind,strength)
    old=membership(points,p0,r);new=local_membership(x,p,r,f)
    inside=old>=0;covered=inside&(new>=0)
    baseline=membership(x,p,r)  # det(F)=1 in both analytic deformations
    safe=np.maximum(new,0)
    local=p0[safe]+np.einsum('bij,bj->bi',np.linalg.inv(f)[safe],x-p[safe])
    rgb=texture(local,new,sites,colors,r)
    reference=texture(points,old,sites,colors,r)
    counts=np.bincount(new[new>=0],minlength=len(p))
    result=dict(identity_loss_pct=100*float((new[inside]!=old[inside]).mean()),
                support_miss_pct=100*float((new[inside]<0).mean()),
                switch_pct_covered=100*float((new[covered]!=old[covered]).mean()),
                spherical_identity_loss_pct=100*float((baseline[inside]!=old[inside]).mean()),
                texture_mae=float(np.abs(rgb-reference)[inside].mean()),
                cells_without_probe_samples=int((counts==0).sum()))
    # Sampling can only detect unobserved cells, not certify mathematically empty cells.
    return result,rgb,new


@torch.no_grad()
def transported_reference(origins,directions,p,r,sites,colors,kind,strength,steps=1024,density=1000.):
    """Convergent ray-march oracle for the EXACT nonlinear transported field.

    This is distinct from exact rendering of the local-F approximation. It holds
    entry color within each sampled material-cell interval, matching the prototype
    shading convention in the limit of step refinement.
    """
    low=(p-r[:,None]).min(0);high=(p+r[:,None]).max(0)
    if kind=='bend':low[2]-=abs(strength);high[2]+=abs(strength)
    else:
        extent=np.hypot(max(abs(low[1]),abs(high[1])),max(abs(low[2]),abs(high[2])))
        low[1:]=-extent;high[1:]=extent
    parallel=np.abs(directions)<1e-12
    a=(low-origins)/np.where(parallel,1.,directions)
    b=(high-origins)/np.where(parallel,1.,directions)
    near=np.maximum(0,np.where(parallel,-np.inf,np.minimum(a,b)).max(1))
    far=np.where(parallel,np.inf,np.maximum(a,b)).min(1)
    hit=(far>near)&~np.any(parallel&((origins<low)|(origins>high)),axis=1)
    output=np.zeros((len(origins),3));alpha_output=np.zeros(len(origins))
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float64,device='cuda:0')
    cp,cr,cs,cc=map(tensor,(p,r,sites,colors))
    sample=(torch.arange(steps,device='cuda:0',dtype=torch.float64)+.5)/steps
    idx=np.flatnonzero(hit)
    for start in range(0,len(idx),64):
        selected=idx[start:start+64]
        o,d=map(tensor,(origins[selected],directions[selected]))
        t0,dt=map(tensor,(near[selected],far[selected]-near[selected]))
        t=t0[:,None]+dt[:,None]*sample
        canonical=inverse_map(o[:,None]+t[:,:,None]*d[:,None],kind,strength)
        flat=canonical.reshape(-1,3)
        power=(flat.square().sum(1)[:,None]+cp.square().sum(1)[None]-2*flat@cp.T-cr.square())
        val,owner=power.min(1);owner=owner.reshape(len(selected),steps)
        owner=torch.where((val.reshape_as(owner)<=0)&(canonical[...,2]<=0),owner,-1)
        change=torch.ones_like(owner,dtype=torch.bool);change[:,1:]=owner[:,1:]!=owner[:,:-1]
        entry_index=torch.where(change,torch.arange(steps,device='cuda:0')[None],0).cummax(1).values
        query=canonical.gather(1,entry_index[:,:,None].expand(-1,-1,3))
        safe=owner.clamp_min(0)
        weights=torch.exp(-10*(query[:,:,None]-cs[safe]).square().sum(3)/cr[safe,None].square())
        rgb=(weights[:,:,:,None]*cc[safe]).sum(2)/weights.sum(2,keepdim=True).clamp_min(1e-20)
        tau=(owner>=0)*(density*dt[:,None]/steps)
        trans=torch.exp(-(tau.cumsum(1)-tau))
        alpha=-torch.expm1(-tau)
        output[selected]=(trans[:,:,None]*alpha[:,:,None]*rgb).sum(1).cpu().numpy()
        alpha_output[selected]=(-torch.expm1(-tau.sum(1))).cpu().numpy()
    return output,alpha_output


def validate_renderer():
    # A shell cell has two separated intervals. A single-interval clipper fails.
    p=np.zeros((2,3));r=np.array([2.,3.]);n=np.tile([0.,0.,1.],(2,1))
    sites=np.zeros((2,1,3));colors=np.array([[[1.,0.,0.]],[[0.,0.,1.]]])
    tracer=QuadraticRayTracer(p,r,n,sites,colors,density=.4)
    f=np.stack((np.eye(3),.5*np.eye(3)))
    out=tracer.render([[-3.,0.,-.1]],[[1.,0.,0.]],p,f,return_intervals=True)
    segments=out['intervals'][0]
    assert [s[2] for s in segments]==[0,1,0]
    extent=np.sqrt(4-.01);cross=np.sqrt((5-.03)/3)
    expected=np.array([[3-extent,3-cross],[3-cross,3+cross],[3+cross,3+extent]])
    np.testing.assert_allclose(np.array(segments)[:,:2],expected,atol=1e-10)
    np.testing.assert_allclose(out['alpha'][0],1-np.exp(-.4*2*extent),atol=1e-12)
    # Shared F must reduce to the previous independent float64 oracle.
    p,r,sites,colors=scene(7,False);n=np.tile([0.,0.,1.],(len(p),1))
    f=np.array([[1.4,.35,.1],[0.,.8,.15],[.1,0.,1.]])
    t=np.array([.1,-.1,.05]);o,d=camera_rays(48,eye=(1.,-1.,7.))
    tracer=QuadraticRayTracer(p,r,n,sites,colors,density=8.)
    got=tracer.render(o,d,p@f.T+t,np.tile(f,(len(p),1,1)))
    ref=oracle(o,d,f,t,p,r,n,sites,colors,8.)
    max_error=float(np.abs(got['rgb']-ref['rgb']).max())
    assert max_error<1e-8,max_error
    # Variable-F intervals versus direct pointwise power classification.
    pcurrent,fs=deformation(p,'twist',.7)
    rng=np.random.default_rng(42)
    o=rng.uniform([-1.2,-1.2,-1.],[1.2,1.2,1.],(48,3))
    d=rng.normal(size=o.shape);d/=np.linalg.norm(d,axis=1,keepdims=True)
    got=tracer.render(o,d,pcurrent,fs,return_intervals=True)
    samples=np.linspace(.000123,4.000123,2048)
    q=(o[:,None]+samples[None,:,None]*d[:,None]).reshape(-1,3)
    labels=local_membership(q,pcurrent,r,fs)
    safe=np.maximum(labels,0)
    local=np.einsum('bij,bj->bi',np.linalg.inv(fs)[safe],q-pcurrent[safe])
    labels[(local[:,2]>0)|(labels<0)]=-1
    predicted=np.full((len(o),len(samples)),-1,dtype=int)
    for i,segments in enumerate(got['intervals']):
        for lo,hi,owner in segments:predicted[i,(samples>=lo)&(samples<hi)]=owner
    assert np.array_equal(predicted.ravel(),labels)
    # Check F is the actual Jacobian, and the nonlinear map is invertible.
    x=rng.uniform(-1.,1.,(100,3));jacobian_error=0.
    for kind in ('bend','twist'):
        current,jac=deformation(x,kind,.4)
        np.testing.assert_allclose(inverse_map(torch.tensor(current),kind,.4).numpy(),x,atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(jac),1.,atol=1e-12)
        for axis in range(3):
            delta=np.zeros(3);delta[axis]=1e-6
            fd=(deformation(x+delta,kind,.4)[0]-deformation(x-delta,kind,.4)[0])/2e-6
            jacobian_error=max(jacobian_error,float(np.abs(fd-jac[:,:,axis]).max()))
    assert jacobian_error<1e-8
    return dict(shell_intervals=expected.tolist(),shared_F_max_rgb_error=max_error,
                interval_point_checks=48*2048,interval_point_mismatches=0,jacobian_max_error=jacobian_error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('output/varying_metric'))
    parser.add_argument('--resolution',type=int,default=128)
    parser.add_argument('--reference-steps',type=int,default=1024)
    parser.add_argument('--animation-frames',type=int,default=16)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.resolution<32 or args.reference_steps<128 or args.animation_frames<0:parser.error('Invalid dimensions')
    checks=validate_renderer();print('Renderer checks:',checks,flush=True)
    p,r,sites,colors=scene(7,False);normals=np.tile([0.,0.,1.],(len(p),1))
    renderer=QuadraticRayTracer(p,r,normals,sites,colors)
    old_renderer=GPURenderer(args.resolution)
    o,d=camera_rays(args.resolution)
    x,y=np.meshgrid(np.linspace(-.95,.95,192),np.linspace(.95,-.95,192))
    surface=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    rng=np.random.default_rng(2026)
    volume=rng.uniform((p-r[:,None]).min(0),(p+r[:,None]).max(0),(32768,3))
    cases=[('identity','bend',0.),('bend_mild','bend',.2),('bend_strong','bend',.4),('twist','twist',.6)]
    fig,axes=plt.subplots(len(cases),3,figsize=(11,12),squeeze=False)
    records=[]
    for row,(name,kind,strength) in enumerate(cases):
        print('Rendering',name,flush=True)
        current,f=deformation(p,kind,strength)
        result=renderer.render(o,d,current,f)
        reference,alpha=transported_reference(o,d,p,r,sites,colors,kind,strength,args.reference_steps)
        normal=np.einsum('ni,nij->nj',normals,np.linalg.inv(f));normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        tex=current[:,None]+np.einsum('nij,nkj->nki',f,sites-p[:,None])
        old,_=old_renderer.render(current,r,tex,colors,np.eye(3),explicit_normals=normal)
        sm,_,_=material_metrics(p,r,sites,colors,kind,strength,surface)
        vm,_,_=material_metrics(p,r,sites,colors,kind,strength,volume)
        active=(alpha>.01)|(result['alpha']>.01)
        record=dict(case=name,kind=kind,strength=strength,surface=sm,volume=vm,
            reference_rgb_mae=float(np.abs(result['rgb']-reference).mean()),
            foreground_rgb_mae=float(np.abs(result['rgb']-reference)[active].mean()),
            spherical_reference_rgb_mae=float(np.abs(old.reshape(-1,3)-reference).mean()),
            max_ray_segments=int(result['segments'].max()),max_reentries=int(result['max_reentries'].max()),
            rays_with_reentries=int((result['max_reentries']>0).sum()))
        for col,(title,image) in enumerate([('Exact map (sampled reference)',reference.reshape(args.resolution,args.resolution,3)),
                  ('Spherical Foam',old),('Per-cell metric Foam',result['rgb'].reshape(args.resolution,args.resolution,3))]):
            axes[row,col].imshow(image);axes[row,col].set_title(name+'\n'+title);axes[row,col].axis('off')
            save_image(args.out/f'{name}_{col}.png',image)
        records.append(record);print(json.dumps(record),flush=True)
    fig.tight_layout();fig.savefig(args.out/'comparison.png',dpi=150);plt.close(fig)
    # Quantify reference integration error separately from representation error.
    ro,rd=camera_rays(64)
    ref_a,_=transported_reference(ro,rd,p,r,sites,colors,'twist',.6,args.reference_steps)
    ref_b,_=transported_reference(ro,rd,p,r,sites,colors,'twist',.6,args.reference_steps*2)
    convergence=dict(case='twist',resolution=64,steps=[args.reference_steps,args.reference_steps*2],
                     rgb_mae=float(np.abs(ref_a-ref_b).mean()),max_error=float(np.abs(ref_a-ref_b).max()))
    print('Reference refinement:',convergence,flush=True)
    refinement=[]
    for grid in (7,11,17):
        cp,cr,cs,cc=refined_scene(grid)
        for kind,strength in [('bend',.4),('twist',.6)]:
            m,_,_=material_metrics(cp,cr,cs,cc,kind,strength,surface)
            refinement.append(dict(grid=grid,cells=len(cp),case=kind,**m))
            print('Cell refinement',refinement[-1],flush=True)
    frames=[];temporal=[];previous_rgb=None
    for frame in range(args.animation_frames):
        strength=.65*np.sin(2*np.pi*frame/max(1,args.animation_frames))
        current,f=deformation(p,'twist',strength)
        result=renderer.render(o,d,current,f)
        normal=np.einsum('ni,nij->nj',normals,np.linalg.inv(f));normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        tex=current[:,None]+np.einsum('nij,nkj->nki',f,sites-p[:,None])
        old,_=old_renderer.render(current,r,tex,colors,np.eye(3),explicit_normals=normal)
        metric,material_rgb,_=material_metrics(p,r,sites,colors,'twist',strength,surface)
        temporal.append(dict(frame=frame,strength=float(strength),**metric,
            material_color_step_mae=None if previous_rgb is None else float(np.abs(material_rgb-previous_rgb).mean())))
        previous_rgb=material_rgb
        canvas=Image.new('RGB',(2*args.resolution,args.resolution+36),'white');draw=ImageDraw.Draw(canvas)
        draw.text((4,8),'Spherical Foam',fill='black');draw.text((args.resolution+4,8),'Per-cell F Foam',fill='black')
        for col,im in enumerate((old,result['rgb'].reshape(args.resolution,args.resolution,3))):
            canvas.paste(Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)),(col*args.resolution,36))
        frames.append(canvas)
        print(f'Animation {frame+1}/{args.animation_frames}',flush=True)
    if frames:frames[0].save(args.out/'motion.gif',save_all=True,append_images=frames[1:],duration=130,loop=0)
    results=dict(checks=checks,cases=records,reference_refinement=convergence,cell_refinement=refinement,
                 temporal=temporal,settings=dict(resolution=args.resolution,reference_steps=args.reference_steps,
                    representation='one local Jacobian per cell; all roots and occupied intervals evaluated',
                    reference='known nonlinear material map; RGB reference uses convergent discrete ray integration',
                    limitations='not MPM, no learning, one seed, no real-time claim; unobserved cells are not certified empty'))
    (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Saved',args.out.resolve(),flush=True)


if __name__=='__main__':main()
