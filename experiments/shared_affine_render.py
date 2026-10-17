"""Validate the shared-metric GPU renderer against an independent float64 oracle.

Run with the existing om environment: python -m experiments.shared_affine_render
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.affine_probe import GPURenderer, membership, save_image, scene, transforms
from powerfoam.affine_raytrace import SharedAffineRayTracer
import matplotlib.pyplot as plt


def camera_rays(resolution, eye=(0., 0., 8.)):
    eye = np.array(eye, dtype=float)
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0., 1., 0.]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    x, y = np.meshgrid(np.linspace(-3/8, 3/8, resolution), np.linspace(3/8, -3/8, resolution))
    rays = forward + x[..., None]*right + y[..., None]*up
    rays = rays.reshape(-1, 3); rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return np.broadcast_to(eye, rays.shape).copy(), rays


def oracle(origins, directions, f, translation, centers, radii, normals, sites, colors, density):
    """Canonical rays, ALL power faces, per-ray intersection sorting, float64.

    Unlike the GPU implementation this neither uses M in world space nor relies
    on a Cech graph or camera-global painter order. Preserve world ray parameter.
    """
    inv = np.linalg.inv(f)
    eye = (origins-translation) @ inv.T
    ray = directions @ inv.T  # Do NOT normalize: t must remain world distance.
    n_rays, n_cells = len(eye), len(centers)
    t_entry = np.full((n_rays, n_cells), np.inf)
    t_exit = np.zeros((n_rays, n_cells))
    rgb_cells = np.zeros((n_rays, n_cells, 3))
    a = np.sum(ray*ray, axis=1)
    for i in range(n_cells):
        offset = eye-centers[i]
        mid = -np.sum(offset*ray, axis=1)/a
        nearest = offset + mid[:, None]*ray
        residual = radii[i]**2 - np.sum(nearest*nearest, axis=1)
        half = np.sqrt(np.maximum(residual, 0)/a)
        near, far = np.maximum(0, mid-half), mid+half
        selected = np.flatnonzero((residual > 0) & (far > near))
        if not len(selected):
            continue
        lo, hi = near[selected], far[selected]
        face_n = centers-centers[i]
        face_b = .5*(np.sum(centers*centers, axis=1)-np.dot(centers[i],centers[i])+radii[i]**2-radii**2)
        numerator = face_b[None]-eye[selected]@face_n.T
        denominator = ray[selected]@face_n.T
        parallel = np.abs(denominator) < 1e-12
        rejected = np.any(parallel & (numerator < -1e-12), axis=1)
        root = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=~parallel)
        lo = np.maximum(lo, np.max(np.where(denominator < -1e-12, root, -np.inf), axis=1))
        hi = np.minimum(hi, np.min(np.where(denominator > 1e-12, root, np.inf), axis=1))
        numerator = (centers[i]-eye[selected])@normals[i]
        denominator = ray[selected]@normals[i]
        parallel = np.abs(denominator) < 1e-12
        rejected |= parallel & (numerator < -1e-12)
        root = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=~parallel)
        lo = np.maximum(lo, np.where(denominator < -1e-12, root, -np.inf))
        hi = np.minimum(hi, np.where(denominator > 1e-12, root, np.inf))
        valid = (hi > lo) & ~rejected
        idx = selected[valid]
        t_entry[idx, i], t_exit[idx, i] = lo[valid], hi[valid]
        query = eye[idx] + lo[valid, None]*ray[idx]
        weights = np.exp(-10*np.sum((query[:, None]-sites[i])**2, axis=-1)/radii[i]**2)
        rgb_cells[idx, i] = weights@colors[i]/np.maximum(weights.sum(1, keepdims=True), 1e-20)
    order = np.argsort(t_entry, axis=1)
    entry = np.take_along_axis(t_entry, order, axis=1)
    exit = np.take_along_axis(t_exit, order, axis=1)
    length = np.maximum(0, exit-entry)
    alpha = -np.expm1(-density*length)
    trans = np.cumprod(np.column_stack((np.ones(n_rays), 1-alpha[:, :-1])), axis=1)
    contribution = trans*alpha
    rgb = np.sum(contribution[..., None]*np.take_along_axis(rgb_cells, order[..., None], axis=1), axis=1)
    world_normals = normals@inv
    world_normals /= np.linalg.norm(world_normals, axis=1, keepdims=True)
    normal = np.sum(contribution[..., None]*world_normals[order], axis=1)
    hit = np.isfinite(entry[:, 0])
    owner = np.where(hit, order[:, 0], -1)
    depth = np.where(hit, entry[:, 0], 0.)
    return dict(rgb=rgb, alpha=1-np.prod(1-alpha, axis=1), depth=depth, owner=owner, normal=normal)


def compare(actual, expected):
    occupied = (expected['alpha'] > .001) & (actual['alpha'] > .001)
    rgb_error = np.abs(actual['rgb']-expected['rgb'])
    result = dict(rgb_mae=float(rgb_error.mean()), rgb_max_error=float(rgb_error.max()),
        alpha_mae=float(np.abs(actual['alpha']-expected['alpha']).mean()),
        depth_mae_on_hits=float(np.abs(actual['depth']-expected['depth'])[occupied].mean()),
        first_cell_mismatch_pct=100*float((actual['owner'][occupied]!=expected['owner'][occupied]).mean()),
        hit_mask_mismatch_pct=100*float(((actual['alpha']>.001)!=(expected['alpha']>.001)).mean()),
        normal_mae=float(np.abs(actual['normal']-expected['normal']).mean()))
    for key in ('rgb','alpha','depth','normal'):
        assert np.isfinite(actual[key]).all(), key
    assert result['rgb_mae'] < 1e-4, result
    assert result['alpha_mae'] < 1e-4, result
    assert result['depth_mae_on_hits'] < 1e-3, result
    assert result['first_cell_mismatch_pct'] < .1, result
    assert result['hit_mask_mismatch_pct'] < .02, result
    assert result['normal_mae'] < 1e-4, result
    return result


def metric_membership_check(f, translation, p0, r0):
    rng = np.random.default_rng(2026)
    x0 = rng.uniform([-1.4,-1.4,-.3],[1.4,1.4,.3],(16384,3))
    inverse = np.linalg.inv(f); metric=inverse.T@inverse
    delta = (x0@f.T+translation)[:,None]-(p0@f.T+translation)
    phi = np.einsum('nki,ij,nkj->nk',delta,metric,delta)-r0*r0
    owner = phi.argmin(1); owner[phi[np.arange(len(phi)),owner]>0]=-1
    original = membership(x0,p0,r0)
    assert np.array_equal(owner,original)
    return int((owner!=original).sum())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('output/shared_affine'))
    parser.add_argument('--resolution',type=int,default=192)
    parser.add_argument('--animation-frames',type=int,default=32)
    args=parser.parse_args()
    if args.resolution<32 or args.animation_frames<0: parser.error('Invalid resolution/frame count')
    args.out.mkdir(parents=True,exist_ok=True)
    p,r,sites,colors=scene(7,False)
    normals=np.tile([0.,0.,1.],(len(p),1))
    tracer=SharedAffineRayTracer(p,r,normals,sites,colors)
    baseline=GPURenderer(args.resolution)
    origins,directions=camera_rays(args.resolution)
    cases=transforms()
    angle=np.deg2rad(35.)
    rx=np.array([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],[0.,np.sin(angle),np.cos(angle)]])
    cases['tilted_stretch']=(rx@np.diag([1.5,.65,1.2]),np.array([.1,-.05,.15]))
    fig,axes=plt.subplots(len(cases),4,figsize=(13,3*len(cases)),squeeze=False)
    records=[]
    for row,(name,(f,t)) in enumerate(cases.items()):
        print(f'Rendering {name}',flush=True)
        expected=oracle(origins,directions,f,t,p,r,normals,sites,colors,1000.)
        actual=tracer.render(origins,directions,f,t)
        stats=compare(actual,expected)
        stats.update(case=name,membership_mismatches=metric_membership_check(f,t,p,r))
        old_rgb,_=baseline.render(p@f.T+t,r*np.cbrt(np.linalg.det(f)),sites@f.T+t,colors,f)
        shape=(args.resolution,args.resolution,3)
        ref_image,our_image=expected['rgb'].reshape(shape),actual['rgb'].reshape(shape)
        stats['old_volume_radius_rgb_mae']=float(np.abs(old_rgb-ref_image).mean())
        if name=='identity':
            assert stats['old_volume_radius_rgb_mae'] < .002, stats
        for col,(title,im) in enumerate([
            (f'{name}\nFloat64 reference rays',ref_image),
            ('Original spherical cells\nvolume radius',old_rgb),
            ('Shared-metric Foam\nworld-space GPU',our_image),
            ('GPU/reference error x100',np.clip(np.abs(our_image-ref_image)*100,0,1))]):
            axes[row,col].imshow(im);axes[row,col].set_title(title);axes[row,col].axis('off')
        save_image(args.out/f'{name}_reference.png',ref_image)
        save_image(args.out/f'{name}_metric_gpu.png',our_image)
        save_image(args.out/f'{name}_old_gpu.png',old_rgb)
        save_image(args.out/f'{name}_alpha.png',actual['alpha'].reshape(args.resolution,args.resolution))
        records.append(stats)
        print(json.dumps(stats),flush=True)
    fig.suptitle('Actual volume rendering: shared anisotropic metric vs spherical radius update')
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(args.out/'comparison.png',dpi=150);plt.close(fig)
    # Transparency makes world-distance parameter mistakes observable, unlike
    # saturated opacity. A 3D tilt also exercises transformed dipole normals.
    transparent=SharedAffineRayTracer(p,r,normals,sites,colors,density=8.)
    f,t=cases['tilted_stretch']
    o,d=camera_rays(96,eye=(2.,-1.,7.))
    expected=oracle(o,d,f,t,p,r,normals,sites,colors,8.)
    transparent_stats=compare(transparent.render(o,d,f,t),expected)
    print('Transparent oblique check',transparent_stats,flush=True)
    # Include rays starting inside and outside the support, parallel to dipoles,
    # and arbitrary origins, which require independent painter orders.
    rng=np.random.default_rng(54)
    o=rng.uniform([-1.5,-1.5,-.3],[1.5,1.5,.3],(1024,3))@f.T+t
    d=rng.normal(size=o.shape);d/=np.linalg.norm(d,axis=1,keepdims=True)
    d[:32]=np.array([1.,0.,0.])@f.T
    d/=np.linalg.norm(d,axis=1,keepdims=True)
    arbitrary_stats=compare(transparent.render(o,d,f,t),oracle(o,d,f,t,p,r,normals,sites,colors,8.))
    print('Arbitrary-origin check',arbitrary_stats,flush=True)
    # A genuinely volumetric arrangement with different dipole orientations
    # exercises occlusion/interval ordering beyond the coplanar checker patch.
    vp=rng.uniform(-.8,.8,(32,3)); vr=rng.uniform(.3,.5,32)
    vn=rng.normal(size=vp.shape);vn/=np.linalg.norm(vn,axis=1,keepdims=True)
    tangent=np.cross(vn,[1.,0.,0.]);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    bitangent=np.cross(vn,tangent)
    offsets=rng.uniform(-.15,.15,(32,8,2))
    vs=vp[:,None]+offsets[...,:1]*tangent[:,None]+offsets[...,1:]*bitangent[:,None]
    vc=rng.uniform(.05,.95,(32,8,3))
    vf=np.array([[1.3,.3,.2],[.1,.8,.1],[0.,.2,1.1]])
    vt=np.array([.2,-.1,.1])
    o,d=camera_rays(64,eye=(2.,1.,5.))
    volumetric=SharedAffineRayTracer(vp,vr,vn,vs,vc,density=8.)
    volumetric_stats=compare(volumetric.render(o,d,vf,vt),oracle(o,d,vf,vt,vp,vr,vn,vs,vc,8.))
    print('Noncoplanar cell check',volumetric_stats,flush=True)
    invalid_rejected=0
    for invalid in (np.zeros((3,3)),np.diag([-1.,1.,1.])):
        try: tracer.render(origins[:1],directions[:1],invalid)
        except ValueError: invalid_rejected+=1
    assert invalid_rejected==2
    frames=[]
    for frame in range(args.animation_frames):
        phase=2*np.pi*frame/max(1,args.animation_frames)
        scale=np.exp(.55*np.sin(phase))
        shear=.8*np.sin(phase+np.pi/3)
        f=np.array([[scale,shear,0.],[0.,1/scale,0.],[0.,0.,1.]])
        new=tracer.render(origins,directions,f)['rgb'].reshape(args.resolution,args.resolution,3)
        old,_=baseline.render(p@f.T,r,sites@f.T,colors,f)
        canvas=Image.new('RGB',(2*args.resolution,args.resolution+36),'white')
        draw=ImageDraw.Draw(canvas)
        draw.text((8,8),'Spherical / volume radius',fill='black')
        draw.text((args.resolution+8,8),'Shared-metric Power Foam',fill='black')
        for col,im in enumerate((old,new)):
            canvas.paste(Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)),(col*args.resolution,36))
        frames.append(canvas)
    if frames:
        frames[0].save(args.out/'motion.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
    results=dict(cases=records,transparent_oblique=transparent_stats,arbitrary_origins=arbitrary_stats,
        noncoplanar_cells=volumetric_stats,
        settings=dict(resolution=args.resolution,animation_frames=args.animation_frames,cells=len(p),
                      geometry='one shared F, world ellipsoids and metric-generated clipping planes',
                      texture='canonical metric, frozen texel colors, zero displacement',
                      density='fixed world-length extinction; 1000 opaque and 8 transparent',
                      oracle='float64 inverse rays, all cell constraints, independently sorted intervals',
                      limitations='no per-cell F, no MPM, no backward pass or speed benchmark'))
    (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'All checks passed. Saved {args.out.resolve()}',flush=True)


if __name__=='__main__':
    main()
