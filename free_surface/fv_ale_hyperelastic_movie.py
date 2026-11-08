"""CPU orthographic ray/sphere rendering of the actual free-body ball union.

No surface smoothing, outlier removal, interpolation or deformation rescaling.
Without vacuum boundaries the union of power-clipped balls equals the ball union.
With vacuum generators, intersect each ball with all its vacuum power planes;
their union is exactly the material domain (no internal real planes needed).
For fixed-horizontal-plane impact tests, clip that union at the same plane.
The camera rotates only about y, so the plane mask is exact along every ray.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--frames-path',type=Path);ap.add_argument('--report-path',type=Path)
    ap.add_argument('--output-prefix',type=Path)
    ap.add_argument('--size',type=int,default=640);args=ap.parse_args()
    root=Path('output/ale');frames=np.load(args.frames_path or root/(args.tag+'_frames.npz'))
    info=json.loads((args.report_path or root/(args.tag+'.json')).read_text())
    completed=info.get('completed',info.get('status')=='completed')
    prefix=args.output_prefix or root/args.tag
    p=frames['p'] if 'p' in frames else frames['x'];r=frames['r'];times=frames['t'];size=args.size
    ground=info.get('ground') if info.get('case')=='impact' else None
    if 'ground_clipped' in frames and bool(frames['ground_clipped']):ground=float(frames['ground_height'])
    angle=.55;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    center=.5*(p[0].min(0)+p[0].max(0));projected=(p-center)@R.T
    low=(projected-r[...,None]).min((0,1));high=(projected+r[...,None]).max((0,1))
    width=max(high[0]-low[0],high[1]-low[1])*1.14
    xycenter=.5*(low[:2]+high[:2]);scale=(size-1)/width
    light=np.array([-.4,.65,.65]);light/=np.linalg.norm(light)
    background=np.array([242,243,245],dtype=np.uint8)
    base=np.array([.22,.57,.78]);images=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for k,t in enumerate(times):
        depth=np.full((size,size),-np.inf);normal=np.zeros((size,size,3))
        vacuum=None
        if 'plane_count' in frames and frames['plane_count'][k]>0:
            count=int(frames['plane_count'][k]);owners=frames['plane_owner'][k,:count]
            d=frames['plane_d'][k,:count];a=frames['plane_a'][k,:count];b=frames['plane_b'][k,:count]
            q=p[k,owners]+d[:,None]*a
            weight=r[k,owners]**2+d*d*(a*a).sum(1)-2*d*(b-(a*p[k,owners]).sum(1))
            vacuum=((q-center)@R.T,weight)
        for c,radius in zip(projected[k],r[k]):
            if radius<=0 or not np.isfinite(c).all():continue
            cx=(c[0]-xycenter[0])*scale+.5*(size-1)
            cy=-(c[1]-xycenter[1])*scale+.5*(size-1)
            radius_px=radius*scale
            x0=max(0,int(np.floor(cx-radius_px)));x1=min(size,int(np.ceil(cx+radius_px))+1)
            y0=max(0,int(np.floor(cy-radius_px)));y1=min(size,int(np.ceil(cy+radius_px))+1)
            if x0>=x1 or y0>=y1:continue
            xx=(np.arange(x0,x1)-cx)[None,:]/scale
            yy=-(np.arange(y0,y1)-cy)[:,None]/scale
            zz2=radius*radius-xx*xx-yy*yy;inside=zz2>=0
            if ground is not None:
                inside=inside&((yy+c[1]+center[1])>=ground)
            zz=np.sqrt(np.maximum(zz2,0));upper=zz.copy();lower=-zz.copy()
            nx=np.broadcast_to(xx,zz.shape).copy()/radius
            ny=np.broadcast_to(yy,zz.shape).copy()/radius;nz=zz/radius
            if vacuum is not None:
                delta=vacuum[0]-c;length=np.linalg.norm(delta,axis=1)
                normals=delta/length[:,None]
                heights=(length**2+radius**2-vacuum[1])/(2*length)
                for nn,h in zip(normals,heights):
                    if h>=radius:continue
                    if h<=-radius:inside[:]=False;break
                    bound=h-nn[0]*xx-nn[1]*yy
                    if nn[2]>0:
                        limit=bound/nn[2];changed=limit<upper;upper=np.minimum(upper,limit)
                        nx[changed]=nn[0];ny[changed]=nn[1];nz[changed]=nn[2]
                    elif nn[2]<0:lower=np.maximum(lower,bound/nn[2])
                    else:inside&=bound>=0
                inside&=upper>=lower
            front=c[2]+upper
            old=depth[y0:y1,x0:x1];visible=inside&(front>old)
            old[visible]=front[visible]
            N=normal[y0:y1,x0:x1]
            N[:,:,0][visible]=nx[visible]
            N[:,:,1][visible]=ny[visible]
            N[:,:,2][visible]=nz[visible]
        occupied=np.isfinite(depth)
        diffuse=np.maximum(normal@light,0)
        half=light+np.array([0.,0.,1.]);half/=np.linalg.norm(half)
        specular=np.maximum(normal@half,0)**40
        rgb=np.broadcast_to(background,(size,size,3)).copy()
        colour=base[None,None,:]*(.32+.68*diffuse[:,:,None])+.16*specular[:,:,None]
        rgb[occupied]=(255*np.clip(colour[occupied],0,1)).astype(np.uint8)
        image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
        if ground is not None:
            floor_pixel=int(round(-(ground-center[1]-xycenter[1])*scale+.5*(size-1)))
            draw.line([(0,floor_pixel),(size-1,floor_pixel)],fill=(100,105,112),width=2)
        title='Neo-Hookean FV | actual clipped domain' if ground is not None else 'Neo-Hookean FV | actual sphere union'
        draw.text((16,12),title,font=font,fill=(30,35,40))
        draw.text((16,38),f't = {1000*float(t):.2f} ms   (slow motion)',font=small,fill=(55,60,65))
        draw.text((16,size-28),'Fixed camera and scale; no outliers removed',font=small,fill=(55,60,65))
        if not completed:
            draw.text((16,60),'Diagnostic stopped run; not a completed rebound',font=small,fill=(150,50,40))
        images.append(image)
    durations=[250]*len(images);durations[0]=600;durations[-1]=900
    path=prefix.with_suffix('.gif')
    images[0].save(path,save_all=True,append_images=images[1:],duration=durations,loop=0,disposal=2)
    images[-1].save(prefix.with_name(prefix.name+'_render.png'))
    print(json.dumps({'gif':str(path),'frames':len(images),'last_time':float(times[-1]),
        'completed':completed,'width_m':float(width)}))


if __name__=='__main__':main()
