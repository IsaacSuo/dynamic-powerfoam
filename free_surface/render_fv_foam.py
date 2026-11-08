"""Original Fourier PowerFoam appearance on saved FV geometry, with true cuts.

Extra halfspaces are expressed as per-cell, zero-opacity power competitors.
Only that cell gets the extra adjacency; dummy spheres never carve other cells.
This leaves the trained shader and renderer unchanged. No dynamics are run.
Dataset images are a visual reference only when the mechanics differ.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from free_surface import render_k1 as rk


@torch.no_grad()
def clipped_inputs(model, positions, radii, shift, frame, k):
    n=len(radii);device=positions.device;dtype=positions.dtype
    owners=[];normals=[];heights=[]
    empty=torch.zeros(n,dtype=torch.bool,device=device)
    def add(ii,nn,hh):
        nonlocal empty
        empty[ii[hh<=-radii[ii]]]=True
        take=(hh<radii[ii])&(hh>-radii[ii])&(radii[ii]>0)
        owners.append(ii[take]);normals.append(nn[take]);heights.append(hh[take])
    if 'ground_clipped' in frame and bool(frame['ground_clipped']):
        floor=float(frame['ground_height'])-float(shift[1])
        ii=torch.arange(n,device=device)
        normal=torch.zeros((n,3),device=device,dtype=dtype);normal[:,1]=-1
        add(ii,normal,positions[:,1]-floor)
    count=int(frame['plane_count'][k]) if 'plane_count' in frame else 0
    if count:
        # Reconstruct the same global vacuum power generators used by FV cuts.
        own=torch.as_tensor(frame['plane_owner'][k,:count],device=device,dtype=torch.long)
        d=torch.as_tensor(frame['plane_d'][k,:count],device=device,dtype=torch.float64)
        a=torch.as_tensor(frame['plane_a'][k,:count],device=device,dtype=torch.float64)
        b=torch.as_tensor(frame['plane_b'][k,:count],device=device,dtype=torch.float64)
        world=torch.as_tensor(frame['x'][k],device=device,dtype=torch.float64)
        rr=torch.as_tensor(frame['r'][k],device=device,dtype=torch.float64)
        virtual=world[own]+d[:,None]*a
        weight=rr[own]**2+d*d*(a*a).sum(1)-2*d*(b-(a*world[own]).sum(1))
        delta=virtual[None,:,:]-world[:,None,:]
        length=torch.linalg.vector_norm(delta,dim=-1)
        h=(length*length+rr[:,None]**2-weight[None,:])/(2*length.clamp_min(1e-30))
        ii,jj=torch.where((h<rr[:,None])&(length>1e-15))
        add(ii,(delta[ii,jj]/length[ii,jj,None]).to(dtype),h[ii,jj].to(dtype))
    owner=torch.cat(owners) if owners else torch.empty(0,dtype=torch.long,device=device)
    normal=torch.cat(normals) if normals else positions[:0]
    height=torch.cat(heights) if heights else radii[:0]
    # Positive dummy radii exist for any cutting plane; no negative weights needed.
    distance=2*(radii[owner]+height.abs()).clamp_min(1e-9)
    dummy_p=positions[owner]+distance[:,None]*normal
    dummy_r=(radii[owner]**2+distance**2-2*distance*height).sqrt()
    reconstructed=(radii[owner].double()**2+distance.double()**2-dummy_r.double()**2)/(2*distance.double())
    plane_error=float((reconstructed-height.double()).abs().max()) if len(owner) else 0.
    offsets=model.adjacency_offsets.long()
    real_sources=torch.repeat_interleave(torch.arange(n,device=device),offsets[1:]-offsets[:-1])
    source=torch.cat([real_sources,owner])
    target=torch.cat([model.adjacency[:len(real_sources)].long(),torch.arange(len(owner),device=device)+n])
    order=source.argsort(stable=True)
    adjacency=target[order].to(torch.int32).contiguous()
    counts=torch.bincount(source,minlength=n+len(owner))
    offsets=torch.cat([torch.zeros(1,dtype=torch.long,device=device),counts.cumsum(0)]).to(torch.int32)
    density=model.get_density().clone();density[empty]=0
    density=torch.cat([density,torch.zeros((len(owner),)+density.shape[1:],device=device,dtype=density.dtype)])
    fourier=model.get_fourier_arg()
    fourier=torch.cat([fourier,torch.zeros((len(owner),)+fourier.shape[1:],device=device,dtype=fourier.dtype)])
    return (torch.cat([positions,dummy_p]),torch.cat([radii,dummy_r]),density,
        fourier,adjacency,offsets,len(owner),int(empty.sum()),plane_error)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--frames',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--run',default=rk.RUN)
    ap.add_argument('--cells',type=Path,default=Path('foams/fourier8k_clean/dem_exact_single/cells.npz'))
    ap.add_argument('--reference',type=Path,default=Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/data/elastic/0/data'))
    ap.add_argument('--views',type=int,nargs='+',default=[0,3])
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    model,data=rk.load(args.run)
    cells=np.load(args.cells);source=torch.as_tensor(cells['source_index'],device=model.device)
    with torch.no_grad():
        for name in rk.PER_CELL:
            value=getattr(model,name,None)
            if value is not None:setattr(model,name,torch.nn.Parameter(value.data[source].clone()))
        model.rebuild_adjacency()
    original_p=model.points.detach().clone();original_r=model.get_radii().detach().clone()
    frames=np.load(args.frames);times=frames['t']
    X=torch.as_tensor(frames['x'],device=model.device,dtype=original_p.dtype)
    R=torch.as_tensor(frames['r'],device=model.device,dtype=original_r.dtype)
    shift=(X[0]-original_p).mean(0)
    spread=float((X[0]-original_p-shift).abs().max())
    assert spread<1e-5, ('initial geometry is not a translation of this trained foam',spread)
    cameras=[data.cameras[v] for v in args.views]
    meta=json.loads((Path(model.args.data_path)/model.args.scene/'transforms_all.json').read_text())
    names=[Path(meta['frames'][v]['file_path']).name.rsplit('_',1)[0] for v in args.views]
    backgrounds=[torch.as_tensor(np.array(Image.open(args.reference/(name+'_-1.png')).convert('RGB').resize((cam.width,cam.height)),dtype=np.float32)/255,device=model.device) for name,cam in zip(names,cameras)]
    untouched=[rk.render(model,cam) for cam in cameras]
    paired=[[] for _ in cameras];solo=[[] for _ in cameras];rows=[];selected={}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    for k,t in enumerate(times):
        # Only compare exact dataset frame times. The extra exact impact frame is not rounded.
        index=int(round(float(t)*24));exact=abs(float(t)-index/24)<1e-8
        with torch.no_grad():
            model.points=torch.nn.Parameter((X[k]-shift).contiguous())
            model.radii=torch.nn.Parameter(rk.inv_softplus(R[k]).contiguous())
            model.rebuild_adjacency()
            p,r,density,fourier,adj,off,ndummy,nempty,err=clipped_inputs(model,model.points,model.get_radii(),shift,frames,k)
            item={'time':float(t),'frame':index if exact else None,'clip_competitors':ndummy,
                'fully_clipped_cells':nempty,'float32_plane_offset_error_m':err,'views':[]}
            for j,cam in enumerate(cameras):
                base=model.get_base_rgb(cam)
                base=torch.cat([base,torch.zeros((ndummy,)+base.shape[1:],device=base.device,dtype=base.dtype)])
                depth=.5*torch.ones(cam.height,cam.width,1,device=model.device)
                rgb,alpha=model.rasterizer.forward(cam,depth,p,r,density,base,fourier,adj,off,None,False)[:2]
                alpha=alpha.reshape(cam.height,cam.width,1).clamp(0,1)
                rendered=(rgb.clamp(0,1)+(1-alpha)*backgrounds[j]).clamp(0,1)
                im=Image.fromarray((rendered.cpu().numpy()*255).astype(np.uint8))
                im.save(args.out/f'view{args.views[j]}_state{k:03d}.png')
                if k==0:
                    error=max(float((rgb.clamp(0,1)-untouched[j][0]).abs().max()),float((alpha.reshape_as(untouched[j][1])-untouched[j][1]).abs().max()))
                    item['views'].append({'view':args.views[j],'initial_update_render_max_error':error})
                    assert error<1e-3,('initial rendering changed',error)
                own=Image.new('RGB',(480,520),'white');own.paste(im.resize((480,480)),(0,40))
                draw=ImageDraw.Draw(own);draw.text((8,5),f'FV Neo-Hookean | t={t:.4f} s',fill='black',font=font)
                draw.text((8,23),'Original PowerFoam appearance + actual cuts',fill='black',font=small)
                solo[j].append(own)
                ref=args.reference/f'{names[j]}_{index}.png'
                if exact and ref.exists():
                    truth=Image.open(ref).convert('RGB')
                    pair=Image.new('RGB',(960,548),'white')
                    pair.paste(im.resize((480,480)),(0,68));pair.paste(truth.resize((480,480)),(480,68))
                    d=ImageDraw.Draw(pair)
                    d.text((8,5),f't={t:.4f} s | DIFFERENT MATERIAL / LAUNCH / CONTACT CONDITIONS',fill=(165,45,20),font=small)
                    d.text((8,25),'Current FV: zero launch velocity; frictionless',fill='black',font=font)
                    d.text((488,25),'Dataset reference: elastic/0',fill='black',font=font)
                    d.text((8,47),'Appearance comparison; not a matched physics validation',fill='black',font=small)
                    paired[j].append(pair)
                    pair.save(args.out/f'compare_view{args.views[j]}_frame{index:02d}.jpg',quality=92)
                    if index in [0,6,8,12]:selected[(j,index)]=pair
        rows.append(item);print(json.dumps(item),flush=True)
    for j,v in enumerate(args.views):
        solo[j][0].save(args.out/f'fv_view{v}.gif',save_all=True,append_images=solo[j][1:],duration=180,loop=0)
        if paired[j]:paired[j][0].save(args.out/f'compare_view{v}.gif',save_all=True,append_images=paired[j][1:],duration=180,loop=0)
        sheet=[selected[j,k] for k in [0,6,8,12] if (j,k) in selected]
        if sheet:
            canvas=Image.new('RGB',(960,548*len(sheet)),'white')
            for ii,pair in enumerate(sheet):canvas.paste(pair,(0,548*ii))
            canvas.save(args.out/f'comparison_sheet_view{v}.jpg',quality=92)
    report={'renderer':'Original trained Fourier PowerFoam volume renderer',
        'source_run':args.run,'frames':str(args.frames.resolve()),'reference':str(args.reference),
        'views':dict(zip(map(str,args.views),names)),'shift_removed':shift.cpu().tolist(),
        'initial_position_spread_m':spread,'rows':rows,
        'comparison':'Different constitutive/launch/friction conditions; image reference, NOT matched ground truth.',
        'texture_limitation':'Saved FV frames contain no material rotation/texture advection. Original per-cell SH/Fourier basis is retained without invented rotations.',
        'geometry':'Actual ball radii/sites, ground and all vacuum power cuts. No outlier removal, smoothing or fitted rigid alignment.',
        'precision':'Same trained renderer, including its original float16 cached adjacency geometry.'}
    (args.out/'render_report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
