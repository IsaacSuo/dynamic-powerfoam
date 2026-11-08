"""Inspect actual rejected geometry, excluding legal zero-volume members."""
import argparse,json
from pathlib import Path
import torch
from free_surface.fv_ale_ground_geometry import GroundClip


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tag');args=ap.parse_args()
    root=Path('output/ale');d=torch.load(root/(args.tag+'.pt'),weights_only=False)
    g=d['groups'].cuda();counts=torch.bincount(g,minlength=len(g));results={}
    for name,source in [('accepted',d),('rejected',d['rejected_geometry'])]:
        p=source['p'].cuda();r=source['r'].cuda();cutter=GroundClip(-10.,'cuda');mesh=cutter.geometry(p,r)
        V=torch.where(mesh.hidden,torch.zeros_like(mesh.volume),mesh.volume)
        bound=64*torch.finfo(r.dtype).eps*(4*torch.pi/3)*r.pow(3)
        negative=torch.nonzero(V<0).flatten();entries=[]
        for k in negative.tolist():
            entries.append({'cell':k,'volume':float(V[k]),'roundoff_bound':float(bound[k]),
                            'ratio':float(-V[k]/bound[k]),'radius':float(r[k]),'group_size':int(counts[g[k]])})
        corrected=torch.where((V<0)&(V>=-bound)&(counts[g]>1),0.,V)
        cutter.recut.pipe.direct_sphere=True
        alternate=cutter.geometry(p,r)
        for entry in entries:
            k=entry['cell'];entry['alternate_complete_arc_volume']=float(alternate.volume[k])
        results[name]={'negative_count':len(entries),'negative_cells':entries,
                       'nonfinite_count':int((~torch.isfinite(V)).sum()),
                       'remaining_negative_count':int((corrected<0).sum()),
                       'zero_count':int((V==0).sum()),'minimum_volume':float(V.min())}
    (root/(args.tag+'_cut_failure.json')).write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
