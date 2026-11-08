"""Inspect converged path data: locate short-lived faces, quantify local errors."""
import json
from pathlib import Path
import numpy as np
import torch
from free_surface.fv_ale_profile import load
from free_surface.fv_ale_swept_check import PathAudit,endpoint,cpu

@torch.no_grad()
def main():
    out=Path('output/ale');rows=[]
    for tag in ['dt8us','cfl20_long']:
        sol,reset,data=load(out/f'fv_state_{tag}.pt');start=reset();end,_=endpoint(sol,start,800e-6)
        audit=PathAudit(sol,start,end,800e-6,data['reference_volume'])
        fine=np.load(out/f'swept_{tag}_800us.npz');keys=fine['keys'];i=keys//audit.n;j=keys%audit.n
        ends=np.union1d(audit.sample(0)['keys'],audit.sample(1)['keys'])
        missing=np.flatnonzero(~np.isin(keys,ends));short=[]
        for k in missing:
            short.append({'i':int(i[k]),'j':int(j[k]),'between_distinct_groups':bool(audit.g[i[k]]!=audit.g[j[k]]),
                'swept_volume_m3':float(fine['flux'][k]),'time_average_area_m2':float(fine['area'][k])})
        row={'snapshot':tag,'short_lived_faces':short,'body_mass_kg':float(start.q[:,0].sum()),'coarse_vs_257_nodes':[]}
        for panels in [2,4,8]:
            coarse=audit.integrate(panels);union=np.union1d(keys,coarse['keys']);ii=union//audit.n;jj=union%audit.n
            delta=np.zeros(len(union));delta[np.searchsorted(union,keys)]+=fine['flux']
            delta[np.searchsorted(union,coarse['keys'])]-=coarse['flux']
            ext=audit.g[ii]!=audit.g[jj]
            score=np.bincount(audit.g[ii[ext]],weights=np.abs(delta[ext]),minlength=audit.ng)
            score+=np.bincount(audit.g[jj[ext]],weights=np.abs(delta[ext]),minlength=audit.ng)
            score+=np.abs(audit.group(fine['cap']-coarse['cap']))
            ordered=np.sort(score)[::-1];count=int(np.searchsorted(np.cumsum(ordered),.9*ordered.sum())+1)
            local=np.minimum(audit.gv0[audit.g[ii]],audit.gv0[audit.g[jj]])
            row['coarse_vs_257_nodes'].append({'nodes':panels+1,'groups_covering_90pct_abs_flux_error':count,
                'group_count':audit.ng,'top10_share':float(ordered[:10].sum()/ordered.sum()),
                'max_external_face_swept_error_relative_adjacent_group_volume':float(np.max(np.abs(delta[ext])/local[ext]))})
        rows.append(row);print(json.dumps(row),flush=True)
    (out/'swept_geometry_summary.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
