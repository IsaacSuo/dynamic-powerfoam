"""Inspect a saved FV state without advancing or modifying it."""
import argparse,json
from pathlib import Path
import torch,warp as wp
from free_surface.fv_ale_solid import SolidALE,State

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshot',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    wp.init();d=torch.load(a.snapshot,map_location='cpu',weights_only=False);g=d['material']
    solver=SolidALE(d['p'],d['r'],g['E'],g['nu'],g['ys'],friction=d.get('friction',10.),interior_radii=d.get('interior_radii','dilate'))
    s=solver.initial;s.q=d['q'].to(s.q.device)
    v,dr,dq,diag=solver.rhs(s);m,_,rho,pressure,stress,sigma,c=solver.primitive(s)
    ids=d.get('ids',torch.arange(len(s.r)))
    contact_force,contact_stiffness,penetration=solver.contact(s)
    items=[]
    chosen=torch.unique(torch.cat((s.mesh.volume.argsort()[:8],torch.tensor([diag['wave_cell']],device=s.p.device))))
    for k in chosen.tolist():
        items.append({'index':k,'original_id':int(ids[k]),'volume':float(s.mesh.volume[k]),'mass':float(m[k]),
            'rho':float(rho[k]),'pressure':float(pressure[k]),'radius':float(s.r[k]),'radius_rate':float(dr[k]),
            'ground_force':contact_force[k].cpu().tolist(),'penetration':float(penetration[k]),
            'ground_acceleration':float(contact_force[k].norm()/m[k]),
            'cap_area':float(s.mesh.cap_area[k]),'surface_area':float(s.mesh.surface_area[k]),'surface':bool(s.mesh.surface[k]),
            'position':s.p[k].cpu().tolist(),'velocity':v[k].cpu().tolist(),
            'normal_wave_dt':float(s.mesh.volume[k]/(s.mesh.surface_area[k]*c[k]))})
    report={'snapshot':str(a.snapshot),'time':d['time'],'diagnostics':diag,'smallest_cells':items}
    txt=json.dumps(report,indent=2);print(txt)
    if a.out:a.out.write_text(txt+'\n')
if __name__=='__main__':main()
