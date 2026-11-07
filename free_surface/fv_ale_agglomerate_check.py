"""Full-foam conservation and internal-geometry invariance of FV agglomeration."""
import argparse,json
from pathlib import Path
import torch,warp as wp
from free_surface.fv_ale_solid import SolidALE,State
from free_surface.fv_ale_agglomerate import merge_cell


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('snapshot',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    wp.init();d=torch.load(a.snapshot,map_location='cpu',weights_only=False);g=d['material']
    sol=SolidALE(d['p'],d['r'],g['E'],g['nu'],g['ys'],surface_motion='coupled')
    s=sol.initial;s.q=d['q'].to(s.q.device)
    if 'reference_volume' in d:sol.contact_length=d['reference_volume']**(1/3);sol.contact_k=sol.bulk/sol.contact_length
    rhs=sol.rhs(s);before_dt=.2*rhs[3]['dt_scale'];k=rhs[3]['wave_cell']
    merged,info=merge_cell(sol,s,k)
    after=sol.rhs(merged);after_dt=.2*after[3]['dt_scale']
    initial_mass=float(merged.q[:,0].sum());initial_energy=float(merged.q[:,4].sum());source=0.
    for _ in range(10):
        rr=sol.rhs(merged);merged,stats=sol.advance(merged,.1*rr[3]['dt_scale'],rr);source+=stats['source_work']
    physical_energy_error=float(merged.q[:,4].sum())-initial_energy-source
    report={'snapshot':str(a.snapshot),'merge':info,'step_before':before_dt,'step_after':after_dt,
        'ten_step_mass_error':(float(merged.q[:,0].sum())-initial_mass)/initial_mass,
        'ten_step_energy_error_joule':physical_energy_error}
    # Build one connected star control volume. Shrink its buried centre sphere
    # until that geometric subcell is hidden. The control volume and exterior
    # must remain unchanged, so its FV RHS must also remain unchanged.
    sol.groups=torch.arange(len(s.r),device=s.r.device);sol.has_groups=False
    k=int(torch.where(~s.mesh.surface,s.mesh.volume,torch.full_like(s.r,float('inf'))).argmin())
    nb=s.mesh.geometry.nbrs[k].long();nb=nb[nb>=0]
    star=torch.unique(torch.cat((nb,torch.tensor([k],device=s.r.device))))
    sol.groups[star]=k;sol.has_groups=True
    s0=State(s.p,s.r,sol.distribute_groups(s.q,s.mesh.volume),s.mesh)
    rhs0=sol.rhs(s0);group_rate0=sol.group_sum(rhs0[2])
    radii=s.r.clone();radii[k]*=.05
    mesh=sol.cut(s.p,radii);s1=State(s.p,radii,sol.distribute_groups(s0.q,mesh.volume),mesh)
    rhs1=sol.rhs(s1);group_rate1=sol.group_sum(rhs1[2])
    V0=sol.group_sum(s.mesh.volume);V1=sol.group_sum(mesh.volume)
    occupied=torch.bincount(sol.groups,minlength=len(s.r))>0
    vol_error=float(((V1[occupied]-V0[occupied])/V0[occupied]).abs().max())
    rate_error=float((group_rate1-group_rate0).norm()/group_rate0.norm().clamp(min=1e-30))
    primitive0=sol.primitive(s0);primitive1=sol.primitive(s1)
    density_error=float((primitive1[2]/primitive0[2]-1).abs().max())
    report['hidden_subcell']={'cell':k,'star_members':len(star),'volume':float(mesh.volume[k]),
        'control_volume_relative_change_max':vol_error,'density_relative_change_max':density_error,
        'aggregate_rhs_relative_change':rate_error,'finite_rhs':bool(torch.isfinite(rhs1[2]).all()),
        'cfl_before':rhs0[3]['dt_scale'],'cfl_after':rhs1[3]['dt_scale']}
    report['checks']={'conservation':info['conservation_relative_max']<1e-12,
        'ten_steps':abs(report['ten_step_mass_error'])<1e-12 and abs(physical_energy_error)<1e-8,
        'hidden_subcell_geometry':float(mesh.volume[k])==0. and vol_error<1e-8,
        'same_control_volume_same_operator':density_error<1e-8 and rate_error<1e-7 and report['hidden_subcell']['finite_rhs']}
    report['passed']=all(report['checks'].values());a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()
