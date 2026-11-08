"""Observe production attempts without modifying their acceptance or RHS."""
import json,runpy,sys
from pathlib import Path
import torch
from free_surface.fv_ale_solid import SolidALE
from free_surface.fv_ale_hyperelastic_config import StepAudit
from free_surface.fv_ale_cfl_diagnose import details

rows=[];pending={};original_advance=SolidALE.advance;original_check=StepAudit.check
out=Path('output/ale/cfl_production_trace.json')

def save():out.write_text(json.dumps(rows,indent=2)+'\n')

def traced(self,s,dt,first=None,cfl=2.):
    saved_rhs=self.rhs;stages=[]
    def rhs(x):
        r=saved_rhs(x);stages.append((x,r));return r
    self.rhs=rhs;row={'dt':dt,'cfl':cfl,'initial_scale':first[3]['dt_scale']}
    pending.clear();pending.update(row=row,s=s,first=first,stages=stages)
    try:
        result=original_advance(self,s,dt,first,cfl);row['advance']='completed';return result
    except Exception as e:
        row['advance']=str(e);raise
    finally:
        self.rhs=saved_rhs
        if stages:
            mid,r=stages[-1];new=mid.mesh.surface&~s.mesh.surface;off=s.mesh.surface&~mid.mesh.surface
            jump=torch.sqrt((r[0]-first[0]).square().sum(1)+(r[1]-first[1]).square())
            k=int(jump.argmax());cell=r[3]['wave_cell'];root=int(self.groups[cell]);members=self.groups==root
            row.update(stage_scale=r[3]['dt_scale'],stage_wave_cell=cell,stage_wave_root=root,
                       new_surfaces=int(new.sum()),lost_surfaces=int(off.sum()),
                       wave_group_switched=bool(((new|off)&members).any()),
                       max_motion_change=float(jump[k]),max_motion_change_cell=k,
                       max_motion_change_switched=bool((new|off)[k]),
                       max_new_surface_motion_change=float(torch.where(new,jump,0).max()),
                       stage_new_surface_area_sum=float(mid.mesh.cap_area[new].sum()))
        rows.append(row)
        if len(rows)%20==0:save()

def checked(self,candidate,dt):
    try:
        value=original_check(self,candidate,dt);pending['row']['audit']='accepted';return value
    except RuntimeError as ex:
        row=pending['row'];row['audit']=str(ex)
        if pending['stages']:
            start=details(self.solver,pending['s'],pending['first']);mid,r=pending['stages'][-1]
            second=details(self.solver,mid,r);end=details(self.solver,candidate,None)
            err=end['energy']-start['energy']-.5*dt*(start['power']+second['power'])
            k=int(err.abs().argmax());root=int(start['roots'][k]);members=self.solver.groups==root
            switched=pending['s'].mesh.surface!=mid.mesh.surface
            row.update(top_work_group=root,top_work_error=float(err[k]),sum_work_error=float(err.sum()),
                       top_work_group_switched=bool((switched&members).any()))
        raise

SolidALE.advance=traced;StepAudit.check=checked
try:
    sys.argv=['free_surface.merged_drop',*sys.argv[1:]]
    runpy.run_module('free_surface.merged_drop',run_name='__main__')
finally:
    save();SolidALE.advance=original_advance;StepAudit.check=original_check
