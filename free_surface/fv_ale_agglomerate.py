"""P0 FV control-volume agglomeration; the power-ball geometry is unchanged.

Cell merging is a standard small-cut-cell treatment for explicit FV schemes.
Here a group carries one conservative state. Its volume, boundary fluxes,
velocity-gradient integrals and CFL scale are summed before updating it.
Internal group faces cancel. Distribution back to geometric subcells is by
their current volume, including zero volume for a hidden generator.

This is local spatial coarsening, not a no-error stabilization. Groups persist
in this trial; splitting and long-run spatial accuracy are not validated.
"""
import torch
from free_surface.fv_ale_solid import State


@torch.no_grad()
def merge_cell(solver,s,k,max_members=32,allow_unavailable=False):
    labels=solver.groups
    target=int(labels[k]);V=solver.group_sum(s.mesh.volume)
    i,j=s.mesh.i,s.mesh.j
    li,lj=labels[i],labels[j]
    incident=(li==target)^(lj==target)
    neighbours=torch.where(li==target,lj,li)[incident]
    area=torch.zeros_like(s.r).index_add(0,neighbours,s.mesh.area[incident])
    counts=torch.bincount(labels,minlength=len(labels))
    raw_area=area.clone()
    area=torch.where((V>0)&(counts+counts[target]<=max_members),area,0.)
    area[target]=0
    anchor=int(area.argmax())
    if float(area[anchor])<=0:
        if allow_unavailable:
            candidates=torch.nonzero((raw_area>0)&(V>0)).flatten()
            # No topology or state change: the caller must use the actual
            # unmerged CFL step instead of insisting on its optional target.
            return s,{'cell':int(k),'group':target,'skipped':True,
                'reason':'no adjacent group within member-count cap',
                'target_members':int(counts[target]),'max_members':max_members,
                'target_active_members':int(((labels==target)&(s.mesh.volume>0)).sum()),
                'candidate_groups':candidates.cpu().tolist(),
                'candidate_member_counts':counts[candidates].cpu().tolist(),
                'candidate_shared_areas':raw_area[candidates].cpu().tolist(),
                'volume_before':float(V[target]),'geometry_changed':False}
        raise RuntimeError('no adjacent admissible FV group for small cell %d'%k)
    updated=labels.clone();updated[updated==target]=anchor
    before=s.q.sum(0)
    solver.groups=updated;solver.has_groups=True
    try:
        q=solver.distribute_groups(s.q,s.mesh.volume)
        result=State(s.p,s.r,q,s.mesh)
        solver.primitive(result)
        error=(q.sum(0)-before).abs()/s.q.abs().sum(0).clamp(min=1e-100)
        if float(error.max())>1e-12:raise RuntimeError('FV group merge lost conserved quantities')
        members=torch.nonzero(updated==anchor).flatten()
        points=s.p[members]
        return result,{'cell':int(k),'members':members.cpu().tolist(),
            'member_count':len(members),'volume_before':float(V[target]),
            'volume_after':float(s.mesh.volume[members].sum()),
            'shared_area':float(area[anchor]),'conservation_relative_max':float(error.max()),
            'surface_members_at_merge':int(s.mesh.surface[members].sum()),
            'merged_mass_fraction':float(q[members,0].sum()/q[:,0].sum()),
            # Bound on the member SITE centres, not on the entire cell domain.
            'group_diameter_bound':float(2*(points-points.mean(0)).norm(dim=1).max()),
            'geometry_changed':False}
    except Exception:
        solver.groups=labels
        solver.has_groups=bool(torch.unique(labels).numel()<len(labels))
        raise
