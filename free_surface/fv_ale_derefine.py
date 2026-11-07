"""Conservative removal of a buried power generator, adapted from AREPO Sec. 6.

For a removed competitor, remaining power cells only grow. If the union of
balls is unchanged, their volume gains are exactly their intersections with the
removed cell. P0 remapping thus needs no general cell-intersection routine.
A cap-free ball alone does NOT prove unchanged domain (it may fill a cavity);
the volume-gain check below is mandatory. This remaps computation, not matter.
"""
import torch
from free_surface.recut import Recut
from free_surface.geo_pipe import GeoPipe
from free_surface.fv_ale_solid import State


@torch.no_grad()
def dissolve(solver,s,k):
    if bool(s.mesh.surface[k]):
        raise RuntimeError('refuse exposed-cell removal: would change free boundary')
    keep=torch.arange(len(s.r),device=s.p.device)!=k
    old=solver.recut
    solver.recut=Recut(int(keep.sum()),str(s.p.device))
    solver.recut.pipe=GeoPipe(int(keep.sum()),str(s.p.device))
    try:
        mesh=solver.cut(s.p[keep],s.r[keep])
        dv=mesh.volume-s.mesh.volume[keep]
        removed=s.mesh.volume[k]
        negative=float((-dv).clamp(min=0).sum()/removed)
        volume_defect=float((dv.sum()-removed).abs()/removed)
        if negative>1e-7 or volume_defect>1e-6:
            raise RuntimeError('removal changes domain or geometry unresolved: negative=%g deficit=%g'%(negative,volume_defect))
        gain=dv.clamp(min=0)
        weights=gain/gain.sum()
        q=s.q[keep]+weights[:,None]*s.q[k]
        result=State(s.p[keep],s.r[keep],q,mesh)
        solver.primitive(result)
        before=s.q.sum(0);after=q.sum(0)
        errors=(after-before).abs()/s.q.abs().sum(0).clamp(min=1e-100)
        if float(errors.max())>1e-12:
            raise RuntimeError('remap failed extensive-channel conservation')
        # Uniform-density preservation tests the transfer geometry separately
        # from the actual nonuniform physical state being remapped.
        uniform=(s.mesh.volume[keep]+weights*removed)/mesh.volume-1
        info={'local_index':int(k),'volume':float(removed),'mass':float(s.q[k,0]),
              'volume_partition_defect':volume_defect,'negative_gain_relative':negative,
              'conservation_relative_max':float(errors.max()),
              'uniform_density_error_max':float(uniform.abs().max()),
              'receiving_cells':int((gain>removed*1e-9).sum()),
              'cells_after':len(q)}
        if info['uniform_density_error_max']>1e-6:
            raise RuntimeError('removal failed uniform-density preservation')
        return result,keep,info
    except Exception:
        solver.recut=old
        raise
