"""Exceptional positive-integral reevaluation of certified tiny surface wedges.

No positive volume floor and no mass/energy/position change. The ordinary
geometry path is retained. Only a negative, roundoff-scale member of an
already agglomerated volume is eligible. The exact local domain must be a
ball intersected by two active planes, with every other power plane proven
inactive by a high-precision enclosing-cap check.
"""
import math
import torch
from free_surface.fv_ale_empty_cell_probe import two_plane_volume


def reevaluate(solver,mesh):
    bad=torch.nonzero(~torch.isfinite(mesh.volume)|(mesh.volume<0)).flatten()
    if not len(bad) or len(bad)>8:return False
    members=solver.group_sum(torch.ones_like(mesh.volume))[solver.groups]
    group_volume=solver.group_sum(mesh.volume.clamp(min=0))[solver.groups]
    volume=mesh.volume.clone();cap=mesh.cap_area.clone();normal=mesh.cap_normal_integral.clone()
    p=mesh.p.cpu().numpy();r=mesh.r.cpu().numpy();records=[];extra=[];vacuum_count=0
    if hasattr(mesh,'planes'):
        # The ordinary two-real-plane formula is valid here ONLY after also
        # proving every vacuum constraint and the fixed floor inactive over
        # the enclosing cap. Do not reuse the old real-only certificate.
        pp,rr,_=mesh.planes.generators(mesh.p,mesh.r,solver.ground)
        p=pp.cpu().numpy();r=rr.cpu().numpy();vacuum_count=len(mesh.planes.owner)
        extra=[([0.,-1.,0.],-solver.ground)]
    eps=torch.finfo(mesh.volume.dtype).eps
    for index in bad.tolist():
        original=float(volume[index]);radius=float(mesh.r[index])
        bound=4096*eps*(4*math.pi/3)*radius**3
        if not math.isfinite(original) or original < -bound or members[index]<=1 or mesh.ground_area[index]>0:
            return False
        if hasattr(mesh,'planes') and bool(((mesh.plane_i==index)&(mesh.plane_area>0)).any()):return False
        mask=(mesh.i==index)|(mesh.j==index)
        faces=torch.nonzero(mask&(mesh.area>0)).flatten()
        neighbours=torch.where(mesh.i[faces]==index,mesh.j[faces],mesh.i[faces])
        if len(faces)!=2:return False
        precise=two_plane_volume(p,r,index,neighbours.cpu().tolist(),dps=60,additional_halfspaces=extra)
        if not precise.get('supported') or precise['volume_m3']<=0:return False
        exact=precise['volume_m3'];delta=exact-original
        if delta>bound or delta>1e-10*float(group_volume[index]):return False
        orientation=torch.where(mesh.i[faces]==index,1.,-1.)
        outward=orientation[:,None]*mesh.normal[faces]
        wall_sum=(mesh.area[faces]*((mesh.centroid[faces]-mesh.p[index])*outward).sum(1)).sum()
        corrected_cap=(3*exact-wall_sum)/radius
        if corrected_cap<=0 or abs(float(corrected_cap-cap[index]))>4096*eps*(4*math.pi*radius**2):return False
        volume[index]=exact;cap[index]=corrected_cap
        normal[index]=-(mesh.area[faces,None]*outward).sum(0)
        records.append({'cell':index,'old_volume_m3':original,'reevaluated_volume_m3':exact,
            'volume_change_m3':delta,'group_relative_volume_change':delta/float(group_volume[index]),
            'active_planes':neighbours.cpu().tolist(),'other_plane_margin':precise['other_plane_margin'],
            'verified_inactive_vacuum_generators':vacuum_count,'floor_inactivity_verified':bool(extra),
            'disjoint_ball_constraints_excluded':precise.get('disjoint_ball_constraints_excluded',0),
            'arbitrary_precision_ball_constraints':precise.get('arbitrary_precision_ball_constraints'),
            'precision_digits':precise['precision_digits']})
    # Several tiny children can belong to one group. Bound their combined
    # correction, not only each child's contribution separately.
    correction=solver.group_sum((volume-mesh.volume).abs())[solver.groups]
    if bool((correction>1e-10*group_volume).any()):return False
    mesh.volume=volume;mesh.cap_area=cap;mesh.cap_normal_integral=normal
    solver._precision_geometry_records.extend(records)
    return True
