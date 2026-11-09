"""Complete power-ball cut with explicit geometric vacuum generators.

The real topology comes from the production recutter. The extra half-spaces
are complete ball/plane tests, followed by the same disk and spherical kernels.
This opt-in correctness path is not yet a performance-optimized recutter.
"""
import math
from types import SimpleNamespace
import torch
import warp as wp
from free_surface.fv_ale_planar_boundary import augmented_structure,material_mesh


def clipped_values(p,r,structure,ground,redundant,material_count,skip_unused=True):
    from free_surface.geometry_warp import GeometryWarp,k_geo_site,BIG
    from free_surface.gpu_geometry import fibonacci
    from free_surface.geo_pipe import GeoPipe
    from free_surface.fv_ale_ground_geometry import GroundClip
    device=str(p.device);g=GeometryWarp(p,r,structure,integrate_sites=not skip_unused);m=len(g.i);n=len(r)
    site=torch.cat((g.i,g.j));ent=site.argsort(stable=True).to(torch.int32)
    off=torch.zeros(n+1,dtype=torch.int32,device=p.device)
    off[1:]=torch.cumsum(torch.bincount(site,minlength=n),0).to(torch.int32)
    Vw=torch.zeros_like(r).index_add(0,g.i,g.wall_A*g.hi).index_add(0,g.j,g.wall_A*g.hj)
    W=lambda x,t:wp.from_torch(x.contiguous(),dtype=t)
    pn=torch.empty_like(p);area=torch.empty_like(r);volume=torch.empty_like(r);pole=torch.empty_like(p)
    fib=W(fibonacci(32,p.dtype,p.device),wp.vec3d)
    wp.launch(k_geo_site,n,inputs=[W(p,wp.vec3d),W(r,wp.float64),W(off,wp.int32),W(ent,wp.int32),m,
        W(g.nn,wp.vec3d),W(g.u,wp.vec3d),W(g.v,wp.vec3d),W(g.rho,wp.float64),W(g.hi,wp.float64),W(g.hj,wp.float64),
        W(g.arc_t0,wp.float64),W(g.arc_t1,wp.float64),W(g.narc,wp.int32),fib,
        32,math.pi/6,BIG,4*math.pi,W(g.nbrs.to(torch.int32),wp.int32),W(g.hidden.to(torch.int32),wp.int32),
        1,0,0.,W(ent,wp.int32),0,W(Vw,wp.float64)],
        outputs=[W(pn,wp.vec3d),W(area,wp.float64),W(volume,wp.float64),W(pole,wp.vec3d)],device=device)
    g.V=torch.where(g.hidden,0.,volume);g.patch_A=area;g.patch_n=pn;g.pole=pole
    pipe=SimpleNamespace(npole=32,piece=math.pi/6,direct_sphere=False)
    GeoPipe._fib[(device,32)]=fib
    clip=GroundClip(ground,device,local=False);clip.size=n
    clip.recut=SimpleNamespace(pipe=pipe,geometry=lambda p,r:(g,None))
    return clip.geometry(p,r,floor_redundant=redundant,floor_owner_count=material_count)


def cut(solver,p,r,planes):
    import os
    fallback = None
    if p.is_cuda and os.environ.get('FV_UNIFIED_PLANES','0') == '1':
        from free_surface.fv_ale_ground_plane import UnifiedCutFallback,plane_geometry
        try:
            return plane_geometry(solver.ground_cutter,p,r,planes)
        except UnifiedCutFallback as exc:
            fallback = str(exc)
    fast=getattr(solver,'fast_planar_geometry',True)
    base=solver.recut.structure(p.contiguous(),r.contiguous()) if fast else solver.recut.geometry(p.contiguous(),r.contiguous())[1]
    pp,rr,redundant=planes.generators(p,r,solver.ground)
    structure=augmented_structure(pp,rr,base,len(r))
    full=clipped_values(pp,rr,structure,solver.ground,redundant,len(r),skip_unused=fast)
    mesh = material_mesh(full,planes,solver.ground)
    if fallback is not None:
        mesh.ground_geometry_info['unified_cut_fallback'] = fallback
    return mesh
