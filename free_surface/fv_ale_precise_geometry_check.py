"""Recompute the saved failed full foam; verify the precision fallback in situ."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
import warp as wp
from free_surface.fv_ale_hyperelastic import HyperelasticALE


def cpu_mesh(p,r,ground=None,structure=None,floor_redundant=None,floor_owner_count=None):
    from free_surface.gpu_geometry import structure_triangulation,fibonacci
    from free_surface.geometry_warp import GeometryWarp,k_geo_site,BIG
    from free_surface.fv_ale_geometry import PowerControlVolumes
    g=GeometryWarp(p,r,structure if structure is not None else structure_triangulation(p.numpy(),r.numpy(),device='cpu'))
    m=len(g.i);n=len(r);site=torch.cat((g.i,g.j));ent=site.argsort(stable=True).to(torch.int32)
    off=torch.zeros(n+1,dtype=torch.int32);off[1:]=torch.cumsum(torch.bincount(site,minlength=n),0).to(torch.int32)
    Vw=torch.zeros_like(r).index_add(0,g.i,g.wall_A*g.hi).index_add(0,g.j,g.wall_A*g.hj)
    W=lambda x,t:wp.from_torch(x.contiguous(),dtype=t)
    pn=torch.empty_like(p);area=torch.empty_like(r);volume=torch.empty_like(r);pole=torch.empty_like(p)
    wp.launch(k_geo_site,n,inputs=[W(p,wp.vec3d),W(r,wp.float64),W(off,wp.int32),W(ent,wp.int32),m,
        W(g.nn,wp.vec3d),W(g.u,wp.vec3d),W(g.v,wp.vec3d),W(g.rho,wp.float64),W(g.hi,wp.float64),W(g.hj,wp.float64),
        W(g.arc_t0,wp.float64),W(g.arc_t1,wp.float64),W(g.narc,wp.int32),W(fibonacci(32,p.dtype,p.device),wp.vec3d),
        32,math.pi/6,BIG,4*math.pi,W(g.nbrs.to(torch.int32),wp.int32),W(g.hidden.to(torch.int32),wp.int32),
        1,0,0.,W(ent,wp.int32),0,W(Vw,wp.float64)],
        outputs=[W(pn,wp.vec3d),W(area,wp.float64),W(volume,wp.float64),W(pole,wp.vec3d)],device='cpu')
    g.V=torch.where(g.hidden,0.,volume);g.patch_A=area;g.patch_n=pn
    if ground is not None:
        # Exercise the SAME full-field disk/sphere ground kernels on CPU;
        # only the neighbour construction bypasses the GPU-only recutter.
        from types import SimpleNamespace
        from free_surface.fv_ale_ground_geometry import GroundClip
        from free_surface.geo_pipe import GeoPipe
        pipe=SimpleNamespace(npole=32,piece=math.pi/6,direct_sphere=False)
        GeoPipe._fib[('cpu',32)]=W(fibonacci(32,p.dtype,p.device),wp.vec3d)
        clip=GroundClip(ground,'cpu',local=False);clip.size=len(r)
        clip.recut=SimpleNamespace(pipe=pipe,geometry=lambda p,r:(g,None))
        return clip.geometry(p,r,floor_redundant=floor_redundant,floor_owner_count=floor_owner_count)
    mesh=PowerControlVolumes.from_geometry(g);mesh.ground_area=torch.zeros_like(r)
    return mesh


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--snapshot',type=Path,default=Path('output/ale/hyperelastic_smooth_stretch2_cfl_fallback10ms.pt'))
    args=ap.parse_args();torch.set_num_threads(1);wp.init();tick=time.perf_counter()
    data=torch.load(args.snapshot,map_location='cpu',weights_only=False);bad=data['rejected_geometry']
    if args.device=='cpu':
        mesh=cpu_mesh(bad['p'],bad['r'])
        solver=object.__new__(HyperelasticALE);solver.groups=data['groups'];solver.has_groups=True
        solver.precision_geometry=True;solver._precision_geometry_records=[]
        solver.normalize_volume_roundoff(mesh)
        before=mesh.volume.clone();success=solver.reevaluate_invalid_geometry(mesh)
        assert success and bool((mesh.volume>=0).all())
    else:
        m=data['material']
        solver=HyperelasticALE(data['p'],data['r'],m['E'],m['nu'],m['ys'],groups=data['groups'],
            ground=-10.,gravity=(0.,0.,0.),contact_model='plane',precision_geometry=True)
        mesh=solver.cut(bad['p'].cuda(),bad['r'].cuda());success=True;before=None
    records=solver._precision_geometry_records
    assert any(row['cell']==929 for row in records),records
    reference=json.loads(args.snapshot.with_name(args.snapshot.stem+'_cell929_empty_probe.json').read_text())
    expected=reference['rejected']['independent_volume']['volume_m3']
    error=abs(float(mesh.volume[929])/expected-1)
    assert error<1e-12,error
    # The repaired cap obeys the volume and constant-field surface identities.
    k=929;faces=torch.nonzero((mesh.i==k)|(mesh.j==k)).flatten()
    signs=torch.where(mesh.i[faces]==k,1.,-1.);normal=signs[:,None]*mesh.normal[faces]
    closure=(mesh.area[faces,None]*normal).sum(0)+mesh.cap_normal_integral[k]
    walls=(mesh.area[faces]*((mesh.centroid[faces]-mesh.p[k])*normal).sum(1)).sum()
    volume_identity=(walls+mesh.r[k]*mesh.cap_area[k])/3
    identity_error=float(abs(volume_identity-mesh.volume[k]))
    assert float(closure.norm())<1e-24 and identity_error<1e-28,(closure,identity_error)
    report={'device':args.device,'success':success,'cells':len(mesh.r),'seconds':time.perf_counter()-tick,
            'reevaluations':records,'positive_integral_relative_error':error,
            'surface_vector_closure_norm_m2':float(closure.norm()),'volume_identity_error_m3':identity_error}
    Path('output/ale/hyperelastic_precise_geometry_'+args.device+'_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
