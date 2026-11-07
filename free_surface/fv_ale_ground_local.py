"""Local ground geometry with fused GPU preparation; full reference retained."""
import math
from types import SimpleNamespace
import torch,warp as wp
from free_surface.geometry_warp import k_geo_site,BIG
from free_surface.geo_pipe import GeoPipe
from free_surface.fv_ale_geometry import PowerControlVolumes


@wp.kernel(enable_backward=False)
def prepare_faces(selected:wp.array(dtype=wp.int64),floor_ids:wp.array(dtype=wp.int64),m:int,
    p:wp.array(dtype=wp.vec3d),r:wp.array(dtype=wp.float64),H:wp.float64,
    si:wp.array(dtype=wp.int64),sj:wp.array(dtype=wp.int64),
    sn:wp.array(dtype=wp.vec3d),su:wp.array(dtype=wp.vec3d),sv:wp.array(dtype=wp.vec3d),sc:wp.array(dtype=wp.vec3d),
    sr:wp.array(dtype=wp.float64),shi:wp.array(dtype=wp.float64),shj:wp.array(dtype=wp.float64),
    sa:wp.array(dtype=wp.float64),swc:wp.array(dtype=wp.vec3d),snc:wp.array(dtype=wp.int32),
    st0:wp.array2d(dtype=wp.float64),st1:wp.array2d(dtype=wp.float64),
    oi:wp.array(dtype=wp.int64),oj:wp.array(dtype=wp.int64),
    on:wp.array(dtype=wp.vec3d),ou:wp.array(dtype=wp.vec3d),ov:wp.array(dtype=wp.vec3d),oc:wp.array(dtype=wp.vec3d),
    orad:wp.array(dtype=wp.float64),ohi:wp.array(dtype=wp.float64),ohj:wp.array(dtype=wp.float64),
    oa:wp.array(dtype=wp.float64),owc:wp.array(dtype=wp.vec3d),onc:wp.array(dtype=wp.int32),
    ot0:wp.array2d(dtype=wp.float64),ot1:wp.array2d(dtype=wp.float64)):
    k=wp.tid();zero=wp.float64(0.);one=wp.float64(1.)
    if k<m:
        f=selected[k]
        oi[k]=si[f];oj[k]=sj[f];on[k]=sn[f];ou[k]=su[f];ov[k]=sv[f];oc[k]=sc[f]
        orad[k]=sr[f];ohi[k]=shi[f];ohj[k]=shj[f];oa[k]=sa[f];owc[k]=swc[f];onc[k]=snc[f]
        for a in range(snc[f]):
            ot0[k,a]=st0[f,a];ot1[k,a]=st1[f,a]
    else:
        s=floor_ids[k-m];centre=p[s];h=centre[1]-H
        centre[1]=H;oi[k]=s;oj[k]=wp.int64(-1)
        on[k]=wp.vec3d(zero,-one,zero);ou[k]=wp.vec3d(one,zero,zero);ov[k]=wp.vec3d(zero,zero,one)
        oc[k]=centre;owc[k]=centre;orad[k]=wp.sqrt(wp.max(zero,r[s]*r[s]-h*h))
        ohi[k]=h;ohj[k]=zero;oa[k]=zero;onc[k]=0


@wp.kernel(enable_backward=False)
def prepare_constraints(work:wp.array(dtype=wp.int64),owners:wp.array(dtype=wp.int64),other:wp.array(dtype=wp.int64),
    nbrs:wp.array2d(dtype=wp.int32),p:wp.array(dtype=wp.vec3d),r:wp.array(dtype=wp.float64),
    centres:wp.array(dtype=wp.vec3d),u:wp.array(dtype=wp.vec3d),v:wp.array(dtype=wp.vec3d),radii:wp.array(dtype=wp.float64),H:wp.float64,
    axes:wp.array2d(dtype=wp.vec2d),bounds:wp.array2d(dtype=wp.float64),enabled:wp.array2d(dtype=wp.int32),limits:wp.array(dtype=wp.int32)):
    row=wp.tid();f=work[row];s=owners[f];zero=wp.float64(0.);two=wp.float64(2.);count=int(0)
    for k in range(nbrs.shape[1]+1):
        a=wp.vec2d();b=zero;keep=int(0)
        if k<nbrs.shape[1]:
            j=nbrs[s,k]
            if j>=0 and wp.int64(j)!=other[f]:
                d=p[j]-p[s]
                a=two*wp.vec2d(wp.dot(d,u[f]),wp.dot(d,v[f]))
                b=wp.dot(d,d)+r[s]*r[s]-r[j]*r[j]-two*wp.dot(d,centres[f]-p[s]);keep=1
        else:
            a=wp.vec2d(-u[f][1],-v[f][1]);b=centres[f][1]-H
            if other[f]>=0:keep=1
        # A half-plane containing the entire disk cannot alter the result.
        # Keep near-tangent planes with conservative relative slack.
        extent=radii[f]*wp.length(a)
        if keep==1 and b<=extent+wp.float64(1e-12)*(wp.abs(b)+extent):
            axes[row,count]=a;bounds[row,count]=b;enabled[row,count]=1;count=count+1
    limits[row]=count


@wp.kernel(enable_backward=False)
def wall_volume(i:wp.array(dtype=wp.int64),j:wp.array(dtype=wp.int64),area:wp.array(dtype=wp.float64),
    hi:wp.array(dtype=wp.float64),hj:wp.array(dtype=wp.float64),volume:wp.array(dtype=wp.float64)):
    f=wp.tid();wp.atomic_add(volume,i[f],area[f]*hi[f])
    if j[f]>=0:wp.atomic_add(volume,j[f],area[f]*hj[f])


def local_geometry(p,r,base,h,plane_ids,touched,ground,device,pipe):
    from free_surface.fv_ale_ground_geometry import disk_clip,MAX_CONSTRAINTS
    n=len(r);dev=p.device;dtype=p.dtype
    affected=h<=r;affected[base.i[touched]]=True;affected[base.j[touched]]=True
    targets=torch.nonzero(affected).flatten()
    selected=torch.nonzero(affected[base.i]|affected[base.j]).flatten()
    m=len(selected);nf=len(plane_ids);total=m+nf;K=base.nbrs.shape[1]+1
    if K>MAX_CONSTRAINTS:raise RuntimeError('ground disk constraint capacity exceeded')
    fields=['i','j','nn','u','v','c','rho','hi','hj','wall_A','wall_c','narc','arc_t0','arc_t1']
    types=[wp.int64,wp.int64,wp.vec3d,wp.vec3d,wp.vec3d,wp.vec3d,wp.float64,wp.float64,wp.float64,wp.float64,wp.vec3d,wp.int32,wp.float64,wp.float64]
    g=SimpleNamespace();W=lambda a,t:wp.from_torch(a.contiguous(),dtype=t)
    width=max(2*K+1,base.arc_t0.shape[1])
    for f in fields:
        a=getattr(base,f);shape=(total,width) if f in ['arc_t0','arc_t1'] else (total,)+a.shape[1:]
        setattr(g,f,torch.empty(shape,dtype=a.dtype,device=dev))
    wp.launch(prepare_faces,total,inputs=[W(selected,wp.int64),W(plane_ids,wp.int64),m,W(p,wp.vec3d),W(r,wp.float64),float(ground)]+
        [W(getattr(base,f),t) for f,t in zip(fields,types)],outputs=[W(getattr(g,f),t) for f,t in zip(fields,types)],device=device)
    work=torch.cat((torch.searchsorted(selected,touched),torch.arange(m,total,device=dev)))
    axes=torch.empty((len(work),K,2),dtype=dtype,device=dev);bounds=torch.empty((len(work),K),dtype=dtype,device=dev)
    enabled=torch.empty((len(work),K),dtype=torch.int32,device=dev);bad=torch.zeros(len(work),dtype=torch.int32,device=dev)
    limits=torch.empty(len(work),dtype=torch.int32,device=dev)
    wp.launch(prepare_constraints,len(work),inputs=[W(work,wp.int64),W(g.i,wp.int64),W(g.j,wp.int64),W(base.nbrs,wp.int32),
        W(p,wp.vec3d),W(r,wp.float64),W(g.c,wp.vec3d),W(g.u,wp.vec3d),W(g.v,wp.vec3d),W(g.rho,wp.float64),float(ground)],
        outputs=[W(axes,wp.vec2d),W(bounds,wp.float64),W(enabled,wp.int32),W(limits,wp.int32)],device=device)
    wp.launch(disk_clip,len(work),inputs=[W(axes,wp.vec2d),W(bounds,wp.float64),W(enabled,wp.int32),W(limits,wp.int32),1,W(g.rho,wp.float64),
        W(work,wp.int64),W(g.c,wp.vec3d),W(g.u,wp.vec3d),W(g.v,wp.vec3d),2*math.pi],
        outputs=[W(g.wall_A,wp.float64),W(g.wall_c,wp.vec3d),W(g.arc_t0,wp.float64),W(g.arc_t1,wp.float64),W(g.narc,wp.int32),W(bad,wp.int32)],device=device)
    sites=torch.cat((g.i,g.j[:m]));entries=torch.cat((torch.arange(total,device=dev),torch.arange(m,device=dev)+total))
    order=torch.argsort(sites,stable=True);ent=entries[order].to(torch.int32)
    off=torch.zeros(n+1,dtype=torch.int32,device=dev);off[1:]=torch.cumsum(torch.bincount(sites,minlength=n),0).to(torch.int32)
    Vw=torch.zeros_like(r)
    wp.launch(wall_volume,total,inputs=[W(g.i,wp.int64),W(g.j,wp.int64),W(g.wall_A,wp.float64),W(g.hi,wp.float64),W(g.hj,wp.float64)],outputs=[W(Vw,wp.float64)],device=device)
    pn=base.patch_n.clone();capA=base.patch_A.clone();volume=base.V.clone();pole=base.pole.clone()
    hidden=base.hidden|(h<=-r)
    wp.launch(k_geo_site,len(targets),inputs=[W(p,wp.vec3d),W(r,wp.float64),W(off,wp.int32),W(ent,wp.int32),total,
        W(g.nn,wp.vec3d),W(g.u,wp.vec3d),W(g.v,wp.vec3d),W(g.rho,wp.float64),W(g.hi,wp.float64),W(g.hj,wp.float64),
        W(g.arc_t0,wp.float64),W(g.arc_t1,wp.float64),W(g.narc,wp.int32),GeoPipe._fib[(str(device),pipe.npole)],
        pipe.npole,float(pipe.piece),BIG,4*math.pi,W(base.nbrs,wp.int32),W(hidden.to(torch.int32),wp.int32),
        1,1,float(ground),W(targets.to(torch.int32),wp.int32),1,W(Vw,wp.float64)],
        outputs=[W(pn,wp.vec3d),W(capA,wp.float64),W(volume,wp.float64),W(pole,wp.vec3d)],device=device)
    volume=torch.where(hidden,torch.zeros_like(volume),volume);hidden=hidden|(volume==0)
    capA=torch.where(hidden,torch.zeros_like(capA),capA);pn=torch.where(hidden[:,None],torch.zeros_like(pn),pn)
    wall_area=base.wall_A.clone();wall_c=base.wall_c.clone();wall_area[selected]=g.wall_A[:m];wall_c[selected]=g.wall_c[:m]
    internal=wall_area>0
    mesh=PowerControlVolumes(p.clone(),r.clone(),volume,base.i[internal],base.j[internal],wall_area[internal],base.nn[internal],wall_c[internal],capA,pn)
    mesh.ground_area=torch.zeros_like(r);mesh.ground_area[plane_ids]=g.wall_A[m:]
    mesh.ground_centroid=p.clone();mesh.ground_centroid[plane_ids]=g.wall_c[m:]
    mesh.ground_normal=torch.zeros_like(p);mesh.ground_normal[:,1]=-1;mesh.hidden=hidden;mesh.geometry=base
    # One host read for geometry checks and diagnostics, rather than separate
    # scalar synchronizations. No check is skipped.
    info=torch.stack((bad.any().to(dtype),(g.wall_A[m:]>0).sum().to(dtype),(volume-base.V).max(),
        (capA-base.patch_A).max(),r.max()**3)).tolist()
    if info[0]:raise RuntimeError('negative ground-clipped disk area')
    if info[2]>1e-10*info[4]:raise RuntimeError('ground clipping increased a cell volume')
    mesh.ground_geometry_info={'method':'direct_halfspace','floor_faces':int(info[1]),'recut_wall_disks':len(touched),
        'stable_sphere_area':True,'local_ground_update':True,'recomputed_cells':len(targets),'gathered_wall_disks':m,
        'max_volume_increase_m3':info[2],'max_cap_area_increase_m2':info[3]}
    return mesh
