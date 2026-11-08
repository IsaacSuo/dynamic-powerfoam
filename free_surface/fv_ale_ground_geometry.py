"""Direct half-space clipping of power-ball cells, without fictitious sites.

Only wall disks crossing the floor are re-clipped. The floor contributes one
additional clipped disk per intersected ball. Their exposed circular arcs feed
the same analytic spherical-sector integral as the original geometry kernel.
All internal faces are computed once and shared. No surface point quadrature.
"""
import math
import torch,warp as wp
from free_surface.recut import Recut
from free_surface.geo_pipe import GeoPipe
from free_surface.geometry_warp import k_geo_site,BIG
from free_surface.fv_ale_geometry import PowerControlVolumes

MAX_CONSTRAINTS=96
Events=wp.types.vector(length=2*MAX_CONSTRAINTS,dtype=wp.float64)
Signs=wp.types.vector(length=2*MAX_CONSTRAINTS,dtype=wp.int32)


@wp.kernel(enable_backward=False)
def disk_clip(axes:wp.array2d(dtype=wp.vec2d),bounds:wp.array2d(dtype=wp.float64),
    valid:wp.array2d(dtype=wp.int32),limits:wp.array(dtype=wp.int32),use_limits:int,radii:wp.array(dtype=wp.float64),
    work:wp.array(dtype=wp.int64),centres:wp.array(dtype=wp.vec3d),
    uu:wp.array(dtype=wp.vec3d),vv:wp.array(dtype=wp.vec3d),two_pi:wp.float64,
    area:wp.array(dtype=wp.float64),centroid:wp.array(dtype=wp.vec3d),
    starts:wp.array2d(dtype=wp.float64),ends:wp.array2d(dtype=wp.float64),
    counts:wp.array(dtype=wp.int32),bad:wp.array(dtype=wp.int32)):
    row=wp.tid();f=work[row];rho=radii[f]
    zero=wp.float64(0.);one=wp.float64(1.);two=wp.float64(2.);pi=two_pi/two
    area[f]=zero;counts[f]=0;centroid[f]=centres[f];bad[row]=0
    if rho<=zero:return
    angles=Events();signs=Signs();ne=int(0);coverage=int(0);empty=int(0)
    K=axes.shape[1]
    if use_limits==1:K=limits[row]
    for k in range(K):
        if valid[row,k]==0:continue
        a=axes[row,k];b=bounds[row,k];length=wp.length(a)
        if length==zero:
            if b<zero:empty=1
            continue
        q=b/(rho*length)
        if q<=-one:empty=1
        elif q<one:
            phi=wp.atan2(a[1],a[0]);half=wp.acos(q)
            lo=phi-half;lo=lo-two*pi*wp.floor(lo/(two*pi))
            hi=phi+half;hi=hi-two*pi*wp.floor(hi/(two*pi))
            if lo>hi:coverage=coverage+1
            angles[ne]=lo;signs[ne]=1;ne=ne+1
            angles[ne]=hi;signs[ne]=-1;ne=ne+1
    if empty==1:return
    for k in range(1,ne):
        angle=angles[k];sg=signs[k];j=k-1
        while j>=0:
            if angles[j]<=angle:break
            angles[j+1]=angles[j];signs[j+1]=signs[j];j=j-1
        angles[j+1]=angle;signs[j+1]=sg
    A=zero;Mx=zero;My=zero;previous=zero;na=int(0)
    for k in range(ne+1):
        endpoint=two*pi
        if k<ne:endpoint=angles[k]
        if coverage==0 and endpoint>previous:
            D=endpoint-previous
            starts[f,na]=previous;ends[f,na]=endpoint;na=na+1
            A=A+rho*rho*D/two
            Mx=Mx+rho*rho*rho*(wp.sin(endpoint)-wp.sin(previous))/wp.float64(3.)
            My=My+rho*rho*rho*(wp.cos(previous)-wp.cos(endpoint))/wp.float64(3.)
        if k<ne:coverage=coverage+signs[k]
        previous=endpoint
    # Straight edges of the disk/half-plane intersection, positively oriented.
    for k in range(K):
        if valid[row,k]==0:continue
        a=axes[row,k];b=bounds[row,k];norm2=wp.dot(a,a)
        if norm2==zero:continue
        foot=a*(b/norm2);foot2=wp.dot(foot,foot)
        if foot2>=rho*rho:continue
        direction=wp.vec2d(-a[1],a[0])/wp.sqrt(norm2)
        half=wp.sqrt(rho*rho-foot2);lo=-half;hi=half;skip=int(0)
        for j in range(K):
            if j==k or valid[row,j]==0:continue
            other=axes[row,j];rhs=bounds[row,j]-wp.dot(other,foot);ad=wp.dot(other,direction)
            if ad>zero:hi=wp.min(hi,rhs/ad)
            elif ad<zero:lo=wp.max(lo,rhs/ad)
            elif rhs<zero:skip=1
        if skip==1 or hi<=lo:continue
        P0=foot+lo*direction;P1=foot+hi*direction
        cross=P0[0]*P1[1]-P0[1]*P1[0]
        A=A+cross/two
        Mx=Mx+cross*(P0[0]+P1[0])/wp.float64(6.)
        My=My+cross*(P0[1]+P1[1])/wp.float64(6.)
    if A < -wp.float64(1e-10)*rho*rho:bad[row]=1
    if A>zero:
        area[f]=A;centroid[f]=centres[f]+(Mx/A)*uu[f]+(My/A)*vv[f]
    counts[f]=na


class GroundClip:
    def __init__(self,ground,device,local=True):
        self.ground=ground;self.device=device;self.recut=None;self.size=0
        self.local=local  # False retains the full-field reference for replay checks.
        import os
        self.plane_cut=os.environ.get('FV_GROUND_PLANE','1')!='0'  # 11-08: the ground as a clip plane of the cut (fv_ale_ground_plane); FV_GROUND_PLANE=0: the local re-clip

    def geometry(self,p,r):
        if self.plane_cut and self.local:
            from free_surface.fv_ale_ground_plane import plane_geometry
            return plane_geometry(self,p,r)
        n=len(r);dev=p.device;dtype=p.dtype
        if n!=self.size:
            self.size=n;self.recut=Recut(n,self.device);self.recut.pipe=GeoPipe(n,self.device)
            self.recut.pipe._cand_storage(1);self.recut.pipe.stable_area=True
        g,_=self.recut.geometry(p.contiguous(),r.contiguous())
        h=p[:,1]-self.ground
        plane_ids=torch.nonzero((h.abs()<r)&~g.hidden).flatten()
        touched=torch.nonzero(g.c[:,1]-g.rho*torch.sqrt(g.u[:,1]**2+g.v[:,1]**2)<self.ground).flatten()
        m=len(g.i);nf=len(plane_ids);total=m+nf
        if nf==0 and len(touched)==0:
            mesh=PowerControlVolumes.from_geometry(g)
            below=h<=-r
            mesh.volume=torch.where(below,torch.zeros_like(mesh.volume),mesh.volume)
            mesh.cap_area=torch.where(below,torch.zeros_like(mesh.cap_area),mesh.cap_area)
            mesh.cap_normal_integral=torch.where(below[:,None],torch.zeros_like(mesh.cap_normal_integral),mesh.cap_normal_integral)
            mesh.ground_area=torch.zeros_like(r);mesh.ground_centroid=p.clone()
            mesh.ground_normal=torch.zeros_like(p);mesh.ground_normal[:,1]=-1
            mesh.hidden=g.hidden|below;mesh.geometry=g
            mesh.ground_geometry_info={'method':'direct_halfspace','floor_faces':0,'recut_wall_disks':0,'stable_sphere_area':True}
            return mesh
        if self.local:
            from free_surface.fv_ale_ground_local import local_geometry
            return local_geometry(p,r,g,h,plane_ids,touched,self.ground,self.device,self.recut.pipe)
        base=g
        K=g.nbrs.shape[1]+1
        if K>MAX_CONSTRAINTS:raise RuntimeError('ground disk constraint capacity exceeded')
        axis=torch.zeros((nf,3),dtype=dtype,device=dev);axis[:,1]=-1
        fu=torch.zeros_like(axis);fu[:,0]=1
        fv=torch.zeros_like(axis);fv[:,2]=1  # u cross v = -ey
        fc=p[plane_ids].clone();fc[:,1]=self.ground
        i=torch.cat((g.i,plane_ids));j=torch.cat((g.j,torch.full_like(plane_ids,-1)))
        normals=torch.cat((g.nn,axis));u=torch.cat((g.u,fu));v=torch.cat((g.v,fv))
        centres=torch.cat((g.c,fc));rho=torch.cat((g.rho,torch.sqrt((r[plane_ids]**2-h[plane_ids]**2).clamp(min=0))))
        hi=torch.cat((g.hi,h[plane_ids]));hj=torch.cat((g.hj,torch.zeros(nf,dtype=dtype,device=dev)))
        A=torch.cat((g.wall_A.clone(),torch.zeros(nf,dtype=dtype,device=dev)))
        C=torch.cat((g.wall_c.clone(),fc));count=torch.cat((g.narc.clone(),torch.zeros(nf,dtype=torch.int32,device=dev)))
        width=max(2*K+1,g.arc_t0.shape[1]);t0=torch.zeros((total,width),dtype=dtype,device=dev);t1=torch.zeros_like(t0)
        t0[:m,:g.arc_t0.shape[1]]=g.arc_t0;t1[:m,:g.arc_t1.shape[1]]=g.arc_t1
        work=torch.cat((touched,torch.arange(m,total,device=dev)))
        owners=i[work];nb=g.nbrs[owners].long();valid=(nb>=0)&(nb!=j[work,None]);safe=nb.clamp(min=0)
        delta=p[safe]-p[owners,None]
        aa=2*torch.stack(((delta*u[work,None]).sum(2),(delta*v[work,None]).sum(2)),2)
        bb=(delta*delta).sum(2)+r[owners,None]**2-r[safe]**2-2*(delta*(centres[work]-p[owners])[:,None]).sum(2)
        ground_axes=-torch.stack((u[work,1],v[work,1]),1)[:,None]
        axes=torch.cat((aa,ground_axes),1).contiguous()
        bounds=torch.cat((bb,(centres[work,1]-self.ground)[:,None]),1).contiguous()
        enabled=torch.cat((valid,(j[work]>=0)[:,None]),1).to(torch.int32).contiguous()
        bad=torch.zeros(len(work),dtype=torch.int32,device=dev)
        W=lambda a,t:wp.from_torch(a.contiguous(),dtype=t)
        wp.launch(disk_clip,len(work),inputs=[W(axes,wp.vec2d),W(bounds,wp.float64),W(enabled,wp.int32),W(bad,wp.int32),0,W(rho,wp.float64),
            W(work,wp.int64),W(centres,wp.vec3d),W(u,wp.vec3d),W(v,wp.vec3d),2*math.pi],
            outputs=[W(A,wp.float64),W(C,wp.vec3d),W(t0,wp.float64),W(t1,wp.float64),W(count,wp.int32),W(bad,wp.int32)],device=self.device)
        if bool(bad.any()):raise RuntimeError('negative ground-clipped disk area')
        # Rebuild oriented face lists for the existing exact sphere-patch kernel.
        sites=torch.cat((i,g.j));entries=torch.cat((torch.arange(total,device=dev),torch.arange(m,device=dev)+total))
        order=torch.argsort(sites,stable=True);ent=entries[order].to(torch.int32)
        off=torch.zeros(n+1,dtype=torch.int32,device=dev)
        off[1:]=torch.cumsum(torch.bincount(sites,minlength=n),0).to(torch.int32)
        Vw=torch.zeros_like(r).index_add(0,i,A*hi).index_add(0,g.j,A[:m]*hj[:m])
        pn=torch.empty_like(p);capA=torch.empty_like(r);volume=torch.empty_like(r);pole=torch.empty_like(p)
        hidden=g.hidden|(h<=-r)
        pipe=self.recut.pipe
        wp.launch(k_geo_site,n,inputs=[W(p,wp.vec3d),W(r,wp.float64),W(off,wp.int32),W(ent,wp.int32),total,
            W(normals,wp.vec3d),W(u,wp.vec3d),W(v,wp.vec3d),W(rho,wp.float64),W(hi,wp.float64),W(hj,wp.float64),
            W(t0,wp.float64),W(t1,wp.float64),W(count,wp.int32),GeoPipe._fib[(str(self.device),pipe.npole)],
            pipe.npole,float(pipe.piece),BIG,4*math.pi,W(g.nbrs.to(torch.int32),wp.int32),W(hidden.to(torch.int32),wp.int32),
            1,1,float(self.ground),W(ent,wp.int32),0,W(Vw,wp.float64)],outputs=[W(pn,wp.vec3d),W(capA,wp.float64),W(volume,wp.float64),W(pole,wp.vec3d)],device=self.device)
        volume=torch.where(hidden,torch.zeros_like(volume),volume)
        # A power cell may lie wholly below the plane even if its ball does
        # not. Mark exact empty intersections as hidden as well.
        hidden=hidden|(volume==0)
        capA=torch.where(hidden,torch.zeros_like(capA),capA);pn=torch.where(hidden[:,None],torch.zeros_like(pn),pn)
        wall_area=A[:m];wall_centroid=C[:m]
        internal=wall_area>0
        mesh=PowerControlVolumes(p.clone(),r.clone(),volume,base.i[internal].clone(),base.j[internal].clone(),
            wall_area[internal],base.nn[internal].clone(),wall_centroid[internal],capA,pn)
        mesh.ground_area=torch.zeros_like(r).index_add(0,plane_ids,A[m:])
        mesh.ground_centroid=p.clone();mesh.ground_centroid[plane_ids]=C[m:]
        mesh.ground_normal=torch.zeros_like(p);mesh.ground_normal[:,1]=-1
        mesh.hidden=hidden;mesh.geometry=base
        mesh.ground_geometry_info={'method':'direct_halfspace','floor_faces':int((A[m:]>0).sum()),
            'recut_wall_disks':len(touched),'stable_sphere_area':True,
            'local_ground_update':self.local,'recomputed_cells':n,'gathered_wall_disks':m,
            'max_volume_increase_m3':float((volume-g.V).max()),
            'max_cap_area_increase_m2':float((capA-g.patch_A).max())}
        if float((volume-g.V).max())>1e-10*float(r.max()**3):
            raise RuntimeError('ground clipping increased a cell volume')
        return mesh
