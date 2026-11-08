"""Exact target-cell geometry: ghosts provide planes, never get clipped.

Uses unmodified production alpha clipping, disk clipping and spherical area.
The alpha kernel only reads r[row] from the input sites, so compact target radii
and compact candidate rows suffice; wall/link IDs remain indices in full p/r.
Output geometry is valid only on requested targets and their incident faces.
"""
import math
from types import SimpleNamespace
import numpy as np
import torch,warp as wp
from free_surface.alpha_complex import AlphaComplex,k_alpha
from free_surface.geometry_warp import k_geo_pair,k_geo_site,BIG
from free_surface.geo_pipe import GeoPipe
from free_surface.fv_ale_ground_geometry import disk_clip,MAX_CONSTRAINTS
from free_surface.fv_ale_ground_local import prepare_faces,prepare_constraints,wall_volume
from free_surface.fv_ale_geometry import PowerControlVolumes


class TargetCut:
    def __init__(self,n,targets,rows,ground,device):
        self.n=n;self.targets=np.asarray(targets,dtype=np.int64);self.rows=rows
        self.ground=ground;self.device=device;self.cache={}
        self.pipe=GeoPipe(1,device)

    def storage(self,batch):
        if batch not in self.cache:
            count=len(self.targets);targets=np.concatenate([self.targets+b*self.n for b in range(batch)])
            lengths=np.array([len(row) for row in self.rows],dtype=np.int64)
            neighbours=np.concatenate([np.concatenate(self.rows)+b*self.n for b in range(batch)])
            owners=np.repeat(np.arange(count*batch),np.tile(lengths,batch))
            cv=lambda a:torch.as_tensor(a,device=self.device,dtype=torch.long)
            self.cache[batch]=(AlphaComplex(count*batch,self.device),cv(targets),cv(owners),cv(neighbours))
        return self.cache[batch]

    def geometry(self,p,r):
        if len(r)%self.n:raise RuntimeError('partial batch size mismatch')
        a,targets,rows,neighbours=self.storage(len(r)//self.n)
        nt=len(targets);n=len(r);dev=p.device;dtype=p.dtype
        W=lambda x,t:wp.from_torch(x.contiguous(),dtype=t)
        owners=targets[rows];delta=p[neighbours]-p[owners];L=delta.norm(dim=1)
        safe=L.clamp(min=1e-300);h=(L*L+r[owners]**2-r[neighbours]**2)/(2*safe)
        keep=(L>0)&(h<r[owners]);row=rows[keep];nb=neighbours[keep];hs=h[keep];normals=delta[keep]/L[keep,None]
        order=torch.argsort(hs,stable=True);order=order[torch.argsort(row[order],stable=True)]
        row=row[order];nb=nb[order];hs=hs[order];normals=normals[order]
        off=torch.zeros(nt+1,dtype=torch.int32,device=dev)
        off[1:]=torch.cumsum(torch.bincount(row,minlength=nt),0).to(torch.int32)
        if getattr(a,'coop',True):
            from free_surface.alpha_coop import clip
            clip(W(r[targets],wp.float64),W(off,wp.int32),W(nb.int(),wp.int32),W(normals,wp.vec3d),W(hs,wp.float64),
                 a.walls,a.links,a.nwall,a.nlink,a.status,nt,device=self.device)
        else:
          wp.launch(k_alpha,nt,inputs=[W(p[targets],wp.vec3d),W(r[targets],wp.float64),W(off,wp.int32),W(nb.int(),wp.int32),
            W(normals,wp.vec3d),W(hs,wp.float64),a.P,a.Pid,a.T,a.V,a.rem,a.ea,a.eb,a.nxt,a.V32,a.ed2],
            outputs=[a.walls,a.links,a.nwall,a.nlink,a.status],device=self.device)
        walls=wp.to_torch(a.walls);counts=wp.to_torch(a.nwall);links=wp.to_torch(a.links);nlink=wp.to_torch(a.nlink)
        status=wp.to_torch(a.status)
        bad,degree=torch.stack((status.max(),counts.max())).tolist()
        if bad>1:raise RuntimeError('target clipping failed, status %d'%bad)
        mask=torch.zeros(n,dtype=torch.bool,device=dev);mask[targets]=True
        slots=torch.arange(walls.shape[1],device=dev)[None,:]
        rr,ss=torch.nonzero(slots<counts[:,None],as_tuple=True)
        own=targets[rr];other=walls[rr,ss].long()
        # One shared face: a selected lower endpoint wins; otherwise its sole
        # selected endpoint supplies the exact edge/link list.
        take=(own<other)|~mask[other];rr=rr[take];ss=ss[take];own=own[take];other=other[take]
        pairs=torch.stack((torch.minimum(own,other),torch.maximum(own,other)),1)
        linklist=links[rr,ss];nl=nlink[rr,ss]
        length=(p[pairs[:,1]]-p[pairs[:,0]]).norm(dim=1)
        hi=(length*length+r[pairs[:,0]]**2-r[pairs[:,1]]**2)/(2*length)
        positive=(r[pairs[:,0]]**2-hi**2)>0
        pairs=pairs[positive];linklist=linklist[positive];nl=nl[positive]
        order=torch.argsort(pairs[:,0]*n+pairs[:,1],stable=True)
        pairs=pairs[order];linklist=linklist[order];nl=nl[order]
        K=max(int(nl.max()) if len(nl) else 0,1);m=len(pairs)
        linklist=linklist[:,:K].clone()
        linklist[torch.arange(K,device=dev)[None]>=nl[:,None]]=-1
        width=max(int(degree),1)
        nbrs=torch.full((n,width),-1,dtype=torch.int32,device=dev)
        compact=walls[:,:width].clone();compact[torch.arange(width,device=dev)[None]>=counts[:,None]]=-1
        nbrs[targets]=compact
        hidden=~mask;hidden[targets]=status==1
        E=lambda *shape:torch.empty(shape,dtype=dtype,device=dev)
        Z=lambda *shape:torch.zeros(shape,dtype=dtype,device=dev)
        base=SimpleNamespace(i=pairs[:,0],j=pairs[:,1])
        for name in ['nn','u','v','c','wall_c']:setattr(base,name,E(m,3))
        for name in ['rho','hi','hj','wall_A']:setattr(base,name,E(m))
        base.arc_t0=E(m,2*K+1);base.arc_t1=E(m,2*K+1);base.narc=torch.zeros(m,dtype=torch.int32,device=dev)
        Vw=Z(n);ea=E(m,2*K);ed=E(m,2*K);el=torch.empty((m,2*K),device=dev,dtype=torch.int32)
        if m:
            wp.launch(k_geo_pair,m,inputs=[W(p,wp.vec3d),W(r,wp.float64),W(pairs.int(),wp.int32),W(linklist.int(),wp.int32),
                K,2*math.pi,BIG,1e299,.9,W(ea,wp.float64),W(ed,wp.float64),W(el,wp.int32)],
                outputs=[W(base.nn,wp.vec3d),W(base.c,wp.vec3d),W(base.u,wp.vec3d),W(base.v,wp.vec3d),
                    W(base.rho,wp.float64),W(base.hi,wp.float64),W(base.hj,wp.float64),W(base.arc_t0,wp.float64),
                    W(base.arc_t1,wp.float64),W(base.narc,wp.int32),W(base.wall_A,wp.float64),W(base.wall_c,wp.vec3d),W(Vw,wp.float64)],device=self.device)
        h=p[:,1]-self.ground;floor_ids=targets[(h[targets].abs()<r[targets])&~hidden[targets]]
        nf=len(floor_ids);total=m+nf;C=width+1
        if C>MAX_CONSTRAINTS:raise RuntimeError('target ground constraint capacity exceeded')
        fields=['i','j','nn','u','v','c','rho','hi','hj','wall_A','wall_c','narc','arc_t0','arc_t1']
        types=[wp.int64,wp.int64,wp.vec3d,wp.vec3d,wp.vec3d,wp.vec3d,wp.float64,wp.float64,wp.float64,wp.float64,wp.vec3d,wp.int32,wp.float64,wp.float64]
        g=SimpleNamespace();arcwidth=max(2*C+1,2*K+1)
        for name in fields:
            src=getattr(base,name);shape=(total,arcwidth) if name in ['arc_t0','arc_t1'] else (total,)+src.shape[1:]
            setattr(g,name,torch.empty(shape,dtype=src.dtype,device=dev))
        if total:
            wp.launch(prepare_faces,total,inputs=[W(torch.arange(m,device=dev),wp.int64),W(floor_ids,wp.int64),m,W(p,wp.vec3d),W(r,wp.float64),float(self.ground)]+
                [W(getattr(base,name),t) for name,t in zip(fields,types)],outputs=[W(getattr(g,name),t) for name,t in zip(fields,types)],device=self.device)
        touched=torch.nonzero(base.c[:,1]-base.rho*torch.sqrt(base.u[:,1]**2+base.v[:,1]**2)<self.ground).flatten()
        work=torch.cat((touched,torch.arange(m,total,device=dev)))
        owner=g.i.clone();other=g.j.clone()
        swap=~mask[owner];owner[swap]=g.j[swap];other[swap]=g.i[swap]
        axes=E(len(work),C,2);bounds=E(len(work),C);enabled=torch.zeros((len(work),C),device=dev,dtype=torch.int32)
        limits=torch.zeros(len(work),device=dev,dtype=torch.int32);flags=torch.zeros_like(limits)
        if len(work):
            wp.launch(prepare_constraints,len(work),inputs=[W(work,wp.int64),W(owner,wp.int64),W(other,wp.int64),W(nbrs,wp.int32),
                W(p,wp.vec3d),W(r,wp.float64),W(g.c,wp.vec3d),W(g.u,wp.vec3d),W(g.v,wp.vec3d),W(g.rho,wp.float64),float(self.ground)],
                outputs=[W(axes,wp.vec2d),W(bounds,wp.float64),W(enabled,wp.int32),W(limits,wp.int32)],device=self.device)
            wp.launch(disk_clip,len(work),inputs=[W(axes,wp.vec2d),W(bounds,wp.float64),W(enabled,wp.int32),W(limits,wp.int32),1,
                W(g.rho,wp.float64),W(work,wp.int64),W(g.c,wp.vec3d),W(g.u,wp.vec3d),W(g.v,wp.vec3d),2*math.pi],
                outputs=[W(g.wall_A,wp.float64),W(g.wall_c,wp.vec3d),W(g.arc_t0,wp.float64),W(g.arc_t1,wp.float64),W(g.narc,wp.int32),W(flags,wp.int32)],device=self.device)
        sites=torch.cat((g.i,base.j));entries=torch.cat((torch.arange(total,device=dev),torch.arange(m,device=dev)+total))
        order=torch.argsort(sites,stable=True);ent=entries[order].int()
        off=torch.zeros(n+1,dtype=torch.int32,device=dev);off[1:]=torch.cumsum(torch.bincount(sites,minlength=n),0).int()
        Vw.zero_()
        if total:wp.launch(wall_volume,total,inputs=[W(g.i,wp.int64),W(g.j,wp.int64),W(g.wall_A,wp.float64),W(g.hi,wp.float64),W(g.hj,wp.float64)],outputs=[W(Vw,wp.float64)],device=self.device)
        pn=Z(n,3);cap=Z(n);vol=Z(n);pole=Z(n,3);hidden=hidden|(h<=-r)
        wp.launch(k_geo_site,nt,inputs=[W(p,wp.vec3d),W(r,wp.float64),W(off,wp.int32),W(ent,wp.int32),total,
            W(g.nn,wp.vec3d),W(g.u,wp.vec3d),W(g.v,wp.vec3d),W(g.rho,wp.float64),W(g.hi,wp.float64),W(g.hj,wp.float64),
            W(g.arc_t0,wp.float64),W(g.arc_t1,wp.float64),W(g.narc,wp.int32),GeoPipe._fib[(self.device,32)],32,float(math.pi/6),BIG,4*math.pi,
            W(nbrs,wp.int32),W(hidden.int(),wp.int32),1,1,float(self.ground),W(targets.int(),wp.int32),1,W(Vw,wp.float64)],
            outputs=[W(pn,wp.vec3d),W(cap,wp.float64),W(vol,wp.float64),W(pole,wp.vec3d)],device=self.device)
        vol=torch.where(hidden,0.,vol);hidden=hidden|(vol==0);cap=torch.where(hidden,0.,cap);pn=torch.where(hidden[:,None],0.,pn)
        positive=g.wall_A[:m]>0
        mesh=PowerControlVolumes(p,r,vol,base.i[positive],base.j[positive],g.wall_A[:m][positive],base.nn[positive],g.wall_c[:m][positive],cap,pn)
        mesh.ground_area=Z(n);mesh.ground_area[floor_ids]=g.wall_A[m:]
        mesh.ground_centroid=p.clone();mesh.ground_centroid[floor_ids]=g.wall_c[m:]
        mesh.ground_normal=Z(n,3);mesh.ground_normal[:,1]=-1;mesh.hidden=hidden
        if bool(flags.any()) or not bool(torch.isfinite(vol[targets]).all()&(vol[targets]>=0).all()):
            raise RuntimeError('invalid target geometry')
        mesh.ground_geometry_info={'target_only':True,'clipped_cells':nt,'constraint_cells':n}
        return mesh
