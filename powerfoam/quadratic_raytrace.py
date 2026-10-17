"""Exact quadratic-event reference tracing for per-cell anisotropic Power Foam.

Forward-only float64 PyTorch implementation. Enumerates all potentially relevant
support, pairwise-power and dipole events, then classifies/merges ray intervals.
No planar-interface, convex-cell or global painter-order assumption is used.
"""

import numpy as np
import torch


def quadratic_roots(a, b, c):
    scale = torch.maximum(torch.maximum(a.abs(), b.abs()), c.abs()).clamp_min(1.)
    linear = a.abs() <= 1e-12 * scale
    discr = b*b - 4*a*c
    valid = ~linear & (discr >= 0)
    # Stable q-form avoids cancellation in one of the two roots.
    q = -.5 * (b + torch.where(b >= 0, 1., -1.) * discr.clamp_min(0).sqrt())
    safe_a = torch.where(linear, torch.ones_like(a), a)
    first = q / safe_a
    second = c / torch.where(q.abs() > 1e-30, q, torch.ones_like(q))
    repeated = q.abs() <= 1e-30
    first = torch.where(repeated, -b/(2*safe_a), first)
    second = torch.where(repeated, first, second)
    inf = torch.full_like(a, float('inf'))
    first, second = torch.where(valid, first, inf), torch.where(valid, second, inf)
    line_valid = linear & (b.abs() > 1e-12*scale)
    first = torch.where(line_valid, -c/torch.where(line_valid,b,torch.ones_like(b)), first)
    return first, second


class QuadraticRayTracer:
    def __init__(self, reference_centers, radii, normals, texel_sites, texel_colors,
                 density=1000., device='cuda:0', batch_size=256):
        self.device, self.batch_size = device, batch_size
        self.tensor = lambda x: torch.as_tensor(np.asarray(x),dtype=torch.float64,device=device)
        self.p0, self.r, self.n0 = map(self.tensor,(reference_centers,radii,normals))
        self.sites, self.colors = map(self.tensor,(texel_sites,texel_colors))
        self.sigma = self.tensor(np.broadcast_to(density,(len(reference_centers),)).copy())
        if len(self.p0)==0 or torch.any(self.r<=0) or torch.any(self.sigma<0):
            raise ValueError('Need nonempty cells, positive radii and nonnegative extinction')

    @torch.no_grad()
    def render(self, origins, directions, centers, deformations, return_intervals=False):
        f=np.asarray(deformations,dtype=float)
        if f.shape!=(len(self.p0),3,3) or not np.isfinite(f).all():
            raise ValueError('Expected one finite 3x3 F per cell')
        if np.any(np.linalg.det(f)<=0) or np.any(np.linalg.cond(f)>1e5):
            raise ValueError('F must preserve orientation and be well-conditioned')
        p=self.tensor(centers); inv=self.tensor(np.linalg.inv(f))
        o,d=map(self.tensor,(origins,directions))
        if o.shape!=d.shape or o.ndim!=2 or o.shape[1]!=3:
            raise ValueError('Rays must have shape [N,3]')
        if not torch.allclose(d.norm(dim=1),torch.ones(len(d),dtype=d.dtype,device=d.device),atol=1e-9,rtol=1e-9):
            raise ValueError('Unit world ray directions required')
        normal=torch.einsum('ni,nij->nj',self.n0,inv)
        normal/=normal.norm(dim=1,keepdim=True)
        chunks=[]; intervals=[]
        for start in range(0,len(o),self.batch_size):
            result,segments=self._batch(o[start:start+self.batch_size],d[start:start+self.batch_size],p,inv,normal)
            chunks.append(result)
            if return_intervals: intervals.extend(segments)
        result={key:np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
        if return_intervals: result['intervals']=intervals
        return result

    def _batch(self, o, d, p, inv, normals):
        b=len(o); device=o.device
        # Shift parameter origin near the object to condition polynomial roots.
        anchor=((p.mean(0)-o)*d).sum(1)
        origin=o+anchor[:,None]*d
        local_o=torch.einsum('nij,bnj->bni',inv,origin[:,None]-p)
        local_d=torch.einsum('nij,bj->bni',inv,d)
        a=local_d.square().sum(2)
        bb=2*(local_o*local_d).sum(2)
        c=local_o.square().sum(2)-self.r.square()
        mid=-bb/(2*a)
        closest=local_o+mid[:,:,None]*local_d
        residual=self.r.square()-closest.square().sum(2)
        half=(residual.clamp_min(0)/a).sqrt()
        near=torch.maximum(mid-half,-anchor[:,None]);far=mid+half
        hit=(residual>0)&(far>near)
        count=hit.sum(1);k=int(count.max())
        empty=dict(rgb=np.zeros((b,3)),alpha=np.zeros(b),depth=np.zeros(b),owner=np.full(b,-1,dtype=int),
                   normal=np.zeros((b,3)),segments=np.zeros(b,dtype=int),max_reentries=np.zeros(b,dtype=int))
        if k==0: return empty,[[] for _ in range(b)]
        # Compact every ray's actual support candidates; no fixed truncation.
        indices=torch.arange(len(p),device=device)[None].expand(b,-1)
        ids=torch.where(hit,indices,len(p)).sort(1).values[:,:k]
        active=ids<len(p);ids=ids.clamp_max(len(p)-1)
        row=torch.arange(b,device=device)[:,None]
        aa,bb,cc=(v[row,ids] for v in (a,bb,c))
        lo,hi=(v[row,ids] for v in (near,far))
        lin_o=(local_o[row,ids]*self.n0[ids]).sum(2)
        lin_d=(local_d[row,ids]*self.n0[ids]).sum(2)
        inf=torch.full_like(lo,float('inf'))
        bounds=torch.cat((torch.where(active,lo,inf),torch.where(active,hi,inf)),1)
        i,j=torch.triu_indices(k,k,offset=1,device=device)
        roots=quadratic_roots(aa[:,i]-aa[:,j],bb[:,i]-bb[:,j],cc[:,i]-cc[:,j])
        lower=torch.maximum(lo[:,i],lo[:,j]);upper=torch.minimum(hi[:,i],hi[:,j])
        pair_active=active[:,i]&active[:,j]
        pair_roots=[torch.where(pair_active&(r>=lower)&(r<=upper),r,torch.full_like(r,float('inf'))) for r in roots]
        plane=-lin_o/torch.where(lin_d.abs()>1e-12,lin_d,torch.ones_like(lin_d))
        plane=torch.where(active&(lin_d.abs()>1e-12)&(plane>=lo)&(plane<=hi),plane,inf)
        events=torch.cat((bounds,*pair_roots,plane),1).sort(1).values
        keep=torch.isfinite(events)
        keep[:,1:] &= (events[:,1:]-events[:,:-1])>1e-10
        lengths=keep.sum(1);width=int(lengths.max())
        if width<2:return empty,[[] for _ in range(b)]
        compact=torch.full((b,width),float('inf'),dtype=o.dtype,device=device)
        target=keep.cumsum(1)-1
        rr=torch.arange(b,device=device)[:,None].expand_as(events)
        compact[rr[keep],target[keep]]=events[keep]
        entry,exit=compact[:,:-1],compact[:,1:]
        valid=torch.isfinite(exit)&(exit>entry)
        midpoint=torch.where(valid,(entry+exit)*.5,torch.zeros_like(entry))
        value=(aa[:,None]*midpoint[:,:,None]+bb[:,None])*midpoint[:,:,None]+cc[:,None]
        value=torch.where(active[:,None],value,torch.full_like(value,float('inf')))
        power,winner=value.min(2)
        winning_plane=lin_o.gather(1,winner)+lin_d.gather(1,winner)*midpoint
        occupied=valid&(power<=0)&(winning_plane<=0)
        labels=torch.where(occupied,ids.gather(1,winner),-1)
        # Merge spurious event subdivisions before querying entry-point color.
        # A genuine departure/re-entry remains separate even for the same owner.
        change=torch.ones_like(labels,dtype=torch.bool)
        change[:,1:]=labels[:,1:]!=labels[:,:-1]
        group=change.cumsum(1)-1;groups=int(group.max())+1
        begin=torch.full((b,groups),float('inf'),dtype=o.dtype,device=device)
        end=torch.full((b,groups),-float('inf'),dtype=o.dtype,device=device)
        begin.scatter_reduce_(1,group,torch.where(valid,entry,float('inf')),reduce='amin')
        end.scatter_reduce_(1,group,torch.where(valid,exit,-float('inf')),reduce='amax')
        owner=torch.full((b,groups),-1,dtype=torch.long,device=device)
        owner.scatter_reduce_(1,group,labels,reduce='amax')
        occupied=(owner>=0)&(end>begin)
        safe_id=owner.clamp_min(0)
        start=torch.where(occupied,begin,torch.zeros_like(begin))
        world=origin[:,None]+start[:,:,None]*d[:,None]
        canonical=self.p0[safe_id]+torch.einsum('bgij,bgj->bgi',inv[safe_id],world-p[safe_id])
        dist=(canonical[:,:,None]-self.sites[safe_id]).square().sum(3)
        weight=torch.exp(-10*dist/self.r[safe_id,None].square())
        color=(weight[:,:,:,None]*self.colors[safe_id]).sum(2)/weight.sum(2,keepdim=True).clamp_min(1e-20)
        interval_length=torch.where(occupied,end-begin,torch.zeros_like(begin))
        alpha=-torch.expm1(-self.sigma[safe_id]*interval_length)
        trans=torch.cumprod(torch.cat((torch.ones((b,1),dtype=o.dtype,device=device),1-alpha[:,:-1]),1),1)
        contribution=alpha*trans
        first=occupied.long().argmax(1)
        has_hit=occupied.any(1)
        first_depth=torch.where(has_hit,begin.gather(1,first[:,None]).squeeze(1)+anchor,0.)
        first_id=torch.where(has_hit,owner.gather(1,first[:,None]).squeeze(1),-1)
        counts=torch.zeros((b,len(p)),dtype=torch.long,device=device)
        counts.scatter_add_(1,safe_id,occupied.long())
        result=dict(rgb=(contribution[:,:,None]*color).sum(1),alpha=1-torch.prod(1-alpha,1),
                    depth=first_depth,owner=first_id,normal=(contribution[:,:,None]*normals[safe_id]).sum(1),
                    segments=occupied.sum(1),max_reentries=(counts.max(1).values-1).clamp_min(0))
        result={key:v.cpu().numpy() for key,v in result.items()}
        # Diagnostic interval output is small for the reference scenes and also
        # supports independent point classification tests.
        starts=(begin+anchor[:,None]).cpu().numpy();ends=(end+anchor[:,None]).cpu().numpy()
        owners=owner.cpu().numpy();mask=occupied.cpu().numpy()
        intervals=[[(float(starts[n,g]),float(ends[n,g]),int(owners[n,g])) for g in np.flatnonzero(mask[n])] for n in range(b)]
        return result,intervals
