"""Condition-aware topological stencils for compatible affine reconstruction.

Only ill-conditioned roots grow beyond shared-face neighbours. Added supports
are reached through actual material faces; no nearest-neighbour search across
vacuum. Their area weight is the maximum bottleneck face area along a path.
The same directed stencil is used in the primal map and its transpose.
"""
import torch
from free_surface.fv_ale_reconstruction import group_centroids
from free_surface.fv_ale_symmetric_eigen import eigh3


def rows(keys,areas,n):
    owner=keys//n;neighbor=keys%n
    counts=torch.bincount(owner,minlength=n)
    width=max(16,((int(counts.max())+15)//16)*16)
    starts=torch.cumsum(counts,0)-counts
    rank=torch.arange(len(keys),device=keys.device)-starts[owner]
    ids=torch.zeros((n,width),dtype=torch.long,device=keys.device)
    weights=torch.zeros((n,width),dtype=areas.dtype,device=areas.device)
    ids[owner,rank]=neighbor;weights[owner,rank]=areas
    return ids,weights


def matrices(centers,ids,areas):
    d=centers[ids]-centers[:,None,:]
    weights=areas/d.square().sum(2).clamp(min=1e-300)
    normal=(weights[:,:,None,None]*d[:,:,:,None]*d[:,:,None,:]).sum(1)
    eig,R=eigh3(normal)
    ratio=eig[:,0]/eig[:,-1].clamp(min=1e-300)
    return d,weights,eig,R,ratio


@torch.no_grad()
def build(b,minimum_ratio=1e-3,max_growth=2,transition_ratio=None):
    g=b['groups'];n=len(g);i,j,A=b['i'],b['j'],b['A']
    gi,gj=g[i],g[j];keep=(gi!=gj)&(A>0)
    raw=torch.cat((gi[keep]*n+gj[keep],gj[keep]*n+gi[keep]))
    keys,inverse=torch.unique(raw,sorted=True,return_inverse=True)
    areas=torch.zeros(len(keys),dtype=A.dtype,device=A.device).index_add(0,inverse,torch.cat((A[keep],A[keep])))
    centers=group_centroids(b);ids,area_rows=rows(keys,areas,n)
    base_ids,base_areas=ids,area_rows
    original_keys=keys
    roots=torch.arange(n,device=g.device)==g
    if transition_ratio is not None:
        if transition_ratio<=minimum_ratio:raise ValueError('stencil transition must exceed the lower ratio')
        original_ratio=matrices(centers,ids,area_rows)[-1]
        x=((transition_ratio-original_ratio)/(transition_ratio-minimum_ratio)).clamp(0,1)
        activation=x*x*(3-2*x)
        # Fixed growth depth, with extra weights vanishing smoothly at the
        # activation boundary. A ratio-based early exit would reintroduce a
        # discontinuous reconstruction/force map.
        selected=torch.nonzero(roots&(original_ratio<transition_ratio)).flatten()
    added=0;history=[];initial_bad=None
    for depth in range(max_growth+1):
        d,weights,eig,R,ratio=matrices(centers,ids,area_rows)
        bad=roots&((ratio<minimum_ratio)|(eig[:,-1]<=0))
        bad_ids=torch.nonzero(bad).flatten() if transition_ratio is None else selected
        if initial_bad is None:initial_bad=bad_ids
        history.append(int(bad.sum()))
        if not len(bad_ids) or depth==max_growth:break
        via=ids[bad_ids];path_area=area_rows[bad_ids]
        neighbor=base_ids[via]
        proxy=torch.minimum(path_area[:,:,None],base_areas[via])
        owner=bad_ids[:,None,None].expand_as(neighbor)
        valid=(proxy>0)&(neighbor!=owner)
        extra_key=owner[valid]*n+neighbor[valid];extra_area=proxy[valid]
        candidate,rev=torch.unique(extra_key,sorted=True,return_inverse=True)
        proxy_area=torch.zeros(len(candidate),dtype=A.dtype,device=A.device)
        proxy_area.scatter_reduce_(0,rev,extra_area,reduce='amax',include_self=True)
        pos=torch.searchsorted(keys,candidate)
        exists=(pos<len(keys))&(keys[pos.clamp(max=max(len(keys)-1,0))]==candidate)
        candidate=candidate[~exists];proxy_area=proxy_area[~exists]
        added+=len(candidate)
        if not len(candidate):break
        keys=torch.cat((keys,candidate));areas=torch.cat((areas,proxy_area))
        order=keys.argsort();keys=keys[order];areas=areas[order]
        ids,area_rows=rows(keys,areas,n)
    if transition_ratio is not None:
        pos=torch.searchsorted(original_keys,keys)
        existing=(pos<len(original_keys))&(original_keys[pos.clamp(max=max(len(original_keys)-1,0))]==keys)
        scaled=areas*torch.where(existing,1.,activation[keys//n])
        ids,area_rows=rows(keys,scaled,n)
        d,weights,eig,R,ratio=matrices(centers,ids,area_rows)
    # If a genuinely lower-dimensional/disconnected stencil cannot be fixed,
    # do not invert its nearly singular directions. Record the fallback.
    good=(ratio>=minimum_ratio)&(eig[:,-1]>0)
    inv=(R*torch.where(good[:,None],eig.clamp(min=1e-300).reciprocal(),0.)[:,None,:])@R.transpose(1,2)
    values={'affine_centers':centers,'affine_neighbors':ids,'affine_weights':weights,'affine_inverse':inv}
    summary={'initial_bad_roots':history[0],'bad_roots_by_growth':history,
        'fallback_roots':int((roots&~good).sum()),'extra_directed_supports':added,
        'width':ids.shape[1],'minimum_root_ratio':float(ratio[roots].min()),
        'minimum_ratio_required':minimum_ratio}
    if transition_ratio is not None:
        summary.update(transition_ratio=transition_ratio,transition_roots=len(selected),
                       fully_extended_roots=int((roots&(activation>=1)).sum()))
    return values,summary


def gradient(b,v):
    if 'affine_owner' in b:
        owner,neighbor,weight=b['affine_owner'],b['affine_neighbor'],b['affine_edge_weight']
        centers=b['affine_centers'];d=centers[neighbor]-centers[owner]
        term=weight[:,None,None]*(v[neighbor]-v[owner])[:,:,None]*d[:,None,:]
        rhs=torch.zeros((len(v),3,3),dtype=v.dtype,device=v.device).index_add(0,owner,term)
        return (rhs@b['affine_inverse'])[b['groups']]
    ids=b['affine_neighbors'];centers=b['affine_centers'];weights=b['affine_weights']
    d=centers[ids]-centers[:,None,:]
    rhs=(weights[:,:,None,None]*(v[ids]-v[:,None,:])[:,:,:,None]*d[:,:,None,:]).sum(1)
    return (rhs@b['affine_inverse'])[b['groups']]


def negative_adjoint(b,conjugate):
    from free_surface.fv_ale_reconstruction import group_sum
    if 'affine_owner' in b:
        owner,neighbor,weight=b['affine_owner'],b['affine_neighbor'],b['affine_edge_weight']
        centers=b['affine_centers'];d=centers[neighbor]-centers[owner]
        C=group_sum(conjugate,b['groups'])@b['affine_inverse']
        h=weight[:,None]*(C[owner]@d[:,:,None])[:,:,0]
        return torch.zeros_like(centers).index_add(0,owner,h).index_add(0,neighbor,-h)
    ids=b['affine_neighbors'];centers=b['affine_centers'];weights=b['affine_weights']
    d=centers[ids]-centers[:,None,:]
    C=group_sum(conjugate,b['groups'])@b['affine_inverse']
    h=weights[:,:,None]*torch.einsum('nab,nkb->nka',C,d)
    return h.sum(1)-torch.zeros_like(centers).index_add(0,ids.reshape(-1),h.reshape(-1,3))


def compact(values,capacity=0):
    """Store only actual directed supports; zero padding gives stable graph shapes.

    The reconstructed operator is unchanged. In particular, no edge is
    dropped and the very same edge list is used in the negative transpose.
    """
    ids,weights=values['affine_neighbors'],values['affine_weights'];keep=weights>0
    owner=torch.arange(len(ids),device=ids.device)[:,None].expand_as(ids)[keep]
    neighbor=ids[keep];weight=weights[keep];count=len(owner)
    capacity=max(capacity,((count+32767)//32768)*32768)
    def pad(x):
        out=torch.zeros(capacity,dtype=x.dtype,device=x.device);out[:count]=x;return out
    return {'affine_centers':values['affine_centers'],'affine_inverse':values['affine_inverse'],
            'affine_owner':pad(owner),'affine_neighbor':pad(neighbor),'affine_edge_weight':pad(weight)},capacity,count
