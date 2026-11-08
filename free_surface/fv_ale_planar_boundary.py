"""Geometric material/vacuum power faces for the opt-in moving boundary.

Vacuum generators have no material variables. Their sole purpose is to retain
a planar material surface after it leaves the fixed ground half-space. All
real-real disks and curved caps still use the original power-ball geometry.
The same geometric state must accompany every stage, save and replay.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch


@dataclass
class Planes:
    owner: torch.Tensor
    d: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor

    def clone(self):
        return Planes(*(getattr(self,k).clone() for k in ['owner','d','a','b']))

    def to(self,device):
        return Planes(*(getattr(self,k).to(device) for k in ['owner','d','a','b']))

    def packed(self):
        return {k:getattr(self,k).detach().cpu().clone() for k in ['owner','d','a','b']}

    @classmethod
    def floor(cls,p,r,owners,ground):
        owners=torch.as_tensor(owners,dtype=torch.long,device=p.device)
        a=p.new_zeros((len(owners),3));a[:,1]=-1
        d=torch.maximum(r.new_ones(len(owners)),2*(p[owners,1]-ground).clamp(min=0)+r[owners])
        return cls(owners,d,a,r.new_full((len(owners),),-ground))

    def generators(self,p,r,ground):
        o,d,a,b=self.owner,self.d,self.a,self.b
        q=p[o]+d[:,None]*a
        weight=r[o].square()+d.square()*a.square().sum(1)-2*d*(b-(a*p[o]).sum(1))
        if bool((weight<=0).any()):raise ValueError('moving vacuum power weight became nonpositive')
        radius=weight.sqrt()
        # Certify the redundant-floor mask using exact input doubles. Avoid
        # counting a coincident floor twice or opening a roundoff-sized gap
        # below it. No mass, volume or state is overwritten by this rounding.
        redundant=torch.zeros(len(r)+len(d),dtype=torch.bool,device=p.device)
        flat=(a[:,0]==0)&(a[:,1]==-1)&(a[:,2]==0)&(-b>=ground)
        if bool(flat.any()):
            import mpmath as mp
            for j in torch.nonzero(flat).flatten().tolist():
                owner=int(o[j]);height=float(-b[j])
                with mp.workdps(60):
                    py=mp.mpf(float(p[owner,1]));qy=mp.mpf(float(q[j,1]));dd=py-qy
                    target=mp.mpf(float(r[owner]))**2+dd**2-2*dd*(py-mp.mpf(height))
                    if target<=0:raise ValueError('nonpositive certified vacuum weight')
                    rad=math.sqrt(float(target))
                    while mp.mpf(rad)**2<target:rad=math.nextafter(rad,math.inf)
                    radius[j]=rad;redundant[owner]=True
        return torch.cat((p,q)),torch.cat((r,radius)),redundant


def augmented_structure(p,r,base,material_count):
    """Append complete cutting half-spaces without triangulating vacuum cells.

    Adding half-spaces cannot create a new real-real face absent from the
    original power diagram. Test every vacuum plane against every real BALL;
    this conservative test includes all potentially active vacuum constraints,
    not just nearby generators. Each new face receives all original owner
    neighbors and all relevant vacuum constraints. Original real links remain.
    """
    n=material_count;ng=len(r)-n
    pairs=base['pairs'].long();links=base['links'].long();nbrs=base['nbrs'].long()
    if len(base['hidden'])!=n:raise ValueError('base topology must contain material generators only')
    delta=p[None,n:]-p[:n,None];length=delta.norm(dim=2)
    if bool((length==0).any()):raise ValueError('coincident material/vacuum generators')
    height=(length.square()+r[:n,None].square()-r[None,n:].square())/(2*length)
    # An already hidden real power cell is empty before adding any vacuum
    # constraint, and intersections cannot make it nonempty. Its original
    # neighbor list may be empty: constructing a new disk with that list
    # would otherwise invent a material/vacuum face inside another real cell.
    candidate=(height<r[:n,None])&(~base['hidden'][:,None])
    hidden=base['hidden']|(height<=-r[:n,None]).any(1)
    ghost_ids=torch.arange(n,len(r),device=p.device)
    additional=torch.where(candidate,ghost_ids[None],-1)
    new_i,new_g=torch.nonzero(candidate,as_tuple=True);new_j=new_g+n
    new_pairs=torch.stack((new_i,new_j),1)
    width=max(links.shape[1],nbrs.shape[1])+ng
    all_links=torch.full((len(pairs)+len(new_pairs),width),-1,dtype=torch.long,device=p.device)
    all_links[:len(pairs),:links.shape[1]]=links
    # No original real constraint shares an index with a geometric vacuum.
    # Tensor assembly preserves exactly the old complete lists without a
    # Python loop over fifty thousand faces or a GPU-to-CPU topology copy.
    start=width-ng
    all_links[:len(pairs),start:]=additional[pairs[:,0]]
    all_links[len(pairs):,:nbrs.shape[1]]=nbrs[new_i]
    all_links[len(pairs):,start:]=torch.where(additional[new_i]!=new_j[:,None],additional[new_i],-1)
    def compact(rows):
        count=max(1,int((rows>=0).sum(1).max()))
        order=torch.argsort((rows<0).to(torch.int32),dim=1,stable=True)
        return torch.gather(rows,1,order[:,:count]).contiguous()
    real_neighbors=compact(torch.cat((nbrs,additional),1))
    all_neighbors=torch.full((len(r),real_neighbors.shape[1]),-1,dtype=torch.long,device=p.device)
    all_neighbors[:n]=real_neighbors
    return {'pairs':torch.cat((pairs,new_pairs)), 'links':compact(all_links),'nbrs':all_neighbors,
            'hidden':torch.cat((hidden,torch.ones(ng,dtype=torch.bool,device=p.device)))}


def material_mesh(full,planes,ground):
    """Separate material/internal and material/vacuum faces without dropping any."""
    from free_surface.fv_ale_geometry import PowerControlVolumes
    n=len(full.r)-len(planes.owner)
    internal=(full.i<n)&(full.j<n);free=(full.i<n)&(full.j>=n)
    mesh=PowerControlVolumes(full.p[:n].clone(),full.r[:n].clone(),full.volume[:n].clone(),
        full.i[internal].clone(),full.j[internal].clone(),full.area[internal].clone(),
        full.normal[internal].clone(),full.centroid[internal].clone(),
        full.cap_area[:n].clone(),full.cap_normal_integral[:n].clone())
    for key in ['ground_area','ground_centroid','ground_normal','hidden']:
        setattr(mesh,key,getattr(full,key)[:n].clone())
    mesh.planes=planes.clone();mesh.plane_i=full.i[free].clone();mesh.plane_j=full.j[free].clone()-n
    mesh.plane_area=full.area[free].clone();mesh.plane_normal=full.normal[free].clone()
    mesh.plane_centroid=full.centroid[free].clone()
    mesh.plane_length=(full.p[full.j[free]]-full.p[full.i[free]]).norm(dim=1)
    mesh.precise_disks=getattr(full,'precise_disks',[])
    mesh.ground_geometry_info=getattr(full,'ground_geometry_info',{})
    mesh.geometry=full.geometry
    return mesh


def geometry_cpu(p,r,planes,ground,base=None,full_hull=False):
    from free_surface.fv_ale_ghost_plane_geometry_check import structure_only
    from free_surface.fv_ale_precise_geometry_check import cpu_mesh
    pp,rr,redundant=planes.generators(p,r,ground)
    if full_hull:
        S=structure_only(pp,rr)
        # The independent hull supplies all real-cell constraints. Vacuum
        # cells themselves are not integrated material domains; remove only
        # vacuum/vacuum faces and suppress their volumes, as the production
        # complete-constraint construction does. Keeping those huge unbounded
        # cells while clipping the floor only for real owners is inconsistent.
        material_face=S['pairs'][:,0]<len(r)
        S['pairs']=S['pairs'][material_face];S['links']=S['links'][material_face]
        S['hidden'][len(r):]=True;S['nbrs'][len(r):]=-1
    else:
        if base is None:base=structure_only(p,r)
        S=augmented_structure(pp,rr,base,len(r))
    return material_mesh(cpu_mesh(pp,rr,ground,S,redundant,len(r)),planes,ground)


def values(mesh):
    p=mesh.planes
    return {'plane_owner':p.owner,'plane_d':p.d,'plane_a':p.a,'plane_b':p.b,
        'plane_i':mesh.plane_i,'plane_j':mesh.plane_j,'plane_A':mesh.plane_area,
        'plane_n':mesh.plane_normal,'plane_c':mesh.plane_centroid,'plane_length':mesh.plane_length}


def geometric_rate(b,w,rd,adot,bdot):
    i,j,c=b['plane_i'],b['plane_j'],b['plane_c'];o=b['plane_owner'][j]
    return (((c-b['p'][i])*w[i]).sum(1)+b['r'][i]*rd[i]
        -((c-b['p'][o])*w[o]).sum(1)-b['r'][o]*rd[o]
        -b['plane_d'][j]*(c*adot[j]).sum(1)+b['plane_d'][j]*bdot[j])/b['plane_length']


def pack_frames(sequence):
    """Numeric padded frame arrays; never require pickle/object-array loading."""
    counts=np.array([0 if item is None else len(item['owner']) for item in sequence],dtype=np.int64)
    width=int(counts.max(initial=0));frames=len(sequence)
    out={'plane_count':counts,'plane_owner':np.zeros((frames,width),dtype=np.int64),
         'plane_d':np.zeros((frames,width)),'plane_a':np.zeros((frames,width,3)),
         'plane_b':np.zeros((frames,width))}
    for frame,item in enumerate(sequence):
        if item is not None:
            for key in ['owner','d','a','b']:out['plane_'+key][frame,:counts[frame]]=np.asarray(item[key])
    return out


def unpack_frames(archive):
    if 'plane_count' not in archive:return [None]*len(archive['t'])
    return [None if count==0 else {key:torch.from_numpy(archive['plane_'+key][frame,:count].copy())
            for key in ['owner','d','a','b']} for frame,count in enumerate(archive['plane_count'])]
