"""Linear cell-average reconstruction on the actual agglomerated power cells.

Area/distance-squared weighted least squares, followed by the componentwise
Barth--Jespersen neighbour-extrema limiter at all shared face centroids.
Rank-deficient groups explicitly fall back to zero slopes. No mesh replacement,
no positive-volume floor and no change to the mechanical constitutive law.
"""
import torch
from free_surface.fv_ale_symmetric_eigen import eigh3


def group_sum(x,g):
    return torch.zeros_like(x).index_add(0,g,x)


def group_centroids(b):
    p,V,g=b['p'],b['V'],b['groups'];i,j,A,n,c=b['i'],b['j'],b['A'],b['n'],b['c']
    di=c-p[i];dj=c-p[j]
    first=torch.zeros_like(p).index_add(0,i,(A*(di*n).sum(1))[:,None]*di)
    first.index_add_(0,j,(-A*(dj*n).sum(1))[:,None]*dj)
    first+=b['r'][:,None].square()*b['N']
    df=b['floor_c']-p
    first+=(b['floor']*(df*b['floor_n']).sum(1))[:,None]*df
    if 'plane_i' in b:
        k=b['plane_i'];d=b['plane_c']-p[k]
        first.index_add_(0,k,(b['plane_A']*(d*b['plane_n']).sum(1))[:,None]*d)
    # Integrate before dividing: hidden members of an agglomerated group may
    # have zero volume; their ill-defined individual centroid is never used.
    moment=V[:,None]*p+first/4
    return group_sum(moment,g)[g]/group_sum(V,g)[g,None]


def slopes(b,values,centers,limited=True,blocks=()):
    g=b['groups'];i,j,A=b['i'],b['j'],b['A'];gi,gj=g[i],g[j]
    delta=centers[j]-centers[i];dist2=delta.square().sum(1)
    weight=torch.where((gi!=gj)&(A>0),A/dist2.clamp(min=1e-300),0.)
    matrix=weight[:,None,None]*delta[:,:,None]*delta[:,None,:]
    normal=torch.zeros((len(g),3,3),dtype=values.dtype,device=values.device)
    normal.index_add_(0,gi,matrix);normal.index_add_(0,gj,matrix)
    rhs=weight[:,None,None]*(values[j]-values[i])[:,:,None]*delta[:,None,:]
    summed=torch.zeros((len(g),values.shape[1],3),dtype=values.dtype,device=values.device)
    summed.index_add_(0,gi,rhs);summed.index_add_(0,gj,rhs)
    eig,R=eigh3(normal)
    good=(eig[:,0]>1e-10*eig[:,-1])&(eig[:,-1]>0)
    inverse=(R*torch.where(good[:,None],eig.clamp(min=1e-300).reciprocal(),0.)[:,None,:])@R.transpose(1,2)
    gradient=(summed@inverse)[g]
    if not limited:return gradient,good[g]
    lo=values.clone();hi=values.clone()
    # Root slots contain the repeated group value, then gather all neighbours.
    for dest,source in [(gi,j),(gj,i)]:
        index=dest[:,None].expand(-1,values.shape[1])
        neighbour=torch.where((A>0)[:,None],values[source],values[dest])
        lo.scatter_reduce_(0,index,neighbour,reduce='amin',include_self=True)
        hi.scatter_reduce_(0,index,neighbour,reduce='amax',include_self=True)
    lo,hi=lo[g],hi[g]
    limiter=torch.ones_like(values)
    for cells,roots in [(i,gi),(j,gj)]:
        increment=(gradient[cells]*(b['c']-centers[cells])[:,None,:]).sum(2)
        bound=torch.where(increment>=0,hi[cells]-values[cells],lo[cells]-values[cells])
        factor=torch.where(increment.abs()>1e-14*values[cells].abs().clamp(min=1.),bound/increment,1.)
        factor=torch.where(((A>0)&(gi!=gj))[:,None],factor.clamp(0,1),1.)
        limiter.scatter_reduce_(0,roots[:,None].expand_as(factor),factor,reduce='amin',include_self=True)
    # Vector/tensor fields use one Frobenius-norm factor for the whole block.
    # Componentwise bounds depend on the coordinate frame and can change a
    # traction merely because the same body is viewed in a rotated frame.
    for start,end in blocks:
        jump=(values[j,start:end]-values[i,start:end]).norm(dim=1)
        jump=torch.where((A>0)&(gi!=gj),jump,0.)
        bound=torch.zeros(len(g),dtype=values.dtype,device=values.device)
        bound.scatter_reduce_(0,gi,jump,reduce='amax',include_self=True)
        bound.scatter_reduce_(0,gj,jump,reduce='amax',include_self=True)
        block_factor=torch.ones_like(bound)
        for cells,roots in [(i,gi),(j,gj)]:
            increment=(gradient[cells,start:end]*(b['c']-centers[cells])[:,None,:]).sum(2).norm(dim=1)
            factor=torch.minimum(torch.ones_like(increment),bound[roots]/increment.clamp(min=1e-300))
            factor=torch.where((A>0)&(gi!=gj)&(increment>0),factor,1.)
            block_factor.scatter_reduce_(0,roots,factor,reduce='amin',include_self=True)
        limiter[:,start:end]=block_factor[:,None]
    return gradient*limiter[g,:,None],good[g]


def face_values(b,values,gradient,centers):
    result=[]
    for k in [b['i'],b['j']]:
        result.append(values[k]+(gradient[k]*(b['c']-centers[k])[:,None,:]).sum(2))
    return result


def compatible_transport(high,low,chi_i,chi_j,pi,pj,relative,thermal_up):
    """Conservative face flux with a nonnegative mechanical mixing rate.

    chi are the derivatives of mechanical energy density with respect to
    (rho,rho*v,rho*H); chi_E=0 and chi.U-eta=p. The first limiter blends only
    the face state, never the material, towards the existing donor state if
    the reconstructed flux would *create* mechanical energy. The work flux
    then splits the remaining pair dissipation equally between the two cells.
    Upwind thermal transport is separate. No post-step energy reset occurs.
    """
    difference=chi_i-chi_j;dp=pi-pj
    dlo=relative*((difference*low).sum(1)-dp)
    dhi=relative*((difference*high).sum(1)-dp)
    fraction=torch.where(dhi<0,dlo.clamp(min=0)/(dlo-dhi).clamp(min=1e-300),1.).clamp(0,1)
    state=low+fraction[:,None]*(high-low)
    # Equal dissipation on both sides plus conservative donor transport of
    # the already accumulated nonelastic remainder.
    state[:,4]=.5*((chi_i+chi_j)*state).sum(1)-.5*(pi+pj)+state[:,0]*thermal_up
    return state,fraction
