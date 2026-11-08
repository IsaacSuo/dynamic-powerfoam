"""Compatible motion and negative-transpose power of moving power faces.

The full solver supplies affine velocities and the balanced sphere-motion
map. This module supplies additional geometric controls, gradient integrals
and their force transpose. It is not a second constitutive law. Coefficients
(stress, impedance, geometry and contact branch) are frozen in this transpose.
"""
import torch
from free_surface.fv_ale_planar_boundary import geometric_rate


def balanced_motion(b,v,G,centers,regularization=0.):
    offset=b['p']-centers;r=b['r'];cap=b['cap'];N=b['N'];Q=b['Q']
    dr=r*G.diagonal(dim1=1,dim2=2).sum(1)/3
    w=v+(G@offset[:,:,None])[:,:,0]
    if regularization>0:
        from free_surface.fv_ale_cap_motion import regularized_map
        B,_=regularized_map(b,regularization);coefficient=(G*B).sum((1,2))
        return w+coefficient[:,None]*N,dr+coefficient*cap
    mean=torch.where(b['surface'][:,None],N/cap.clamp(min=1e-300)[:,None],0.)
    defect=r*(G*Q).sum((1,2))/cap.clamp(min=1e-300)-dr
    correction=torch.where(b['surface'],defect/(1+mean.square().sum(1)),0.)
    return w+correction[:,None]*mean,dr+correction


def balanced_transpose(b,centers,aw,ar,regularization=0.):
    cap,N,Q,r=b['cap'],b['N'],b['Q'],b['r'];eye=torch.eye(3,dtype=r.dtype,device=r.device)
    if regularization>0:
        from free_surface.fv_ale_cap_motion import regularized_map
        B,_=regularized_map(b,regularization)
        C=aw[:,:,None]*(b['p']-centers)[:,None,:]+(ar*r/3)[:,None,None]*eye
        C+=((aw*N).sum(1)+ar*cap)[:,None,None]*B
        return -aw,C
    mean=torch.where(b['surface'][:,None],N/cap.clamp(min=1e-300)[:,None],0.)
    coefficient=((aw*mean).sum(1)+ar)*r/(1+mean.square().sum(1))
    C=aw[:,:,None]*(b['p']-centers)[:,None,:]+(ar*r/3)[:,None,None]*eye
    C+=torch.where(b['surface'][:,None,None],
        coefficient[:,None,None]*(Q/cap.clamp(min=1e-300)[:,None,None]-eye/3),0.)
    return -aw,C


def motion(b,v,G,sigma,Z,centers,ground,force_contact=None,regularization=0.):
    i,j,A,n,c=b['plane_i'],b['plane_j'],b['plane_A'],b['plane_n'],b['plane_c']
    o=b['plane_owner'];d=b['plane_d'];a=b['plane_a'];offset=b['plane_b']
    plane_count=len(o);summed=lambda x:torch.zeros((plane_count,)+x.shape[1:],dtype=x.dtype,device=x.device).index_add(0,j,x)
    w,dr=balanced_motion(b,v,G,centers,regularization)
    at_floor=(a[:,0]==0)&(a[:,1]==-1)&(a[:,2]==0)&(offset==-ground)
    adot=-(G[o].transpose(1,2)@a[:,:,None])[:,:,0]
    adot=torch.where(at_floor[:,None],0.,adot)
    base=geometric_rate(b,w,dr,adot,torch.zeros_like(offset))
    k=d[j]/b['plane_length']
    affine=v[i]+(G[i]@(c-centers[i])[:,:,None])[:,:,0]
    target=(affine*n).sum(1)
    traction=(sigma[i]@n[:,:,None])[:,:,0];sn=(traction*n).sum(1)
    denominator=summed(Z[i]*A*k.square())
    elastic_force=summed(A*sn*k)
    numerator=summed(Z[i]*A*k*(target-base))
    safe=denominator.clamp(min=1e-300)
    bkin=numerator/safe;bfree=bkin-elastic_force/safe
    # A redundant plane without exposed faces follows its owner's affine map.
    # No force, heat or arbitrary deletion is assigned to such a plane.
    inactive_rate=(a*(v[o]-(G[o]@centers[o,:,None])[:,:,0])).sum(1)
    bfree=torch.where(denominator>0,bfree,inactive_rate)
    active=at_floor&(bfree>=0)
    if force_contact is not None:active=force_contact
    bdot=torch.where(active,0.,bfree)
    speed=base+k*bdot[j]
    velocity=affine+(speed-target)[:,None]*n
    # Active own faces are exactly coincident floor faces and use the existing
    # fixed-wall operator. Cross-pair faces retain their own geometry.
    fixed=active[j]&(i==o[j]);live=~fixed
    liveA=A*live
    floor=b['floor'].index_add(0,i,A*fixed)
    floor_moment=b['floor'][:,None]*b['floor_c']
    floor_moment=floor_moment.index_add(0,i,(A*fixed)[:,None]*c)
    floor_c=torch.where((floor>0)[:,None],floor_moment/floor.clamp(min=1e-300)[:,None],b['floor_c'])
    # Analytic coefficients of the SAME kinematic power, including the fact
    # that bkin is a weighted projection involving all this plane's faces.
    L=torch.where(active[j],0.,(elastic_force/safe)[j]*Z[i]*A*k)
    B=(A*sn-L)*live
    T=(A[:,None]*(traction-sn[:,None]*n)+L[:,None]*n)*live[:,None]
    direct=torch.zeros_like(v).index_add(0,i,-T)
    conjugate=torch.zeros_like(G).index_add(0,i,T[:,:,None]*(c-centers[i])[:,None,:])
    factor=B/b['plane_length'];owner=o[j]
    aw=torch.zeros_like(v).index_add(0,i,factor[:,None]*(c-b['p'][i]))
    aw.index_add_(0,owner,-factor[:,None]*(c-b['p'][owner]))
    ar=torch.zeros_like(b['r']).index_add(0,i,factor*b['r'][i])
    ar.index_add_(0,owner,-factor*b['r'][owner])
    ca=summed(-(d[j]*factor)[:,None]*c)
    ca=torch.where(at_floor[:,None],0.,ca)
    conjugate.index_add_(0,o,-a[:,:,None]*ca[:,None,:])
    dw,dC=balanced_transpose(b,centers,aw,ar,regularization);direct+=dw;conjugate+=dC
    plane_heat=torch.where(active,0.,elastic_force.square()/safe)
    face_heat=plane_heat[j]*(Z[i]*A*k.square()/safe[j])
    heat=torch.zeros_like(b['r']).index_add(0,i,face_heat)
    gradient=torch.zeros_like(G).index_add(0,i,liveA[:,None,None]*(velocity-v[i])[:,:,None]*n[:,None,:])
    volume_rate=torch.zeros_like(b['r']).index_add(0,i,A*speed)
    area=torch.zeros_like(b['r']).index_add(0,i,liveA)
    # These independent scalar powers are also useful to the frozen audit.
    kinematic_velocity=affine+(base+k*torch.where(active,0.,bkin)[j]-target)[:,None]*n
    kinematic_power=(liveA*(traction*kinematic_velocity).sum(1)).sum()
    actual_power=(liveA*(traction*velocity).sum(1)).sum()
    return {'adot':adot,'bdot':bdot,'active':active,'fixed_faces':fixed,
        'direct':direct,'conjugate':conjugate,'gradient':gradient,'volume_rate':volume_rate,
        'heat':heat,'area':area,'floor':floor,'floor_c':floor_c,
        'velocity':velocity,'normal_speed':speed,'kinematic_power':kinematic_power,
        'actual_power':actual_power,'plane_heat':plane_heat,
        'stress_force':elastic_force,'denominator':denominator}
