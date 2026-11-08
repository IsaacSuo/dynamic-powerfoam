"""Affine kinematics and its negative-adjoint force on the actual FV mesh.

Experimental free-body path. This is a compatible semidiscrete power operator,
not an independent constitutive model or an energy reset. The same least-
squares map used in the velocity gradient is transposed in the force. Its
wide stencil is essential; an unchanged two-cell traction is not its adjoint.
Fixed-plane frictionless wall variants are explicit opt-ins.
"""
import torch
from free_surface.fv_ale_reconstruction import group_centroids,group_sum
from free_surface.fv_ale_symmetric_eigen import eigh3
from free_surface.fv_ale_hyperelastic_material import response,symmetric_deviator


def affine_map(b,v,centers):
    g=b['groups'];i,j,A=b['i'],b['j'],b['A'];gi,gj=g[i],g[j]
    d=centers[j]-centers[i]
    weight=torch.where((gi!=gj)&(A>0),A/d.square().sum(1).clamp(min=1e-300),0.)
    outer=weight[:,None,None]*d[:,:,None]*d[:,None,:]
    normal=torch.zeros((len(g),3,3),dtype=v.dtype,device=v.device).index_add(0,gi,outer).index_add(0,gj,outer)
    eig,R=eigh3(normal)
    good=(eig[:,0]>1e-10*eig[:,-1])&(eig[:,-1]>0)
    inv=(R*torch.where(good[:,None],eig.clamp(min=1e-300).reciprocal(),0.)[:,None,:])@R.transpose(1,2)
    term=weight[:,None,None]*(v[j]-v[i])[:,:,None]*d[:,None,:]
    rhs=torch.zeros_like(normal).index_add(0,gi,term).index_add(0,gj,term)
    return (rhs@inv)[g],inv[g],d,weight


def kinematics(ops,b,v,sigma,Z,baseline):
    i,j,A,n,g=b['i'],b['j'],b['A'],b['n'],b['groups']
    w0,dr0,radial,cg0,cr0,_,_,ti,tj=baseline
    extended=ops.s.affine_stencil!='one_ring'
    if extended:
        from free_surface.fv_ale_affine_stencil import build,gradient
        if 'affine_neighbors' not in b and 'affine_owner' not in b:
            values,_=build(b,transition_ratio=.01 if ops.s.affine_stencil=='smooth' else None);b=dict(b,**values)
        centers=b['affine_centers'];G=gradient(b,v)
    else:
        centers=group_centroids(b)
        G,inv,d,weight=affine_map(b,v,centers)
    oi,oj=b['c']-centers[i],b['c']-centers[j]
    vl=v[i]+(G[i]@oi[:,:,None])[:,:,0]
    vr=v[j]+(G[j]@oj[:,:,None])[:,:,0]
    den=Z[i]+Z[j];ai=Z[i]/den;aj=Z[j]/den
    delta_t=ti-tj;jump=vr-vl;alpha=ops.s.interface_dissipation
    us=ai[:,None]*vl+aj[:,None]*vr-alpha*delta_t/den[:,None]
    # ts is retained for diagnostics only: the actual force below includes
    # the transpose of the reconstruction, not just this face traction.
    ts=aj[:,None]*ti+ai[:,None]*tj+alpha*(Z[i]*Z[j]/den)[:,None]*jump
    N,Q,cap,r=b['N'],b['Q'],b['cap'],b['r'];offset=b['p']-centers
    moment=offset[:,:,None]*N[:,None,:]+r[:,None,None]*Q
    eye=torch.eye(3,dtype=v.dtype,device=v.device)
    radius_volume_map=cap[:,None,None]*eye/3
    if ops.s.affine_cap_motion!='isotropic':
        # Integral of the affine normal velocity on this actual exposed cap:
        # r * integral n.G.n dA. Using tr(G)/3 is only correct for a full
        # sphere or isotropic dilation, not a general clipped cap.
        radius_volume_map=torch.where(b['surface'][:,None,None],Q,radius_volume_map)
    dw=(G@offset[:,:,None])[:,:,0]
    dr_kin=r*G.diagonal(dim1=1,dim2=2).sum(1)/3
    if ops.s.affine_cap_motion=='volume':
        dr_kin=torch.where(b['surface'],r*(G*radius_volume_map).sum((1,2))/cap.clamp(min=1e-300),dr_kin)
    elif ops.s.affine_cap_motion=='balanced':
        # Least Euclidean change of (site velocity, radius velocity) subject
        # to the same exact affine cap-volume constraint. Both controls have
        # velocity units. This avoids assigning the entire correction to r.
        eps=getattr(ops.s,'cap_motion_regularization',0.)
        if eps>0:
            # Minimize |delta_w|^2 + delta_r^2 + (a.delta-D)^2/(eps*S)^2,
            # a=(N,cap), D=r*G:(Q-cap*I/3). Unlike the hard constraint,
            # this integrated form tends to zero as a cap is born.
            # Use actual cell boundary area as a dimensionally consistent scale.
            from free_surface.fv_ale_cap_motion import regularized_map
            B,radius_volume_map=regularized_map(b,eps)
            coefficient=(G*B).sum((1,2))
            dw=dw+coefficient[:,None]*N;dr_kin=dr_kin+coefficient*cap
            # The force must transpose the ACTUAL softened volume map.
            # Leaving Q here would break the power identity.
        else:
            mean_n=torch.where(b['surface'][:,None],N/cap.clamp(min=1e-300)[:,None],0.)
            defect=r*(G*Q).sum((1,2))/cap.clamp(min=1e-300)-dr_kin
            correction=torch.where(b['surface'],defect/(1+mean_n.square().sum(1)),0.)
            dw=dw+correction[:,None]*mean_n;dr_kin=dr_kin+correction
    dcg=G@moment;dcr=(dw*N).sum(1)+cap*dr_kin
    dcg=dcg+(dcr-dcg.diagonal(dim1=1,dim2=2).sum(1))[:,None,None]*eye/3
    w=w0+dw;dr=dr0+dr_kin;cg=cg0+dcg;cr=cr0+dcr
    # Elastic power = sum A*delta_t.u_kin + sigma*N.v + C:G.
    # Its negative derivative with respect to all group velocities is force.
    direct=torch.zeros_like(v).index_add(0,i,-(A*ai)[:,None]*delta_t)
    direct.index_add_(0,j,-(A*aj)[:,None]*delta_t)
    direct-=(sigma@N[:,:,None])[:,:,0]
    conjugate=torch.zeros_like(sigma).index_add(0,i,(A*ai)[:,None,None]*delta_t[:,:,None]*oi[:,None,:])
    conjugate.index_add_(0,j,(A*aj)[:,None,None]*delta_t[:,:,None]*oj[:,None,:])
    pressure=-sigma.diagonal(dim1=1,dim2=2).sum(1)/3
    shear=sigma+pressure[:,None,None]*eye
    if ops.s.affine_cap_motion!='isotropic':
        conjugate+=shear@moment.transpose(1,2)-pressure[:,None,None]*(N[:,:,None]*offset[:,None,:]+r[:,None,None]*radius_volume_map)
    else:
        conjugate+=shear@moment.transpose(1,2)-pressure[:,None,None]*(N[:,:,None]*offset[:,None,:]+(r*cap/3)[:,None,None]*eye)
    # Dissipation is a quadratic form of the reconstructed velocity jump.
    damping=alpha*A*Z[i]*Z[j]/den
    dj=damping[:,None]*jump
    direct.index_add_(0,i,dj);direct.index_add_(0,j,-dj)
    conjugate.index_add_(0,i,-dj[:,:,None]*oi[:,None,:])
    conjugate.index_add_(0,j,dj[:,:,None]*oj[:,None,:])
    planar=None;floor=b['floor'];floor_c=b.get('floor_c')
    if 'plane_i' in b:
        if ops.s.cap_dissipation!=0 or ops.s.affine_cap_motion!='balanced' or ops.s.affine_stencil=='one_ring':
            raise ValueError('moving planes require balanced compatible extended kinematics and zero cap damping')
        from free_surface.fv_ale_planar_mechanics import motion
        planar=motion(b,v,G,sigma,Z,centers,ops.s.ground,
                      regularization=getattr(ops.s,'cap_motion_regularization',0.))
        direct+=planar['direct'];conjugate+=planar['conjugate']
        floor=planar['floor'];floor_c=planar['floor_c']
    wall_velocity=v;wall_normal_velocity=torch.zeros_like(b['floor'])
    wall_heat=torch.zeros_like(b['floor'])
    wall_mode=getattr(ops.s,'wall_mode','unsupported')
    if wall_mode in ['compatible_slip','compatible_slip_acoustic']:
        # Fixed floor normal velocity is zero; the tangential material
        # velocity is reconstructed with the same affine map. The wall's
        # elastic power is A*(sigma*n).u_t. Differentiate that SAME power,
        # so net force is the normal reaction and wall work remains zero.
        floor_offset=floor_c-centers
        wall_velocity=v+(G@floor_offset[:,:,None])[:,:,0]
        wall_normal_velocity=wall_velocity[:,1].clone()
        wall_velocity=wall_velocity.clone();wall_velocity[:,1]=0.
        wall_velocity=torch.where((floor>0)[:,None],wall_velocity,v)
        tangent=-sigma[:,:,1].clone();tangent[:,1]=0.
        weighted=floor[:,None]*tangent
        direct-=weighted
        conjugate+=weighted[:,:,None]*floor_offset[:,None,:]
        if wall_mode=='compatible_slip_acoustic':
            # Rigid-wall impedance correction p*=-sigma_yy-Z*v_face,y.
            # Only this correction is added: the elastic reaction is above.
            # Its force is minus the SAME reconstruction transpose applied
            # to d(.5*A*Z*v_face,y^2)/dv; the exact lost mechanical power
            # A*Z*v_face,y^2 is retained as nonelastic energy.
            damping_normal=floor*Z*wall_normal_velocity
            correction=torch.zeros_like(v);correction[:,1]=damping_normal
            direct-=correction
            conjugate+=correction[:,:,None]*floor_offset[:,None,:]
            wall_heat=damping_normal*wall_normal_velocity
    if extended:
        from free_surface.fv_ale_affine_stencil import negative_adjoint
        direct+=negative_adjoint(b,conjugate)
    else:
        C=group_sum(conjugate,g)[g]@inv
        h=weight[:,None]*((C[i]+C[j])@d[:,:,None])[:,:,0]
        direct.index_add_(0,i,h);direct.index_add_(0,j,-h)
    mass=b['q'][:,0];gm=group_sum(mass,g)[g]
    force=group_sum(direct,g)[g]*(mass/gm)[:,None]
    stress_heat=alpha*A*delta_t.square().sum(1)/den
    velocity_heat=damping*jump.square().sum(1)
    cap_heat=-(sigma*cg0).sum((1,2))
    pair=.5*(stress_heat+velocity_heat)
    heat=torch.zeros_like(mass).index_add(0,i,pair).index_add(0,j,pair)+cap_heat+wall_heat
    if planar is not None:heat+=planar['heat']
    return (w,dr,radial,cg,cr,us,ts,ti,tj),{
        'force':force,'heat':heat,'stress_heat':stress_heat,'velocity_heat':velocity_heat,
        'cap_heat':cap_heat,'gradient':G,'conjugate':conjugate,
        'kinematic_cap_gradient':dcg,'stress_cap_gradient':cg0,'wall_velocity':wall_velocity,
        'wall_normal_velocity':wall_normal_velocity,'wall_heat':wall_heat,'planar':planar,'floor':floor}


def energy_rhs(ops,b,dq,total,groupV,pressure,relative,donor,mw,mc,cache):
    mass=total[:,0];v=total[:,1:4]/mass[:,None];rho=mass/groupV
    H=total[:,5:14].reshape(-1,3,3)/mass[:,None,None]
    _,_,_,elastic,_,B,_=response(H,rho,ops.s.mu,ops.s.bulk,ops.s.rho0)
    T=ops.s.mu/(2*ops.s.rho0)*symmetric_deviator(B)
    chi=torch.zeros_like(total)
    chi[:,0]=elastic+pressure/rho-(T*H).sum((1,2))-.5*v.square().sum(1)
    chi[:,1:4]=v;chi[:,5:14]=T.reshape(-1,9)
    i,j=b['i'],b['j']
    volume_rate=torch.zeros_like(mass).index_add(0,i,mw).index_add(0,j,-mw)+mc
    mixing=relative*(((chi[i]-chi[j])*donor).sum(1)-(pressure[i]-pressure[j]))
    heat=cache['heat']+torch.zeros_like(mass).index_add(0,i,.5*mixing).index_add(0,j,.5*mixing)
    remainder=total[:,4]/mass-.5*v.square().sum(1)-elastic
    thermal_flux=relative*donor[:,0]*torch.where(relative>=0,remainder[i],remainder[j])
    thermal_transport=torch.zeros_like(mass).index_add(0,i,-thermal_flux).index_add(0,j,thermal_flux)
    # Differential compatible work, including the measured geometric volume
    # rate. No step-end energy or deformation projection is performed.
    mechanical_rate=(chi*dq).sum(1)-pressure*volume_rate
    dq[:,4]=mechanical_rate+heat+thermal_transport
    cache['mechanical_rate_chain_W']=mechanical_rate.sum()
    cache['mechanical_energy_joule']=(b['q'][:,0]*(elastic+.5*v.square().sum(1))).sum()
    cache['mixing_heat']=mixing
    cache['internal_acoustic_W']=-(cache['stress_heat']+cache['velocity_heat']).sum()
    cache['donor_mixing_W']=-mixing.sum()
    cache['free_cap_W']=-cache['cap_heat'].sum()
    if cache['planar'] is not None:cache['free_cap_W']-=cache['planar']['heat'].sum()
    cache['wall_acoustic_W']=-cache['wall_heat'].sum()
    cache['wall_friction_W']=torch.zeros((),dtype=dq.dtype,device=dq.device)
    cache['gravity_W']=(b['q'][:,1:4]*ops.s.g).sum()
    return dq
