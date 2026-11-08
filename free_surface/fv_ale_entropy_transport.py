"""Mechanical-entropy conservative ALE transport state for (rho,rho*v,rho*H).

It satisfies [chi].U_ec=[p], with chi=d mechanical_energy_density/dU.
This is the Tadmor algebraic condition applied to this particular athermal
neo-Hookean/log-shape state, not a borrowed Euler-gas formula. Mixing toward
the positive-density upwind state supplies explicitly measured dissipation.
Semidiscrete entropy balance alone is NOT a full nonlinear stability proof.
"""
import torch


def entropy_state(rho,v,H,T,shape_energy,rho0,i,j):
    Ji,Jj=rho0/rho[i],rho0/rho[j]
    avg=.5*(Ji+Jj);z=(Ji-Jj)/(Ji+Jj);z2=z*z
    # (log Ji-log Jj)/(Ji-Jj), stable at coincident states.
    series=1+z2/3+z2*z2/5+z2*z2*z2/7
    safe_z=torch.where(z.abs()>1e-4,z,torch.ones_like(z))
    quotient=.5*(Ji.log()-Jj.log())/safe_z
    inverse_log_mean=torch.where(z.abs()>1e-4,quotient,series)/avg
    face_rho=2*rho0*(1+1/(Ji*Jj))/(Ji+Jj+2*inverse_log_mean)
    face_v=.5*(v[i]+v[j]);face_H=.5*(H[i]+H[j]);dT=T[i]-T[j]
    dH=H[i]-H[j]
    defect=(.5*(T[i]+T[j])*dH).sum((1,2))-(shape_energy[i]-shape_energy[j])
    norm2=dT.square().sum((1,2))
    scale=(T[i].square().sum((1,2))+T[j].square().sum((1,2))).clamp(min=1.)
    # The trapezoidal defect is cubic in dH. Below 1e-6 its direct subtraction
    # is dominated by roundoff; the arithmetic mean has a cubic entropy defect
    # there, while dividing noisy energies by |dT| would destroy continuity.
    resolved=dH.square().sum((1,2))>1e-12
    correction=torch.where(resolved&(norm2>1e-24*scale),defect/norm2.clamp(min=1e-300),0.)
    face_H=face_H+correction[:,None,None]*dT
    out=torch.zeros((len(i),16),dtype=rho.dtype,device=rho.device)
    out[:,0]=face_rho;out[:,1:4]=face_rho[:,None]*face_v
    out[:,5:14]=face_rho[:,None]*face_H.reshape(-1,9)
    return out


def transport_state(total,groupV,material,mu,rho0,i,j,relative,dissipation):
    pressure,_,_,elastic,_,B,_=material
    from free_surface.fv_ale_hyperelastic_material import symmetric_deviator
    mass=total[:,0];rho=mass/groupV;v=total[:,1:4]/mass[:,None]
    H=total[:,5:14].reshape(-1,3,3)/mass[:,None,None]
    T=mu/(2*rho0)*symmetric_deviator(B)
    shape=mu/(2*rho0)*(B.diagonal(dim1=1,dim2=2).sum(1)-3)
    ec=entropy_state(rho,v,H,T,shape,rho0,i,j)
    low=torch.where((relative>=0)[:,None],total[i]/groupV[i,None],total[j]/groupV[j,None])
    out=ec+dissipation*(low-ec)
    chi=torch.zeros_like(total)
    chi[:,0]=elastic+pressure/rho-(T*H).sum((1,2))-.5*v.square().sum(1)
    chi[:,1:4]=v;chi[:,5:14]=T.reshape(-1,9)
    remainder=total[:,4]/mass-.5*v.square().sum(1)-elastic
    thermal=torch.where(relative>=0,remainder[i],remainder[j])
    out[:,4]=.5*((chi[i]+chi[j])*out).sum(1)-.5*(pressure[i]+pressure[j])+out[:,0]*thermal
    return out
