"""Exposed spherical-surface quadrature and penalty/dashpot contact traction.

Surface penalty contact follows the established FV traction-boundary approach
(Cardiff et al., the repository PDF
is linked in ALE_CONTACT.json). This curved-surface acoustic coupling is our
adaptation, not a reproduction of their implicit displacement solver.

For penetration delta, t_y = max(0, k*delta - eta*u_star_y), with
u_star = v + (t - sigma@n)/Z. The scalar equation is solved algebraically.
Tangential traction solves the Coulomb stick/slip condition against u_star.
Frictional work is returned as heat to the body. Normal work is external work
of the compliant obstacle, explicitly recorded; no exact penalty-potential
identity is claimed for evolving surface area.

Quadrature resolves the exposed region using the production power-cell arcs:
split latitude at every exposed arc endpoint and vertical extremum; at each
latitude split azimuth at ALL neighbour-circle intersections. Exposure is
constant within each azimuth interval. No point sampling of tiny caps and no
whole-ball contact force. Gauss order convergence must still be checked.
"""
import math
import numpy as np
import torch


def reference_surface_quadrature(g, ground, order=8, ids=None):
    dev=g.p.device; dtype=g.p.dtype
    if ids is None:
        ids=torch.nonzero((g.patch_A>1e-12*g.wall_A.max()) & (g.p[:,1]-g.r<ground)).flatten()
    ga,gw=np.polynomial.legendre.leggauss(order)
    ga=torch.as_tensor(ga,dtype=dtype,device=dev)
    gw=torch.as_tensor(gw,dtype=dtype,device=dev)
    owners=[]; normals=[]; weights=[]; gaps=[]
    npr=len(g.i)
    for k in ids.tolist():
        r=g.r[k]; upper=torch.minimum((ground-g.p[k,1])/r,r.new_zeros(()))
        if float(upper)<=-1:continue
        entries=g.site_ent[g.site_off[k]:g.site_off[k+1]].long()
        pairs=entries % npr if npr else entries
        if len(pairs):
            counts=g.narc[pairs].long()
            pair=torch.repeat_interleave(pairs,counts)
            slots=torch.arange(len(pair),device=dev)-torch.repeat_interleave(torch.cumsum(counts,0)-counts,counts)
            t0=g.arc_t0[pair,slots]; t1=g.arc_t1[pair,slots]
            # Heights of the exact exposed arcs, independent of their orientation.
            uy=g.u[pair,1]; vy=g.v[pair,1]
            centre=(g.c[pair,1]-g.p[k,1])/r; radius=g.rho[pair]/r
            roots=torch.atan2(vy,uy)[:,None]+math.pi*torch.arange(-2,5,device=dev)[None]
            roots=torch.where((roots>=t0[:,None])&(roots<=t1[:,None]),roots,t0[:,None])
            angles=torch.cat((t0[:,None],t1[:,None],roots),1)
            heights=(centre[:,None]+radius[:,None]*(uy[:,None]*torch.cos(angles)+vy[:,None]*torch.sin(angles))).flatten()
            cuts=torch.cat((heights,torch.stack((upper,r.new_tensor(-1.)))))
        else:
            cuts=torch.stack((upper,r.new_tensor(-1.)))
        cuts=torch.sort(cuts.clamp(min=-1).minimum(upper))[0]
        low,high=cuts[:-1],cuts[1:]
        valid=high-low>4e-15
        low,high=low[valid],high[valid]
        if not len(low):continue
        # Sine mapping smooths the square-root azimuth width at circle extrema.
        z=((low+high)[:,None]/2+(high-low)[:,None]/2*torch.sin(math.pi/2*ga)).flatten()
        wz=((high-low)[:,None]/2*(math.pi/2)*torch.cos(math.pi/2*ga)*gw).flatten()
        nb=g.nbrs[k].long();nb=nb[nb>=0]
        displacement=g.p[nb]-g.p[k];distance=displacement.norm(dim=1)
        a=displacement/distance[:,None]
        b=(distance**2+r*r-g.r[nb]**2)/(2*distance*r)
        horizontal=torch.sqrt((1-z*z).clamp(min=0))
        aznorm=torch.sqrt(a[:,0]**2+a[:,2]**2)
        denominator=horizontal[:,None]*aznorm
        q=(b-z[:,None]*a[:,1])/denominator.clamp(min=1e-300)
        phi0=torch.atan2(a[:,2],a[:,0])
        theta=torch.acos(q.clamp(min=-1,max=1))
        crossings=(denominator>0)&(q>-1)&(q<1)
        left=torch.where(crossings,torch.remainder(phi0-theta,2*math.pi),0.)
        right=torch.where(crossings,torch.remainder(phi0+theta,2*math.pi),0.)
        ends=torch.cat((z[:,None]*0,z[:,None]*0+2*math.pi,left,right),1).sort(1)[0]
        lo,hi=ends[:,:-1],ends[:,1:]
        mid=(lo+hi)/2
        nx=horizontal[:,None]*torch.cos(mid); nz=horizontal[:,None]*torch.sin(mid)
        if len(nb):
            exposed=(nx[:,:,None]*a[:,0]+z[:,None,None]*a[:,1]+nz[:,:,None]*a[:,2]<=b+2e-14).all(2)
        else:exposed=torch.ones_like(mid,dtype=torch.bool)
        good=(hi-lo>4e-15)&exposed
        row,col=torch.nonzero(good,as_tuple=True)
        l,h=lo[row,col],hi[row,col]
        phi=(l+h)[:,None]/2+(h-l)[:,None]/2*ga
        zz=z[row,None].expand_as(phi)
        rr=horizontal[row,None]
        n=torch.stack((rr*torch.cos(phi),zz,rr*torch.sin(phi)),2).reshape(-1,3)
        w=(r*r*wz[row,None]*(h-l)[:,None]/2*gw).flatten()
        delta=(ground-g.p[k,1]-r*zz).flatten().clamp(min=0)
        owners.append(torch.full((len(w),),k,dtype=torch.long,device=dev))
        normals.append(n);weights.append(w);gaps.append(delta)
    if not owners:
        return (torch.empty(0,dtype=torch.long,device=dev),g.p.new_empty((0,3)),g.r.new_empty(0),g.r.new_empty(0))
    return tuple(torch.cat(x) for x in (owners,normals,weights,gaps))


def surface_quadrature(g,ground,order=8,ids=None):
    """Same arc/latitude quadrature, batched across all contact caps on GPU."""
    dev=g.p.device;dtype=g.p.dtype
    if ids is None:
        ids=torch.nonzero((g.patch_A>1e-12*g.wall_A.max())&(g.p[:,1]-g.r<ground)).flatten()
    upper=((ground-g.p[ids,1])/g.r[ids]).clamp(max=0)
    ids=ids[upper>-1];upper=upper[upper>-1];nc=len(ids)
    if not nc:
        return (torch.empty(0,dtype=torch.long,device=dev),g.p.new_empty((0,3)),g.r.new_empty(0),g.r.new_empty(0))
    ga,gw=np.polynomial.legendre.leggauss(order)
    ga=torch.as_tensor(ga,dtype=dtype,device=dev);gw=torch.as_tensor(gw,dtype=dtype,device=dev)
    counts=(g.site_off[ids+1]-g.site_off[ids]).long()
    cell_ent=torch.repeat_interleave(torch.arange(nc,device=dev),counts)
    ent_offsets=torch.cumsum(counts,0)-counts
    ent=g.site_ent[g.site_off[ids[cell_ent]]+torch.arange(len(cell_ent),device=dev)-ent_offsets[cell_ent]].long()
    pair=ent%len(g.i) if len(g.i) else ent
    if len(pair):
        arc_counts=g.narc[pair].long()
        repeated_ent=torch.repeat_interleave(torch.arange(len(pair),device=dev),arc_counts)
        cell_arc=cell_ent[repeated_ent];pairs=pair[repeated_ent]
        slot=torch.arange(len(pairs),device=dev)-(torch.cumsum(arc_counts,0)-arc_counts)[repeated_ent]
        cell_counts=torch.bincount(cell_arc,minlength=nc)
        width=9*int(cell_counts.max())+2
        cuts=upper[:,None].expand(-1,width).clone()
        if len(pairs):
            ta=g.arc_t0[pairs,slot];tb=g.arc_t1[pairs,slot]
            uy=g.u[pairs,1];vy=g.v[pairs,1]
            roots=torch.atan2(vy,uy)[:,None]+math.pi*torch.arange(-2,5,device=dev)[None]
            roots=torch.where((roots>=ta[:,None])&(roots<=tb[:,None]),roots,ta[:,None])
            angles=torch.cat((ta[:,None],tb[:,None],roots),1)
            own=ids[cell_arc];r=g.r[own]
            heights=(g.c[pairs,1]-g.p[own,1])[:,None]/r[:,None]+(g.rho[pairs]/r)[:,None]*(uy[:,None]*torch.cos(angles)+vy[:,None]*torch.sin(angles))
            local_arc=torch.arange(len(pairs),device=dev)-(torch.cumsum(cell_counts,0)-cell_counts)[cell_arc]
            cuts[cell_arc[:,None],2+9*local_arc[:,None]+torch.arange(9,device=dev)]=heights
    else:cuts=upper[:,None].expand(-1,2).clone()
    cuts[:,0]=-1;cuts[:,1]=upper
    cuts=cuts.clamp(min=-1).minimum(upper[:,None]).sort(1)[0]
    low,high=cuts[:,:-1],cuts[:,1:]
    cell,interval=torch.nonzero(high-low>4e-15,as_tuple=True)
    lo=low[cell,interval];hi=high[cell,interval]
    z=((lo+hi)[:,None]/2+(hi-lo)[:,None]/2*torch.sin(math.pi/2*ga)).flatten()
    wz=((hi-lo)[:,None]/2*(math.pi/2)*torch.cos(math.pi/2*ga)*gw).flatten()
    cell_z=cell.repeat_interleave(order)
    nb=g.nbrs[ids].long();valid=nb>=0;nb=nb.clamp(min=0)
    displacement=g.p[nb]-g.p[ids,None];length=displacement.norm(dim=2).clamp(min=1e-300)
    a=displacement/length[:,:,None]
    b=(length**2+g.r[ids,None]**2-g.r[nb]**2)/(2*length*g.r[ids,None])
    horizontal=(1-z*z).clamp(min=0).sqrt()
    aznorm=(a[:,:,0]**2+a[:,:,2]**2).sqrt()
    denominator=horizontal[:,None]*aznorm[cell_z]
    q=(b[cell_z]-z[:,None]*a[cell_z,:,1])/denominator.clamp(min=1e-300)
    centre=torch.atan2(a[:,:,2],a[:,:,0])[cell_z]
    theta=torch.acos(q.clamp(min=-1,max=1))
    crossing=(denominator>0)&(q>-1)&(q<1)&valid[cell_z]
    left=torch.where(crossing,torch.remainder(centre-theta,2*math.pi),0.)
    right=torch.where(crossing,torch.remainder(centre+theta,2*math.pi),0.)
    ends=torch.cat((z[:,None]*0,z[:,None]*0+2*math.pi,left,right),1).sort(1)[0]
    l,h=ends[:,:-1],ends[:,1:]
    row,col=torch.nonzero(h-l>4e-15,as_tuple=True)
    mid=(l[row,col]+h[row,col])/2
    normal_mid=torch.stack((horizontal[row]*torch.cos(mid),z[row],horizontal[row]*torch.sin(mid)),1)
    exposed=torch.where(valid[cell_z[row]],(normal_mid[:,None,:]*a[cell_z[row]]).sum(2)<=b[cell_z[row]]+2e-14,True).all(1)
    row=row[exposed];col=col[exposed];l=l[row,col];h=h[row,col]
    phi=(l+h)[:,None]/2+(h-l)[:,None]/2*ga
    zz=z[row,None].expand_as(phi);rr=horizontal[row,None]
    normals=torch.stack((rr*torch.cos(phi),zz,rr*torch.sin(phi)),2).reshape(-1,3)
    own=ids[cell_z[row]]
    weights=(g.r[own,None]**2*wz[row,None]*(h-l)[:,None]/2*gw).flatten()
    delta=(ground-g.p[own,1,None]-g.r[own,None]*zz).flatten().clamp(min=0)
    return own.repeat_interleave(order),normals,weights,delta


def contact_flux(owner,n,w,delta,v,sigma,Z,stiffness,friction,damping=1.):
    """Return pointwise traction and material boundary velocity using one law."""
    uf=v[owner]-(sigma[owner]@n[:,:,None])[:,:,0]/Z[owner,None]
    pressure=((stiffness*delta-damping*Z[owner]*uf[:,1])/(1+damping)).clamp(min=0)
    t=torch.zeros_like(uf);t[:,1]=pressure
    tangent=uf[:,[0,2]];speed=tangent.norm(dim=1)
    amount=torch.minimum(friction*pressure,Z[owner]*speed)
    t[:,[0,2]]=-amount[:,None]*tangent/speed.clamp(min=1e-300)[:,None]
    us=uf+t/Z[owner,None]
    friction_power=(t[:,[0,2]]*us[:,[0,2]]).sum(1)
    return t,us,friction_power


def project_cap_motion(v,sigma,Z,A,N,force,cap_rate):
    """L2 fit of material cap velocity to sphere velocity w_site + r_dot*n.

    Normal equations are A*w + N*r_dot = integral(u_star) and
    N.w + A*r_dot = integral(u_star.n). Only exact first/second cap moments
    and the integrated traction are needed. The second equation is the
    integrated no-mass-flow boundary condition. Nearly flat tiny caps cannot
    distinguish site translation from radial change. On almost flat caps
    (normal variance < 1e-3), keep the original radial volume rate and fit
    only mean tangential translation. This avoids dividing cancellation in
    geometric moments by a tiny variance, and preserves integrated volume.
    This changes mesh
    motion only; material momentum is not projected.
    """
    area=torch.where(A>0,A,torch.ones_like(A)); mean_n=N/area[:,None]
    mean_du=(-(sigma@mean_n[:,:,None])[:,:,0]+force/area[:,None])/Z[:,None]
    mean_normal_du=cap_rate/area
    variance=1-(mean_n*mean_n).sum(1)
    numerator=mean_normal_du-(mean_du*mean_n).sum(1)
    resolved=variance>1e-3
    radial=torch.where(resolved,numerator/variance.clamp(min=1e-300),torch.zeros_like(A))
    change=mean_du-radial[:,None]*mean_n
    tangent=mean_du-mean_n*((mean_du*mean_n).sum(1)/(mean_n*mean_n).sum(1).clamp(min=1e-300))[:,None]
    change=torch.where(resolved[:,None],change,tangent)
    radial=torch.where(resolved,radial,mean_normal_du)
    return v+change,radial,~resolved
