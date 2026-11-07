"""Experimental cell-centred solid FV on the existing sphere-clipped power mesh.

Constitutive reference: Guisset et al., CMA 217 (2026) 379-401, Wilkins/Jaumann
hypoelasticity plus perfect von Mises plasticity. This is NOT a reproduction of
its polyhedral nodal solver. Here a first-order acoustic face solver, ALE donor
transport and SSPRK2 are adapted to curved power cells. The adaptation and its
limits must be reported with every run.

SI units. State: cell mass, momentum, TOTAL energy, mass-weighted deviatoric
Cauchy stress, equivalent plastic strain, plastic-work diagnostic. No k1 F or L
is evolved. Density = mass / actual cut volume. A stiffened-gas EOS (gamma=2)
initially matches the specified bulk modulus, but differs from k1's log energy.

The free cap has zero traction. Its acoustic velocity v*=v-SIGMA n/Z is
represented only in the INTEGRATED normal sense by moving the site at v and
r_dot=-SIGMA:Q/(Z A), Q=int_cap n tensor n. It is not pointwise impermeable;
a diagnostic explicitly reports this approximation. Interior radii stay fixed by default (an ALE mesh choice); the initial 'dilate'
trial used r_dot=r div(v)/3 and drove a buried cell to zero volume.
Internal mass exchange is retained under either choice.

KNOWN FAILED CONTACT ADAPTATION (retained only for diagnostic replay):
A nonzero cap does not imply the ball bottom is exposed. The inherited force
can remain finite while FV mass and exposed area vanish. This code is NOT a
valid continuum contact boundary treatment. See ALE_TRIAL.json.
Ground contact: same Hertz normal force formula as k1, on cap cells, applied
as an external body force with its measured power. Tangential Coulomb impulse
is applied after a step and its lost kinetic energy is booked as heat (total
energy unchanged). This is NOT k1's Mindlin sticking history.
"""
from dataclasses import dataclass
import math
import torch
from free_surface.recut import Recut
from free_surface.geo_pipe import GeoPipe
from free_surface.fv_ale_geometry import PowerControlVolumes


def deviator(a):
    a=.5*(a+a.transpose(-1,-2))
    eye=torch.eye(3,dtype=a.dtype,device=a.device)
    return a-a.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]*eye/3


def acoustic_face(vl,vr,tl,tr,zl,zr):
    """Tension-positive traction, normal oriented left -> right.
    Scalar acoustic impedance is also used for tangential dissipation. It is
    consistent, but not an exact characteristic solver for shear waves.
    """
    den=(zl+zr)[:,None]
    vel=(zl[:,None]*vl+zr[:,None]*vr+tr-tl)/den
    traction=(zr[:,None]*tl+zl[:,None]*tr+(zl*zr)[:,None]*(vr-vl))/den
    return vel,traction


@dataclass
class State:
    p: torch.Tensor
    r: torch.Tensor
    q: torch.Tensor
    mesh: PowerControlVolumes


class SolidALE:
    def __init__(self,p,r,E,nu,yield_stress,rho0=1000.,gravity=(0.,-9.8,0.),
                 ground=-.05,friction=10.,device='cuda',interior_radii='fixed'):
        self.interior_radii=interior_radii
        self.E=float(E); self.nu=float(nu); self.mu=E/(2*(1+nu))
        self.bulk=E/(3*(1-2*nu)); self.ys=float(yield_stress); self.rho0=float(rho0)
        self.gamma=2.; self.pinf=self.bulk/self.gamma
        self.e0=self.bulk/((self.gamma-1)*rho0)
        self.g=torch.tensor(gravity,dtype=torch.float64,device=device)
        self.ground=ground; self.friction=friction; self.contact_E=E/(1-nu*nu)
        self.recut=Recut(len(p),device); self.recut.pipe=GeoPipe(len(p),device)
        self.geometry_calls=0
        p=torch.as_tensor(p,dtype=torch.float64,device=device).clone()
        r=torch.as_tensor(r,dtype=torch.float64,device=device).clone()
        mesh=self.cut(p,r)
        q=torch.zeros((len(p),16),dtype=p.dtype,device=p.device)
        q[:,0]=rho0*mesh.volume; q[:,4]=q[:,0]*self.e0
        self.initial=State(p,r,q,mesh)

    def cut(self,p,r):
        if not bool(torch.isfinite(p).all() and torch.isfinite(r).all() and (r>0).all()):
            raise RuntimeError('invalid site/radius; state not accepted')
        g,_=self.recut.geometry(p.contiguous(),r.contiguous()); self.geometry_calls+=1
        mesh=PowerControlVolumes.from_geometry(g)
        if not bool(torch.isfinite(mesh.volume).all() and (mesh.volume>0).all()):
            bad=torch.nonzero(~torch.isfinite(mesh.volume)|(mesh.volume<=0)).flatten()
            raise RuntimeError('hidden/nonpositive cells, not silently removed: '+str(bad[:20].tolist()))
        # Reconstruct the cap's exact second moment from the first divergence moment.
        outer=lambda a,b:a[:,:,None]*b[:,None,:]
        moment=torch.zeros((len(p),3,3),dtype=p.dtype,device=p.device)
        moment.index_add_(0,mesh.i,mesh.area[:,None,None]*outer(mesh.centroid-p[mesh.i],mesh.normal))
        moment.index_add_(0,mesh.j,-mesh.area[:,None,None]*outer(mesh.centroid-p[mesh.j],mesh.normal))
        eye=torch.eye(3,dtype=p.dtype,device=p.device)
        mesh.cap_Q=(mesh.volume[:,None,None]*eye-moment)/r[:,None,None]
        mesh.cap_Q=.5*(mesh.cap_Q+mesh.cap_Q.transpose(1,2))
        # Only roundoff-size caps are ignored, and their area is recorded.
        mesh.surface=mesh.cap_area>1e-12*mesh.area.max()
        mesh.cap_Q=torch.where(mesh.surface[:,None,None],mesh.cap_Q,torch.zeros_like(mesh.cap_Q))
        mesh.ignored_cap_area=float(mesh.cap_area[~mesh.surface].abs().sum())
        mesh.surface_area=torch.zeros_like(r).index_add(0,mesh.i,mesh.area).index_add(0,mesh.j,mesh.area)+mesh.cap_area.clamp(min=0)
        return mesh

    def set_velocity(self,s,v):
        v=torch.as_tensor(v,dtype=s.p.dtype,device=s.p.device).expand_as(s.p)
        s.q[:,1:4]=s.q[:,:1]*v
        s.q[:,4]=s.q[:,0]*(self.e0+.5*(v*v).sum(1))
        return s

    def primitive(self,s):
        m=s.q[:,0]; v=s.q[:,1:4]/m[:,None]
        rho=m/s.mesh.volume
        internal=s.q[:,4]/m-.5*(v*v).sum(1)
        pressure=(self.gamma-1)*rho*internal-self.gamma*self.pinf
        csq=(self.gamma*(pressure+self.pinf)+4*self.mu/3)/rho
        if not bool(torch.isfinite(s.q).all() and (m>0).all() and (csq>0).all() and (pressure+self.pinf>0).all()):
            raise RuntimeError('nonpositive mass/sound-speed or nonfinite state; no floor applied')
        stress=deviator(s.q[:,5:14].reshape(-1,3,3)/m[:,None,None])
        eye=torch.eye(3,dtype=s.p.dtype,device=s.p.device)
        sigma=stress-pressure[:,None,None]*eye
        return m,v,rho,pressure,stress,sigma,torch.sqrt(csq)

    def plastic_return(self,q,mesh):
        q=q.clone(); m=q[:,0]
        stress=deviator(q[:,5:14].reshape(-1,3,3)/m[:,None,None])
        equivalent=torch.sqrt(1.5*(stress*stress).sum((1,2)))
        ratio=(self.ys/equivalent.clamp(min=1e-300)).clamp(max=1.)
        returned=stress*ratio[:,None,None]
        loss=mesh.volume*((stress*stress-returned*returned).sum((1,2)))/(4*self.mu)
        dep=(equivalent-self.ys).clamp(min=0)/(3*self.mu)
        q[:,5:14]=(m[:,None,None]*returned).reshape(-1,9)
        q[:,14]+=m*dep; q[:,15]+=loss
        return q

    def contact(self,s):
        n=len(s.r); force=torch.zeros_like(s.p); kn=torch.zeros_like(s.r)
        delta=torch.zeros_like(s.r)
        if self.ground is not None:
            delta=(s.r-s.p[:,1]+self.ground).clamp(min=0)*s.mesh.surface
            sr=torch.sqrt(s.r*delta)
            force[:,1]=4/3*self.contact_E*sr*delta
            kn=2*self.contact_E*sr
        return force,kn,delta

    def rhs(self,s):
        m,v,rho,pressure,stress,sigma,sound=self.primitive(s)
        mesh=s.mesh; i,j,n,A=mesh.i,mesh.j,mesh.normal,mesh.area
        z=rho*sound
        traction=lambda x,normal:(x@normal[:,:,None])[:,:,0]
        ti,tj=traction(sigma[i],n),traction(sigma[j],n)
        vs,ts=acoustic_face(v[i],v[j],ti,tj,z[i],z[j])
        # Integral gradient with the actual spherical normal moment, no G0 fit.
        grad=torch.zeros_like(stress)
        grad.index_add_(0,i,A[:,None,None]*(vs-v[i])[:,:,None]*n[:,None,:])
        grad.index_add_(0,j,-A[:,None,None]*(vs-v[j])[:,:,None]*n[:,None,:])
        grad-=sigma@mesh.cap_Q/z[:,None,None]
        grad/=mesh.volume[:,None,None]
        trace=grad.diagonal(dim1=1,dim2=2).sum(1)
        dr=s.r*trace/3 if self.interior_radii == 'dilate' else torch.zeros_like(s.r)
        cap_rate=-(sigma*mesh.cap_Q).sum((1,2))/z
        dr=torch.where(mesh.surface,cap_rate/mesh.cap_area.clamp(min=1e-300),dr)
        mw,mc=mesh.mesh_flux(v,dr)
        # The mass boundary condition holds only in the integrated, P0 sense.
        material_cap=(mesh.cap_normal_integral*v).sum(1)+cap_rate
        cap_defect=mc-material_cap
        relative=A*(vs*n).sum(1)-mw
        donor=torch.where(relative[:,None]>=0,s.q[i]/mesh.volume[i,None],s.q[j]/mesh.volume[j,None])
        dq=-mesh.incidence(relative[:,None]*donor)
        dq[:,1:4]+=mesh.incidence(A[:,None]*ts)
        dq[:,4]+=mesh.incidence(A*(ts*vs).sum(1))
        sym=.5*(grad+grad.transpose(1,2)); spin=.5*(grad-grad.transpose(1,2))
        stressdot=2*self.mu*deviator(sym)+spin@stress-stress@spin
        dq[:,5:14]+=(m[:,None,None]*stressdot).reshape(-1,9)
        ground_force,kn,delta=self.contact(s)
        dq[:,1:4]+=m[:,None]*self.g+ground_force
        source_power=(s.q[:,1:4]*self.g+ground_force*v).sum()
        dq[:,4]+=(s.q[:,1:4]*self.g+ground_force*v).sum(1)
        # Conservative wave/advection/contact time-scale estimates, not a proof
        # of stability of the complete curved-boundary adaptation.
        outgoing=torch.zeros_like(m).index_add(0,i,relative.clamp(min=0)).index_add(0,j,(-relative).clamp(min=0))
        wave=mesh.surface_area*sound+outgoing
        dt_wave=mesh.volume/wave.clamp(min=1e-300)
        dt_contact=torch.sqrt(m/kn.clamp(min=1e-300))
        dt_radius=s.r/dr.abs().clamp(min=1e-300)
        cfl_dt=float(torch.minimum(torch.minimum(dt_wave,dt_contact),dt_radius).min())
        diagnostics={'dt_scale':cfl_dt,'wave_cell':int(dt_wave.argmin()),
          'rho_min':float(rho.min()),'rho_max':float(rho.max()),
          'pressure_max_abs':float(pressure.abs().max()),'stress_vm_max':float(torch.sqrt(1.5*(stress*stress).sum((1,2))).max()),
          'cap_integrated_flux_defect':float(cap_defect.abs().sum()),
          'mass_rate_sum':float(dq[:,0].sum()),'force_sum':dq[:,1:4].sum(0).cpu().tolist(),
          'source_power':float(source_power),'energy_rate_sum':float(dq[:,4].sum()),
          'normal_contact_power':float((ground_force*v).sum()),
          'contact_cells':int((delta>0).sum()),'penetration_max':float(delta.max()),
          'ignored_cap_area':mesh.ignored_cap_area}
        return v,dr,dq,diagnostics

    def advance(self,s,dt,first=None,cfl=.2):
        first=self.rhs(s) if first is None else first
        vp,vr,dq,d0=first
        if dt>cfl*d0['dt_scale']*(1+1e-12):
            raise RuntimeError('requested step exceeds FV CFL; no hidden substep')
        p1=s.p+dt*vp; r1=s.r+dt*vr
        mesh1=self.cut(p1,r1)
        q1=self.plastic_return(s.q+dt*dq,mesh1)
        stage=State(p1,r1,q1,mesh1)
        vp1,vr1,dq1,d1=self.rhs(stage)
        if dt>cfl*d1['dt_scale']*1.25:
            raise RuntimeError('stage CFL changed too much; retry from unchanged state')
        p2=s.p+.5*dt*(vp+vp1); r2=s.r+.5*dt*(vr+vr1)
        mesh2=self.cut(p2,r2)
        q2=self.plastic_return(s.q+.5*dt*(dq+dq1),mesh2)
        # Coulomb sliding impulse limited by the time-integrated normal force.
        cf0,_,_=self.contact(s); cf1,_,_=self.contact(stage)
        jn=.5*dt*(cf0[:,1]+cf1[:,1])
        pt=q2[:,[1,3]]; plen=pt.norm(dim=1)
        impulse=torch.minimum(self.friction*jn,plen)
        fraction=impulse/plen.clamp(min=1e-300)
        q2[:,[1,3]]=pt*(1-fraction[:,None])
        friction_heat=float((((pt*pt).sum(1)-(q2[:,[1,3]]**2).sum(1))/(2*q2[:,0])).sum())
        result=State(p2,r2,q2,mesh2)
        self.primitive(result)
        stats=d1.copy(); stats.update(dt=dt,friction_heat=friction_heat,
            source_work=.5*dt*(d0['source_power']+d1['source_power']),
            normal_contact_work=.5*dt*(d0['normal_contact_power']+d1['normal_contact_power']))
        return result,stats

    @torch.no_grad()
    def cap_velocity_mismatch(self,s,samples=512):
        """Pointwise boundary mismatch, sampled on exposed spherical caps.
        RMS is area-weighted; this is diagnostic quadrature, not the force rule.
        """
        _,v,_,_,_,sigma,c=self.primitive(s)
        rho=s.q[:,0]/s.mesh.volume; z=rho*c
        _,dr,_,_=self.rhs(s)
        k=torch.arange(samples,dtype=s.p.dtype,device=s.p.device)+.5
        zz=1-2*k/samples; az=math.pi*(3-math.sqrt(5))*k
        dirs=torch.stack((torch.sqrt(1-zz*zz)*torch.cos(az),torch.sqrt(1-zz*zz)*torch.sin(az),zz),1)
        _,structure=self.recut.geometry(s.p,s.r)
        nb=structure['nbrs'].long(); total=0.; norm2=0.; maxerr=0.; surface=torch.nonzero(s.mesh.surface).flatten()
        for ids in surface.split(32):
            nn=nb[ids]; valid=nn>=0; nn=nn.clamp(min=0)
            y=s.p[ids,None]+s.r[ids,None,None]*dirs[None]
            # Exposure means no other sphere has lower power than zero there.
            diff=y[:,:,None,:]-s.p[nn][:,None,:,:]
            exposed=torch.where(valid[:,None,:],(diff*diff).sum(-1)>=s.r[nn][:,None,:]**2,True).all(2)
            normal_stress=torch.einsum('qa,nab,qb->nq',dirs,sigma[ids],dirs)
            err=dr[ids,None]+normal_stress/z[ids,None]
            weight=4*math.pi*s.r[ids,None]**2/samples*exposed
            total+=float(weight.sum()); norm2+=float((weight*err**2).sum())
            if bool(exposed.any()): maxerr=max(maxerr,float(err[exposed].abs().max()))
        return {'sample_count':samples,'sampled_area':total,'rms_normal_speed_mismatch':math.sqrt(norm2/max(total,1e-300)),
                'max_normal_speed_mismatch':maxerr}
