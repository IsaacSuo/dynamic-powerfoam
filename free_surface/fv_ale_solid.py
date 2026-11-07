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

The default coupled boundary solves contact directly in translating/dilating
sphere velocity unknowns (fv_ale_boundary.py); this same velocity supplies
geometry, gradient and boundary work. Friction is viscously regularized Coulomb,
not exact static sticking. The following projection description applies to the
retained radial/least-squares diagnostic controls only.
The uncontacted free cap has zero traction. Surface contact uses an integrated
penalty/dashpot traction with acoustic boundary velocity and Coulomb friction,
implemented in fv_ale_contact.py. The same traction enters force, work, gradient
and the integrated mesh boundary velocity. This remains an experimental coupling.
The free cap has zero traction. Its acoustic velocity v*=v-SIGMA n/Z is
projected onto a translating/dilating sphere with project_cap_motion. It
preserves integrated normal flux; locally flat caps fit mean tangential velocity
and radial volume rate. The radial-only control holds site velocity at v and
uses r_dot=-SIGMA:Q/(Z A), Q=int_cap n tensor n. It is not pointwise impermeable;
a diagnostic explicitly reports this approximation. Interior radii stay fixed by default (an ALE mesh choice); the initial 'dilate'
trial used r_dot=r div(v)/3 and drove a buried cell to zero volume.
Internal mass exchange is retained under either choice.

LEGACY contact_model=legacy-ball ONLY (retained for diagnostic replay):
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
from free_surface.fv_ale_contact import surface_quadrature,contact_flux,project_cap_motion
from free_surface.fv_ale_boundary import solve_boundary


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
                 ground=-.05,friction=10.,device='cuda',interior_radii='fixed',
                 contact_model='surface',contact_order=8,contact_scale=1.,contact_damping=1.,surface_motion='coupled',groups=None):
        self.surface_motion=surface_motion
        self.contact_model=contact_model; self.contact_order=contact_order
        self.contact_damping=contact_damping; self.contact_scale=contact_scale
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
        self.groups=torch.arange(len(p),device=p.device) if groups is None else torch.as_tensor(groups,device=p.device,dtype=torch.long).clone()
        self.has_groups=bool(torch.unique(self.groups).numel()<len(p))
        mesh=self.cut(p,r)
        self.contact_length=float(mesh.volume.median())**(1/3)
        self.contact_k=contact_scale*self.bulk/self.contact_length
        q=torch.zeros((len(p),16),dtype=p.dtype,device=p.device)
        q[:,0]=rho0*mesh.volume; q[:,4]=q[:,0]*self.e0
        self.initial=State(p,r,q,mesh)

    def cut(self,p,r):
        if not bool(torch.isfinite(p).all() and torch.isfinite(r).all() and (r>0).all()):
            raise RuntimeError('invalid site/radius; state not accepted')
        g,_=self.recut.geometry(p.contiguous(),r.contiguous()); self.geometry_calls+=1
        mesh=PowerControlVolumes.from_geometry(g)
        if self.has_groups:
            mesh.volume=torch.where(g.hidden,torch.zeros_like(mesh.volume),mesh.volume)
        valid_volume=mesh.volume>=0 if self.has_groups else mesh.volume>0
        if not bool(torch.isfinite(mesh.volume).all() and valid_volume.all()):
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
        mesh.geometry=g
        mesh.contact_quad=surface_quadrature(g,self.ground,self.contact_order) if self.ground is not None and self.contact_model=='surface' else None
        return mesh

    def group_sum(self,x):
        return torch.zeros_like(x).index_add(0,self.groups,x)

    def distribute_groups(self,q,volume):
        if not self.has_groups:return q.clone()
        totals=self.group_sum(q);volumes=self.group_sum(volume)
        if not bool((volumes[self.groups]>0).all() and (totals[self.groups,0]>0).all()):
            raise RuntimeError('a whole FV group lost its volume or mass')
        return totals[self.groups]*(volume/volumes[self.groups])[:,None]

    def set_velocity(self,s,v):
        v=torch.as_tensor(v,dtype=s.p.dtype,device=s.p.device).expand_as(s.p)
        s.q[:,1:4]=s.q[:,:1]*v
        s.q[:,4]=s.q[:,0]*(self.e0+.5*(v*v).sum(1))
        return s

    def primitive(self,s):
        q=self.group_sum(s.q)[self.groups] if self.has_groups else s.q
        volume=self.group_sum(s.mesh.volume)[self.groups] if self.has_groups else s.mesh.volume
        effective_mass=q[:,0];m=s.q[:,0];v=q[:,1:4]/effective_mass[:,None]
        rho=effective_mass/volume
        internal=q[:,4]/effective_mass-.5*(v*v).sum(1)
        pressure=(self.gamma-1)*rho*internal-self.gamma*self.pinf
        csq=(self.gamma*(pressure+self.pinf)+4*self.mu/3)/rho
        if not bool(torch.isfinite(s.q).all() and (effective_mass>0).all() and (csq>0).all() and (pressure+self.pinf>0).all()):
            raise RuntimeError('nonpositive mass/sound-speed or nonfinite state; no floor applied')
        stress=deviator(q[:,5:14].reshape(-1,3,3)/effective_mass[:,None,None])
        eye=torch.eye(3,dtype=s.p.dtype,device=s.p.device)
        sigma=stress-pressure[:,None,None]*eye
        return m,v,rho,pressure,stress,sigma,torch.sqrt(csq)

    def plastic_return(self,q,mesh):
        q=self.distribute_groups(q,mesh.volume);m=q[:,0]
        effective=self.group_sum(q)[self.groups] if self.has_groups else q
        stress=deviator(effective[:,5:14].reshape(-1,3,3)/effective[:,0,None,None])
        equivalent=torch.sqrt(1.5*(stress*stress).sum((1,2)))
        ratio=(self.ys/equivalent.clamp(min=1e-300)).clamp(max=1.)
        returned=stress*ratio[:,None,None]
        loss=mesh.volume*((stress*stress-returned*returned).sum((1,2)))/(4*self.mu)
        dep=(equivalent-self.ys).clamp(min=0)/(3*self.mu)
        q[:,5:14]=(m[:,None,None]*returned).reshape(-1,9)
        q[:,14]+=m*dep; q[:,15]+=loss
        return q

    def surface_contact(self,s,v,sigma,z):
        force=torch.zeros_like(s.p); moment=torch.zeros_like(sigma)
        area=torch.zeros_like(s.r); delta=torch.zeros_like(s.r)
        power=torch.zeros_like(s.r); heat=torch.zeros_like(s.r)
        if s.mesh.contact_quad is not None:
            owner,n,w,penetration=s.mesh.contact_quad
            t,us,fp=contact_flux(owner,n,w,penetration,v,sigma,z,self.contact_k,self.friction,self.contact_damping)
            force.index_add_(0,owner,w[:,None]*t)
            moment.index_add_(0,owner,(w/z[owner])[:,None,None]*t[:,:,None]*n[:,None,:])
            area.index_add_(0,owner,w)
            delta.scatter_reduce_(0,owner,penetration,reduce='amax',include_self=True)
            power.index_add_(0,owner,w*t[:,1]*us[:,1])
            heat.index_add_(0,owner,-w*fp)
        return force,moment,power,heat,self.contact_k*area,delta,area

    def contact(self,s):
        if self.contact_model=='surface':
            _,v,rho,_,_,sigma,c=self.primitive(s)
            if self.surface_motion=='coupled':
                b=solve_boundary(s.mesh,v,sigma,rho*c,self.contact_k,self.friction,self.contact_damping)[4]
                return b[0],b[3],b[4]
            force,_,_,_,kn,delta,_=self.surface_contact(s,v,sigma,rho*c)
            return force,kn,delta
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
        boundary=None;coupled=None
        if self.contact_model=='surface':
            if self.surface_motion=='coupled':
                coupled=solve_boundary(mesh,v,sigma,z,self.contact_k,self.friction,self.contact_damping)
                cb=coupled[4];boundary=(cb[0],None,cb[1],cb[2],cb[3],cb[4],cb[5])
            else:boundary=self.surface_contact(s,v,sigma,z)
        traction=lambda x,normal:(x@normal[:,:,None])[:,:,0]
        ti,tj=traction(sigma[i],n),traction(sigma[j],n)
        vs,ts=acoustic_face(v[i],v[j],ti,tj,z[i],z[j])
        # Integral gradient with the actual spherical normal moment, no G0 fit.
        grad=torch.zeros_like(stress)
        grad.index_add_(0,i,A[:,None,None]*(vs-v[i])[:,:,None]*n[:,None,:])
        grad.index_add_(0,j,-A[:,None,None]*(vs-v[j])[:,:,None]*n[:,None,:])
        if coupled is not None:grad+=coupled[2]
        else:
            grad-=sigma@mesh.cap_Q/z[:,None,None]
            if boundary is not None:grad+=boundary[1]
        if self.has_groups:
            grad=self.group_sum(grad)[self.groups]/self.group_sum(mesh.volume)[self.groups,None,None]
        else:grad/=mesh.volume[:,None,None]
        trace=grad.diagonal(dim1=1,dim2=2).sum(1)
        dr=s.r*trace/3 if self.interior_radii == 'dilate' else torch.zeros_like(s.r)
        cap_rate=-(sigma*mesh.cap_Q).sum((1,2))/z
        if coupled is not None:cap_rate=coupled[3]
        elif boundary is not None:cap_rate+=boundary[1].diagonal(dim1=1,dim2=2).sum(1)
        dr=torch.where(mesh.surface,cap_rate/mesh.cap_area.clamp(min=1e-300),dr)
        mesh_velocity=v
        flat_fit=torch.zeros_like(mesh.surface)
        if coupled is not None:
            mesh_velocity,dr=coupled[0],coupled[1]
        elif self.surface_motion=='least-squares':
            surface_force=boundary[0] if boundary is not None else torch.zeros_like(v)
            fitted,radial,flat_fit=project_cap_motion(v,sigma,z,mesh.cap_area,mesh.cap_normal_integral,surface_force,cap_rate)
            mesh_velocity=torch.where(mesh.surface[:,None],fitted,v)
            dr=torch.where(mesh.surface,radial,dr)
        mw,mc=mesh.mesh_flux(mesh_velocity,dr)
        # The mass boundary condition holds only in the integrated, P0 sense.
        material_cap=(mesh.cap_normal_integral*v).sum(1)+cap_rate
        cap_defect=mc-material_cap
        relative=A*(vs*n).sum(1)-mw
        density=self.group_sum(s.q)[self.groups]/self.group_sum(mesh.volume)[self.groups,None] if self.has_groups else s.q/mesh.volume[:,None]
        donor=torch.where(relative[:,None]>=0,density[i],density[j])
        dq=-mesh.incidence(relative[:,None]*donor)
        dq[:,1:4]+=mesh.incidence(A[:,None]*ts)
        dq[:,4]+=mesh.incidence(A*(ts*vs).sum(1))
        sym=.5*(grad+grad.transpose(1,2)); spin=.5*(grad-grad.transpose(1,2))
        stressdot=2*self.mu*deviator(sym)+spin@stress-stress@spin
        dq[:,5:14]+=(m[:,None,None]*stressdot).reshape(-1,9)
        if boundary is None:
            ground_force,kn,delta=self.contact(s)
            normal_power=(ground_force*v).sum(1)
            friction_heat_rate=0.; contact_area=0.; contact_mismatch_power=0.; contact_mismatch_abs_power=0.; contact_normal_rms=0.; contact_normal_max=0.
        else:
            ground_force,_,normal_power,heat,kn,delta,areas=boundary
            friction_heat_rate=float(heat.sum());contact_area=float(areas.sum())
            contact_mismatch_power=0.;contact_mismatch_abs_power=0.;contact_normal_rms=0.;contact_normal_max=0.
            if mesh.contact_quad is not None:
                owner,n,w,penetration=mesh.contact_quad
                if coupled is not None and len(owner):
                    t=coupled[5]['point_traction'];us=coupled[5]['point_velocity']
                else:t,us,_=contact_flux(owner,n,w,penetration,v,sigma,z,self.contact_k,self.friction,self.contact_damping)
                actual=mesh_velocity[owner]+dr[owner,None]*n
                power_difference=w*(t*(us-actual)).sum(1)
                contact_mismatch_power=float(power_difference.sum())
                contact_mismatch_abs_power=float(power_difference.abs().sum())
                normal_error=((us-actual)*n).sum(1)
                contact_normal_rms=float(torch.sqrt((w*normal_error.square()).sum()/w.sum().clamp(min=1e-300)))
                contact_normal_max=float(normal_error.abs().max()) if len(w) else 0.
        dq[:,1:4]+=m[:,None]*self.g+ground_force
        source=(s.q[:,1:4]*self.g).sum(1)+normal_power
        source_power=source.sum()
        dq[:,4]+=source
        # Conservative wave/advection/contact time-scale estimates, not a proof
        # of stability of the complete curved-boundary adaptation.
        external=self.groups[i]!=self.groups[j] if self.has_groups else torch.ones_like(i,dtype=torch.bool)
        external_relative=torch.where(external,relative,0.)
        outgoing=torch.zeros_like(m).index_add(0,i,external_relative.clamp(min=0)).index_add(0,j,(-external_relative).clamp(min=0))
        external_area=mesh.cap_area.clamp(min=0)+torch.zeros_like(m).index_add(0,i,A*external).index_add(0,j,A*external)
        wave=external_area*sound+outgoing
        if self.has_groups:
            group_volumes=self.group_sum(mesh.volume)[self.groups]
            dt_wave=group_volumes/self.group_sum(wave)[self.groups].clamp(min=1e-300)
            dt_contact=torch.sqrt(self.group_sum(m)[self.groups]/self.group_sum(kn)[self.groups].clamp(min=1e-300))
        else:
            dt_wave=mesh.volume/wave.clamp(min=1e-300)
            dt_contact=torch.sqrt(m/kn.clamp(min=1e-300))
        dt_radius=s.r/dr.abs().clamp(min=1e-300)
        cfl_dt=float(torch.minimum(torch.minimum(dt_wave,dt_contact),dt_radius).min())
        diagnostics={'dt_scale':cfl_dt,'wave_cell':int(dt_wave.argmin()),'wave_dt_scale':float(dt_wave.min()),
          'minimum_geometric_volume':float(mesh.volume.min()),'hidden_subcells':int((mesh.volume==0).sum()),
          'rho_min':float(rho.min()),'rho_max':float(rho.max()),
          'pressure_max_abs':float(pressure.abs().max()),'stress_vm_max':float(torch.sqrt(1.5*(stress*stress).sum((1,2))).max()),
          'cap_integrated_flux_defect':float(cap_defect.abs().sum()),
          'mass_rate_sum':float(dq[:,0].sum()),'force_sum':dq[:,1:4].sum(0).cpu().tolist(),
          'source_power':float(source_power),'energy_rate_sum':float(dq[:,4].sum()),
          'normal_contact_power':float(normal_power.sum()),
          'friction_heat_rate':friction_heat_rate,'contact_area':contact_area,
          'contact_boundary_motion_power_defect':contact_mismatch_power,
          'contact_boundary_motion_abs_power_defect':contact_mismatch_abs_power,
          'contact_normal_speed_mismatch_rms':contact_normal_rms,'contact_normal_speed_mismatch_max':contact_normal_max,
          'contact_cells':int((delta>0).sum()),'penetration_max':float(delta.max()),
          'ignored_cap_area':mesh.ignored_cap_area,
          'nearly_flat_cap_fits':coupled[5]['translation_only_caps'] if coupled is not None else int((flat_fit&mesh.surface).sum()),
          'boundary_local_iterations':coupled[5]['iterations'] if coupled is not None else 0,
          'boundary_local_relative_residual':coupled[5]['relative_residual'] if coupled is not None else None,
          'mesh_material_speed_difference_max':float((mesh_velocity-v).norm(dim=1).max())}
        return mesh_velocity,dr,dq,diagnostics

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
        if self.contact_model=='legacy-ball':
            # Coulomb sliding impulse limited by the time-integrated normal force.
            cf0,_,_=self.contact(s); cf1,_,_=self.contact(stage)
            jn=.5*dt*(cf0[:,1]+cf1[:,1])
            pt=q2[:,[1,3]]; plen=pt.norm(dim=1)
            impulse=torch.minimum(self.friction*jn,plen)
            fraction=impulse/plen.clamp(min=1e-300)
            q2[:,[1,3]]=pt*(1-fraction[:,None])
            friction_heat=float((((pt*pt).sum(1)-(q2[:,[1,3]]**2).sum(1))/(2*q2[:,0])).sum())
        else:
            friction_heat=.5*dt*(d0['friction_heat_rate']+d1['friction_heat_rate'])
        result=State(p2,r2,q2,mesh2)
        self.primitive(result)
        stats=d1.copy(); stats.update(dt=dt,friction_heat=friction_heat,
            source_work=.5*dt*(d0['source_power']+d1['source_power']),
            boundary_work_defect=.5*dt*(d0['contact_boundary_motion_power_defect']+d1['contact_boundary_motion_power_defect']),
            boundary_abs_work_defect=.5*dt*(d0['contact_boundary_motion_abs_power_defect']+d1['contact_boundary_motion_abs_power_defect']),
            normal_contact_work=.5*dt*(d0['normal_contact_power']+d1['normal_contact_power']))
        return result,stats

    @torch.no_grad()
    def cap_velocity_mismatch(self,s,samples=512):
        """Pointwise boundary mismatch, sampled on exposed spherical caps.
        RMS is area-weighted; this is diagnostic quadrature, not the force rule.
        """
        _,v,rho,_,_,sigma,c=self.primitive(s)
        z=rho*c
        mesh_velocity,dr,_,_=self.rhs(s)
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
            err=dr[ids,None]+normal_stress/z[ids,None]+(mesh_velocity[ids]-v[ids])@dirs.T
            if self.contact_model=='surface' and self.ground is not None:
                owner=ids[:,None].expand(-1,samples).flatten()
                normals=dirs[None].expand(len(ids),-1,-1).reshape(-1,3)
                penetration=(self.ground-y[:,:,1]).flatten()
                traction,_,_=contact_flux(owner,normals,torch.ones_like(penetration),penetration.clamp(min=0),v,sigma,z,self.contact_k,self.friction,self.contact_damping)
                active=(penetration>0)&(normals[:,1]<0)
                change=(traction*normals).sum(1)/z[owner]*active
                err-=change.reshape(len(ids),samples)
            weight=4*math.pi*s.r[ids,None]**2/samples*exposed
            total+=float(weight.sum()); norm2+=float((weight*err**2).sum())
            if bool(exposed.any()): maxerr=max(maxerr,float(err[exposed].abs().max()))
        if self.surface_motion=='coupled':
            return {'boundary_velocity':'sphere velocity is the material boundary unknown; equality is structural, not an accuracy test',
                    'rms_normal_speed_mismatch':0.,'max_normal_speed_mismatch':0.,
                    'rms_difference_to_unprojected_pointwise_acoustics':math.sqrt(norm2/max(total,1e-300)),
                    'sample_count':samples,'sampled_area':total}
        return {'sample_count':samples,'sampled_area':total,'rms_normal_speed_mismatch':math.sqrt(norm2/max(total,1e-300)),
                'max_normal_speed_mismatch':maxerr}
