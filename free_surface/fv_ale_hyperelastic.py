"""Opt-in neo-Hookean branch of the complete explicit power-ball ALE solver.

Same geometry, grouping, shared-face force/work, ground clipping, SSPRK2 and
CUDA replay. Channels 5:14 hold mass-weighted log(Bbar), NOT stress. Old FV
snapshots therefore cannot be resumed as hyperelastic snapshots.
Log states are advected with mass. Mixing is spatial numerical dissipation;
it is not plasticity. Total energy is never reset to match elastic energy.
The elastic/kinetic/thermal remainder is exposed for independent auditing.
"""
import math
import torch
from free_surface.fv_ale_solid import SolidALE, State
from free_surface.fv_ale_device import PlaneDeviceOps
from free_surface.fv_ale_hyperelastic_material import response, log_metric_rate, symmetric_deviator


class HyperelasticDeviceOps(PlaneDeviceOps):
    def wall_state(self,b,v,sigma,Z):
        mode=getattr(self.s,'wall_mode','unsupported')
        if mode not in ['compatible_slip','compatible_slip_acoustic']:
            return super().wall_state(b,v,sigma,Z)
        uv=self._compatible['wall_velocity'];active=self._compatible['floor']>0
        raw=-sigma[:,1,1]
        if mode=='compatible_slip_acoustic':raw=raw-Z*self._compatible['wall_normal_velocity']
        pressure=raw.clamp(min=0)*active
        scale=sigma.norm(dim=(1,2))+self.s.mu
        if mode=='compatible_slip_acoustic':scale=scale+Z*self._compatible['wall_normal_velocity'].abs()
        tensile=active&(raw < -1e-8*scale)
        # Elastic reaction and optional impedance correction are already in
        # the compatible transpose. Adding them here would count them twice.
        return uv,torch.zeros_like(v),torch.zeros_like(raw),raw,pressure,tensile,torch.zeros_like(active)

    def extra_rhs_diagnostics(self):
        if self.s.mechanical_reconstruction!='compatible':return {}
        result={key:self._compatible[key] for key in ['mechanical_rate_chain_W','mechanical_energy_joule',
            'internal_acoustic_W','donor_mixing_W','free_cap_W','wall_acoustic_W','wall_friction_W','gravity_W']}
        if self.s.mesh_fit_iterations:
            result.update({key:self._mesh_motion_diagnostics[k] for k,key in enumerate([
                'mesh_fit_weighted_slip_before','mesh_fit_weighted_slip_after','mesh_fit_max_site_correction_m_s',
                'mesh_fit_max_radius_correction_m_s','mesh_fit_relative_residual','mesh_fit_accepted'])})
        return result

    def energy_guard_gpu(self,b):
        total=self.group(b['q'],b['groups'])[b['groups']]
        volume=self.group(b['V'],b['groups'])[b['groups']]
        mass=total[:,0];v=total[:,1:4]/mass[:,None]
        H=total[:,5:14].reshape(-1,3,3)/mass[:,None,None]
        elastic=response(H,mass/volume,self.s.mu,self.s.bulk,self.s.rho0)[3]
        mechanical=elastic+.5*v.square().sum(1)
        negative=(mechanical-total[:,4]/mass).clamp(min=0)
        relative=negative/(elastic+self.s.mu/self.s.rho0).clamp(min=1e-300)
        return torch.stack(((b['q'][:,0]*mechanical).sum(),(b['q'][:,0]*negative).sum(),negative.max(),relative.max()))

    def energy_guard(self,state):
        return self.energy_guard_full(state)[:2]

    def energy_guard_full(self,state):
        values={'q':state.q,'V':state.mesh.volume,'groups':self.s.groups}
        return self.call('energy_guard',self.energy_guard_gpu,values).tolist()

    def mechanical_kinematics(self,b,v,sigma,Z):
        baseline=super().mechanical_kinematics(b,v,sigma,Z)
        if self.s.mechanical_reconstruction=='constant':return baseline
        if self.s.mechanical_reconstruction=='compatible':
            from free_surface.fv_ale_compatible_mechanics import kinematics
            result,self._compatible=kinematics(self,b,v,sigma,Z,baseline)
            return result
        from free_surface.fv_ale_reconstruction import group_centroids,slopes,face_values
        w,dr,radial,cg,cr,u0,t0,ti,tj=baseline
        i,j,n=b['i'],b['j'],b['n'];centers=group_centroids(b)
        # Velocity is affine exact, including rigid spin. Traction extrapolation
        # uses the neighbour-extrema limiter. Additional mechanical-work limits
        # below prevent the reconstruction from creating mechanical energy.
        L,_=slopes(b,v,centers,limited=False)
        gs,_=slopes(b,sigma.reshape(-1,9),centers,limited=True,blocks=[(0,9)])
        vl,vr=face_values(b,v,L,centers)
        sl,sr=face_values(b,sigma.reshape(-1,9),gs,centers)
        tl=(sl.reshape(-1,3,3)@n[:,:,None])[:,:,0]
        tr=(sr.reshape(-1,3,3)@n[:,:,None])[:,:,0]
        den=(Z[i]+Z[j])[:,None];a=self.s.interface_dissipation
        uh=(Z[i,None]*vl+Z[j,None]*vr+a*(tr-tl))/den
        th=(Z[j,None]*tl+Z[i,None]*tr+a*(Z[i]*Z[j])[:,None]*(vr-vl))/den
        def work(u,t):
            return (ti*(u-v[i])).sum(1)-(tj*(u-v[j])).sum(1)+(t*(v[i]-v[j])).sum(1)
        low,high=work(u0,t0),work(uh,th)
        eps=128*torch.finfo(v.dtype).eps
        work_tol=eps*(Z[i]+Z[j])*((vl-v[i]).square().sum(1)+(vr-v[j]).square().sum(1)+(v[i]-v[j]).square().sum(1))
        # A stress-free state has O(eps*K) pressure after grouped summation.
        # Do not let that representational residue switch off affine motion.
        work_tol+=eps*(self.s.mu+self.s.bulk)*((vl-v[i]).norm(dim=1)+(vr-v[j]).norm(dim=1)+(v[i]-v[j]).norm(dim=1))
        fraction=torch.where(high>work_tol,(-low).clamp(min=0)/(high-low).clamp(min=1e-300),1.).clamp(0,1)
        us=u0+fraction[:,None]*(uh-u0);ts=t0+fraction[:,None]*(th-t0)
        # Exact affine surface-gradient moments. The spherical geometry can
        # follow translations/spin and isotropic dilation; a deviatoric affine
        # normal field is only approximated. Match its integrated trace to the
        # actual sphere motion so density and shape still use one volume rate.
        N,Q,cap=b['N'],b['Q'],b['cap'];offset=b['p']-centers
        dw=(L@offset[:,:,None])[:,:,0]
        dr_extra=b['r']*L.diagonal(dim1=1,dim2=2).sum(1)/3
        moment=offset[:,:,None]*N[:,None,:]+b['r'][:,None,None]*Q
        dcg=L@moment;dcr=(dw*N).sum(1)+dr_extra*cap
        dcg+=(dcr-dcg.diagonal(dim1=1,dim2=2).sum(1))[:,None,None]*torch.eye(3,dtype=v.dtype,device=v.device)/3
        cap_low=(sigma*cg).sum((1,2));cap_high=cap_low+(sigma*dcg).sum((1,2))
        cap_tol=eps*Z*cap.clamp(min=0)*(dw.square().sum(1)+dr_extra.square())
        cap_tol+=eps*(self.s.mu+self.s.bulk)*cap.clamp(min=0)*(dw.norm(dim=1)+dr_extra.abs())
        cap_fraction=torch.where(cap_high>cap_tol,(-cap_low).clamp(min=0)/(cap_high-cap_low).clamp(min=1e-300),1.).clamp(0,1)
        cap_fraction=torch.where(b['surface'],cap_fraction,1.)
        cap_fraction=torch.where(b['floor']>0,0.,cap_fraction)
        w=w+cap_fraction[:,None]*dw;dr=dr+cap_fraction*dr_extra
        cg=cg+cap_fraction[:,None,None]*dcg;cr=cr+cap_fraction*dcr
        return w,dr,radial,cg,cr,us,ts,ti,tj

    def mechanical_force(self,b,ts):
        if self.s.mechanical_reconstruction=='compatible':return self._compatible['force']
        return super().mechanical_force(b,ts)

    def finalize_energy(self,b,dq,total,groupV,pressure,relative,donor,mw,mc):
        if self.s.mechanical_reconstruction!='compatible':return dq
        from free_surface.fv_ale_compatible_mechanics import energy_rhs
        return energy_rhs(self,b,dq,total,groupV,pressure,relative,donor,mw,mc,self._compatible)

    def rhs(self,state):
        if self.s.mechanical_reconstruction=='compatible' and getattr(self.s,'wall_mode','unsupported') not in ['compatible_slip','compatible_slip_acoustic'] and bool((state.mesh.ground_area>0).any()):
            raise RuntimeError('compatible mechanical reconstruction: wall work not yet implemented')
        return super().rhs(state)

    def mesh_motion(self,b,w,dr,us):
        if not self.s.mesh_fit_iterations:return w,dr
        from free_surface.fv_ale_mesh_motion import fit_motion
        w,dr,self._mesh_motion_diagnostics=fit_motion(b,w,dr,us,self.s.mesh_fit_iterations,self.s.mesh_fit_regularization,self.s.mesh_fit_radius)
        return w,dr

    def extra_rhs_values(self,mesh):
        if self.s.transport_reconstruction=='constant' and self.s.mechanical_reconstruction=='constant':return {}
        values={'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
        if hasattr(mesh,'planes'):
            from free_surface.fv_ale_planar_boundary import values as plane_values
            values.update(plane_values(mesh))
        if self.s.mechanical_reconstruction=='compatible' and self.s.affine_stencil!='one_ring':
            cached=getattr(self,'_affine_stencil_cache',None)
            if cached is None or cached[0] is not mesh or cached[1] is not self.s.groups or cached[2]!=self.s.groups._version:
                from free_surface.fv_ale_affine_stencil import build
                b={'p':mesh.p,'r':mesh.r,'V':mesh.volume,'groups':self.s.groups,
                    'i':mesh.i,'j':mesh.j,'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,
                    'N':mesh.cap_normal_integral,'floor':mesh.ground_area,**values}
                import os
                if self.s.affine_stencil=='smooth' and os.environ.get('FAST_STENCIL','1')!='0':
                    # 11-08: build + compact in Warp kernels, the reference when a root would grow (FAST_STENCIL=0: always the reference)
                    from free_surface.fv_ale_affine_stencil_fast import build_compact
                    extra,self.stencil_summary,self._affine_edge_capacity,edge_count=build_compact(b,getattr(self,'_affine_edge_capacity',0))
                    self.stencil_summary.update(directed_supports=edge_count,edge_capacity=self._affine_edge_capacity)
                else:
                    extra,self.stencil_summary=build(b,transition_ratio=.01 if self.s.affine_stencil=='smooth' else None)
                    if self.s.affine_stencil=='smooth':
                        from free_surface.fv_ale_affine_stencil import compact
                        extra,self._affine_edge_capacity,edge_count=compact(extra,getattr(self,'_affine_edge_capacity',0))
                        self.stencil_summary.update(directed_supports=edge_count,edge_capacity=self._affine_edge_capacity)
                stats=getattr(self,'stencil_statistics',{'builds':0,'bad_root_events':0,'fallback_root_events':0,'maximum_bad_roots':0,'maximum_width':0})
                stats['builds']+=1;stats['bad_root_events']+=self.stencil_summary['initial_bad_roots']
                stats['fallback_root_events']+=self.stencil_summary['fallback_roots']
                stats['maximum_bad_roots']=max(stats['maximum_bad_roots'],self.stencil_summary['initial_bad_roots'])
                stats['maximum_width']=max(stats['maximum_width'],self.stencil_summary['width'])
                self.stencil_statistics=stats
                cached=(mesh,self.s.groups,self.s.groups._version,extra)
                self._affine_stencil_cache=cached
            values.update(cached[3])
        return values

    def advective_donor(self,b,total,groupV,relative):
        if self.s.transport_reconstruction=='constant':
            return super().advective_donor(b,total,groupV,relative)
        if self.s.transport_reconstruction=='entropy':
            from free_surface.fv_ale_entropy_transport import transport_state
            H=total[:,5:14].reshape(-1,3,3)/total[:,0,None,None]
            material=response(H,total[:,0]/groupV,self.s.mu,self.s.bulk,self.s.rho0)
            return transport_state(total,groupV,material,self.s.mu,self.s.rho0,b['i'],b['j'],relative,self.s.transport_dissipation)
        from free_surface.fv_ale_reconstruction import group_centroids,slopes,face_values,compatible_transport
        mass=total[:,0];rho=mass/groupV;v=total[:,1:4]/mass[:,None]
        H=total[:,5:14].reshape(-1,3,3)/mass[:,None,None]
        material=response(H,rho,self.s.mu,self.s.bulk,self.s.rho0)
        pressure,elastic,B=material[0],material[3],material[5]
        remainder=total[:,4]/mass-.5*v.square().sum(1)-elastic
        values=torch.cat((rho[:,None],v,H.reshape(-1,9)),1)
        centers=group_centroids(b);gradient,good=slopes(b,values,centers,blocks=[(1,4),(4,13)])
        left,right=face_values(b,values,gradient,centers)
        face=torch.where(relative[:,None]>=0,left,right)
        # CUDA graph face padding has c=0, not a physical face centroid.
        # Do not evaluate nonlinear material functions at extrapolated padding:
        # multiplying the resulting NaN by its zero area would not remove it.
        face=torch.where((b['A']>0)[:,None],face,values[b['i']])
        rf=face[:,0];vf=face[:,1:4];hf=symmetric_deviator(face[:,4:13].reshape(-1,3,3))
        donor=torch.zeros((len(relative),16),dtype=total.dtype,device=total.device)
        donor[:,0]=rf;donor[:,1:4]=rf[:,None]*vf
        donor[:,5:14]=rf[:,None]*hf.reshape(-1,9)
        T=self.s.mu/(2*self.s.rho0)*symmetric_deviator(B)
        chi=torch.zeros_like(total)
        chi[:,0]=elastic+pressure/rho-(T*H).sum((1,2))-.5*v.square().sum(1)
        chi[:,1:4]=v;chi[:,5:14]=T.reshape(-1,9)
        i,j=b['i'],b['j']
        low=super().advective_donor(b,total,groupV,relative)
        thermal_up=torch.where(relative>=0,remainder[i],remainder[j])
        donor,_=compatible_transport(donor,low,chi[i],chi[j],pressure[i],pressure[j],relative,thermal_up)
        return donor

    def transport_dt(self,b,donor,relative,total):
        if self.s.transport_reconstruction=='constant':
            return super().transport_dt(b,donor,relative,total)
        i,j,g=b['i'],b['j'],b['groups']
        flux=torch.where(g[i]!=g[j],relative*donor[:,0],0.)
        out=torch.zeros_like(b['V']).index_add(0,i,flux.clamp(min=0))
        out.index_add_(0,j,(-flux).clamp(min=0))
        return (total[:,0]/self.group(out,g)[g].clamp(min=1e-300)).min()

    def acoustic_interface(self,v,Z,ti,tj,i,j):
        a=self.s.interface_dissipation
        if a==1.:return super().acoustic_interface(v,Z,ti,tj,i,j)
        den=(Z[i]+Z[j])[:,None]
        us=(Z[i,None]*v[i]+Z[j,None]*v[j]+a*(tj-ti))/den
        ts=(Z[j,None]*ti+Z[i,None]*tj+a*(Z[i]*Z[j])[:,None]*(v[j]-v[i]))/den
        return us,ts

    def free_cap_motion(self,v,sigma,Z,N,Q,cap,surface):
        w,dr,radial=super().free_cap_motion(v,sigma,Z,N,Q,cap,surface)
        a=self.s.cap_dissipation
        if a==1.:return w,dr,radial
        return v+a*(w-v),a*dr,radial

    def interface_energy_flux(self,us,ts,v,ti,tj,i,j):
        if self.s.interface_dissipation==1. and self.s.mechanical_reconstruction=='constant':
            return super().interface_energy_flux(us,ts,v,ti,tj,i,j)
        # Symmetric compatible work flux: each side receives half of the
        # nonnegative pair dissipation. ts.us is only locally dissipative
        # for the full acoustic solver and must not be reused after blending.
        return .5*((v[i]+v[j])*ts+(ti+tj)*us-ti*v[i]-tj*v[j]).sum(1)

    def primitive(self, q, V, g):
        total = self.group(q, g)[g]; volume = self.group(V, g)[g]
        mass = total[:, 0]; v = total[:, 1:4] / mass[:, None]; rho = mass / volume
        H = total[:, 5:14].reshape(-1, 3, 3) / mass[:, None, None]
        pressure, stress, sigma, elastic, c2, B, J = response(H, rho, self.s.mu, self.s.bulk, self.s.rho0,self.s.acoustic_bound)
        valid = torch.isfinite(q).all() & (mass > 0).all() & (volume > 0).all()
        valid = valid & torch.isfinite(sigma).all() & torch.isfinite(elastic).all() & (c2 > 0).all() & torch.isfinite(c2).all()
        return total, volume, v, rho, pressure, stress, sigma, c2.sqrt(), valid

    def material_rate(self, grad, stress, total, mass):
        # total is repeated group Q; mass passed by RHS is each geometric
        # member's mass and must NOT be used to divide a group shape state.
        H = total[:, 5:14].reshape(-1, 3, 3) / total[:, 0, None, None]
        return log_metric_rate(H, grad)

    def plastic_gpu(self, b):
        q, V, g = b['q'], b['V'], b['groups']
        total = self.group(q, g); volume = self.group(V, g)
        out = total[g] * (V / volume[g])[:, None]
        # The log metric evolves in the symmetric traceless subspace. Remove
        # only representational drift; no eigenvalue clipping or yield return.
        out[:, 5:14] = symmetric_deviator(out[:, 5:14].reshape(-1, 3, 3)).reshape(-1, 9)
        valid = self.primitive(out, V, g)[-1]
        return out, valid


class HyperelasticALE(SolidALE):
    constitutive = 'neo_hookean'
    state_schema = 'fv_neo_hookean_log_bbar_v1'

    def __init__(self, *args, **kwargs):
        E=kwargs.get('E',args[2] if len(args)>2 else None)
        nu=kwargs.get('nu',args[3] if len(args)>3 else None)
        rho0=kwargs.get('rho0',args[5] if len(args)>5 else 1000.)
        if E is None or not math.isfinite(float(E)) or E<=0:
            raise ValueError('hyperelastic Young modulus must be positive and finite')
        if nu is None or not math.isfinite(float(nu)) or not -1<nu<.5:
            raise ValueError('compressible hyperelastic Poisson ratio must lie strictly between -1 and 0.5')
        if not math.isfinite(float(rho0)) or rho0<=0:
            raise ValueError('hyperelastic reference density must be positive and finite')
        # Experimental entropy-compatible acoustic blending; defaults retain
        # the baseline. These coefficients do not alter E, nu or wave CFL.
        self.interface_dissipation=float(kwargs.pop('interface_dissipation',1.))
        self.precision_geometry=bool(kwargs.pop('precision_geometry',False))
        self._precision_geometry_records=[]
        self.acoustic_bound=kwargs.pop('acoustic_bound','conservative')
        if self.acoustic_bound not in ['conservative','tight']:
            raise ValueError('acoustic_bound must be conservative or tight')
        self.cap_dissipation=float(kwargs.pop('cap_dissipation',1.))
        self.transport_reconstruction=kwargs.pop('transport_reconstruction','constant')
        self.transport_dissipation=float(kwargs.pop('transport_dissipation',.1))
        if not 0<=self.transport_dissipation<=1:
            raise ValueError('transport dissipation must be in [0,1]')
        self.mechanical_reconstruction=kwargs.pop('mechanical_reconstruction','constant')
        self.wall_mode=kwargs.pop('wall_mode','unsupported')
        if self.wall_mode not in ['unsupported','compatible_slip','compatible_slip_acoustic']:
            raise ValueError('unrecognized hyperelastic wall mode')
        if self.wall_mode in ['compatible_slip','compatible_slip_acoustic'] and (self.mechanical_reconstruction!='compatible' or kwargs.get('friction',.5)!=0):
            raise ValueError('compatible_slip requires compatible mechanics and explicitly friction=0; release is still guarded')
        self.affine_stencil=kwargs.pop('affine_stencil','one_ring')
        self.affine_cap_motion=kwargs.pop('affine_cap_motion','isotropic')
        self.cap_motion_regularization=float(kwargs.pop('cap_motion_regularization',0.))
        if not math.isfinite(self.cap_motion_regularization) or self.cap_motion_regularization<0:
            raise ValueError('cap motion regularization must be finite and nonnegative')
        if self.cap_motion_regularization and (self.affine_cap_motion!='balanced' or self.cap_dissipation!=0):
            raise ValueError('regularized cap motion requires balanced motion and zero cap damping')
        if self.affine_cap_motion not in ['isotropic','volume','balanced']:
            raise ValueError('affine cap motion must be isotropic, volume or balanced')
        if self.affine_stencil not in ['one_ring','extended','smooth']:
            raise ValueError('affine_stencil must be one_ring, extended or smooth')
        if self.mechanical_reconstruction not in ['constant','affine','compatible']:
            raise ValueError('mechanical_reconstruction must be constant, affine or compatible')
        if self.affine_cap_motion!='isotropic' and self.mechanical_reconstruction!='compatible':
            raise ValueError('affine cap-volume motion requires its compatible force transpose')
        self.mesh_fit_iterations=int(kwargs.pop('mesh_fit_iterations',0))
        self.mesh_fit_regularization=float(kwargs.pop('mesh_fit_regularization',.1))
        self.mesh_fit_radius=bool(kwargs.pop('mesh_fit_radius',False))
        if self.mechanical_reconstruction=='compatible' and self.mesh_fit_iterations:
            raise ValueError('compatible interior mesh fitting is blocked: frozen identities pass but full trajectories develop local energy deficits')
        if self.mesh_fit_iterations<0 or self.mesh_fit_regularization<=0:
            raise ValueError('mesh fit requires nonnegative iterations and positive regularization')
        if self.transport_reconstruction not in ['constant','linear','entropy']:
            raise ValueError('transport_reconstruction must be constant, linear or entropy')
        if not (0<=self.interface_dissipation<=1 and 0<=self.cap_dissipation<=1):
            raise ValueError('acoustic blend coefficients must be in [0,1]')
        if kwargs.get('contact_model', 'surface') != 'plane':
            raise ValueError('neo-Hookean trial currently requires the existing plane-ground geometry path')
        if kwargs.get('surface_motion', 'coupled') != 'coupled':
            raise ValueError('neo-Hookean trial requires coupled free-cap motion')
        if kwargs.get('device_step', True) is False:
            raise ValueError('use the integrated GPU path for neo-Hookean; the old reference RHS stores stress')
        super().__init__(*args, **kwargs)
        self.e0 = 0.
        self.initial.q[:, 4] = 0.
        self.initial.q[:, 5:] = 0.

    def device_ops(self):
        if self._device_ops is None: self._device_ops = HyperelasticDeviceOps(self)
        return self._device_ops

    def advance(self, s, dt, first=None, cfl=.5):
        return super().advance(s, dt, first, cfl)

    def normalize_volume_roundoff(self, mesh):
        """Only zero roundoff-scale negatives in already agglomerated cells.

        The ball-cap formula subtracts full-sphere terms, so its absolute
        cancellation scale is the ball volume, not the vanishing cell volume.
        256 eps times that volume is a conservative numerical guard, not a
        proof of the geometric algorithm's total error. No positive floor,
        no mass/energy changes and no acceptance of larger negative volumes.
        Counts include rejected trial cuts as well as accepted cuts.
        """
        if not hasattr(self, '_volume_roundoff_stats'):
            self._volume_roundoff_stats = torch.zeros(3, dtype=mesh.volume.dtype, device=mesh.volume.device)
        if not self.has_groups:return
        bound = 256*torch.finfo(mesh.volume.dtype).eps*(4*torch.pi/3)*mesh.r.pow(3)
        members = self.group_sum(torch.ones_like(mesh.volume))[self.groups]
        repair = (mesh.volume < 0)&(mesh.volume >= -bound)&(members > 1)
        correction = torch.where(repair, -mesh.volume, torch.zeros_like(mesh.volume))
        # A near-vanishing group cannot be rescued by this numerical guard.
        # Even many eligible members must change its volume by at most 1e-10.
        group_correction=self.group_sum(correction)[self.groups]
        group_positive=self.group_sum(mesh.volume.clamp(min=0))[self.groups]
        repair=repair&(group_correction<=1e-10*group_positive)
        correction=torch.where(repair,-mesh.volume,torch.zeros_like(mesh.volume))
        mesh.volume = mesh.volume + correction
        self._volume_roundoff_stats[0] += repair.sum()
        self._volume_roundoff_stats[1] += correction.sum()
        self._volume_roundoff_stats[2] = torch.maximum(self._volume_roundoff_stats[2], correction.max())

    def reevaluate_invalid_geometry(self,mesh):
        if not self.precision_geometry or not self.has_groups:return False
        from free_surface.fv_ale_precise_geometry import reevaluate
        return reevaluate(self,mesh)

    def primitive(self, s):
        total, volume, v, rho, pressure, stress, sigma, sound, valid = self.device_ops().primitive(s.q, s.mesh.volume, self.groups)
        if not bool(valid): raise RuntimeError('invalid hyperelastic mass/volume/metric/wave speed')
        return s.q[:, 0], v, rho, pressure, stress, sigma, sound

    def set_velocity(self, s, v):
        # Unlike a reset to the stress-free reference energy, a velocity
        # assignment must retain stored elastic energy and numerical heat.
        old = self.primitive(s)[1]
        v = torch.as_tensor(v, dtype=s.p.dtype, device=s.p.device).expand_as(s.p)
        s.q[:, 4] += .5*s.q[:, 0]*(v.square().sum(1)-old.square().sum(1))
        s.q[:, 1:4] = s.q[:, :1]*v
        return s

    def initialize_strain(self, s, F):
        """Initialize a homogeneous prestrained body, no geometry is fabricated.

        Reference material coordinates are F^{-1} times this actual domain.
        This is initialization only, NEVER a step-end energy/volume repair.
        """
        F = torch.as_tensor(F, dtype=s.p.dtype, device=s.p.device)
        J = torch.linalg.det(F)
        if not bool(torch.isfinite(F).all() and J > 0): raise ValueError('initial F must have positive determinant')
        Bbar = (F @ F.T) / J**(2/3)
        eigen, R = torch.linalg.eigh(Bbar)
        H = symmetric_deviator((R * eigen.log()[None, :]) @ R.T)
        v = self.primitive(s)[1]
        mass = self.rho0 / J * s.mesh.volume
        q = torch.zeros_like(s.q); q[:, 0] = mass; q[:, 1:4] = mass[:, None] * v
        q[:, 5:14] = mass[:, None] * H.reshape(1, 9)
        elastic = response(H[None], (self.rho0 / J).reshape(1), self.mu, self.bulk, self.rho0)[3][0]
        q[:, 4] = mass * (elastic + .5 * v.square().sum(1))
        return State(s.p, s.r, q, s.mesh)

    def energy_audit(self, s):
        total = self.group_sum(s.q)[self.groups]
        volume = self.group_sum(s.mesh.volume)[self.groups]
        m = total[:, 0]; v = total[:, 1:4] / m[:, None]
        H = total[:, 5:14].reshape(-1, 3, 3) / m[:, None, None]
        _, _, _, elastic, _, _, J = response(H, m / volume, self.mu, self.bulk, self.rho0)
        kinetic = .5 * v.square().sum(1)
        remainder = total[:, 4] / m - kinetic - elastic
        return {'elastic_energy_joule': float((s.q[:, 0] * elastic).sum()),
                'negative_specific_remainder_max_joule_per_kg':float((-remainder).clamp(min=0).max()),
                'negative_remainder_over_local_elastic_plus_mu_over_rho0':float(((-remainder).clamp(min=0)/(elastic+self.mu/self.rho0).clamp(min=1e-300)).max()),
                'precision_geometry_reevaluations':len(self._precision_geometry_records),
                'kinetic_energy_joule': float((s.q[:, 0] * kinetic).sum()),
                'nonelastic_remainder_joule': float((s.q[:, 0] * remainder).sum()),
                'nonelastic_specific_min': float(remainder.min()),
                'negative_remainder_deficit_joule': float((s.q[:, 0] * (-remainder).clamp(min=0.)).sum()),
                'negative_remainder_mass_fraction': float(s.q[remainder < -1e-9, 0].sum()/s.q[:, 0].sum()),
                'roundoff_volume_corrections_all_trial_cuts': int(self._volume_roundoff_stats[0]),
                'roundoff_volume_max_correction_m3': float(self._volume_roundoff_stats[2]),
                'J_min': float(J.min()), 'J_max': float(J.max()),
                'mass_fraction_J_outside_half_to_two':float(s.q[(J<.5)|(J>2),0].sum()/s.q[:,0].sum()),
                'log_shape_max_abs_eigenvalue': float(torch.linalg.eigvalsh(symmetric_deviator(H)).abs().max())}
