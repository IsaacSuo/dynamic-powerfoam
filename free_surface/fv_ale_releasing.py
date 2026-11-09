"""Opt-in integration of material/vacuum power faces in the full FV solver.

Boundary activation is an explicit unchanged-material-state operation handled
by the driver. A trial requesting release is discarded; no hidden time step,
adhesive reaction, energy reset or clamped material update is accepted.
"""
import os
import torch
from free_surface.fv_ale_hyperelastic import HyperelasticALE,HyperelasticDeviceOps
from free_surface.fv_ale_solid import State
from free_surface.fv_ale_planar_boundary import Planes


class ReleaseRequired(RuntimeError):
    def __init__(self,cells):
        self.cells=cells
        super().__init__('material floor faces require moving-boundary activation: '+str(cells))


class MovingFaceRHS(tuple):
    """Keep device samples separate from the existing JSON diagnostics."""

    def __new__(cls, values, boundary):
        result = super().__new__(cls, values)
        result.boundary = boundary
        return result


class ReleasingDeviceOps(HyperelasticDeviceOps):
    def wall_state(self,b,v,sigma,Z):
        result=super().wall_state(b,v,sigma,Z)
        self._release_mask=result[5]
        return result

    def rhs_gpu(self,b):
        original=super().rhs_gpu(b)
        if 'plane_i' in b:
            boundary=self._compatible['planar'];adot=boundary['adot'];bdot=boundary['bdot']
        else:
            adot=b['p'].new_zeros((0,3));bdot=b['r'].new_zeros((0,))
        result = (*original,adot,bdot,self._release_mask)
        if self.s.moving_face_gauss and 'plane_i' in b:
            if self.s.moving_face_endpoints:
                return (*result, boundary['normal_speed'], boundary['plane_heat'], self._compatible['volume_rate'])
            return (*result, *self._compatible['moving_face_rhs'], self._compatible['volume_rate'])
        return result

    def rhs_values(self,state):
        m=state.mesh
        b={'q':state.q,'V':m.volume,'groups':self.s.groups,'p':state.p,'r':state.r,
            'i':m.i,'j':m.j,'A':m.area,'n':m.normal,'c':m.centroid,'Q':m.cap_Q,'N':m.cap_normal_integral,
            'cap':m.cap_area,'surface':m.surface,'floor':m.ground_area}
        b.update(self.extra_rhs_values(m))
        return b

    def moving_face_sample_gpu(self, b):
        from free_surface.fv_ale_affine_stencil import gradient
        from free_surface.fv_ale_planar_mechanics import motion, moving_face_rhs

        total, volume, v, rho, pressure, shear, sigma, sound, valid = self.primitive(b['q'], b['V'], b['groups'])
        planar = motion(b, v, gradient(b, v), sigma, rho * sound, b['affine_centers'], self.s.ground,
                        regularization=self.s.cap_motion_regularization)
        shape_conjugate = (0.5 / rho)[:, None, None] * shear
        rates = moving_face_rhs(self, b, planar, total, volume, pressure, v, shape_conjugate)
        return *rates, planar['adot'], planar['bdot'], valid

    def moving_face_sample(self, state):
        b = self.rhs_values(state)
        # The affine stencil above already contains the internal mesh data.
        for key in ('i', 'j', 'A', 'n', 'c'):
            del b[key]
        q, volume, work, adot, bdot, valid = self.call('moving_face_sample', self.moving_face_sample_gpu, b)
        if not bool(valid):
            raise RuntimeError('moving-face quadrature encountered an invalid material state')
        n = len(state.mesh.planes.owner)
        return q.clone(), volume.clone(), work.clone(), adot[:n].clone(), bdot[:n].clone()

    def rhs(self,state):
        m=state.mesh
        b=self.rhs_values(state)
        w,dr,dq,pack,adot,bdot,release,*boundary=self.call('rhs_releasing',self.rhs_gpu,b,('i','j','A','n','c'))
        d=dict(zip(self.rhs_keys,pack.tolist()))
        if not d.pop('valid'):raise RuntimeError('nonpositive mass/sound-speed or nonfinite state; no floor applied')
        if d['wall_tensile_prediction_cells']:raise ReleaseRequired(torch.nonzero(release).flatten().tolist())
        for key in ['wave_cell','hidden_subcells','contact_cells','nearly_flat_cap_fits','wall_tensile_prediction_cells','wall_sliding_cells']:d[key]=int(d[key])
        if not d['contact_cells']:d['wall_pressure_min']=0.
        d['force_sum']=[d.pop('force_x'),d.pop('force_y'),d.pop('force_z')]
        for key in ['normal_contact_power','contact_boundary_motion_power_defect','contact_boundary_motion_abs_power_defect',
            'contact_normal_speed_mismatch_rms','contact_normal_speed_mismatch_max','penetration_max','wall_normal_velocity_max','boundary_local_relative_residual']:d[key]=0.
        d.update(ignored_cap_area=m.ignored_cap_area,boundary_local_iterations=0,ground_geometry=m.ground_geometry_info)
        # Lists are small boundary controls, explicitly owned and serializable.
        # No reference to a later overwritten CUDA replay buffer is retained.
        count=len(m.planes.owner) if hasattr(m,'planes') else 0
        d['plane_rates']={'a':adot[:count].tolist(),'b':bdot[:count].tolist()}
        result = w.clone(),dr.clone(),dq.clone(),d
        if boundary:
            if self.s.moving_face_endpoints:
                boundary = (boundary[0][:len(m.plane_i)], boundary[1][:count], boundary[2])
            return MovingFaceRHS(result, tuple(x.clone() for x in boundary))
        return result


class ReleasingHyperelasticALE(HyperelasticALE):
    boundary_mode='releasing_planes'
    moving_face_gauss = False

    def __init__(self,*args,**kwargs):
        quadrature = os.environ.get('FV_MOVING_FACE_GAUSS', '0')
        self.moving_face_gauss = quadrature in ('1', 'midpoint', 'endpoint')
        self.moving_face_midpoint = quadrature == 'midpoint'
        self.moving_face_endpoints = quadrature == 'endpoint'
        required={'wall_mode':'compatible_slip_acoustic','mechanical_reconstruction':'compatible',
            'affine_stencil':'smooth','affine_cap_motion':'balanced','cap_dissipation':0.,'friction':0.}
        for key,value in required.items():
            if kwargs.get(key)!=value:raise ValueError('releasing planes require '+key+'='+str(value))
        self.boundary_insertions=[];self.boundary_prunings=[]
        super().__init__(*args,**kwargs)

    def device_ops(self):
        if self._device_ops is None:self._device_ops=ReleasingDeviceOps(self)
        return self._device_ops

    def cut(self,p,r,planes=None):
        if planes is None or len(planes.owner)==0:return super().cut(p,r)
        if not bool(torch.isfinite(p).all()&torch.isfinite(r).all()&(r>0).all()):
            raise RuntimeError('invalid site/radius; state not accepted')
        from free_surface.fv_ale_planar_floor_reuse import cut
        mesh=cut(self,p,r,planes);self.geometry_calls+=1
        self.normalize_volume_roundoff(mesh)
        valid=mesh.volume>=0 if self.has_groups else mesh.volume>0
        if not bool(torch.isfinite(mesh.volume).all()&valid.all()):
            # The exceptional positive integral now certifies all vacuum and
            # floor half-spaces as well. Unsupported geometry still fails;
            # no volume floor, q change or real-only fallback is permitted.
            repaired=self.reevaluate_invalid_geometry(mesh)
            valid=mesh.volume>=0 if self.has_groups else mesh.volume>0
            if not repaired or not bool(torch.isfinite(mesh.volume).all()&valid.all()):
                raise RuntimeError('moving-plane cut has nonpositive material volume; no incompatible fallback used')
        self.device_ops().moments(mesh);mesh.contact_quad=None
        return mesh

    def activate(self,state,cells):
        old=getattr(state.mesh,'planes',None)
        if old is not None:
            floor=(old.a[:,0]==0)&(old.a[:,1]==-1)&(old.a[:,2]==0)&(old.b==-self.ground)
            existing=set(old.owner[floor].tolist())
            duplicates=existing.intersection(cells)
            if duplicates:raise RuntimeError('already active floor plane still has tensile reaction: '+str(sorted(duplicates)))
        new=Planes.floor(state.p,state.r,cells,self.ground)
        if old is not None:
            new=Planes(*(torch.cat((getattr(old,k),getattr(new,k))) for k in ['owner','d','a','b']))
        mesh=self.cut(state.p,state.r,new)
        before=self.group_sum(state.mesh.volume)[self.groups];after=self.group_sum(mesh.volume)[self.groups]
        error=float(((after-before).abs()/before.clamp(min=1e-300)).max())
        if error>1e-9:raise RuntimeError('boundary insertion changed a material group volume: '+str(error))
        record={'owners':list(cells),'plane_count':len(new.owner),'maximum_group_volume_relative_change':error}
        from free_surface.fv_ale_energy_ledger import mechanical
        record['mechanical_energy_roundoff_change_joule']=float(mechanical(self,state.q,mesh.volume)-mechanical(self,state.q,state.mesh.volume))
        self.boundary_insertions.append(record)
        return State(state.p,state.r,state.q,mesh),record

    def compact_boundaries(self,state):
        """Remove only unexposed geometric constraints after a domain check.

        A zero exposed area alone is NOT proof: a generator may completely
        hide a cell. Removing constraints can only enlarge each real power
        cell, so independently recutting and checking every group's volume
        excludes that case. q and all visible boundary controls are retained.
        No positive face is removed and no energy/state is repaired.
        """
        if not hasattr(state.mesh,'planes'):return state,None
        mesh=state.mesh;planes=mesh.planes
        area=torch.zeros_like(planes.b).index_add(0,mesh.plane_j,mesh.plane_area)
        # A just-inserted floor plane may have zero area at the accepted point
        # but is needed by the imminent trial's newly opening contact face.
        # Retain that active-set seed; deleting it during the same retry would
        # repeat activation forever without advancing physical time.
        on_floor=(planes.a[:,0]==0)&(planes.a[:,1]==-1)&(planes.a[:,2]==0)&(planes.b==-self.ground)
        keep=(area>0)|on_floor
        if bool(keep.all()):return state,None
        reduced=Planes(*(getattr(planes,key)[keep] for key in ['owner','d','a','b']))
        try:cut=self.cut(state.p,state.r,reduced)
        except RuntimeError:return state,None
        before=self.group_sum(mesh.volume)[self.groups];after=self.group_sum(cut.volume)[self.groups]
        error=float(((after-before).abs()/before.clamp(min=1e-300)).max())
        cap_error=float((cut.cap_area-mesh.cap_area).norm()/mesh.cap_area.norm().clamp(min=1e-300))
        floor_error=float((cut.ground_area-mesh.ground_area).norm()/mesh.ground_area.norm().clamp(min=1e-300))
        if max(error,cap_error,floor_error)>1e-10:return state,None
        record={'removed':int((~keep).sum()),'remaining':int(keep.sum()),
            'maximum_group_volume_relative_change':error,'cap_area_relative_change':cap_error,
            'floor_area_relative_change':floor_error,'removed_positive_faces':0}
        from free_surface.fv_ale_energy_ledger import mechanical
        record['mechanical_energy_roundoff_change_joule']=float(mechanical(self,state.q,cut.volume)-mechanical(self,state.q,mesh.volume))
        self.boundary_prunings.append(record)
        return State(state.p,state.r,state.q,cut),record

    @staticmethod
    def plane_rates(state,rhs):
        rates=rhs[3]['plane_rates'];pl=state.mesh.planes
        return pl.a.new_tensor(rates['a']),pl.b.new_tensor(rates['b'])

    def advance(self,s,dt,first=None,cfl=.5):
        if not hasattr(s.mesh,'planes'):return super().advance(s,dt,first,cfl)
        first=self.rhs(s) if first is None else first;d0=first[3]
        if dt>cfl*d0['dt_scale']*(1+1e-12):raise RuntimeError('requested step exceeds FV CFL; no hidden substep')
        pa,pb=self.plane_rates(s,first);planes=s.mesh.planes
        p1,r1,qtrial=self.device_ops().update(s,dt,first)
        pl1=planes.clone();pl1.a+=dt*pa;pl1.b+=dt*pb
        mesh1=self.cut(p1,r1,pl1);q1=self.plastic_return(qtrial,mesh1)
        stage=State(p1,r1,q1,mesh1);second=self.rhs(stage);d1=second[3]
        if dt>cfl*d1['dt_scale']*1.25:raise RuntimeError('stage CFL changed too much; retry from unchanged state')
        pa1,pb1=self.plane_rates(stage,second)
        p2,r2,qtrial=self.device_ops().update(s,dt,first,second)
        pl2=planes.clone();pl2.a+=.5*dt*(pa+pa1);pl2.b+=.5*dt*(pb+pb1)
        sweep = None
        if self.moving_face_gauss and not self.moving_face_endpoints:
            from free_surface.fv_ale_moving_face_step import correct
            qtrial, pl2, sweep = correct(self, s, dt, first, second, p2, r2, qtrial, pl2)
        mesh2=self.cut(p2,r2,pl2)
        if self.moving_face_gauss and self.moving_face_endpoints:
            from free_surface.fv_ale_moving_face_step import correct_endpoints
            qtrial, sweep = correct_endpoints(self, s, stage, State(p2,r2,qtrial,mesh2), dt, first, second)
        q2=self.plastic_return(qtrial,mesh2)
        result=State(p2,r2,q2,mesh2);totals=self.device_ops().validate(result)
        stats=d1.copy();stats.update(dt=dt,friction_heat=.5*dt*(d0['friction_heat_rate']+d1['friction_heat_rate']),
            source_work=.5*dt*(d0['source_power']+d1['source_power']),boundary_work_defect=0.,
            boundary_abs_work_defect=0.,normal_contact_work=0.,total_mass=totals[0],total_energy=totals[1],
            moving_plane_count=len(planes.owner))
        if sweep is not None:
            work, integrated_volume, diagnostics = sweep
            result.integrated_mechanical_work = work
            residual = self.group_sum(mesh2.volume - integrated_volume)[self.groups]
            stats.update(diagnostics, moving_face_volume_residual_m3=float(residual.abs().max()))
        return result,stats
