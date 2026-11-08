"""GPU-resident numerical segments of the plane-contact FV step.

The dynamic geometric cuts remain the same. Fixed-size tensor work between
cuts is captured in CUDA graphs; face buffers are capacity-padded with zero
area. Checks and diagnostics cross to the host together at the same algorithmic
boundaries. SolidALE.device_step=False retains the original implementation.
"""
import time
from collections import OrderedDict
import torch


def dev(a):
    a=.5*(a+a.transpose(-1,-2))
    return a-a.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]*torch.eye(3,dtype=a.dtype,device=a.device)/3


class Replay:
    def __init__(self,fn,values,face_keys=(),capacity=None,padding=None):
        tick=time.perf_counter()
        self.face_keys=face_keys;self.buffers={};self.padding=padding or {}
        for k,v in values.items():
            shape=(capacity,)+v.shape[1:] if k in face_keys else v.shape
            if k in self.padding:shape=(self.padding[k],)+v.shape[1:]
            self.buffers[k]=torch.zeros(shape,dtype=v.dtype,device=v.device)
        self.copy(values)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(self.buffers)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.outputs=fn(self.buffers)
        torch.cuda.synchronize()
        self.build_seconds=time.perf_counter()-tick

    def copy(self,values):
        for k,v in values.items():
            if k in self.padding:
                self.buffers[k][:len(v)].copy_(v)
                # Zero-area dummy faces and inactive dummy controls. Positive
                # lengths avoid 0/0 before multiplication by zero face area.
                self.buffers[k][len(v):].fill_(1 if k in ['plane_d','plane_length'] else 0)
            elif k in self.face_keys:self.buffers[k][:len(v)].copy_(v)
            else:self.buffers[k].copy_(v)
        if self.face_keys:
            # All padding indices stay valid. Only its area must be zero.
            self.buffers['A'][len(values['A']):].zero_()

    def __call__(self,values):
        self.copy(values);self.graph.replay();return self.outputs


class PlaneDeviceOps:
    def __init__(self,solver):
        self.s=solver;self.cache=OrderedDict();self.builds=[]

    def call(self,name,fn,values,face_keys=()):
        cap=((len(values['A'])+1023)//1024)*1024 if face_keys else None
        cap=max(1,cap) if cap is not None else None
        key=(name,cap,tuple((k,tuple(v.shape[1:] if k in face_keys else v.shape),v.dtype) for k,v in values.items()))
        padding={}
        if 'plane_i' in values and getattr(self,'pad_planar_graph',True):
            face_cap=max(256,((len(values['plane_i'])+255)//256)*256)
            control_cap=max(32,((len(values['plane_owner'])+31)//32)*32)
            padding={k:face_cap for k in ['plane_i','plane_j','plane_A','plane_n','plane_c','plane_length']}
            padding.update({k:control_cap for k in ['plane_owner','plane_d','plane_a','plane_b']})
            key=(name,cap,tuple((k,((padding[k],)+tuple(v.shape[1:])) if k in padding else
                tuple(v.shape[1:] if k in face_keys else v.shape),v.dtype) for k,v in values.items()))
        if key not in self.cache:
            # Bound per-operation graph memory as face counts evolve.
            previous=[k for k in self.cache if k[0]==name]
            if len(previous)>=3:del self.cache[previous[0]]
            self.cache[key]=Replay(fn,values,face_keys,cap,padding)
            self.builds.append({'operation':name,'face_capacity':cap,'seconds':self.cache[key].build_seconds})
        self.cache.move_to_end(key)
        return self.cache[key](values)

    @staticmethod
    def group(x,g):return torch.zeros_like(x).index_add(0,g,x)

    def primitive(self,q,V,g):
        total=self.group(q,g)[g];volume=self.group(V,g)[g]
        mass=total[:,0];v=total[:,1:4]/mass[:,None];rho=mass/volume
        pressure=(self.s.gamma-1)*rho*(total[:,4]/mass-.5*(v*v).sum(1))-self.s.gamma*self.s.pinf
        csq=(self.s.gamma*(pressure+self.s.pinf)+4*self.s.mu/3)/rho
        valid=torch.isfinite(q).all()&(mass>0).all()&(csq>0).all()&(pressure+self.s.pinf>0).all()
        stress=dev(total[:,5:14].reshape(-1,3,3)/mass[:,None,None])
        sigma=stress-pressure[:,None,None]*torch.eye(3,dtype=q.dtype,device=q.device)
        return total,volume,v,rho,pressure,stress,sigma,torch.sqrt(csq),valid

    def material_rate(self,grad,stress,total,mass):
        sym=.5*(grad+grad.transpose(1,2));spin=.5*(grad-grad.transpose(1,2))
        return 2*self.s.mu*dev(sym)+spin@stress-stress@spin

    def free_cap_motion(self,v,sigma,Z,N,Q,cap,surface):
        area=cap.clamp(min=1e-300);mean_n=N/area[:,None]
        mean_f=-(sigma@mean_n[:,:,None])[:,:,0]/Z[:,None]
        mean_b=-(sigma*Q).sum((1,2))/(Z*area);variance=1-(mean_n*mean_n).sum(1)
        radial=surface&(variance>1e-3)
        dr=torch.where(radial,(mean_b-(mean_f*mean_n).sum(1))/variance.clamp(min=1e-300),0.)
        w=torch.where(surface[:,None],v+mean_f-dr[:,None]*mean_n,v)
        return w,dr,radial

    def acoustic_interface(self,v,Z,ti,tj,i,j):
        den=(Z[i]+Z[j])[:,None]
        us=(Z[i,None]*v[i]+Z[j,None]*v[j]+tj-ti)/den
        ts=(Z[j,None]*ti+Z[i,None]*tj+(Z[i]*Z[j])[:,None]*(v[j]-v[i]))/den
        return us,ts

    def interface_energy_flux(self,us,ts,v,ti,tj,i,j):
        return (ts*us).sum(1)

    def advective_donor(self,b,total,groupV,relative):
        density=total/groupV[:,None]
        return torch.where(relative[:,None]>=0,density[b['i']],density[b['j']])

    def extra_rhs_values(self,mesh):
        return {}

    def transport_dt(self,b,donor,relative,total):
        return torch.full((),float('inf'),dtype=total.dtype,device=total.device)

    def mesh_motion(self,b,w,dr,us):
        return w,dr

    def mechanical_force(self,b,ts):
        f=b['A'][:,None]*ts
        return torch.zeros_like(b['p']).index_add(0,b['i'],f).index_add(0,b['j'],-f)

    def finalize_energy(self,b,dq,total,groupV,pressure,relative,donor,mw,mc):
        return dq

    def extra_rhs_diagnostics(self):
        return {}

    def mechanical_kinematics(self,b,v,sigma,Z):
        N,Q,cap=b['N'],b['Q'],b['cap'];i,j,n=b['i'],b['j'],b['n']
        w,dr,radial=self.free_cap_motion(v,sigma,Z,N,Q,cap,b['surface'])
        cap_grad=(w-v)[:,:,None]*N[:,None,:]+dr[:,None,None]*Q
        cap_rate=((w-v)*N).sum(1)+dr*cap
        ti=(sigma[i]@n[:,:,None])[:,:,0];tj=(sigma[j]@n[:,:,None])[:,:,0]
        us,ts=self.acoustic_interface(v,Z,ti,tj,i,j)
        return w,dr,radial,cap_grad,cap_rate,us,ts,ti,tj

    def wall_state(self,b,v,sigma,Z):
        """Original impermeable acoustic wall, unchanged for old materials."""
        floor=b['floor'];active=floor>0;free=v+sigma[:,:,1]/Z[:,None];rawP=-Z*free[:,1]
        P=rawP.clamp(min=0)*active;tangent=torch.stack((free[:,0],free[:,2]),1);speed=tangent.norm(dim=1)
        amount=torch.minimum(self.s.friction*P,Z*speed)
        traction=torch.stack((-amount*tangent[:,0]/speed.clamp(min=1e-300),P,-amount*tangent[:,1]/speed.clamp(min=1e-300)),1)
        uv=free+traction/Z[:,None];uv=torch.stack((uv[:,0],torch.zeros_like(P),uv[:,2]),1)
        uv=torch.where(active[:,None],uv,v);force=floor[:,None]*traction
        heat=-floor*(traction[:,0]*uv[:,0]+traction[:,2]*uv[:,2])
        tensile=active&(rawP < -1e-8*(Z*v.norm(dim=1)+sigma.norm(dim=(1,2))+1.))
        sliding=active&(Z*speed>self.s.friction*P)
        return uv,force,heat,rawP,P,tensile,sliding

    def rhs_gpu(self,b):
        q,V,g=b['q'],b['V'],b['groups'];p,r=b['p'],b['r'];i,j,A,n,c=b['i'],b['j'],b['A'],b['n'],b['c']
        total,groupV,v,rho,pressure,stress,sigma,sound,valid=self.primitive(q,V,g)
        mass=q[:,0];Z=rho*sound;Q,N,cap=b['Q'],b['N'],b['cap'];surface=b['surface'];floor=b['floor']
        w,dr,radial,cap_grad,cap_rate,us,ts,ti,tj=self.mechanical_kinematics(b,v,sigma,Z)
        boundary=self._compatible['planar'] if 'plane_i' in b else None
        if boundary is not None:floor=boundary['floor']
        active=floor>0
        uv,force,heat,rawP,P,tensile,sliding=self.wall_state(b,v,sigma,Z)
        w,dr=self.mesh_motion(b,w,dr,us)
        grad=torch.zeros_like(stress).index_add(0,i,A[:,None,None]*(us-v[i])[:,:,None]*n[:,None,:])
        grad.index_add_(0,j,-A[:,None,None]*(us-v[j])[:,:,None]*n[:,None,:])
        grad+=cap_grad
        if boundary is not None:grad+=boundary['gradient']
        grad[:,:,1]-=floor[:,None]*(uv-v)
        grad=self.group(grad,g)[g]/groupV[:,None,None]
        length=(p[j]-p[i]).norm(dim=1);length=torch.where(A>0,length,torch.ones_like(length))
        mw=A/length*(((c-p[i])*w[i]).sum(1)-((c-p[j])*w[j]).sum(1)+r[i]*dr[i]-r[j]*dr[j])
        mc=(N*w).sum(1)+cap*dr;cap_defect=mc-((N*v).sum(1)+cap_rate)
        if boundary is not None:mc+=boundary['volume_rate']
        relative=A*(us*n).sum(1)-mw
        donor=self.advective_donor(b,total,groupV,relative)
        def incidence(x):return torch.zeros((len(q),)+x.shape[1:],dtype=x.dtype,device=x.device).index_add(0,i,x).index_add(0,j,-x)
        dq=-incidence(relative[:,None]*donor)
        dq[:,1:4]+=self.mechanical_force(b,ts)
        dq[:,4]+=incidence(A*self.interface_energy_flux(us,ts,v,ti,tj,i,j))
        stressdot=self.material_rate(grad,stress,total,mass)
        dq[:,5:14]+=(mass[:,None,None]*stressdot).reshape(-1,9)
        dq[:,1:4]+=mass[:,None]*self.s.g+force;source=(q[:,1:4]*self.s.g).sum(1);dq[:,4]+=source
        dq=self.finalize_energy(b,dq,total,groupV,pressure,relative,donor,mw,mc)
        external=g[i]!=g[j];rel=torch.where(external,relative,0.)
        outgoing=torch.zeros_like(mass).index_add(0,i,rel.clamp(min=0)).index_add(0,j,(-rel).clamp(min=0))
        extarea=cap.clamp(min=0)+floor+torch.zeros_like(mass).index_add(0,i,A*external).index_add(0,j,A*external)
        if boundary is not None:extarea+=boundary['area']
        wave=extarea*sound+outgoing;dt_wave=groupV/self.group(wave,g)[g].clamp(min=1e-300)
        dt_contact=torch.sqrt(self.group(mass,g)[g]/1e-300);dt_radius=r/dr.abs().clamp(min=1e-300)
        diag={
            'valid':valid,'wall_tensile_prediction_cells':tensile.sum(),
            'dt_scale':torch.minimum(torch.minimum(torch.minimum(dt_wave,dt_contact),dt_radius).min(),self.transport_dt(b,donor,relative,total)),
            'wave_cell':dt_wave.argmin(),'wave_dt_scale':dt_wave.min(),'minimum_geometric_volume':V.min(),
            'hidden_subcells':(V==0).sum(),'rho_min':rho.min(),'rho_max':rho.max(),
            'pressure_max_abs':pressure.abs().max(),'stress_vm_max':torch.sqrt(1.5*(stress*stress).sum((1,2))).max(),
            'cap_integrated_flux_defect':cap_defect.abs().sum(),'mass_rate_sum':dq[:,0].sum(),
            'force_x':dq[:,1].sum(),'force_y':dq[:,2].sum(),'force_z':dq[:,3].sum(),
            'source_power':source.sum(),'energy_rate_sum':dq[:,4].sum(),'friction_heat_rate':heat.sum(),
            'contact_area':floor.sum(),'contact_cells':active.sum(),
            'nearly_flat_cap_fits':(surface&~radial).sum(),'mesh_material_speed_difference_max':(w-v).norm(dim=1).max(),
            'wall_pressure_min':torch.where(active,rawP,torch.full_like(rawP,float('inf'))).min(),
            'wall_pressure_max':P.max(),'wall_tensile_prediction_area':(floor*tensile).sum(),
            'wall_sliding_cells':sliding.sum()}
        diag.update(self.extra_rhs_diagnostics())
        self.rhs_keys=tuple(diag)
        return w,dr,dq,torch.stack([x.to(q.dtype) for x in diag.values()])

    def rhs(self,state):
        m=state.mesh
        values={'q':state.q,'V':m.volume,'groups':self.s.groups,'p':state.p,'r':state.r,
            'i':m.i,'j':m.j,'A':m.area,'n':m.normal,'c':m.centroid,'Q':m.cap_Q,'N':m.cap_normal_integral,
            'cap':m.cap_area,'surface':m.surface,'floor':m.ground_area}
        values.update(self.extra_rhs_values(m))
        w,dr,dq,pack=self.call('rhs',self.rhs_gpu,values,('i','j','A','n','c'))
        d=dict(zip(self.rhs_keys,pack.tolist()))
        if not d.pop('valid'):raise RuntimeError('nonpositive mass/sound-speed or nonfinite state; no floor applied')
        if d['wall_tensile_prediction_cells']:raise RuntimeError('planar contact requires release: '+str(d))
        for key in ['wave_cell','hidden_subcells','contact_cells','nearly_flat_cap_fits','wall_tensile_prediction_cells','wall_sliding_cells']:d[key]=int(d[key])
        if not d['contact_cells']:d['wall_pressure_min']=0.
        d['force_sum']=[d.pop('force_x'),d.pop('force_y'),d.pop('force_z')]
        for key in ['normal_contact_power','contact_boundary_motion_power_defect','contact_boundary_motion_abs_power_defect',
            'contact_normal_speed_mismatch_rms','contact_normal_speed_mismatch_max','penetration_max','wall_normal_velocity_max','boundary_local_relative_residual']:d[key]=0.
        d.update(ignored_cap_area=m.ignored_cap_area,boundary_local_iterations=0,ground_geometry=m.ground_geometry_info)
        # Graph outputs are persistent. Return owned arrays: stage 2 must not
        # overwrite stage 1 or the saved RHS used for step retries.
        return w.clone(),dr.clone(),dq.clone(),d

    def plastic_gpu(self,b):
        q,V,g=b['q'],b['V'],b['groups'];total=self.group(q,g);vol=self.group(V,g)
        valid=(vol[g]>0).all()&(total[g,0]>0).all()
        q=total[g]*(V/vol[g])[:,None];mass=q[:,0]
        effective=self.group(q,g)[g]
        stress=dev(effective[:,5:14].reshape(-1,3,3)/effective[:,0,None,None])
        eq=torch.sqrt(1.5*(stress*stress).sum((1,2)));ratio=(self.s.ys/eq.clamp(min=1e-300)).clamp(max=1.)
        returned=stress*ratio[:,None,None];loss=V*((stress*stress-returned*returned).sum((1,2)))/(4*self.s.mu)
        q[:,5:14]=(mass[:,None,None]*returned).reshape(-1,9)
        q[:,14]+=mass*(eq-self.s.ys).clamp(min=0)/(3*self.s.mu);q[:,15]+=loss
        return q,valid

    def plastic(self,q,mesh):
        out,valid=self.call('plastic',self.plastic_gpu,{'q':q,'V':mesh.volume,'groups':self.s.groups})
        if not bool(valid):raise RuntimeError('a whole FV group lost its volume or mass')
        return out.clone()

    def validate(self,state):
        def check(b):
            good=self.primitive(b['q'],b['V'],b['groups'])[-1]
            return torch.stack((good.to(b['q'].dtype),b['q'][:,0].sum(),b['q'][:,4].sum()))
        values=self.call('validate',check,{'q':state.q,'V':state.mesh.volume,'groups':self.s.groups}).tolist()
        if not values[0]:raise RuntimeError('nonpositive mass/sound-speed or nonfinite state; no floor applied')
        return values[1:]

    def update(self,state,dt,first,second=None):
        def step(b):
            return b['p']+b['dt']*b['vp'],b['r']+b['dt']*b['vr'],b['q']+b['dt']*b['dq']
        def average(b):
            return b['p']+.5*b['dt']*(b['vp']+b['vp2']),b['r']+.5*b['dt']*(b['vr']+b['vr2']),b['q']+.5*b['dt']*(b['dq']+b['dq2'])
        values={'p':state.p,'r':state.r,'q':state.q,'dt':torch.tensor(dt,dtype=state.p.dtype,device=state.p.device),
            'vp':first[0],'vr':first[1],'dq':first[2]}
        if second is not None:values.update(vp2=second[0],vr2=second[1],dq2=second[2])
        result=self.call('update2' if second is not None else 'update1',average if second is not None else step,values)
        return tuple(x.clone() for x in result)

    def moments_gpu(self,b):
        p,r,V=b['p'],b['r'],b['V'];i,j,A,n,c=b['i'],b['j'],b['A'],b['n'],b['c']
        outer=lambda a,z:a[:,:,None]*z[:,None,:]
        moment=torch.zeros((len(p),3,3),dtype=p.dtype,device=p.device)
        moment.index_add_(0,i,A[:,None,None]*outer(c-p[i],n));moment.index_add_(0,j,-A[:,None,None]*outer(c-p[j],n))
        moment+=b['floor'][:,None,None]*outer(b['floor_c']-p,b['floor_n'])
        if 'plane_i' in b:
            k=b['plane_i'];moment.index_add_(0,k,b['plane_A'][:,None,None]*outer(b['plane_c']-p[k],b['plane_n']))
        Q=(V[:,None,None]*torch.eye(3,dtype=p.dtype,device=p.device)-moment)/r[:,None,None];Q=.5*(Q+Q.transpose(1,2))
        scale=torch.maximum(b['cap'].max(),A.max());surface=b['cap']>1e-12*scale
        Q=torch.where(surface[:,None,None],Q,torch.zeros_like(Q))
        ignored=torch.where(surface,torch.zeros_like(b['cap']),b['cap'].abs()).sum()
        area=torch.zeros_like(r).index_add(0,i,A).index_add(0,j,A)+b['cap'].clamp(min=0)+b['floor']
        if 'plane_i' in b:area.index_add_(0,b['plane_i'],b['plane_A'])
        return Q,surface,area,ignored

    def moments(self,mesh):
        m=mesh
        values={'p':m.p,'r':m.r,'V':m.volume,'i':m.i,'j':m.j,'A':m.area,'n':m.normal,'c':m.centroid,
            'floor':m.ground_area,'floor_c':m.ground_centroid,'floor_n':m.ground_normal,'cap':m.cap_area}
        if hasattr(m,'planes'):
            from free_surface.fv_ale_planar_boundary import values as plane_values
            values.update(plane_values(m))
        Q,surface,area,ignored=self.call('moments',self.moments_gpu,values,('i','j','A','n','c'))
        m.cap_Q=Q.clone();m.surface=surface.clone();m.surface_area=area.clone();m.ignored_cap_area=float(ignored)
