"""Opt-in first-order implicit FV trial with lagged face geometry.

Uses the production plane-contact flux and plastic return. Group volumes are
unknowns of the same backward-Euler solve, using the OLD mesh's exact volume
derivative. One final actual cut measures the finite-motion geometry defect.
This is not a fully implicit moving-mesh/GCL-exact scheme: a large defect is a
failed trial, not permission to reset mass, energy, or the computed volume.
No state is committed on nonlinear or geometry failure.
"""
import time
import torch
from free_surface.fv_ale_device import Replay
from free_surface.fv_ale_solid import State


def gmres(product,b,rtol=.05,restart=40,cycles=3):
    """GPU vectors, twice orthogonalized Arnoldi; true residual every restart."""
    x=torch.zeros_like(b);bn=b.norm();calls=0
    for cycle in range(cycles):
        r=b if cycle==0 else b-product(x)
        if cycle:calls+=1
        beta=r.norm()
        if float(beta)<=rtol*float(bn):return x,calls,float(beta/bn)
        Q=torch.empty((b.numel(),restart+1),dtype=b.dtype,device=b.device)
        H=torch.zeros((restart+1,restart),dtype=b.dtype,device=b.device)
        Q[:,0]=r/beta
        updated=False
        for k in range(restart):
            w=product(Q[:,k]);calls+=1
            basis=Q[:,:k+1]
            h=basis.T@w;w=w-basis@h
            h2=basis.T@w;w=w-basis@h2;h+=h2
            norm=w.norm();H[:k+1,k]=h;H[k+1,k]=norm
            breakdown=float(norm)<1e-13
            if not breakdown:Q[:,k+1]=w/norm
            if (k+1)%8==0 or breakdown or k+1==restart:
                hk=H[:k+2,:k+1];rhs=torch.zeros(k+2,dtype=b.dtype,device=b.device);rhs[0]=beta
                y=torch.linalg.lstsq(hk,rhs).solution
                if float((rhs-hk@y).norm()/bn)<=rtol or breakdown:
                    x+=basis@y
                    true_relative=float((b-product(x)).norm()/bn);calls+=1
                    if true_relative<=rtol:return x,calls,true_relative
                    updated=True
                    break
        if not updated:x+=Q[:,:restart]@y
    return x,calls+1,float((b-product(x)).norm()/bn)


class ImplicitStep:
    def __init__(self,solver,state,dt):
        if solver.contact_model!='plane' or solver.interior_radii!='fixed':
            raise ValueError('implicit trial supports the current plane/fixed-interior mode only')
        self.s=solver;self.old=state;self.dt=float(dt);self.ops=solver.device_ops()
        m=state.mesh;self.labels,g=torch.unique(solver.groups,sorted=True,return_inverse=True)
        self.g=g;ng=len(self.labels)
        def group(a):return torch.zeros((ng,)+a.shape[1:],dtype=a.dtype,device=a.device).index_add(0,g,a)
        self.group=group;self.Q0=group(state.q);self.V0=group(m.volume)
        self.fraction=m.volume/self.V0[g]
        scales=self.Q0[:,:1]*torch.ones_like(self.Q0)
        sound=(solver.bulk/solver.rho0+4*solver.mu/(3*solver.rho0))**.5
        scales[:,1:4]*=sound;scales[:,4]*=solver.e0;scales[:,5:14]*=solver.bulk
        scales[:,15]=self.V0*solver.bulk
        self.scale=torch.cat((scales,self.V0[:,None]),1)
        self.zero=torch.zeros_like(self.scale)
        self.base={'groups':g,'p':state.p,'r':state.r,
            'i':m.i,'j':m.j,'A':m.area,'n':m.normal,'c':m.centroid,'Q':m.cap_Q,'N':m.cap_normal_integral,
            'cap':m.cap_area,'surface':m.surface,'floor':m.ground_area}
        self.replay=Replay(self.evaluate,{'y':self.zero})
        self.evaluations=0

    def evaluate(self,inputs):
        y=inputs['y'];old=self.old;g=self.g;f=self.fraction
        Q=self.Q0+y[:,:16]*self.scale[:,:16];V=self.V0+y[:,16]*self.V0
        q=Q[g]*f[:,None];vol=V[g]*f
        b=dict(self.base,q=q,V=vol)
        w,dr,dq,diagnostic=self.ops.rhs_gpu(b)
        # This identity uses exactly the same old-face mesh flux as transport.
        Vdot=self.group(old.mesh.volume_rate(w,dr))
        trial=self.Q0+self.dt*self.group(dq)
        returned,plastic_valid=self.ops.plastic_gpu({'q':trial[g]*f[:,None],'V':vol,'groups':g})
        target=self.group(returned)
        res=torch.cat(((Q-target)/self.scale[:,:16],((V-self.V0-self.dt*Vdot)/self.V0)[:,None]),1)
        # The explicit-looking target is a residual evaluation, not a proposed
        # state. At large dt its mass can be negative before Newton converges;
        # only the actual candidate Q,V must be physical during globalization.
        valid=(V>0).all()&diagnostic[0].bool()&torch.isfinite(res).all()
        return res,w,dr,Q,V,diagnostic,valid

    def call(self,y):
        self.evaluations+=1
        return tuple(a.clone() for a in self.replay({'y':y}))

    def solve(self,max_newton=20,volume_tolerance=1e-3):
        tick=time.perf_counter();y=self.zero.clone();history=[];linear_calls=0
        result=self.call(y)
        for it in range(max_newton+1):
            residual=result[0];rnorm=float(residual.norm());rmax=float(residual.abs().max())
            if not bool(result[-1]):raise RuntimeError('implicit nonphysical iterate')
            if rmax<1e-8:break
            if it==max_newton:raise RuntimeError(f'implicit Newton budget; scaled max residual {rmax:g}; history {history}')
            def product(v):
                direction=v.reshape_as(y)
                h=1e-6/float(direction.abs().max().clamp(min=1e-300))
                return ((self.call(y+h*direction)[0]-self.call(y-h*direction)[0])/(2*h)).flatten()
            step,nprod,linres=gmres(product,-residual.flatten())
            linear_calls+=nprod;step=step.reshape_as(y)
            accepted=False
            for ls in range(14):
                alpha=2.**(-ls);candidate=self.call(y+alpha*step)
                if bool(candidate[-1]) and float(candidate[0].norm())<(1-1e-4*alpha)*rnorm:
                    y=y+alpha*step;result=candidate;accepted=True;break
            history.append({'newton':it,'max_residual':rmax,'gmres_products':nprod,'linear_residual':linres,'alpha':alpha})
            if not accepted:raise RuntimeError(f'implicit line search failed; history {history}')
        res,w,dr,Q,V,diag,valid=result
        diagnostics=dict(zip(self.ops.rhs_keys,diag.tolist()))
        if diagnostics['wall_tensile_prediction_cells']:
            raise RuntimeError('implicit planar contact requires release')
        p=self.old.p+self.dt*w;r=self.old.r+self.dt*dr
        mesh=self.s.cut(p,r)
        actualV=self.group(mesh.volume)
        defect=(actualV-V)/self.V0
        info={'dt':self.dt,'newton_iterations':len(history),'linear_products':linear_calls,
            'residual_evaluations':self.evaluations,'scaled_max_residual':float(res.abs().max()),
            'group_geometry_defect_max':float(defect.abs().max()),
            'group_geometry_defect_rms':float(defect.square().mean().sqrt()),
            'source_work':self.dt*diagnostics['source_power'],'history':history,
            'graph_build_seconds':self.replay.build_seconds,'solve_and_cut_seconds':time.perf_counter()-tick}
        # q stays conservative; actual cut volume is authoritative in the next step.
        q=Q[self.g]*(mesh.volume/actualV[self.g])[:,None]
        end=State(p,r,q,mesh);self.s.primitive(end)
        info['mass_relative_error']=float((q[:,0].sum()-self.old.q[:,0].sum())/self.old.q[:,0].sum())
        info['energy_balance_joule']=float((q[:,4]-self.old.q[:,4]).sum())-info['source_work']
        info['geometry_accepted']=info['group_geometry_defect_max']<=volume_tolerance
        return end,info
