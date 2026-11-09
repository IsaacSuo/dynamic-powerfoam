"""Two-way linear elastic wave on a free bar, with conservative full-step fluxes."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

C = 1.
L = 1.
RHO = 1.
DT = 1/30
STEPS = 60
AMP = .02
LOADS = {
    "fit": [(1,.8,.2),(2,.35,-.2)],
    "pressure_release": [(1,.1,-.9),(2,-.3,.1)],
    "asymmetric": [(1,-.25,.55),(2,.55,.2),(3,.25,-.35)],
}


def primitive(x, load):
    x = np.asarray(x, dtype=float)
    total = np.zeros_like(x)
    for n, cosine, sine in load:
        k = np.pi*n/L
        total += AMP*(cosine*np.sin(k*x)-sine*np.cos(k*x))/k
    return total


def overlaps(edges, left, right):
    a, b = np.asarray(left)[:,None], np.asarray(right)[:,None]
    w, m = np.zeros((len(a),len(edges)-1)), np.zeros((len(a),len(edges)-1))
    for shift in (-2*L,0,2*L):
        l, r = edges[:-1]+shift,edges[1:]+shift
        low, high = np.maximum(a,l),np.minimum(b,r)
        length = np.maximum(high-low,0)
        w += length
        m += length*((low+high)-(l+r))*.5
    return w,m


class Bar:
    def __init__(self, thin):
        e = np.sort(np.r_[np.linspace(0,L,25), .375+thin])
        self.edges = np.r_[e,2*L-e[-2::-1]]
        self.n = len(self.edges)-1
        self.mass = np.diff(self.edges)
        self.center = (self.edges[:-1]+self.edges[1:])*.5
        self.xleft = self.center-np.r_[self.center[-1]-2*L,self.center[:-1]]
        self.xright = np.r_[self.center[1:],self.center[0]+2*L]-self.center
        d = C*DT
        I, J = overlaps(self.edges,self.edges[:-1]-d,self.edges[1:]-d)
        self.avg, self.grad = I/self.mass[:,None],J/self.mass[:,None]
        self.face, self.face_grad = overlaps(self.edges,self.edges-d,self.edges)
        self.ends = [0,len(e)-1]
        self.thin = np.flatnonzero(self.mass < .01)
        assert len(self.thin) == 2
        self.patch = np.unique(np.r_[self.thin,(self.thin-1)%self.n,(self.thin+1)%self.n])
        np.testing.assert_allclose(self.avg.sum(1), 1., atol=1e-7, rtol=0)
        np.testing.assert_allclose(self.mass@self.avg,self.mass,atol=1e-15,rtol=0)
        np.testing.assert_allclose(self.mass@self.grad,0,atol=1e-15,rtol=0)

    def energy(self, w):
        return RHO/4*(self.mass@(w*w))

    def normsq(self, w):
        return self.mass@(w*w)

    def cap(self, low, corr, old):
        if self.normsq(low+corr) <= old:
            return 1.
        a = self.normsq(corr)
        b = 2*self.mass@(low*corr)
        c = min(0.,self.normsq(low)-old)
        disc = np.sqrt(max(0.,b*b-4*a*c))
        if b>=0:
            return -2*c/(b+disc) if b+disc else 0.
        return (-b+disc)/(2*a) if a else 1.

    def slopes(self, w):
        a = (w-np.roll(w,1))/self.xleft
        b = (np.roll(w,-1)-w)/self.xright
        return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)

    def run(self, load, alpha, *, placement="patch", method="limited"):
        w = np.diff(primitive(self.edges,load))/self.mass
        E0 = self.energy(w)
        m0 = self.mass@w
        min_w,max_w = min(w),max(w)
        jumps = np.zeros(2)
        path = [jumps.copy()]
        max_conservation, max_flux_resid, max_growth = 0.,0.,0.
        cap_hits, cap_min, first_energy_jump = 0,1.,None
        mask = self.patch if placement=="patch" else self.thin
        n = 0
        status = "ok"
        for n in range(1,STEPS+1):
            oldE = self.energy(w)
            if method=="upwind":
                I = C*DT*np.roll(w,1)
                I = np.r_[I,I[0]]
                q = w+(I[:-1]-I[1:])/self.mass
            else:
                lo = self.avg@w
                slope = np.zeros_like(w) if method=="low" else self.slopes(w)
                other = slope.copy()
                other[mask]=0
                cap = min(1.,self.cap(lo,self.grad@other,self.normsq(w)),
                             self.cap(lo,self.grad@slope,self.normsq(w)))
                cap_min = min(cap_min,cap)
                cap_hits += cap < 1-1e-12
                slope[mask] *= alpha
                slope *= cap
                q = lo+self.grad@slope
                I = self.face@w+self.face_grad@slope
            max_flux_resid = max(max_flux_resid,float(np.max(abs(self.mass*(q-w)-(I[:-1]-I[1:])))))
            w = q
            E = self.energy(w)
            max_conservation = max(max_conservation,abs(self.mass@w-m0))
            max_growth = max(max_growth,(E-oldE)/E0)
            if E > oldE+1e-11*E0 and first_energy_jump is None:
                first_energy_jump = n
            jumps += I[self.ends]/C
            path.append(jumps.copy())
            if method=="upwind" and E>100*E0:
                status = "growth"
                break
            if method!="upwind":
                assert E <= oldE+1e-11*E0
                assert min(w) >= min_w-1e-10 and max(w) <= max_w+1e-10
        times = DT*np.arange(n+1)
        endpoint = np.array([0.,L])
        oracle = (primitive(endpoint,load)-primitive(endpoint-times[:,None]*C,load))/C
        path = np.array(path)
        return {"steps":n,"status":status,"alpha":float(alpha),"placement":placement,"method":method,
                "rms_percent":float(100*np.sqrt(np.mean((path-oracle)**2))/np.max(abs(oracle))),
                "max_percent":float(100*np.max(abs(path-oracle))/np.max(abs(oracle))),
                "energy_loss_percent":float(100*(1-self.energy(w)/E0)),
                "max_conservation_error":float(max_conservation),"max_energy_growth":float(max_growth),
                "max_flux_balance_error":max_flux_resid,"energy_limiter_steps":int(cap_hits),
                "min_energy_scale":float(cap_min),"first_energy_jump":first_energy_jump,
                "endpoints":path.tolist(),"reference":oracle.tolist()}


def summary(r):
    return {k:v for k,v in r.items() if k not in ("endpoints","reference")}


def main():
    thin = 1e-6
    b = Bar(thin)
    profile = LOADS["fit"]
    obj = lambda p:b.run(profile,p)["rms_percent"]
    candidates = [(obj(x),x) for x in np.linspace(0,1,21)]
    fit = minimize_scalar(obj,method="bounded",bounds=(0,1),options={"xatol":1e-6})
    opt = min(candidates+[(fit.fun,fit.x)])[1]
    data = {"dt":DT,"base_dx":L/24,"fit_alpha":float(opt),"cases":{},"stability_sweep":[]}
    for name, load in LOADS.items():
        choices = {"low":b.run(load,0,method="low"),
                   "patch_zero":b.run(load,0),
                   "patch_full":b.run(load,1),
                   "patch_fitted":b.run(load,opt),
                   "thin_zero":b.run(load,0,placement="thin")}
        data["cases"][name] = choices
        print(name,json.dumps({k:summary(v) for k,v in choices.items()}),flush=True)
    for eps in [5e-3,1e-4,1e-6,1e-8]:
        bar = Bar(eps)
        run = {"width":eps,"thin_cfl":C*DT/eps,"bypass":1-eps/(C*DT),
               "raw_upwind":summary(bar.run(profile,0,method="upwind")),
               "low":summary(bar.run(profile,0,method="low")),
               "limited":summary(bar.run(profile,opt))}
        data["stability_sweep"].append(run)
        print("sweep",json.dumps(run),flush=True)
    Path("cutcell_wave_flux.json").write_text(json.dumps(data,indent=2)+"\n")


if __name__=="__main__":
    main()
