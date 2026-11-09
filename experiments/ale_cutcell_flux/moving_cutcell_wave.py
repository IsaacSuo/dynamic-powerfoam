"""Conservative ALE characteristic transport on a moving-cut linear elastic bar."""
import json
from pathlib import Path

import numpy as np

from cutcell_wave_flux import AMP as WAVE_AMPLITUDE
from cutcell_wave_flux import C, DT, L, LOADS, RHO, STEPS, primitive

LD = np.longdouble
X0 = LD(".375")
AMP = LD(".006")


def grid(thin, time, motion):
    base = np.linspace(LD(0), LD(1), 25, dtype=LD)
    i = 9
    delta = LD(0)
    width = LD(str(thin))
    if motion != "fixed":
        a, f = (AMP, LD("1.3")) if motion == "slow" else (LD(".019"), LD("17"))
        delta = a*np.sin(2*np.pi*f*time)
        width *= 1+LD(".7")*np.sin(2*np.pi*LD("2.7")*time)
    base[i] = X0+delta
    e = np.insert(base,i+1,base[i]+width)
    assert np.all(np.diff(e)>0)
    return np.r_[e,2-e[-2::-1]], len(e)-1


def integrals(edges, left, right):
    # Integrate 1, x-center, (x-center)^2 over periodic overlaps.
    a,b = np.asarray(left,dtype=LD)[:,None],np.asarray(right,dtype=LD)[:,None]
    signs = np.where(a <= b, 1., -1.)
    lo,hi = np.minimum(a,b),np.maximum(a,b)
    moments = np.zeros((3,len(a),len(edges)-1),dtype=LD)
    for shift in (-2,0,2):
        l,r = edges[:-1]+shift,edges[1:]+shift
        aa,bb = np.maximum(lo,l),np.minimum(hi,r)
        h = np.maximum(bb-aa,0)
        d = .5*((aa+bb)-(l+r))
        moments[0] += h
        moments[1] += h*d
        moments[2] += h*(d*d+h*h/12)
    return np.asarray(moments*signs,dtype=float)


def exact_average(edges, widths, load, offset=0.):
    if load is None:
        return np.full(len(widths), .011)
    mid = np.asarray((edges[:-1]+edges[1:])*.5,dtype=float)-offset
    average = np.zeros_like(widths)
    for n,co,si in load:
        k = np.pi*n/L
        average += WAVE_AMPLITUDE*(co*np.cos(k*mid)+si*np.sin(k*mid))*np.sinc(k*widths/(2*np.pi))
    return average


def slopes(edges,w):
    mid = np.asarray((edges[:-1]+edges[1:])*.5,dtype=float)
    L = (w-np.roll(w,1))/(mid-np.r_[mid[-1]-2,mid[:-1]])
    R = (np.roll(w,-1)-w)/(np.r_[mid[1:],mid[0]+2]-mid)
    return np.where(L*R>0,np.sign(L)*np.minimum(abs(L),abs(R)),0.)


def safe_scale(lo, corr, old_energy_norm, widths):
    if widths@((lo+corr)**2) <= old_energy_norm:
        return 1.
    a,b,c = widths@(corr*corr), 2*widths@(lo*corr), min(0.,widths@(lo*lo)-old_energy_norm)
    root = np.sqrt(max(0,b*b-4*a*c))
    return min(1., -2*c/(b+root) if b>=0 and b+root>0 else (-b+root)/(2*a) if a else 0.)


def physical_state(W,h,n):
    plus,minus = W[:n],W[n:][::-1]
    # Linearized mass and momentum, with the unperturbed reference density.
    return np.array([RHO*h[:n]*(1+(plus-minus)/(2*C)),
                     RHO*h[:n]*(plus+minus)/2])


def run(thin,motion,load,order):
    edges,n = grid(thin,LD(0),motion)
    h = np.asarray(np.diff(edges),dtype=float)
    w = exact_average(edges,h,load)
    old_min,old_max = min(w),max(w)
    E0 = RHO/4*h@(w*w)
    dx = LD(str(DT))*C
    path=[np.zeros(2)]
    m0,p0 = physical_state(w,h,n).sum(1)
    checks = dict.fromkeys(("volume","constant","char_flux","mass","momentum","energy",
                             "energy_growth","reconstruction_energy","projection_loss",
                             "total_mass","total_momentum","sweep_omitted_error",
                             "outer_energy_flux"),0.)
    limits, negative = 0,0
    wave_error_sq, wave_reference_sq = 0.,0.
    smallest = float(min(h))
    for k in range(STEPS):
        new,_ = grid(thin,(k+1)*LD(str(DT)),motion)
        newh = np.asarray(np.diff(new),dtype=float)
        smallest=min(smallest,float(min(newh)))
        sweep = np.asarray(new-edges,dtype=float)
        A,B,D=integrals(edges,new[:-1]-dx,new[1:]-dx)
        P,Q,S=integrals(edges,new-dx,edges)
        negative += int(np.sum(P.sum(1)<0))
        checks["volume"]=max(checks["volume"],float(np.max(abs(newh-h-np.diff(sweep)))))
        if load is None:
            checks["sweep_omitted_error"]=max(checks["sweep_omitted_error"],float(np.max(abs(h/newh-1))))
        low = (A@w)/newh
        s = slopes(edges,w) if order==1 else np.zeros_like(w)
        scale = safe_scale(low,(B@s)/newh,h@(w*w),newh)
        limits += int(scale<1-1e-12)
        s *= scale
        result = low+(B@s)/newh
        I = P@w+Q@s
        I2 = RHO/4*(P@(w*w)+2*Q@(w*s)+S@(s*s))
        recon = RHO/4*s*s*h**3/12
        adv = RHO/4*(A@(w*w)+2*B@(w*s)+D@(s*s))
        drop = adv-RHO/4*newh*result*result
        assert np.min(drop)>-2e-18

        before=physical_state(w,h,n)
        after=physical_state(result,newh,n)
        Im, Ip = I[:n+1], -I[::-1][:n+1]
        transfer = np.array([RHO/(2*C)*(Im-Ip)-RHO*sweep[:n+1], RHO/2*(Im+Ip)])
        checks["char_flux"]=max(checks["char_flux"],float(np.max(abs(newh*result-h*w+np.diff(I)))))
        for j,key in enumerate(("mass","momentum")):
            err = after[j]-before[j]+np.diff(transfer[j])
            checks[key]=max(checks[key],float(np.max(abs(err))))
        # Physical acoustic work minus swept acoustic energy, plus the
        # explicitly accounted reconstruction and averaging terms.
        energy_err = RHO/4*(newh*result*result-h*w*w)+np.diff(I2)-recon+drop
        Eold,Enew=RHO/4*(h@(w*w)),RHO/4*(newh@(result*result))
        checks["energy"]=max(checks["energy"],float(np.max(abs(energy_err))))
        checks["energy_growth"]=max(checks["energy_growth"],float((Enew-Eold)/E0))
        checks["reconstruction_energy"]+=float(recon.sum())
        checks["projection_loss"]+=float(drop.sum())
        path.append(path[-1]+I[[0,n]]/C)
        # The linearized equations live on the reference interval. At its
        # ends p=0 but v need not vanish. Account for the material end motion
        # when computing the mass between the two free ends.
        total_mass = after[0].sum()+RHO*(path[-1][1]-path[-1][0])
        checks["total_mass"]=max(checks["total_mass"],abs(total_mass-m0))
        checks["outer_energy_flux"]=max(checks["outer_energy_flux"],float(abs(I2[-1]-I2[0])))
        checks["total_momentum"]=max(checks["total_momentum"],abs(after[1].sum()-p0))
        if load is None:
            checks["constant"]=max(checks["constant"],float(np.max(abs(result-.011))))
        else:
            exact = exact_average(new,newh,load,float((k+1)*dx))
            wave_error_sq += newh@((result-exact)**2)
            wave_reference_sq += newh@(exact*exact)
        assert min(result)>old_min-1e-10 and max(result)<old_max+1e-10
        w,h,edges=result,newh,new
    assert max(checks[k] for k in ("volume","char_flux","mass","momentum","energy",
                                  "total_mass","total_momentum","outer_energy_flux")) < 2e-13, checks
    assert checks["energy_growth"] < 1e-10
    assert checks["constant"] < 1e-10
    assert abs((Enew-E0)-(checks["reconstruction_energy"]-checks["projection_loss"]))<5e-17
    row = {"width":thin,"smallest_width":smallest,"motion":motion,"order":order,
           "steps":STEPS,"negative_net_flux_intervals":negative,"limiter_steps":limits,
           "energy_loss_percent":float(100*(1-Enew/E0)),**checks}
    if load is not None:
        boundary=np.array([0.,1.])
        oracle=(primitive(boundary,load)-primitive(boundary-DT*np.arange(STEPS+1)[:,None]*C,load))/C
        err=np.array(path)-oracle
        row.update(rms_wave_percent=float(100*np.sqrt(wave_error_sq/wave_reference_sq)),
                   rms_endpoint_percent=float(100*np.sqrt(np.mean(err**2))/np.max(abs(oracle))),
                   endpoints=np.array(path).tolist(),reference=oracle.tolist())
    return row


def check_analytic_face():
    # Independent time quadrature on the analytic smooth wave, using
    # a sinusoidal face whose velocity reverses the relative flow.
    nodes,weights=np.polynomial.legendre.leggauss(48)
    a, omega = .019,2*np.pi*17
    maxerr=0.
    reverse=0
    load=LOADS["fit"]
    for k in range(STEPS):
        t0=k*DT
        for c in [-1.,1.]:
            time=t0+.5*DT*(1+nodes)
            xp=.375+a*np.sin(omega*time)
            xp0=.375+a*np.sin(omega*t0)
            xp1=.375+a*np.sin(omega*(t0+DT))
            x0=xp-c*(time-t0)
            q=sum(WAVE_AMPLITUDE*(co*np.cos(np.pi*n*x0)+si*np.sin(np.pi*n*x0)) for n,co,si in load)
            vrel=c-a*omega*np.cos(omega*time)
            direct=.5*DT*weights@(vrel*q)
            endpoint=primitive(xp0,load)-primitive(xp1-c*DT,load)
            maxerr=max(maxerr,abs(direct-endpoint))
            reverse += int(min(vrel)<0<max(vrel))
    assert maxerr < 5e-15
    return {"max_flux_error":maxerr,"sign_reversals":reverse}


if __name__=="__main__":
    oracle=check_analytic_face()
    cases = []
    for eps in [5e-3,1e-6,1e-8]:
        for motion in ["slow","fast"]:
            cases.append(run(eps,motion,None,1))
    for motion in ["fixed","slow","fast"]:
        for order in [0,1]:
            for name,load in LOADS.items():
                cases.append({"load":name,**run(1e-8,motion,load,order)})
    for eps in [5e-3,1e-6]:
        name = "asymmetric"
        cases.append({"load":name,**run(eps,"fast",LOADS[name],1)})
    result={"dt":DT,"wave_speed":C,"analytic_face":oracle,"cases":cases}
    for r in cases:
        print(json.dumps({k:v for k,v in r.items() if k not in ("endpoints","reference")}),flush=True)
    Path("moving_cutcell_wave.json").write_text(json.dumps(result,indent=2)+"\n")
