"""Three-cell nonlinear ALE flux predictor with a bounded wave-event budget.

The only input is (tau, velocity) in each cell, the two face paths, and dt.
Compression fronts are exact. Expansion fans use small conservative jumps;
their entropy error is reported separately. Beyond the event budget, reject.
"""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from nonlinear_cutcell_impact import DT, Impact, fields


def riemann(left, right, fan_step, fan_limit=None):
    """Resolve one p-system discontinuity into ordered (speed, left, right) fronts."""
    a, ul = left
    b, ur = right
    if left == right:
        return []
    la, lb = math.log(a), math.log(b)

    def curve(d, count):
        if count == 0 and d >= 0:
            return d
        n = max(1, count) if d >= 0 else 1
        return 2*n*math.sinh(d/(2*n))

    def solve(count_l, count_r):
        return brentq(lambda x: curve(x-la, count_l) + curve(x-lb, count_r) + ul-ur,
                      min(la, lb)-abs(ul-ur)-1, max(la, lb)+abs(ul-ur)+1, xtol=5e-15)

    ls = solve(0, 0)
    nl = max(1, math.ceil((ls-la)/fan_step))
    nr = max(1, math.ceil((ls-lb)/fan_step))
    if fan_limit is not None:
        nl, nr = min(nl, fan_limit), min(nr, fan_limit)
    if ls > min(la, lb) + 1e-13:
        ls = solve(nl, nr)
    us = ul + curve(ls-la, nl)
    star = (math.exp(ls), us)
    fronts = []
    for l0, l1, n, sign, initial, final in [(la, ls, nl, -1, left, star),
                                            (ls, lb, nr, 1, star, right)]:
        if abs(l1-l0) < 2e-13:
            continue
        current = initial
        for j in range(1, n+1):
            d = (l1-l0)/n
            target = final if j == n else (math.exp(l0+j*d),
                                          current[1] - sign*2*math.sinh(d/2))
            fronts.append((sign / math.sqrt(current[0]*target[0]), current, target))
            current = target
    return fronts


def fluxes(q, w):
    tau, u = q
    p = 1/tau - 1
    e = u*u/2 + tau - 1 - math.log(tau)
    return np.array([-u-w*tau, p-w*u, p*u-w*e, p, p*u])


def intervals(begin, end, events):
    times = [begin, *sorted(t for t in events if begin < t < end), end]
    return zip(times[:-1], times[1:])


def face_interval(left, fronts, starts, speeds, time, stop, faces_only=False):
    # fronts store their position at 'time'. Face paths are linear in X.
    accum = np.zeros((2, 6 if faces_only else 5))
    entropy = np.zeros(2)
    net_gain = 0.
    for i, (x0, w) in enumerate(zip(starts, speeds)):
        face = x0 + w*time
        hits = (time+(x-face)/(w-s) for x,s,_,_ in fronts if w != s)
        for a, b in intervals(time, stop, hits):
            mid = (a+b)/2
            q = left
            for x, s, _, r in fronts:
                if x+s*(mid-time) > x0+w*mid:
                    break
                q = r
            accum[i, :5] += (b-a)*fluxes(q,w)
            if faces_only:
                accum[i, 5] -= (b-a)*q[1]
    if faces_only:
        return accum, entropy, net_gain
    for x,s,l,r in fronts:
        ql, fl = fields(l)
        qr, fr = fields(r)
        production = fr[2]-fl[2]-s*(qr[2]-ql[2])
        net_gain += max(0.,production)*(stop-time)
        hits = (time+(x-(x0+w*time))/(w-s)
                for x0,w in zip(starts,speeds) if w != s)
        for a,b in intervals(time,stop,hits):
            mid = (a+b)/2
            position = x+s*(mid-time)
            if starts[0]+speeds[0]*mid < position < starts[1]+speeds[1]*mid:
                entropy[int(production > 0)] += (b-a)*production
    return accum, entropy, net_gain


def mean_in_cell(left, fronts, start, end):
    mass_integral = np.zeros(3)
    q = left
    low = start
    for x, _, _, r in fronts:
        if x >= end:
            break
        if x > low:
            mass_integral += (x-low)*fields(q)[0]
            low = x
        q = r
    mass_integral += (end-low)*fields(q)[0]
    return mass_integral/(end-start)


def predict(states, starts, speeds, dt, fan_step=.025, event_budget=24, *, fan_limit=None, faces_only=False):
    end = starts+dt*speeds
    assert end[1] > end[0]
    initial = [riemann(l,r,fan_step,fan_limit) for l,r in zip(states[:-1], states[1:])]
    fronts = [(x,s,l,r) for x, fan in zip(starts,initial) for s,l,r in fan]
    if not faces_only:
        frozen = []
        for w, fan, q in zip(speeds, initial, states):
            for s, _, r in fan:
                if s > w:
                    break
                q = r
            frozen.append(dt*fluxes(q,w))
        q0 = fields(states[1])[0]
        old_integral = (starts[1]-starts[0])*q0
        frozen_end = (old_integral - np.diff(frozen,axis=0)[0,:3])/(end[1]-end[0])
    accum = np.zeros((2, 6 if faces_only else 5))
    entropy = np.zeros(2)
    whole_patch_gain = 0.
    time = 0.
    events = 0
    max_fronts = len(fronts)
    while time < dt:
        stop, collision = dt, None
        for i, (a,b) in enumerate(zip(fronts[:-1],fronts[1:])):
            if a[1] > b[1]:
                hit = time + max(0.,b[0]-a[0])/(a[1]-b[1])
                if hit < stop:
                    stop, collision = hit, i
        if collision is not None and events >= event_budget:
            return {"status": "budget", "events": events, "riemann_solves": 2+events,
                    "max_fronts": max_fronts, "covered_fraction": time/dt}
        f, d, gain = face_interval(states[0],fronts,starts,speeds,time,stop,faces_only)
        whole_patch_gain += gain
        accum += f
        entropy += d
        fronts = [(x+s*(stop-time),s,l,r) for x,s,l,r in fronts]
        time = stop
        if collision is not None:
            a,b = fronts[collision:collision+2]
            x = (a[0]+b[0])/2
            fan = riemann(a[2],b[3],fan_step,fan_limit)
            fronts[collision:collision+2] = [(x,s,l,r) for s,l,r in fan]
            events += 1
            max_fronts = max(max_fronts,len(fronts))
    if faces_only:
        return dict(status='ok', face_fluxes=accum, events=events)
    mean = mean_in_cell(states[0],fronts,*end)
    balance = old_integral - np.diff(accum,axis=0)[0,:3]
    balance[2] += entropy.sum()
    return {"status": "ok", "events": events, "riemann_solves": 2+events, "max_fronts": max_fronts,
            "stretch": float(mean[0]), "velocity": float(mean[1]),
            "frozen_stretch": float(frozen_end[0]), "face_fluxes": accum.tolist(),
            "balance_error": float(np.max(abs(balance-mean*(end[1]-end[0])))),
            "normalized_balance_error": float(np.max(abs(balance/(end[1]-end[0])-mean))),
            "expansion_entropy_gain": float(entropy[1]), "whole_patch_expansion_gain": float(whole_patch_gain), "shock_entropy_loss": float(-entropy[0])}


def case_geometry(width, motion):
    starts = np.array([-width/2,width/2])
    speeds = np.array({"fixed": (0.,0.), "drift": (.85,.85),
                       "squeeze": (10*width,-10*width)}[motion])
    return starts, speeds


def analytic_tests():
    results = []
    for compression in (.65,.8):
        for width in (5e-3,1e-5,1e-8):
            for motion in ("fixed","drift","squeeze"):
                oracle = Impact(width,compression)
                states = (oracle.left,oracle.middle,oracle.right)
                starts, speeds = case_geometry(width,motion)
                row = predict(states,starts,speeds,DT)
                assert row["status"] == "ok" and row["events"] == 1
                reference = np.array([oracle.face_integral(x,w)[:2]
                                      for x,w in zip(starts,speeds)])
                expected = np.column_stack((reference[:,0],reference[:,1,1:]))
                err = np.max(abs(np.array(row["face_fluxes"])-expected),axis=0)
                assert max(err) < 1e-13 and abs(row["stretch"]-oracle.star[0]) < 1e-11
                results.append(dict(width=width,motion=motion,compression=compression,
                                    max_flux_error=err.tolist(),**row))
    return results


def samples():
    rng = np.random.default_rng(183)
    for kind in ("compression","mixed","expansion"):
        for i in range(16):
            tau = rng.uniform(.65,1.15,3)
            u = rng.uniform(-.22,.22,3)
            if kind != "mixed":
                sign = 1 if kind == "compression" else -1
                u += sign*np.array([.45,0,-.45])
            yield kind,i,list(zip(tau,u))


def stress_tests(sample, fan_step=.025):
    rows = []
    for kind,i,states in sample:
        for width in (5e-3,1e-5,1e-8):
            starts,speeds=case_geometry(width,("fixed","drift","squeeze")[i%3])
            row = predict(states,starts,speeds,DT,fan_step=fan_step)
            if row["status"] == "ok":
                assert row["balance_error"] < 2e-12 and row["stretch"] > 0.
            rows.append(dict(kind=kind,index=i,states=states,width=width,**row))
    return rows


def fine_reference(states, starts, speeds, count):
    """Independent MUSCL/Rusanov integration on a fine uniform reference grid."""
    edges = np.linspace(-.1,.1,count+1)
    dx = edges[1]-edges[0]
    centers = (edges[1:]+edges[:-1])/2
    q = np.asarray(states)[np.searchsorted(starts,centers)]
    initial = q.copy()
    reference = np.zeros((2,5))

    def stencil(u):
        ext = np.vstack((states[0],u,states[2]))
        dl, dr = ext[1:-1]-ext[:-2],ext[2:]-ext[1:-1]
        slope = np.where(dl*dr > 0,np.sign(dl)*np.minimum(abs(dl),abs(dr)),0.)
        l = np.vstack((states[0],u+.5*slope))
        r = np.vstack((u-.5*slope,states[2]))
        f = lambda a: np.column_stack((-a[:,1],1/a[:,0]-1))
        c = (1/np.minimum(l[:,0],r[:,0]))[:,None]
        flux = (f(l)+f(r)-c*(r-l))/2
        return -np.diff(flux,axis=0)/dx, slope

    def sample(q,slope,time):
        x = starts+time*speeds
        i = np.searchsorted(edges,x)-1
        state = q[i]+slope[i]*((x-centers[i])/dx)[:,None]
        return np.array([fluxes(a,w) for a,w in zip(state,speeds)])

    time = 0.
    while time < DT:
        step = min(DT-time,.30*dx*np.min(q[:,0]))
        rate, slope = stencil(q)
        stage = q+step*rate
        rate1, slope1 = stencil(stage)
        reference += step/2*(sample(q,slope,time)+sample(stage,slope1,time+step))
        q = (q+stage+step*rate1)/2
        assert np.min(q[:,0]) > 0
        time += step
    # Recover the conservative fluxes from the left interval balance,
    # independent of the sampled pressure/work traces above.
    end = starts+DT*speeds
    _, slope = stencil(q)

    def cumulative(q,slope,x):
        i = np.searchsorted(edges,x)-1
        length = x-edges[i]
        return (q[:i].sum(axis=0)*dx + length*q[i]
                + .5*length*(length/dx-1)*slope[i])

    boundary = DT*fluxes(states[0],0)[:2]
    final_integrals = []
    for i,(a,b) in enumerate(zip(starts,end)):
        final_integrals.append(cumulative(q,slope,b))
        reference[i,:2] = boundary + cumulative(initial,np.zeros_like(initial),a) - final_integrals[-1]
    final = (final_integrals[1]-final_integrals[0])/(end[1]-end[0])
    return reference, final


def reference_tests(sample, by_fan_step):
    rows = []
    for kind,ids in (("compression",(0,1)),("mixed",(0,1)),("expansion",(1,6))):
        for case,index,states in sample:
            if case != kind or index not in ids:
                continue

            starts,speeds = case_geometry(.005,("fixed","drift","squeeze")[index%3])
            fine_n = 10240 if index==1 and case in ("mixed","expansion") else 5120
            ref_low, mean_low = fine_reference(states,starts,speeds,fine_n//2)
            ref, mean = fine_reference(states,starts,speeds,fine_n)
            for fan_step, stress in by_fan_step.items():
                a = next(r for r in stress if r["kind"]==case and r["index"]==index and r["width"]==.005)
                if a["status"] != "ok":
                    continue
                err = np.max(abs(np.array(a["face_fluxes"])-ref),axis=0)
                scale = np.max(abs(ref),axis=0)
                norm_err = np.divide(err,scale,out=np.zeros_like(err),where=scale>1e-12)
                rows.append(dict(kind=case,index=index,fan_step=fan_step,
                                 reference_flux=ref.tolist(),reference_cells=fine_n,
                                 reference_stretch=float(mean[0]),predicted_stretch=a["stretch"],
                                 stretch_error=float(abs(a["stretch"]-mean[0])),
                                 stretch_refinement_difference=float(abs(mean_low[0]-mean[0])),
                                 flux_error=err.tolist(),relative_flux_error=norm_err.tolist(),
                                 flux_refinement_difference=np.max(abs(ref-ref_low),axis=0).tolist()))
    return rows


def thin_fan_sensitivity(sample, coarse):
    # A resolution sensitivity check, not an independent PDE reference.
    width = 1e-8
    rows = []
    for kind, index, states in sample:
        a = next(r for r in coarse if r["kind"]==kind and r["index"]==index and r["width"]==width)
        assert a["status"] == "ok"
        starts,speeds = case_geometry(width,("fixed","drift","squeeze")[index%3])
        b = predict(states,starts,speeds,DT,fan_step=.025,event_budget=256)
        row = dict(kind=kind,index=index,reference_status=b["status"],reference_events=b["events"])
        if b["status"] == "ok":
            ref = np.asarray(b["face_fluxes"])
            err = np.max(abs(ref-np.asarray(a["face_fluxes"])),axis=0)
            scale = np.max(abs(ref),axis=0)
            row.update(stretch_difference=abs(a["stretch"]-b["stretch"]),
                       flux_difference=err.tolist(),
                       relative_flux_difference=(err/np.maximum(scale,1e-10)).tolist())
        rows.append(row)
    return rows


if __name__ == "__main__":
    sample = list(samples())
    analytic = analytic_tests()
    fan_steps = (.025,.08,.2)
    by_fan_step = {step: stress_tests(sample,step) for step in fan_steps}
    stress = by_fan_step[fan_steps[0]]
    print("analytic",len(analytic),"stress",Counter(r["status"] for r in stress),flush=True)
    reference = reference_tests(sample,by_fan_step)
    thin_sensitivity = thin_fan_sensitivity(sample,by_fan_step[.2])
    print("independent reference",len(reference),
          "thin resolution",Counter(r["reference_status"] for r in thin_sensitivity))
    Path("nonlinear_cutcell_predictor.json").write_text(
        json.dumps({"dt":DT,"analytic":analytic,"stress_by_fan_step":by_fan_step,"reference":reference,"thin_fan_sensitivity":thin_sensitivity},indent=2)+"\n")
