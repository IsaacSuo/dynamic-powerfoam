"""k1_slave.anchor_map in Warp (11-06: the follow map's rows were 226 ms of a 270 ms rebuild per cut, Python loops over the
walls and the slaved particles). The same rule, the same output:
  each slaved particle s takes an anchor k, a non-slaved wall neighbour (largest shared wall first; ties as Python's
  sorted on (area, index), descending), and moves with k's affine motion u_s = u_k + G_k (x_s - x_k), G_k k's
  least-squares fit over its own non-slaved neighbours with weights A / L: weights 1 - sum c_o on k and
  c_o = w_o d_o^T M_k^-1 (x_s - x_k) on each fit neighbour o. A fit needs 3 neighbours and lambda_min > 1e-3 lambda_max.
  Of the first three anchors the first with max |w| <= wmax, else the one with the smallest max |w|.
  k_fit    one thread per particle: M_k^-1 over its non-slaved neighbours, or not valid
  k_rows   one thread per slaved particle: the anchor, the row in the particle's slots (1 + the anchor's degree)
Returns (s, k, w) triplets, the slaved left free (no anchor with a spanning fit), the max |w| per slaved particle."""
import numpy as np
import torch
import warp as wp

f64 = torch.float64


@wp.kernel(enable_backward=False)
def k_fit(x: wp.array(dtype=wp.vec3d), off: wp.array(dtype=wp.int32), nbr: wp.array(dtype=wp.int32), nw: wp.array(dtype=wp.float64),
          sl: wp.array(dtype=wp.int32), ratio: wp.float64, Mi: wp.array(dtype=wp.mat33d), ok: wp.array(dtype=wp.int32)):
    k = wp.tid()
    ok[k] = 0
    if sl[k] != 0:
        return
    M = wp.mat33d()
    cnt = int(0)
    for e in range(off[k], off[k+1]):
        o = nbr[e]
        if sl[o] == 0:
            d = x[o]-x[k]
            M = M+nw[e]*wp.outer(d, d)
            cnt += 1
    if cnt < 3:
        return
    Q, ev = wp.eig3(M)
    lo = wp.min(ev[0], wp.min(ev[1], ev[2]))
    hi = wp.max(ev[0], wp.max(ev[1], ev[2]))
    if lo > ratio*hi:
        Mi[k] = wp.inverse(M)
        ok[k] = 1


@wp.kernel(enable_backward=False)
def k_rows(x: wp.array(dtype=wp.vec3d), off: wp.array(dtype=wp.int32), nbr: wp.array(dtype=wp.int32), na: wp.array(dtype=wp.float64),
           nw: wp.array(dtype=wp.float64), sl: wp.array(dtype=wp.int32), Mi: wp.array(dtype=wp.mat33d), ok: wp.array(dtype=wp.int32),
           sid: wp.array(dtype=wp.int32), cap: int, wmax: wp.float64,
           rk: wp.array2d(dtype=wp.int32), rw: wp.array2d(dtype=wp.float64), rn: wp.array(dtype=wp.int32), rmax: wp.array(dtype=wp.float64)):
    t = wp.tid()
    s = sid[t]
    rn[t] = 0
    # the three largest walls to non-slaved neighbours, (area, index) descending
    a1 = wp.float64(-1.0)
    a2 = wp.float64(-1.0)
    a3 = wp.float64(-1.0)
    k1 = int(-1)
    k2 = int(-1)
    k3 = int(-1)
    for e in range(off[s], off[s+1]):
        k = nbr[e]
        if sl[k] != 0:
            continue
        a = na[e]
        if a > a1 or (a == a1 and k > k1):
            a3 = a2
            k3 = k2
            a2 = a1
            k2 = k1
            a1 = a
            k1 = k
        elif a > a2 or (a == a2 and k > k2):
            a3 = a2
            k3 = k2
            a2 = a
            k2 = k
        elif a > a3 or (a == a3 and k > k3):
            a3 = a
            k3 = k
    best = int(-1)
    bestm = wp.float64(1e300)
    for ci in range(3):
        k = k1
        if ci == 1:
            k = k2
        elif ci == 2:
            k = k3
        if k < 0:
            continue
        if ok[k] == 0:
            continue
        b = Mi[k]@(x[s]-x[k])
        csum = wp.float64(0.0)
        mx = wp.float64(0.0)
        for e in range(off[k], off[k+1]):
            o = nbr[e]
            if sl[o] == 0:
                c = nw[e]*wp.dot(x[o]-x[k], b)
                csum = csum+c
                mx = wp.max(mx, wp.abs(c))
        mx = wp.max(mx, wp.abs(wp.float64(1.0)-csum))
        if mx < bestm:
            bestm = mx
            best = k
        if mx <= wmax:
            break
    if best < 0:
        return
    k = best
    b = Mi[k]@(x[s]-x[k])
    csum = wp.float64(0.0)
    m = int(1)
    for e in range(off[k], off[k+1]):
        o = nbr[e]
        if sl[o] == 0:
            c = nw[e]*wp.dot(x[o]-x[k], b)
            csum = csum+c
            rk[t, m] = o
            rw[t, m] = c
            m += 1
    rk[t, 0] = k
    rw[t, 0] = wp.float64(1.0)-csum
    rn[t] = m
    rmax[t] = bestm


def neighbours(x, wi, wj, wa):
    """the walls as each particle's CSR list: neighbour, shared area, A / L"""
    n = len(x)
    src, dst, a = torch.cat([wi, wj]), torch.cat([wj, wi]), torch.cat([wa, wa])
    L = (x[dst]-x[src]).norm(dim=1)
    o = torch.argsort(src, stable=True)
    src, dst, a, L = src[o], dst[o], a[o], L[o]
    off = torch.zeros(n+1, dtype=torch.long, device=x.device); off[1:] = torch.cumsum(torch.bincount(src, minlength=n), 0)
    return off, dst, a, a/L


def anchor_map_warp(x, wi, wj, wa, slave, wmax=4., ratio=1e-3, nb=None):
    dev = x.device; n = len(x)
    off, dst, a, w = neighbours(x, wi, wj, wa) if nb is None else nb
    I = lambda t_: wp.from_torch(t_.to(torch.int32).contiguous(), dtype=wp.int32)
    F = lambda t_: wp.from_torch(t_.to(f64).contiguous(), dtype=wp.float64)
    xw = wp.from_torch(x.detach().contiguous(), dtype=wp.vec3d)
    slw = I(slave)
    Mi = wp.zeros(n, dtype=wp.mat33d, device=str(dev)); ok = wp.zeros(n, dtype=wp.int32, device=str(dev))
    offw, nbw, aw, ww = I(off), I(dst), F(a), F(w)
    wp.launch(k_fit, n, inputs=[xw, offw, nbw, ww, slw, float(ratio)], outputs=[Mi, ok], device=str(dev))
    sid = torch.nonzero(slave).flatten()
    ns = len(sid)
    if ns == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z, torch.zeros(0, dtype=f64, device=dev), [], np.zeros(0)
    cap = 1+int((off[1:]-off[:-1]).max())
    rk = wp.zeros((ns, cap), dtype=wp.int32, device=str(dev)); rw = wp.zeros((ns, cap), dtype=wp.float64, device=str(dev))
    rn = wp.zeros(ns, dtype=wp.int32, device=str(dev)); rmax = wp.zeros(ns, dtype=wp.float64, device=str(dev))
    wp.launch(k_rows, ns, inputs=[xw, offw, nbw, aw, ww, slw, Mi, ok, I(sid), cap, float(wmax)], outputs=[rk, rw, rn, rmax], device=str(dev))
    T = lambda a_: wp.to_torch(a_, requires_grad=False)
    rk_, rw_, rn_, rm_ = T(rk).long(), T(rw), T(rn).long(), T(rmax)
    m = torch.arange(cap, device=dev)[None, :] < rn_[:, None]
    s_ = sid[:, None].expand(-1, cap)[m]; k_ = rk_[m]; w_ = rw_[m]
    got = rn_ > 0
    free = sid[~got].tolist()
    return s_, k_, w_, free, rm_[got].cpu().numpy()
