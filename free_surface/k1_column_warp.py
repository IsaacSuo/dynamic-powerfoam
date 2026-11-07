"""k1_slave.column_omega in Warp (11-06: 16-22 ms of the follow map's rebuild, a few million coefficients expanded and sorted
by torch.unique). The same estimate per cell k: omega_k^2 = [(lam + 2 mu) sum_s V_s |c_sk|^2 + its springs' k] / m_k,
c_sk the coefficient of u_k in V_s G_s summed over k1's terms (term_triplets): the walls ((1 - t) A n on i, t A n on j,
signed by the side), the open face a_s, tau (each of s's walls, both sides' fits: (g . tau) A / 2 n), the cap (s's fit:
Mc^T g); a fit's entries are its own G0 rows (O, R) and (S, -R), or for an aggregated tiny cell the weighted rows of its
aggregation partners. k_col: one thread per cell s, its (k, c) pairs merged in its own scratch row (linear search), then
V_s |c / V_s|^2 (lam + 2 mu) added to k."""
import time

import torch
import warp as wp

f64 = torch.float64
CAP = 640
_BUF = {}
TIMING = False
_T = {}


@wp.func
def acc(sk: wp.array2d(dtype=wp.int32), sc: wp.array2d(dtype=wp.vec3d), s: int, cnt: int, k: int, c: wp.vec3d, cap: int):
    """add c to k's slot in s's row; returns the new count (cap + 1 marks an overflow)"""
    for q in range(cnt):
        if sk[s, q] == k:
            sc[s, q] = sc[s, q]+c
            return cnt
    if cnt >= cap:
        return cap+1
    sk[s, cnt] = k
    sc[s, cnt] = c
    return cnt+1


@wp.kernel(enable_backward=False)
def k_col(wi: wp.array(dtype=wp.int32), wj: wp.array(dtype=wp.int32), wa: wp.array(dtype=wp.float64), wn: wp.array(dtype=wp.vec3d),
          wt: wp.array(dtype=wp.float64), wtau: wp.array(dtype=wp.vec3d), offW: wp.array(dtype=wp.int32), lstW: wp.array(dtype=wp.int32),
          offG: wp.array(dtype=wp.int32), gO: wp.array(dtype=wp.int32), gR: wp.array(dtype=wp.vec3d),
          offA: wp.array(dtype=wp.int32), aO: wp.array(dtype=wp.int32), aW: wp.array(dtype=wp.float64), used: wp.array(dtype=wp.int32),
          a_: wp.array(dtype=wp.vec3d), Mc: wp.array(dtype=wp.mat33d), capc: wp.array(dtype=wp.int32), Vi: wp.array(dtype=wp.float64),
          l2m: wp.float64, cap: int, sk: wp.array2d(dtype=wp.int32), sc: wp.array2d(dtype=wp.vec3d), K: wp.array(dtype=wp.float64),
          over: wp.array(dtype=wp.int32)):
    s = wp.tid()
    one = wp.float64(1.0)
    half = wp.float64(0.5)
    cnt = int(0)
    # the walls and tau
    for e in range(offW[s], offW[s+1]):
        w = lstW[e]
        i = wi[w]
        j = wj[w]
        sg = one
        if s != i:
            sg = -one
        A = wa[w]
        n_ = wn[w]
        cnt = acc(sk, sc, s, cnt, i, (sg*A*(one-wt[w]))*n_, cap)
        cnt = acc(sk, sc, s, cnt, j, (sg*A*wt[w])*n_, cap)
        base = sg*A*half
        tau = wtau[w]
        for side in range(2):
            c = i
            if side == 1:
                c = j
            if used[c] == 0:
                for f in range(offG[c], offG[c+1]):
                    co = base*wp.dot(gR[f], tau)
                    cnt = acc(sk, sc, s, cnt, gO[f], co*n_, cap)
                    cnt = acc(sk, sc, s, cnt, c, -co*n_, cap)
            else:
                for b in range(offA[c], offA[c+1]):
                    o = aO[b]
                    ww = aW[b]
                    for f in range(offG[o], offG[o+1]):
                        co = base*ww*wp.dot(gR[f], tau)
                        cnt = acc(sk, sc, s, cnt, gO[f], co*n_, cap)
                        cnt = acc(sk, sc, s, cnt, o, -co*n_, cap)
    # the open face
    cnt = acc(sk, sc, s, cnt, s, a_[s], cap)
    # the cap through s's fit
    if capc[s] != 0:
        M = wp.transpose(Mc[s])
        if used[s] == 0:
            for f in range(offG[s], offG[s+1]):
                cnt = acc(sk, sc, s, cnt, gO[f], M@gR[f], cap)
                cnt = acc(sk, sc, s, cnt, s, -(M@gR[f]), cap)
        else:
            for b in range(offA[s], offA[s+1]):
                o = aO[b]
                ww = aW[b]
                for f in range(offG[o], offG[o+1]):
                    cnt = acc(sk, sc, s, cnt, gO[f], ww*(M@gR[f]), cap)
                    cnt = acc(sk, sc, s, cnt, o, -ww*(M@gR[f]), cap)
    if cnt > cap:
        over[0] = 1
        return
    vi = Vi[s]
    for q in range(cnt):
        cv = sc[s, q]*vi
        wp.atomic_add(K, sk[s, q], l2m/vi*wp.dot(cv, cv))


def csr(keys, n, *vals):
    o = torch.argsort(keys, stable=True)
    off = torch.zeros(n+1, dtype=torch.long, device=keys.device); off[1:] = torch.cumsum(torch.bincount(keys, minlength=n), 0)
    return (off,)+tuple(v[o] for v in vals)


def column_omega_warp(W):
    if TIMING:
        torch.cuda.synchronize(); _T['start'] = time.perf_counter()
    t, n = W.tul, W.n; ml, m = W._ml, W._w[5]; dev = t['a'].device; sd = str(dev)
    S, O, R = t['S'][:2*ml].long(), t['O'][:2*ml].long(), t['R'][:2*ml]
    wi, wj = t['wi'][:m].long(), t['wj'][:m].long()
    offW, lstW = csr(torch.cat([wi, wj]), n, torch.cat([torch.arange(m, device=dev)]*2))
    offG, gO, gR = csr(S, n, O, R)
    As, Ao, Aw = t['As'].long(), t['Ao'].long(), t['Aw']; ok = Aw != 0
    offA, aO, aW = csr(As[ok], n, Ao[ok], Aw[ok])
    capc = (t['Mc'].abs().sum((1, 2)) > 0)
    I = lambda x_: wp.from_torch(x_.to(torch.int32).contiguous(), dtype=wp.int32)
    F = lambda x_: wp.from_torch(x_.to(f64).contiguous(), dtype=wp.float64)
    V3 = lambda x_: wp.from_torch(x_.to(f64).contiguous(), dtype=wp.vec3d)
    key = (sd, n)
    if key not in _BUF:                                                              # (the scratch rows kept: only the first cnt slots are read)
        _BUF[key] = (wp.empty((n, CAP), dtype=wp.int32, device=sd), wp.empty((n, CAP), dtype=wp.vec3d, device=sd))
    sk, sc = _BUF[key]
    K = wp.zeros(n, dtype=wp.float64, device=sd); over = wp.zeros(1, dtype=wp.int32, device=sd)
    if TIMING:
        torch.cuda.synchronize(); _T['prep'] = time.perf_counter()
    wp.launch(k_col, n, inputs=[I(wi), I(wj), F(t['wa'][:m]), V3(t['wn'][:m]), F(t['wt'][:m]), V3(t['wtau'][:m]), I(offW), I(lstW), I(offG), I(gO), V3(gR),
                                I(offA), I(aO), F(aW), I(t['used']), V3(t['a']), wp.from_torch(t['Mc'].contiguous(), dtype=wp.mat33d), I(capc), F(t['Vi']),
                                float(W.kap+4*W.mu/3), CAP], outputs=[sk, sc, K, over], device=sd)
    if TIMING:
        torch.cuda.synchronize(); _T['kernel'] = time.perf_counter()
    if int(over.numpy()[0]):
        raise RuntimeError('column_omega_warp: a cell\'s stencil over %d points' % CAP)
    Kt = wp.to_torch(K, requires_grad=False)
    skk = W.tsp['sk'][:W._m_s]
    Kt = Kt.index_add(0, W.tlo, skk).index_add(0, W.thi, skk)
    return torch.sqrt(Kt/W.V0)


# ---- by sort (11-06: the per-cell merge in global scratch was 17.6 ms; emit every contribution, radix sort by s n + k, sum runs)
@wp.func
def fitn(c: int, used: wp.array(dtype=wp.int32), offG: wp.array(dtype=wp.int32), offA: wp.array(dtype=wp.int32), aO: wp.array(dtype=wp.int32)):
    """the number of a cell's fit entries (each G0 row gives two)"""
    if used[c] == 0:
        return 2*(offG[c+1]-offG[c])
    t = int(0)
    for b in range(offA[c], offA[c+1]):
        o = aO[b]
        t += 2*(offG[o+1]-offG[o])
    return t


@wp.kernel(enable_backward=False)
def k_count(wi: wp.array(dtype=wp.int32), wj: wp.array(dtype=wp.int32), offW: wp.array(dtype=wp.int32), lstW: wp.array(dtype=wp.int32),
            offG: wp.array(dtype=wp.int32), offA: wp.array(dtype=wp.int32), aO: wp.array(dtype=wp.int32), used: wp.array(dtype=wp.int32),
            capc: wp.array(dtype=wp.int32), cnt: wp.array(dtype=wp.int32)):
    s = wp.tid()
    t = int(1)
    for e in range(offW[s], offW[s+1]):
        w = lstW[e]
        t += 2+fitn(wi[w], used, offG, offA, aO)+fitn(wj[w], used, offG, offA, aO)
    if capc[s] != 0:
        t += fitn(s, used, offG, offA, aO)
    cnt[s] = t


@wp.func
def emit(keys: wp.array(dtype=wp.int32), idx: wp.array(dtype=wp.int32), cv: wp.array(dtype=wp.vec3d), p: int, s: int, n: int, k: int, c: wp.vec3d):
    keys[p] = s*n+k
    idx[p] = p
    cv[p] = c
    return p+1


@wp.kernel(enable_backward=False)
def k_emit(wi: wp.array(dtype=wp.int32), wj: wp.array(dtype=wp.int32), wa: wp.array(dtype=wp.float64), wn: wp.array(dtype=wp.vec3d),
           wt: wp.array(dtype=wp.float64), wtau: wp.array(dtype=wp.vec3d), offW: wp.array(dtype=wp.int32), lstW: wp.array(dtype=wp.int32),
           offG: wp.array(dtype=wp.int32), gO: wp.array(dtype=wp.int32), gR: wp.array(dtype=wp.vec3d),
           offA: wp.array(dtype=wp.int32), aO: wp.array(dtype=wp.int32), aW: wp.array(dtype=wp.float64), used: wp.array(dtype=wp.int32),
           a_: wp.array(dtype=wp.vec3d), Mc: wp.array(dtype=wp.mat33d), capc: wp.array(dtype=wp.int32), start: wp.array(dtype=wp.int32), n: int,
           keys: wp.array(dtype=wp.int32), idx: wp.array(dtype=wp.int32), cv: wp.array(dtype=wp.vec3d)):
    s = wp.tid()
    one = wp.float64(1.0)
    half = wp.float64(0.5)
    p = start[s]
    for e in range(offW[s], offW[s+1]):
        w = lstW[e]
        i = wi[w]
        j = wj[w]
        sg = one
        if s != i:
            sg = -one
        A = wa[w]
        n_ = wn[w]
        p = emit(keys, idx, cv, p, s, n, i, (sg*A*(one-wt[w]))*n_)
        p = emit(keys, idx, cv, p, s, n, j, (sg*A*wt[w])*n_)
        base = sg*A*half
        tau = wtau[w]
        for side in range(2):
            c = i
            if side == 1:
                c = j
            if used[c] == 0:
                for f in range(offG[c], offG[c+1]):
                    co = base*wp.dot(gR[f], tau)
                    p = emit(keys, idx, cv, p, s, n, gO[f], co*n_)
                    p = emit(keys, idx, cv, p, s, n, c, -co*n_)
            else:
                for b in range(offA[c], offA[c+1]):
                    o = aO[b]
                    ww = aW[b]
                    for f in range(offG[o], offG[o+1]):
                        co = base*ww*wp.dot(gR[f], tau)
                        p = emit(keys, idx, cv, p, s, n, gO[f], co*n_)
                        p = emit(keys, idx, cv, p, s, n, o, -co*n_)
    p = emit(keys, idx, cv, p, s, n, s, a_[s])
    if capc[s] != 0:
        M = wp.transpose(Mc[s])
        if used[s] == 0:
            for f in range(offG[s], offG[s+1]):
                p = emit(keys, idx, cv, p, s, n, gO[f], M@gR[f])
                p = emit(keys, idx, cv, p, s, n, s, -(M@gR[f]))
        else:
            for b in range(offA[s], offA[s+1]):
                o = aO[b]
                ww = aW[b]
                for f in range(offG[o], offG[o+1]):
                    p = emit(keys, idx, cv, p, s, n, gO[f], ww*(M@gR[f]))
                    p = emit(keys, idx, cv, p, s, n, o, -ww*(M@gR[f]))


@wp.kernel(enable_backward=False)
def k_runs(keys: wp.array(dtype=wp.int32), idx: wp.array(dtype=wp.int32), cv: wp.array(dtype=wp.vec3d), tot: int, n: int,
           Vi: wp.array(dtype=wp.float64), l2m: wp.float64, K: wp.array(dtype=wp.float64)):
    """a thread at each run's start sums the run (sorted keys) and adds V_s |c / V_s|^2 (lam + 2 mu) to k"""
    q = wp.tid()
    key = keys[q]
    if q > 0:
        if keys[q-1] == key:
            return
    c = wp.vec3d()
    r = q
    while r < tot:
        if keys[r] != key:
            break
        c = c+cv[idx[r]]
        r += 1
    s = key/n
    k = key-s*n
    vi = Vi[s]
    cc = c*vi
    wp.atomic_add(K, k, l2m/vi*wp.dot(cc, cc))


def column_omega_sort(W):
    if TIMING:
        torch.cuda.synchronize(); _T['start'] = time.perf_counter()
    t, n = W.tul, W.n; ml, m = W._ml, W._w[5]; dev = t['a'].device; sd = str(dev)
    S, O, R = t['S'][:2*ml].long(), t['O'][:2*ml].long(), t['R'][:2*ml]
    wi, wj = t['wi'][:m].long(), t['wj'][:m].long()
    offW, lstW = csr(torch.cat([wi, wj]), n, torch.cat([torch.arange(m, device=dev)]*2))
    offG, gO, gR = csr(S, n, O, R)
    As, Ao, Aw = t['As'].long(), t['Ao'].long(), t['Aw']; ok = Aw != 0
    offA, aO, aW = csr(As[ok], n, Ao[ok], Aw[ok])
    capc = (t['Mc'].abs().sum((1, 2)) > 0)
    I = lambda x_: wp.from_torch(x_.to(torch.int32).contiguous(), dtype=wp.int32)
    F = lambda x_: wp.from_torch(x_.to(f64).contiguous(), dtype=wp.float64)
    V3 = lambda x_: wp.from_torch(x_.to(f64).contiguous(), dtype=wp.vec3d)
    wiw, wjw, offWw, lstWw, offGw, offAw, aOw, usedw, capw = I(wi), I(wj), I(offW), I(lstW), I(offG), I(offA), I(aO), I(t['used']), I(capc)
    cnt = wp.zeros(n, dtype=wp.int32, device=sd)
    wp.launch(k_count, n, inputs=[wiw, wjw, offWw, lstWw, offGw, offAw, aOw, usedw, capw], outputs=[cnt], device=sd)
    cn = wp.to_torch(cnt).long(); start = torch.cumsum(cn, 0)-cn; tot = int(cn.sum())
    assert n*n < 2**31, 'int32 keys s n + k'
    key = (sd, 'sort')
    if key not in _BUF or _BUF[key][0].shape[0] < 2*tot:
        cap = int(1.3*tot)+1024
        _BUF[key] = (wp.empty(2*cap, dtype=wp.int32, device=sd), wp.empty(2*cap, dtype=wp.int32, device=sd), wp.empty(cap, dtype=wp.vec3d, device=sd))
    keys, idx, cv = _BUF[key]
    wp.launch(k_emit, n, inputs=[wiw, wjw, F(t['wa'][:m]), V3(t['wn'][:m]), F(t['wt'][:m]), V3(t['wtau'][:m]), offWw, lstWw, offGw, I(gO), V3(gR),
                                 offAw, aOw, F(aW), usedw, V3(t['a']), wp.from_torch(t['Mc'].contiguous(), dtype=wp.mat33d), capw, I(start), n],
              outputs=[keys, idx, cv], device=sd)
    if TIMING:
        torch.cuda.synchronize(); _T['prep'] = time.perf_counter()
    wp.utils.radix_sort_pairs(keys, idx, tot)
    K = wp.zeros(n, dtype=wp.float64, device=sd)
    wp.launch(k_runs, tot, inputs=[keys, idx, cv, tot, n, F(t['Vi']), float(W.kap+4*W.mu/3)], outputs=[K], device=sd)
    if TIMING:
        torch.cuda.synchronize(); _T['kernel'] = time.perf_counter()
    Kt = wp.to_torch(K, requires_grad=False)
    skk = W.tsp['sk'][:W._m_s]
    Kt = Kt.index_add(0, W.tlo, skk).index_add(0, W.thi, skk)
    return torch.sqrt(Kt/W.V0)
