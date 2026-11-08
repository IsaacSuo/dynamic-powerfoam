"""geometry_warp.k_geo_site with one 32-lane warp per site (11-08), as a Warp native CUDA snippet: the same samples, the
same pole rule, the same per-piece formulas; only the sums' order differs (warp reductions), so the patch normal
integral, the area and the volume agree with k_geo_site to rounding.
  arcs      the site's (entry, arc) list in shared memory (entry order, as k_geo_site loops), each arc's piece count
            (ceil(|D| / piece), at least 1) and the pieces (arc, kk) listed after it
  pass 1    patch_n: one lane per arc, a warp sum
  pass 2    the samples (piece start and middle, k_geo_site's arithmetic) one lane per piece into shared memory; then
            one lane per candidate (32 Fibonacci directions, +nd, -nd) its least dot over all samples (min: exact), the
            largest, the lowest index on a tie (k_geo_site's strict > scan)
  pass 3    the area: one lane per piece (the same Om, beta, cs |dt|, stable_area's chord form), a warp sum; the
            exposure test over the neighbours lane-strided
A site whose arcs or pieces exceed the shared capacity (AMAX, PMX) is flagged and done by k_geo_site afterwards (indexed by
the flagged ids). site(dim, inputs, outputs) takes k_geo_site's launch arguments."""
import os

import warp as wp

from free_surface import geometry_warp as GW

WPB = 4
AMAX = int(os.environ.get('SITE_COOP_AMAX', 32))                                  # (the shared capacity: arcs, pieces a site;
PMX = int(os.environ.get('SITE_COOP_PMX', 128))                                   # smaller only to test the fallback)
SNIP = r'''
#define FULLM 0xffffffffu
const int WPB = %(WPB)d, AMAX = %(AMAX)d, PMX = %(PMX)d, NPK = %(PMX)d/32;
typedef wp::vec_t<3,double> v3d;
__shared__ int sAq[WPB][AMAX];
__shared__ int sAm[WPB][AMAX];
__shared__ int sAf[WPB][AMAX];
__shared__ int sAn[WPB][AMAX];
__shared__ int sPa[WPB][PMX];
__shared__ int sPk[WPB][PMX];
__shared__ double sMa[WPB][PMX][3];
__shared__ double sMm[WPB][PMX][3];
const int w = threadIdx.x >> 5, L = threadIdx.x & 31;
const int sidx = tid >> 5;
if (sidx >= nsites) return;
int s = sidx;
if (indexed == 1) s = site_ids.data[sidx];
const double zero = 0.0, one = 1.0, half_ = 0.5;
const double* P = (const double*)p.data;
const double rs = r.data[s];
const double* NN = (const double*)nn_o.data; const double* U = (const double*)u_o.data; const double* V = (const double*)v_o.data;
const double* A0 = (const double*)at0.data; const double* A1 = (const double*)at1.data;
const double* FB = (const double*)fib.data;
const int e0 = off.data[s], e1 = off.data[s+1];
auto ld = [&](const double* a, int q) -> v3d { return v3d(a[3*q], a[3*q+1], a[3*q+2]); };
auto emv = [&](double t, v3d u, v3d v) -> v3d { return wp::add(wp::mul(wp::cos(t), u), wp::mul(wp::sin(t), v)); };
// the arcs, in k_geo_site's order (entries, then arcs)
int nar = 0, ovf = 0;
for (int eb = e0; eb < e1; eb += 32) {
    int e = eb+L, na = 0, q = 0, first = 0;
    if (e < e1) { int t = ent.data[e]; q = t %% npr; first = t < npr; na = narc.data[q]; }
    int inc = na;
    for (int o = 1; o < 32; o <<= 1) { int x = __shfl_up_sync(FULLM, inc, o); if (L >= o) inc += x; }
    int tot = __shfl_sync(FULLM, inc, 31);
    for (int m = 0; m < na; ++m) {
        int a = nar+inc-na+m;
        if (a < AMAX) { sAq[w][a] = q; sAm[w][a] = m; sAf[w][a] = first; } else ovf = 1;
    }
    nar += tot;
}
ovf = __any_sync(FULLM, ovf) || nar > AMAX;
int npc_tot = 0;
if (!ovf) {
    __syncwarp();
    // pieces per arc, listed
    for (int ab = 0; ab < nar; ab += 32) {
        int a = ab+L, npc = 0;
        if (a < nar) {
            int q = sAq[w][a], m = sAm[w][a];
            double ta = A0[(size_t)q*AW+m], tb = A1[(size_t)q*AW+m];
            if (sAf[w][a]) { double x = ta; ta = tb; tb = x; }
            double D = tb-ta;
            npc = (int)(wp::ceil(wp::abs(D)/piece));
            if (npc < 1) npc = 1;
            sAn[w][a] = npc;
        }
        int inc = npc;
        for (int o = 1; o < 32; o <<= 1) { int x = __shfl_up_sync(FULLM, inc, o); if (L >= o) inc += x; }
        int tot = __shfl_sync(FULLM, inc, 31);
        for (int kk = 0; kk < npc; ++kk) {
            int pi = npc_tot+inc-npc+kk;
            if (pi < PMX) { sPa[w][pi] = a; sPk[w][pi] = kk; } else ovf = 1;
        }
        npc_tot += tot;
    }
    ovf = __any_sync(FULLM, ovf) || npc_tot > PMX;
}
if (ovf) { if (L == 0) flag.data[s] = 1; return; }
__syncwarp();
// pass 1: int m dA
v3d pn(0.0, 0.0, 0.0);
for (int a = L; a < nar; a += 32) {
    int q = sAq[w][a], m = sAm[w][a], first = sAf[w][a];
    v3d axis = ld(NN, q); double h = hi_o.data[q];
    if (!first) { axis = wp::neg(ld(NN, q)); h = hj_o.data[q]; }
    v3d u = ld(U, q), v = ld(V, q); double rho = rho_o.data[q];
    double ta = A0[(size_t)q*AW+m], tb = A1[(size_t)q*AW+m];
    if (first) { double x = ta; ta = tb; tb = x; }
    double D = tb-ta;
    pn = wp::add(pn, wp::mul(half_, wp::add(wp::mul(rho, wp::cross(wp::mul(h, axis), wp::sub(emv(tb, u, v), emv(ta, u, v)))), wp::mul(rho*rho*D, wp::cross(u, v)))));
}
for (int o = 16; o > 0; o >>= 1) { pn[0] += __shfl_xor_sync(FULLM, pn[0], o); pn[1] += __shfl_xor_sync(FULLM, pn[1], o); pn[2] += __shfl_xor_sync(FULLM, pn[2], o); }
double npn = wp::length(pn);
v3d nd(0.0, 0.0, 0.0);
if (npn > zero) nd = wp::div(pn, npn);
// pass 2: the samples, one lane per piece (kept for pass 3)
v3d kma[NPK], kmb[NPK]; double kpa[NPK], kpb[NPK], kcs[NPK], ksn[NPK], ksg[NPK]; v3d ku[NPK], kv[NPK];
#pragma unroll
for (int k = 0; k < NPK; ++k) {
    int pi = k*32+L;
    if (pi < npc_tot) {
        int a = sPa[w][pi], kk = sPk[w][pi];
        int q = sAq[w][a], m = sAm[w][a], first = sAf[w][a], npc = sAn[w][a];
        v3d axis = ld(NN, q); double h = hi_o.data[q];
        if (!first) { axis = wp::neg(ld(NN, q)); h = hj_o.data[q]; }
        v3d u = ld(U, q), v = ld(V, q);
        double cs = h/rs, sn = rho_o.data[q]/rs;
        double ta = A0[(size_t)q*AW+m], tb = A1[(size_t)q*AW+m];
        if (first) { double x = ta; ta = tb; tb = x; }
        double D = tb-ta;
        double pa = ta+D*((double)kk/(double)npc);
        double pb = ta+D*((double)(kk+1)/(double)npc);
        v3d ma = wp::add(wp::mul(cs, axis), wp::mul(sn, emv(pa, u, v)));
        v3d mm = wp::add(wp::mul(cs, axis), wp::mul(sn, emv(half_*(pa+pb), u, v)));
        v3d mb = wp::add(wp::mul(cs, axis), wp::mul(sn, emv(pb, u, v)));
        sMa[w][pi][0] = ma[0]; sMa[w][pi][1] = ma[1]; sMa[w][pi][2] = ma[2];
        sMm[w][pi][0] = mm[0]; sMm[w][pi][1] = mm[1]; sMm[w][pi][2] = mm[2];
        kma[k] = ma; kmb[k] = mb; kpa[k] = pa; kpb[k] = pb; kcs[k] = cs; ksn[k] = sn; ku[k] = u; kv[k] = v;
        ksg[k] = D > zero ? one : (D < zero ? -one : zero);
    }
}
__syncwarp();
// the candidates: lane L has L (a Fibonacci direction) and 32 + L (L = 0: +nd, L = 1: -nd)
double bestv = -big*10.0; int best = 0;
for (int c2 = 0; c2 < 2; ++c2) {
    int ci = c2*32+L;
    double sc = big;
    if (ci < nfib+2) {
        v3d dir = (ci < nfib) ? ld(FB, ci) : nd;
        for (int pi = 0; pi < npc_tot; ++pi) {
            v3d ma(sMa[w][pi][0], sMa[w][pi][1], sMa[w][pi][2]), mm(sMm[w][pi][0], sMm[w][pi][1], sMm[w][pi][2]);
            if (ci == nfib+1) sc = wp::min(sc, wp::min(-wp::dot(dir, ma), -wp::dot(dir, mm)));
            else sc = wp::min(sc, wp::min(wp::dot(dir, ma), wp::dot(dir, mm)));
        }
        if (npn == zero && ci >= nfib) sc = -big*100.0;
        if (sc > bestv) { bestv = sc; best = ci; }
    }
}
for (int o = 16; o > 0; o >>= 1) {
    double ov = __shfl_xor_sync(FULLM, bestv, o); int oi = __shfl_xor_sync(FULLM, best, o);
    if (ov > bestv || (ov == bestv && oi < best)) { bestv = ov; best = oi; }
}
v3d pole = nd;
if (best < nfib) pole = ld(FB, best);
else if (best == nfib+1) pole = wp::neg(nd);
// pass 3: the area, fan from the pole
double Au = 0.0;
#pragma unroll
for (int k = 0; k < NPK; ++k) {
    int pi = k*32+L;
    if (pi < npc_tot) {
        v3d ma = kma[k], mb = kmb[k]; double pa = kpa[k], pb = kpb[k], cs = kcs[k], sn = ksn[k], sg = ksg[k]; v3d u = ku[k], v = kv[k];
        double Om = 2.0*wp::atan2(wp::dot(pole, wp::cross(ma, mb)), one+wp::dot(pole, ma)+wp::dot(ma, mb)+wp::dot(mb, pole));
        v3d tau = wp::mul(sg, wp::add(wp::mul(-wp::sin(pa), u), wp::mul(wp::cos(pa), v)));
        v3d gg = wp::sub(mb, wp::mul(wp::dot(ma, mb), ma));
        if (stable_area == 1) {
            double mid = half_*(pa+pb);
            v3d tm = wp::add(wp::mul(-wp::sin(mid), u), wp::mul(wp::cos(mid), v));
            v3d chord = wp::mul(2.0*sn*wp::sin(half_*(pb-pa)), tm);
            Om = 2.0*wp::atan2(wp::dot(pole, wp::cross(ma, chord)), 2.0+2.0*wp::dot(pole, ma)+wp::dot(pole, chord)+wp::dot(ma, chord));
            gg = wp::mul(sg, wp::sub(tm, wp::mul(wp::dot(ma, tm), ma)));
        }
        double bet = wp::atan2(wp::dot(wp::cross(tau, gg), ma), wp::dot(tau, gg));
        Au += Om+2.0*bet+cs*wp::abs(pb-pa);
    }
}
for (int o = 16; o > 0; o >>= 1) Au += __shfl_xor_sync(FULLM, Au, o);
// is the pole's antipode exposed
if (hidden.data[s] == 0) {
    int ex = 1;
    v3d ps = ld(P, s);
    if (floor_enabled == 1 && ps[1]-rs*pole[1] < floor_height) ex = 0;
    const int* NB = (const int*)nbrs.data+(size_t)s*DN;
    for (int b = L; b < DN; b += 32) {
        int k = NB[b];
        if (k >= 0) {
            v3d dX = wp::sub(wp::sub(ps, ld(P, k)), wp::mul(rs, pole));
            double rk = r.data[k];
            if (!(wp::dot(dX, dX)-rk*rk > zero)) ex = 0;
        }
    }
    ex = __all_sync(FULLM, ex);
    if (ex) Au += fourpi;
}
if (L == 0) {
    double pA = rs*rs*Au;
    double* PN = (double*)pn_o.data; PN[3*s] = pn[0]; PN[3*s+1] = pn[1]; PN[3*s+2] = pn[2];
    pA_o.data[s] = pA;
    V_o.data[s] = (Vw.data[s]+rs*pA)/3.0;
    double* PO = (double*)pole_o.data; PO[3*s] = pole[0]; PO[3*s+1] = pole[1]; PO[3*s+2] = pole[2];
    flag.data[s] = 0;
}
''' % dict(WPB=WPB, AMAX=AMAX, PMX=PMX)


@wp.func_native(SNIP)
def _site(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), off: wp.array(dtype=wp.int32), ent: wp.array(dtype=wp.int32), npr: int,
          nn_o: wp.array(dtype=wp.vec3d), u_o: wp.array(dtype=wp.vec3d), v_o: wp.array(dtype=wp.vec3d), rho_o: wp.array(dtype=wp.float64),
          hi_o: wp.array(dtype=wp.float64), hj_o: wp.array(dtype=wp.float64), at0: wp.array2d(dtype=wp.float64), at1: wp.array2d(dtype=wp.float64),
          narc: wp.array(dtype=wp.int32), fib: wp.array(dtype=wp.vec3d), nfib: int, piece: wp.float64, big: wp.float64, fourpi: wp.float64,
          nbrs: wp.array2d(dtype=wp.int32), hidden: wp.array(dtype=wp.int32), stable_area: int, floor_enabled: int, floor_height: wp.float64,
          site_ids: wp.array(dtype=wp.int32), indexed: int, Vw: wp.array(dtype=wp.float64),
          pn_o: wp.array(dtype=wp.vec3d), pA_o: wp.array(dtype=wp.float64), V_o: wp.array(dtype=wp.float64), pole_o: wp.array(dtype=wp.vec3d),
          flag: wp.array(dtype=wp.int32), nsites: int, AW: int, DN: int, tid: int):
    ...


@wp.kernel(enable_backward=False)
def k_site_coop(p: wp.array(dtype=wp.vec3d), r: wp.array(dtype=wp.float64), off: wp.array(dtype=wp.int32), ent: wp.array(dtype=wp.int32), npr: int,
                nn_o: wp.array(dtype=wp.vec3d), u_o: wp.array(dtype=wp.vec3d), v_o: wp.array(dtype=wp.vec3d), rho_o: wp.array(dtype=wp.float64),
                hi_o: wp.array(dtype=wp.float64), hj_o: wp.array(dtype=wp.float64), at0: wp.array2d(dtype=wp.float64), at1: wp.array2d(dtype=wp.float64),
                narc: wp.array(dtype=wp.int32), fib: wp.array(dtype=wp.vec3d), nfib: int, piece: wp.float64, big: wp.float64, fourpi: wp.float64,
                nbrs: wp.array2d(dtype=wp.int32), hidden: wp.array(dtype=wp.int32), stable_area: int, floor_enabled: int, floor_height: wp.float64,
                site_ids: wp.array(dtype=wp.int32), indexed: int, Vw: wp.array(dtype=wp.float64),
                pn_o: wp.array(dtype=wp.vec3d), pA_o: wp.array(dtype=wp.float64), V_o: wp.array(dtype=wp.float64), pole_o: wp.array(dtype=wp.vec3d),
                flag: wp.array(dtype=wp.int32), nsites: int, AW: int, DN: int):
    _site(p, r, off, ent, npr, nn_o, u_o, v_o, rho_o, hi_o, hj_o, at0, at1, narc, fib, nfib, piece, big, fourpi, nbrs, hidden, stable_area,
          floor_enabled, floor_height, site_ids, indexed, Vw, pn_o, pA_o, V_o, pole_o, flag, nsites, AW, DN, wp.tid())


def site(dim, inputs, outputs, device=None):
    """k_geo_site(dim, inputs, outputs) by k_site_coop; the sites it could not hold (flagged: one count read) by
    k_geo_site itself, indexed by their ids"""
    if dim == 0:
        return
    import torch
    at0, nbrs, p = inputs[11], inputs[19], inputs[0]
    assert inputs[15] == 32, 'k_site_coop: one lane a Fibonacci candidate'
    key = (p.device.alias, len(p))
    flag = site.flags.get(key)
    if flag is None:
        flag = wp.zeros(len(p), dtype=wp.int32, device=p.device)
        site.flags[key] = flag
    wp.launch(k_site_coop, dim*32, inputs=list(inputs)+list(outputs)+[flag, dim, at0.shape[1], nbrs.shape[1]], block_dim=32*WPB, device=device)
    ft = wp.to_torch(flag)
    site.last = int(ft.sum())
    site.ids = torch.nonzero(ft).flatten().to(torch.int32) if site.last else None
    if site.last:
        ft.zero_()
        if 'site_ids' in [a.label for a in GW.k_geo_site.adj.args]:
            ids = site.ids
            ins = list(inputs); ins[24] = wp.from_torch(ids, dtype=wp.int32); ins[25] = 1
            wp.launch(GW.k_geo_site, len(ids), inputs=ins, outputs=list(outputs), device=device)
        else:                                                                     # (k1's k_geo_site: no site_ids; all sites again)
            wp.launch(GW.k_geo_site, dim, inputs=list(inputs[:21])+[inputs[26]], outputs=list(outputs), device=device)


site.flags = {}
site.last = 0
