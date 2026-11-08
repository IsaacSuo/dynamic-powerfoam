"""Warp-cooperative clip (11-08; the default clip of AlphaComplex / GeoPipe / TargetCut, A.coop = False for the original): alpha_complex.k_alpha with one 32-lane warp per cell
and the polytope in shared memory, as a Warp native CUDA snippet (NVRTC, no nvcc). Same predicates, same order, same
outputs (walls, links, nwall, nlink, status) bit for bit; the scratch arrays (P, Pid, T, V, ...) are not written (no
caller reads them). Vertex slots t = 32 k + lane; the f64 vertex is recomputed from its three planes when the float32
test falls in the tolerance band (it always equals vert3 of its triple); the hole's boundary, the compaction and the new
vertices by ballots / prefix sums in the original (t, e) order; the walls phase one lane per face, then the faces' order
for the breaks (status 4 / 6) and 5. Candidates read 32 at a time into shared memory.
  make_pair() -> tier 1 (PC 64 planes, TC 128 vertices, CN 48 face edges; beyond: status 7), tier 2 (PMAX 192, TMAX 512,
  only the status-7 cells; its overflow status 2 / 3 as the original).
  clip(r, coff, cj, cn, ch, walls, links, nwall, nlink, status, n): both tiers.
Checked (alpha_coop_check.py): 70 states (ALE gravity drop 17 frames x 3 offsets, k1 plasticine0 19 frames) identical;
6.2 -> 3.0 ms (ALE), 6.0-6.3 -> 2.6-3.3 ms (k1, 1-2 tier-2 cells after the impact), RTX 5070 Ti laptop.
Diagnostic flags of make: NOWALL (skip the walls phase), FASTV (vertices in float32: timing only, outputs differ), CNT."""
import sys
import warp as wp

SNIP_T = r'''
#define FULLM 0xffffffffu
#define VERT3 %(V3)s
const int WPB = %(WPB)d, PC = %(PC)d, TC = %(TC)d, CN = %(CN)d, NOWALL = %(NOWALL)d, CNT = %(CNT)d, NCH = %(TC)d/32, TIER2 = %(TIER2)d;
typedef wp::vec_t<3,double> v3d;
typedef wp::vec_t<3,float> v3f;
__shared__ double sP[WPB][PC][4];
__shared__ int sPid[WPB][PC];
__shared__ int sT[WPB][TC][3];
__shared__ float sV32[WPB][TC][3];
__shared__ int sEA[WPB][TC];
__shared__ int sEB[WPB][TC];
__shared__ double sCH[WPB][32];
__shared__ double sCN[WPB][32][3];
__shared__ int sCJ[WPB][32];
const int w = threadIdx.x >> 5, L = threadIdx.x & 31;
const int i = tid >> 5;
if (i >= n) return;
if (TIER2 && status.data[i] != 7) return;
auto vert3 = [&](int a, int b, int c) -> v3d {
    v3d na(sP[w][a][0], sP[w][a][1], sP[w][a][2]);
    v3d nb(sP[w][b][0], sP[w][b][1], sP[w][b][2]);
    v3d nc(sP[w][c][0], sP[w][c][1], sP[w][c][2]);
    v3d bc = wp::cross(nb, nc); v3d ca = wp::cross(nc, na); v3d ab = wp::cross(na, nb);
    double det = wp::dot(na, bc);
    return wp::div(wp::add(wp::add(wp::mul(sP[w][a][3], bc), wp::mul(sP[w][b][3], ca)), wp::mul(sP[w][c][3], ab)), det);
};
auto vert3f = [&](int a, int b, int c) -> v3d {
    v3f na((float)sP[w][a][0], (float)sP[w][a][1], (float)sP[w][a][2]);
    v3f nb((float)sP[w][b][0], (float)sP[w][b][1], (float)sP[w][b][2]);
    v3f nc((float)sP[w][c][0], (float)sP[w][c][1], (float)sP[w][c][2]);
    v3f bc = wp::cross(nb, nc); v3f ca = wp::cross(nc, na); v3f ab = wp::cross(na, nb);
    float det = wp::dot(na, bc);
    v3f r_ = wp::div(wp::add(wp::add(wp::mul((float)sP[w][a][3], bc), wp::mul((float)sP[w][b][3], ca)), wp::mul((float)sP[w][c][3], ab)), det);
    return v3d(r_[0], r_[1], r_[2]);
};
auto segd = [&](v3d a, v3d b) -> double {
    v3d e = wp::sub(b, a);
    double ee = wp::dot(e, e);
    double t = 0.0;
    if (ee > 0.0) t = wp::clamp(wp::div(wp::neg(wp::dot(a, e)), ee), 0.0, 1.0);
    v3d q = wp::add(a, wp::mul(t, e));
    return wp::dot(q, q);
};
const double R = r.data[i];
const int* co = (const int*)coff.data; const int* cjp = (const int*)cj.data;
const double* cnp = (const double*)cn.data; const double* chp = (const double*)ch.data;
// the cube: planes 0..5, its 8 corners
if (L < 6) {
    int ax = L >> 1; double s = (L & 1) ? -1.0 : 1.0;
    sP[w][L][0] = ax == 0 ? s : 0.0; sP[w][L][1] = ax == 1 ? s : 0.0; sP[w][L][2] = ax == 2 ? s : 0.0; sP[w][L][3] = R;
    sPid[w][L] = -1;
}
__syncwarp();
if (L < 8) {
    int sx = L >> 2, sy = (L >> 1) & 1, sz = L & 1;
    int a = sx, b = 2+sy, c = 4+sz;
    int t0 = a, t1 = b, t2 = c;
    if ((sx+sy+sz) %% 2 == 1) { t0 = b; t1 = a; }
    sT[w][L][0] = t0; sT[w][L][1] = t1; sT[w][L][2] = t2;
    v3d v = VERT3(t0, t1, t2);
    sV32[w][L][0] = (float)(v[0]); sV32[w][L][1] = (float)(v[1]); sV32[w][L][2] = (float)(v[2]);
}
__syncwarp();
int nt = 8, nplane = 6, st = 0, ovf = 0, origin_in = 1;
double R2 = 3.0*R*R;
float sR2 = (float)(wp::sqrt(R2));
const int cend = co[i+1];
int done = 0, nvis = 0, ncut = 0;
for (int c0 = co[i]; c0 < cend && !done; c0 += 32) {
  __syncwarp();
  if (c0+L < cend) { int cc = c0+L; sCH[w][L] = chp[cc]; sCN[w][L][0] = cnp[3*cc]; sCN[w][L][1] = cnp[3*cc+1]; sCN[w][L][2] = cnp[3*cc+2]; sCJ[w][L] = cjp[cc]; }
  __syncwarp();
  const int lim = min(32, cend-c0);
  for (int cb = 0; cb < lim; ++cb) {
    double h = sCH[w][cb]; nvis++;
    if (h > 0.0 && h*h >= R2) { done = 1; break; }
    v3d nv(sCN[w][cb][0], sCN[w][cb][1], sCN[w][cb][2]);
    v3f n32((float)(nv[0]), (float)(nv[1]), (float)(nv[2]));
    float h32 = (float)(h);
    float tol = 9.5367431640625e-07f*(sR2+wp::abs(h32));
    unsigned rw[NCH]; int nrem = 0;
    #pragma unroll
    for (int k = 0; k < NCH; ++k) {
        int t = k*32+L; int o = 0;
        if (k*32 >= nt) { rw[k] = 0u; continue; }
        if (t < nt) {
            v3f vt(sV32[w][t][0], sV32[w][t][1], sV32[w][t][2]);
            float d32 = wp::dot(n32, vt)-h32;
            if (d32 > tol) o = 1;
            else if (d32 >= -tol) { if (wp::dot(nv, VERT3(sT[w][t][0], sT[w][t][1], sT[w][t][2])) > h) o = 1; }
        }
        rw[k] = __ballot_sync(FULLM, o); nrem += __popc(rw[k]);
    }
    if (nrem == 0) continue;
    if (h <= 0.0) origin_in = 0;
    if (nrem == nt) { nt = 0; done = 1; break; }
    if (nplane >= PC) { if (TIER2) st = 2; else ovf = 1; done = 1; break; }
    int q = nplane; ncut++;
    if (L == 0) { sP[w][q][0] = nv[0]; sP[w][q][1] = nv[1]; sP[w][q][2] = nv[2]; sP[w][q][3] = h; sPid[w][q] = sCJ[w][cb]; }
    nplane++;
    __syncwarp();
    // the hole's boundary, in (t, e) order
    int base = 0;
    #pragma unroll
    for (int k = 0; k < NCH; ++k) {
        if (k*32 >= nt) break;
        int t = k*32+L; int fl = 0, cb_ = 0;
        int x0 = 0, x1 = 0, x2 = 0;
        if ((rw[k] >> L) & 1u) {
            x0 = sT[w][t][0]; x1 = sT[w][t][1]; x2 = sT[w][t][2];
            for (int e = 0; e < 3; ++e) {
                int x = e == 0 ? x0 : (e == 1 ? x1 : x2), y = e == 0 ? x1 : (e == 1 ? x2 : x0);
                int inner = 0;
                for (int k2 = 0; k2 < NCH && !inner; ++k2) {
                    unsigned m2 = rw[k2];
                    while (m2) {
                        int b = __ffs(m2)-1; m2 &= m2-1u; int t2 = k2*32+b;
                        if (t2 != t) {
                            int u0 = sT[w][t2][0], u1 = sT[w][t2][1], u2 = sT[w][t2][2];
                            if ((u0 == y && u1 == x) || (u1 == y && u2 == x) || (u2 == y && u0 == x)) { inner = 1; break; }
                        }
                    }
                }
                if (!inner) { fl |= 1 << e; cb_++; }
            }
        }
        int inc = cb_;
        for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(FULLM, inc, o); if (L >= o) inc += v; }
        int tot = __shfl_sync(FULLM, inc, 31);
        int pos = base+inc-cb_;
        if (fl & 1) { sEA[w][pos] = x0; sEB[w][pos] = x1; pos++; }
        if (fl & 2) { sEA[w][pos] = x1; sEB[w][pos] = x2; pos++; }
        if (fl & 4) { sEA[w][pos] = x2; sEB[w][pos] = x0; pos++; }
        base += tot;
    }
    int nb = base;
    // compact the kept triangles (stable), then close the hole
    int kt[NCH][3]; float kv[NCH][3]; int np_[NCH]; int kp[NCH]; int m = 0;
    #pragma unroll
    for (int k = 0; k < NCH; ++k) {
        int t = k*32+L;
        kp[k] = 0;
        if (k*32 >= nt) continue;
        kp[k] = (t < nt) && !((rw[k] >> L) & 1u);
        unsigned km = __ballot_sync(FULLM, kp[k]);
        np_[k] = m+__popc(km & ((1u << L)-1u)); m += __popc(km);
        if (kp[k]) { kt[k][0] = sT[w][t][0]; kt[k][1] = sT[w][t][1]; kt[k][2] = sT[w][t][2]; kv[k][0] = sV32[w][t][0]; kv[k][1] = sV32[w][t][1]; kv[k][2] = sV32[w][t][2]; }
    }
    if (m+nb > TC) { if (TIER2) st = 3; else ovf = 1; done = 1; break; }
    __syncwarp();
    #pragma unroll
    for (int k = 0; k < NCH; ++k) if (kp[k]) { int d = np_[k]; sT[w][d][0] = kt[k][0]; sT[w][d][1] = kt[k][1]; sT[w][d][2] = kt[k][2]; sV32[w][d][0] = kv[k][0]; sV32[w][d][1] = kv[k][1]; sV32[w][d][2] = kv[k][2]; }
    __syncwarp();
    for (int e = L; e < nb; e += 32) {
        int a = sEA[w][e], b = sEB[w][e], t = m+e;
        sT[w][t][0] = a; sT[w][t][1] = b; sT[w][t][2] = q;
        v3d vn = VERT3(a, b, q);
        sV32[w][t][0] = (float)(vn[0]); sV32[w][t][1] = (float)(vn[1]); sV32[w][t][2] = (float)(vn[2]);
    }
    nt = m+nb;
    __syncwarp();
    float R2f = 0.0f;
    #pragma unroll
    for (int k = 0; k < NCH; ++k) { int t = k*32+L; if (t < nt) { v3f vt(sV32[w][t][0], sV32[w][t][1], sV32[w][t][2]); R2f = wp::max(R2f, wp::dot(vt, vt)); } }
    for (int o = 16; o > 0; o >>= 1) R2f = wp::max(R2f, __shfl_xor_sync(FULLM, R2f, o));
    R2 = (double)(R2f)*1.00000095367431640625;
    sR2 = (float)(wp::sqrt(R2));
  }
}
// walls and links in the ball: one lane per face, then the faces' order
int nw = 0;
const double R2b = R*R;
int* wl = (int*)walls.data+(size_t)i*WM; int* ll = (int*)links.data+(size_t)i*WM*LM; int* nll = (int*)nlink.data+(size_t)i*WM;
if (!ovf && st == 0 && nt > 0 && !NOWALL) {
    int nface = nplane-6, any5 = 0, brk = 0;
    for (int k = 0; k*32 < nface; ++k) {
        int j = k*32+L, f = 6+j, cnt = 0, start = -1, closed = 1, wall = 0, bad = 0, lov = 0;
        int kl[CN], lv[CN], el[CN]; double ed[CN];
        if (j < nface) {
            for (int t = 0; t < nt; ++t) {
                int a0 = sT[w][t][0], a1 = sT[w][t][1], a2 = sT[w][t][2];
                int kk_ = -1, l_ = -1;
                if (a0 == f) { kk_ = a1; l_ = a2; } else if (a1 == f) { kk_ = a2; l_ = a0; } else if (a2 == f) { kk_ = a0; l_ = a1; }
                if (kk_ >= 0) { if (cnt < CN) { kl[cnt] = kk_; lv[cnt] = l_; } start = kk_; cnt++; }
            }
            if (cnt > CN) lov = 1;
            else if (cnt >= 3) {
                auto succ = [&](int key) -> int { for (int u = cnt-1; u >= 0; --u) if (kl[u] == key) return lv[u]; return -1; };
                v3d nf(sP[w][f][0], sP[w][f][1], sP[w][f][2]);
                double hf = sP[w][f][3];
                v3d q0 = wp::mul(hf, nf);
                double d2 = hf*hf;
                int pos = 0, neg = 0;
                double dmin = 1.0e300;
                int kk = start;
                int l = succ(kk);
                if (l < 0) closed = 0;
                else {
                    v3d va = VERT3(f, kk, l);
                    for (int s = 0; s < cnt; ++s) {
                        int lp = succ(l);
                        if (lp < 0) { closed = 0; break; }
                        v3d vb = VERT3(f, l, lp);
                        double sg = wp::dot(wp::cross(wp::sub(vb, va), wp::sub(q0, va)), nf);
                        if (sg > 0.0) pos++;
                        if (sg < 0.0) neg++;
                        double dd = segd(va, vb);
                        dmin = wp::min(dmin, dd);
                        el[s] = l; ed[s] = dd;
                        va = vb; kk = l; l = lp;
                    }
                }
                if (!closed) bad = 1;
                else {
                    if (pos == 0 || neg == 0) dmin = wp::min(dmin, d2);
                    if (dmin < R2b) wall = 1;
                }
            }
        }
        if (__any_sync(FULLM, lov)) { ovf = 1; break; }
        unsigned bm = __ballot_sync(FULLM, bad), wm = __ballot_sync(FULLM, wall);
        int jb = bm ? __ffs(bm)-1 : 32;
        unsigned wm2 = wm & (jb < 32 ? ((1u << jb)-1u) : FULLM);
        int nwc = __popc(wm2);
        int my = nw+__popc(wm2 & ((1u << L)-1u));
        int s5 = 0;
        if (((wm2 >> L) & 1u) && my < WM) {
            wl[my] = sPid[w][f];
            int nl = 0;
            for (int s = 0; s < cnt; ++s) {
                int l = el[s];
                if (ed[s] < R2b && sPid[w][l] >= 0) {
                    if (nl >= LM) s5 = 1;
                    else { ll[(size_t)my*LM+nl] = sPid[w][l]; nl++; }
                }
            }
            nll[my] = nl;
        }
        any5 |= __any_sync(FULLM, s5);
        if (nw+nwc > WM) { st = 4; nw = WM; brk = 1; break; }
        nw += nwc;
        if (jb < 32) { st = 6; brk = 1; break; }
    }
    if (!brk && any5) st = 5;
}
if (L == 0) {
    if (ovf) { status.data[i] = 7; }
    else {
        nwall.data[i] = nw;
        if (st == 0 && (nt == 0 || (nw == 0 && origin_in == 0))) st = 1;
        status.data[i] = st;
    }
    if (CNT) { nwall.data[i] = nvis; nll[0] = ncut; nll[1] = co[i+1]-co[i]; }
}
'''

_made = {}
def make(WPB=4, PC=64, TC=128, CN=48, NOWALL=0, FASTV=0, CNT=0, TIER2=0):
    key = (WPB, PC, TC, CN, NOWALL, FASTV, CNT, TIER2)
    if key in _made:
        return _made[key]
    snip = SNIP_T % dict(WPB=WPB, PC=PC, TC=TC, CN=CN, NOWALL=NOWALL, CNT=CNT, TIER2=TIER2, V3='vert3f' if FASTV else 'vert3')
    @wp.func_native(snip)
    def coop(r: wp.array(dtype=wp.float64), coff: wp.array(dtype=int), cj: wp.array(dtype=int), cn: wp.array(dtype=wp.vec3d), ch: wp.array(dtype=wp.float64),
             walls: wp.array2d(dtype=int), links: wp.array3d(dtype=int), nwall: wp.array(dtype=int), nlink: wp.array2d(dtype=int), status: wp.array(dtype=int),
             n: int, WM: int, LM: int, tid: int):
        ...
    @wp.kernel(enable_backward=False, module="unique")
    def k_coop(r: wp.array(dtype=wp.float64), coff: wp.array(dtype=int), cj: wp.array(dtype=int), cn: wp.array(dtype=wp.vec3d), ch: wp.array(dtype=wp.float64),
               walls: wp.array2d(dtype=int), links: wp.array3d(dtype=int), nwall: wp.array(dtype=int), nlink: wp.array2d(dtype=int), status: wp.array(dtype=int),
               n: int, WM: int, LM: int):
        coop(r, coff, cj, cn, ch, walls, links, nwall, nlink, status, n, WM, LM, wp.tid())
    _made[key] = (k_coop, WPB)
    return _made[key]


def make_pair():
    """tier 1 (small shared capacity) for all cells, tier 2 (the original kernel's capacity: PMAX 192 planes, TMAX 512
    vertices; its overflow reported as the original does, status 2 / 3) for the cells tier 1 marked 7"""
    return make(), make(WPB=1, PC=192, TC=512, CN=192, TIER2=1)


_PAIR = {}


def clip(r, coff, cj, cn, ch, walls, links, nwall, nlink, status, n, device=None):
    """alpha_complex.k_alpha's outputs (walls, links, nwall, nlink, status) for n cells from its candidate arrays (warp
    arrays, contiguous): tier 1 on every cell, then tier 2 on the cells tier 1 left at status 7"""
    if n == 0:
        return
    if 'pair' not in _PAIR:
        _PAIR['pair'] = make_pair()
    (k1, w1), (k2, w2) = _PAIR['pair']
    args = [r, coff, cj, cn, ch, walls, links, nwall, nlink, status, n, walls.shape[1], links.shape[2]]
    wp.launch(k1, n*32, inputs=args, block_dim=32*w1, device=device)
    wp.launch(k2, n*32, inputs=args, block_dim=32*w2, device=device)
