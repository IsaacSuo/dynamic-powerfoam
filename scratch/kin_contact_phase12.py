"""the kinematic closure of a touching cap's column with the fitted skin's own energy (the reviewer's task, phases 1 and 2;
PAC-NeRF elastic scene 0's foam, GT E, nu, no viscosity). E(x, c): the model's energy with the cap columns exposed as
inputs (ClosureWarp.set_cap_columns' kernel; every other cell's column held at its free completion C_free(x), read from
the original model's F_s m), so E(x, C_free(x)) is the original model (checked). One cap point r of cell B:
    d = m . r,  y0(x) = x_B + G P r = (1 - sum_e w_e) x_B + sum_e w_e x_k  (w_e = dX_e^T M2^-1 P r),  y = y0 + d c;
held at a: c = (a - y0) / d, Ehat(x; a) = E(x, C_contact), f = E_c / d, grad Ehat . v = E_x . v + sum_{k != B} E_ck . (D C_free,k v) - f . (A v), the partials at the contact state.
Phase 1 (the algebra, at a chosen configuration): y = a; path A (central differences of Ehat along v) against path B (the
formula; D C_free v by central differences of the kinematic C_free, the energy partials analytic); dEhat/da against f;
the distribution coefficients sum to 1, their moment = a x f.
Phase 2 (the switch): the point just touching (a = its free position): c_contact = c_free, Ehat = E, but the reaction
f = E_c(x, C_free) / d; pressing the body rigidly by delta: Ehat(delta) = E(x, c_free + delta up / d) (translation
invariance), f(delta); the jump at the switch f(0) against the pressing stiffness k = df_n/ddelta, as an equivalent depth
delta* = f_n(0) / k. Configurations: rest (just touching), uniform uniaxial compression 10 % (free completion), the
impact state (frame 6 of out_inverse/pacnerf_e0_show_capc30, the original model with the cap contact). Writes
out_inverse/kin_contact_phase12.npz"""
import math
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
from experiments.pacnerf_dem_forward import build
from free_surface.cap_contact import cap_quadrature
from free_surface.closure_warp import ClosureWarp
from free_surface.sphere_closure import ClosureTerms
f64, dev = torch.float64, 'cuda'
li.set_scene('pacnerf_e0')
cells, g, site_np = li.load_foam()
m_ = build(g, li.GT['E'], li.GT['nu'], dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
UP = torch.tensor([0., 1., 0.], dtype=f64, device=dev)


def make(cols):
    st = ClosureTerms(m_, g, cells['points'], cells['radii'])
    W = ClosureWarp(st)
    W.set_viscosity(0., 0., li.DT); W.visc_weight = 2.; W.total_mode()
    site = torch.as_tensor(site_np, device=dev).long()
    W.set_ball(site, torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site], (0., 1., 0.), -10., 0.)
    W.set_cap_positions(); W.set_cap_floor(0.)
    if cols:
        W.set_cap_columns()
    W.mass_mode = 'lumped'
    W.prepare()
    return W, st


Wf, st = make(False); Wc, _ = make(True)
n = Wf.n; mA, PA = st.mA, st.PA; cidx = Wc._cidx_t.long()
P_ = Wc._pos; off, pk, dX, M2i = P_['off'], P_['pk'], P_['dX'], P_['M2i']
cl, roff, wq = cap_quadrature(cells['points'], cells['radii'], st.c_body.cpu().numpy(), cache=str(li.FOAM/'dem_exact_single'/'cap_quadrature.npz'), verbose=False)
cl = torch.as_tensor(cl, device=dev).long(); roff = torch.as_tensor(roff, dtype=f64, device=dev)
mu, lam = Wf.mu, Wf.lam
has_pts = torch.zeros(n, dtype=torch.bool, device=dev); has_pts[cl] = True
CAPS = [B for B in Wc.cap_cells.tolist() if bool(has_pts[B])]                       # (caps with quadrature points: 183 tiny ones have none)


def C_free(x):
    """every cell's free-completion column from the original model: F_s m (zero off the caps)"""
    Wf.grad(x)
    return torch.einsum('nij,nj->ni', Wf.tFs, mA)


def E_old(x):
    e, gr = Wf.grad(x)
    return float(e), gr.clone()


def E_parts(x, C):
    """E(x, C) with the columns C (n, 3; the cap cells' rows used): E, E_x (n, 3), E_c (n, 3; zero off the caps)"""
    q = torch.cat([x, C[Wc.cap_cells]], 0)
    e, gr = Wc.grad(q)
    Ec = torch.zeros(n, 3, dtype=f64, device=dev); Ec[Wc.cap_cells] = gr[n:]
    return float(e), gr[:n].clone(), Ec


def G_of(x, B):
    e = slice(int(off[B]), int(off[B+1]))
    return ((x[pk[e]]-x[B])[:, :, None]*dX[e][:, None, :]).sum(0)@M2i[B]


def y0_and_A(x, B, r):
    """y0 = x_B + G P r and its distribution: (site indices, coefficients) with sum 1"""
    e = slice(int(off[B]), int(off[B+1]))
    w = dX[e]@(M2i[B]@(PA[B]@r))
    idx = torch.cat([torch.tensor([B], device=dev), pk[e]]); coef = torch.cat([1-w.sum()[None], w])
    y0 = (coef[:, None]*x[idx]).sum(0)
    return y0, idx, coef


def lowest_point(x, C, B):
    """cell B's cap point lowest in the free completion: (r, d, its position)"""
    r = roff[cl == B]
    y0 = x[B][None]+(G_of(x, B)@PA[B]@r.T).T
    d = r@mA[B]
    y = y0+d[:, None]*C[B][None]
    j = int(torch.argmin(y[:, 1]))
    return r[j], float(d[j]), y[j]


def Ehat(x, C, B, r, d, a):
    """E with cell B's column from the kinematic closure at a (the others at C)"""
    Cc = C.clone(); y0, _, _ = y0_and_A(x, B, r); Cc[B] = (a-y0)/d
    return E_parts(x, Cc)


gen = torch.Generator(device='cpu').manual_seed(0)
rnd = lambda *s: torch.randn(*s, generator=gen, dtype=f64).to(dev)
out = {}


def phase1(name, x, B):
    print('\n== phase 1 (%s), cell %d' % (name, B), flush=True)
    C = C_free(x); r, d, yf = lowest_point(x, C, B)
    a = yf.clone()                                                                  # (held where the free completion put it, then moved)
    a = a+torch.tensor([.5e-3, -1e-3, -.3e-3], dtype=f64, device=dev)
    print('   d = m . r = %.4f m (|r| %.4f), the free point %s, held at a %s' % (d, float(r.norm()), np.round(yf.cpu().numpy(), 4), np.round(a.cpu().numpy(), 4)))
    y0, idx, coef = y0_and_A(x, B, r)
    cc = (a-y0)/d
    print('   closure: |y0 + d c - a| = %.2e m; distribution coefficients: %d sites, sum %.12f, |coef| max %.3f' % (float((y0+d*cc-a).norm()), len(idx), float(coef.sum()), float(coef.abs().max())))
    e_hat, Ex, Ec = Ehat(x, C, B, r, d, a); f = Ec[B]/d
    e_old, g_old = E_old(x)
    print('   E_old %.8e, Ehat %.8e; f = E_c / d = %s (|f| %.4g, per unit density)' % (e_old, e_hat, np.round(f.cpu().numpy(), 5), float(f.norm())))
    # path A vs path B along directions
    ring = torch.zeros(n, dtype=torch.bool, device=dev); ring[idx] = True
    dirs = [('all sites', rnd(n, 3)), ('cell + ring', torch.where(ring[:, None], rnd(n, 3), torch.zeros(n, 3, dtype=f64, device=dev))),
            ('translation', UP.expand(n, 3).clone()), ('cell B only', torch.zeros(n, 3, dtype=f64, device=dev))]
    dirs[3][1][B] = rnd(3)
    rows = []
    for nm, v in dirs:
        v = v/v.abs().max(); h = 1e-6
        eA = (Ehat(x+h*v, C_free(x+h*v), B, r, d, a)[0]-Ehat(x-h*v, C_free(x-h*v), B, r, d, a)[0])/(2*h)
        DC = (C_free(x+h*v)-C_free(x-h*v))/(2*h); DC[B] = 0.                     # (every other cell's free column along v; B's is the closure's)
        Av = (coef[:, None]*v[idx]).sum(0)
        eB = float((Ex*v).sum())+float((Ec*DC).sum())-float(f@Av)                 # (the partials at the contact state)
        rows.append((nm, eA, eB)); print('   %-12s path A (FD of Ehat) %+.10e, path B (formula) %+.10e, rel. diff %.1e' % (nm, eA, eB, abs(eA-eB)/max(abs(eA), 1e-300)))
    # dEhat/da against f; virtual work of the contact force; moment
    fd = torch.zeros(3, dtype=f64, device=dev); h = 1e-7
    for k in range(3):
        ek = torch.zeros(3, dtype=f64, device=dev); ek[k] = h
        fd[k] = (Ehat(x, C, B, r, d, a+ek)[0]-Ehat(x, C, B, r, d, a-ek)[0])/(2*h)
    print('   dEhat/da (FD) %s against f %s: rel. diff %.1e' % (np.round(fd.cpu().numpy(), 6), np.round(f.cpu().numpy(), 6), float((fd-f).norm()/f.norm())))
    F = coef[:, None]*f[None]                                                         # (the force on the sites: the contact force distributed)
    mom = torch.linalg.cross(x[idx], F).sum(0); print('   distributed force: sum %s = f; moment about the origin %s against y0 x f %s (|diff| %.1e; the resultant acts at y0, |y0 - a| %.2f mm)' % (
        np.round(F.sum(0).cpu().numpy(), 6), np.round(mom.cpu().numpy(), 7), np.round(torch.linalg.cross(y0, f).cpu().numpy(), 7), float((mom-torch.linalg.cross(y0, f)).norm()), 1e3*float((y0-a).norm())))
    out[name+'_phase1'] = np.array([[ea, eb] for _, ea, eb in rows])


def phase2(name, x, nsel=40, press=None):
    """f(0) = E_c(x, C_free) / d at the lowest point of each of the nsel lowest cells; the pressing slice for `press` cells"""
    print('\n== phase 2 (%s)' % name, flush=True)
    C = C_free(x); e0, Ex, Ec = E_parts(x, C)
    e_old = E_old(x)[0]
    print('   E(x, C_free) %.8e against the original model %.8e (|diff| %.1e); max |E_c| over the caps %.3e' % (e0, e_old, abs(e0-e_old), float(Ec.norm(dim=1).max())))
    capc = Wc.cap_cells
    low = []
    for B in CAPS:
        r, d, y = lowest_point(x, C, B); low.append((float(y[1]), B, r, d))
    low.sort(key=lambda t: t[0])
    sel = low[:nsel]; rec = []
    for yl, B, r, d in sel:
        f = Ec[B]/d; fn = float(f@UP); ft = float((f-fn*UP).norm())
        # the pressing stiffness: Ehat(delta) = E(x, c_free + delta up / d): f_n at delta = 1 mm
        Cd = C.clone(); Cd[B] = C[B]+1e-3*UP/d
        _, _, Ecd = E_parts(x, Cd); fn1 = float(Ecd[B]@UP)/d
        k = (fn1-fn)/1e-3; dstar = fn/k if k > 0 else float('nan')
        rec.append((B, yl, d, fn, ft, k, dstar))
    rec = np.array(rec)
    print('   the %d lowest cap points (cell, height over the lowest, d, f_n(0), |f_t(0)|, k = df_n/ddelta, delta* = f_n(0)/k):' % nsel)
    for row in rec[:12]:
        print('     %5d  %+6.2f mm  d %.3f  f_n %+9.4g  f_t %9.4g  k %9.4g /m  delta* %+8.3f mm' % (int(row[0]), 1e3*(row[1]-rec[0, 1]), row[2], row[3], row[4], row[5], 1e3*row[6]))
    fn, ft, k, ds = rec[:, 3], rec[:, 4], rec[:, 5], rec[:, 6]
    print('   over the %d: f_n(0) pushing (> 0) %d, pulling (< 0) %d; |f_n(0)| median %.3g, |f_t(0)| median %.3g; k median %.3g; |delta*| median %.2f mm, max %.2f mm; k <= 0 on %d' % (
        nsel, (fn > 0).sum(), (fn < 0).sum(), np.median(np.abs(fn)), np.median(ft), np.median(k), 1e3*np.nanmedian(np.abs(ds)), 1e3*np.nanmax(np.abs(ds)), (k <= 0).sum()))
    out[name+'_phase2'] = rec
    if press is not None:
        for B in press:
            B = int(B); r, d, y = lowest_point(x, C, B)
            print('   pressing cell %d (lowest point held, the body moved rigidly): delta mm | Ehat - E_free | f_n | f_t' % B)
            path = []
            for delta in [0., .5e-3, 1e-3, 2e-3, 3e-3, 2e-3, 1e-3, .5e-3, 0.]:
                Cd = C.clone(); Cd[B] = C[B]+delta*UP/d
                e, _, Ecd = E_parts(x, Cd); f = Ecd[B]/d; fn_ = float(f@UP)
                path.append((delta, e-e0, fn_, float((f-fn_*UP).norm())))
                print('     %5.2f | %+.4e | %+9.4g | %9.4g' % (1e3*delta, e-e0, fn_, float((f-fn_*UP).norm())))
            out['%s_press_%d' % (name, B)] = np.array(path)


# ---------------------------------------------------------------- configurations
X = st.X.clone()
# (a) rest, just touching: translate so the lowest free cap point is on the plane y = 0
C0 = C_free(X); ymin = min(lowest_point(X, C0, B)[2][1] for B in CAPS)
xa = X.clone(); xa[:, 1] -= float(ymin)
Ba = min(CAPS, key=lambda B: float(lowest_point(xa, C0, B)[2][1]))
# (b) uniform uniaxial compression 10 % (lateral free), free completion
la = .9; lt = 1.
for _ in range(100):
    fv = 2*mu*lt*(lt-1)+lam*math.log(la*lt*lt); lt -= fv/(2*mu*(2*lt-1)+2*lam/lt)
F0 = torch.diag(torch.tensor([lt, la, lt], dtype=f64, device=dev)); c_ = X.mean(0)
xb = c_+(X-c_)@F0.T
# (c) the impact state: frame 6 of the original model's run with the cap contact
xc = torch.as_tensor(np.load(ROOT/'out_inverse'/'pacnerf_e0_show_capc30'/'show_frames.npz')['x'][6], dtype=f64, device=dev)
print('configurations: (a) rest just touching (cell %d lowest); (b) uniaxial 10 %% (lt %.4f); (c) impact frame 6 (sites\' lowest height over the ground %.1f mm)' % (
    Ba, lt, 1e3*(float(xc[:, 1].min())-li.C0)), flush=True)
phase1('rest', xa, Ba)
Cc_ = C_free(xc); Bc = min(CAPS, key=lambda B: float(lowest_point(xc, Cc_, B)[2][1]))
phase1('impact', xc, Bc)
phase2('rest', xa, press=[Ba])
phase2('uniaxial', xb, press=None)
phase2('impact', xc, press=[Bc])
np.savez(ROOT/'out_inverse'/'kin_contact_phase12.npz', **out)
print('\nsaved out_inverse/kin_contact_phase12.npz', flush=True)
