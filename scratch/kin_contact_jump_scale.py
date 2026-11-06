"""the switch jump of the kinematic closure (kin_contact_phase12.py's phase 2: f(0) = E_c(x, C_free) / d at a cap's
lowest point) against what the contact there actually carries: the original model with the cap contact (alpha 30, the
run of out_inverse/pacnerf_e0_show_capc30) to frame 6; at that step per touching cell B the contact force F_B = sum over
its points of (f_n n + f_t) (the midpoint rule's forces, per unit density as f) and the cell's momentum V_B v_B. Reports
per touching cell |f_n(0)| / F_n,B, |f_t(0)| / |F_t,B|, and the jump's one-step impulse |f(0)| dt against the cell's
momentum V_B |v_B| and against the contact's one-step impulse |F_B| dt. args: frame (default 6)"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
f64, dev = torch.float64, 'cuda'
FR = int(sys.argv[1]) if len(sys.argv) > 1 else 6
li.set_scene('pacnerf_e0'); li.CAPC = 30.
S0 = FR*li.SPF
dump = {}


class Stop(Exception):
    pass


def probe(s, W, ST, cbody, Rb):
    if s < S0:
        return
    cq = W.tcq
    dump.update(x=ST.tx[:W.n].clone(), v=ST.tv[:W.n].clone(), cb=cq['cb'].long().clone(), fnq=cq['fnq'].clone(), ffq=cq['ffq'].clone(), V=W.st.V.clone())
    raise Stop


try:
    li.forward(np.array([10.04361, .30079413, -5.809143]), FR+1, verbose=False, on_step=probe)
except Stop:
    pass
n = len(dump['x']); UP = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
Fn = torch.zeros(n, dtype=f64, device=dev).index_add_(0, dump['cb'], dump['fnq'])
Ft = torch.zeros(n, 3, dtype=f64, device=dev).index_add_(0, dump['cb'], dump['ffq'])
touch = torch.nonzero(Fn > 0)[:, 0]
print('frame %d (step %d): touching cells %d, total normal force %.4g, |tangential| %.4g (per unit density); dt %.3e' % (
    FR, S0, len(touch), float(Fn.sum()), float(Ft.sum(0).norm()), li.DT), flush=True)

# ---- the jump at the same state (kin_contact_phase12's machinery)
sys.argv = [sys.argv[0]]
src = open(ROOT/'scratch'/'kin_contact_phase12.py').read().split('# ---------------------------------------------------------------- configurations')[0]
exec(compile(src, 'kin_contact_phase12', 'exec'))
x = dump['x']; C = C_free(x); e0, Ex, Ec = E_parts(x, C)
rows = []
for B in touch.tolist():
    if not bool(has_pts[B]):
        continue
    r, d, y = lowest_point(x, C, B)
    f = Ec[B]/d; fn0 = float(f@UP); ft0 = float((f-fn0*UP).norm())
    FnB = float(Fn[B]); FtB = float(Ft[B].norm()); FB = float((Fn[B]*UP+Ft[B]).norm())
    mom = float(dump['V'][B]*dump['v'][B].norm())
    rows.append((B, fn0, ft0, FnB, FtB, float(f.norm())*li.DT/mom, float(f.norm())/FB, float(y[1])-li.C0))
R = np.array(rows)
_, fn0, ft0, FnB, FtB, imp_mom, imp_con, h = R.T
q = lambda a: '%.3g [%.3g, %.3g]' % (np.median(a), np.percentile(a, 10), np.percentile(a, 90))
print('over the %d touching cells (median [10 %%, 90 %%]):' % len(R))
print('  actual normal force F_n,B %s, |F_t,B| %s; the jump f_n(0) %s (pushing on %d, pulling on %d), |f_t(0)| %s' % (
    q(FnB), q(FtB), q(fn0), (fn0 > 0).sum(), (fn0 < 0).sum(), q(ft0)))
print('  |f_n(0)| / F_n,B %s; |f_t(0)| / |F_t,B| %s; |f(0)| / |F_B| %s' % (q(np.abs(fn0)/FnB), q(ft0/np.maximum(FtB, 1e-30)), q(imp_con)))
print('  the jump\'s one-step impulse |f(0)| dt / the cell\'s momentum V_B |v_B| %s' % q(imp_mom))
print('  (the contact\'s own one-step impulse |F_B| dt / V_B |v_B| %s)' % q(imp_mom/imp_con))
wN = FnB/FnB.sum()
print('  weighted by F_n,B: |f_n(0)| / F_n,B %.3f, |f(0)| / |F_B| %.3f; the jumps\' sum over the cells %.4g (normal, signed %+.4g) against the total normal force %.4g' % (
    (wN*np.abs(fn0)/FnB).sum(), (wN*imp_con).sum(), np.abs(fn0).sum(), fn0.sum(), FnB.sum()))
np.savez(str(ROOT/'out_inverse'/('kin_contact_jump_scale_f%d.npz' % FR)), rows=R, cols='cell, f_n(0), |f_t(0)|, F_n,B, |F_t,B|, |f(0)| dt / V|v|, |f(0)| / |F_B|, lowest point height')
print('saved out_inverse/kin_contact_jump_scale_f%d.npz' % FR)
