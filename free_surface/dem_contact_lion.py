"""hertz_drop.py's lion run (the foam in --out, GT E nu, mu 0, dt 1/1200, 1.2 s) at one beta, with the bodies' x and F_B saved
every --every steps (F_B: the closure at the step's end x, as closure_scene.Scene.run) for rendering"""
import argparse, math, time
import numpy as np, torch, warp as wp
from pathlib import Path
from experiments.pacnerf_dem_forward import build
from free_surface.closure_drop import colouring, probe_blocks
from free_surface.closure_warp import ClosureWarp, MidpointStep, TrustCG
from free_surface.sphere_closure import ClosureTerms
f64, dev = torch.float64, 'cuda'
a_ = argparse.ArgumentParser()
a_.add_argument('--beta', type=float, default=.03)
a_.add_argument('--every', type=int, default=20)
a_.add_argument('--T', type=float, default=1.2)
a_.add_argument('--out', type=Path, required=True)
a_.add_argument('--sweep-mult', type=int, default=1)
a_.add_argument('--tag', default='')
a_.add_argument('--self-contact', type=int, default=0)
a_.add_argument('--sc-sigma', type=float, default=0.)     # > 0: the colour code for the contacts (set_self_contact, 10-30)
a_.add_argument('--sc-cap', type=int, default=None)       # the contact list's capacity (default the rest walls)
a_.add_argument('--save', type=Path, default=None)        # where the frames go (default --out, the foam's directory)
a_.add_argument('--a-floor', type=float, default=0.)      # set_cap_floor: the floor on the one-way cap's normal (10-30)
a_.add_argument('--watch', default='')                     # cells to log at every step from --watch-from on
a_.add_argument('--watch-from', type=int, default=400)
a_.add_argument('--fine-until', type=float, default=0.)   # a frame every 4 steps up to this time
a_.add_argument('--cap', choices=('rim', 'pos', 'wall', 'ul'), default='pos')     # 10-29 user: the one-way cap by default  # pos: the one-way cap; dir: the free direction (set_free_direction)
args = a_.parse_args()
OUT = args.out; E, NU, DT, UP, C0, V0 = 10648.614517785627, .3912951704721983, 1/1200, 2, -.2, (0., 0., -.245)
cells = np.load(OUT/'dem_exact_single'/'cells.npz'); g = dict(np.load(OUT/'dem_exact_single'/'geometry.npz')); bsk = dict(np.load(OUT/'boundary_skin.npz'))
m = build(g, E, NU, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
ClosureWarp.sweep_mult = args.sweep_mult
st = ClosureTerms(m, g, cells['points'], cells['radii'], bsk); W = ClosureWarp(st)
W.set_viscosity(args.beta*W.mu, args.beta*(W.lam+2*W.mu/3), DT)
W.visc_weight = 2.; W.total_mode()
boc = torch.as_tensor(np.asarray(g['body_of_cell']), device=dev).long(); site = torch.as_tensor(bsk['patch_site'], device=dev).long()
cbody = boc[site]; Rb = torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site]
upt = tuple(1. if k == UP else 0. for k in range(3))
W.set_ball(cbody, Rb, upt, C0, 0.)
if args.self_contact:
    W.set_self_contact(torch.as_tensor(cells['radii'], dtype=f64, device=dev), K=args.sc_cap, sigma=args.sc_sigma)
    if args.self_contact == 2:                             # (2: only the cut for --cap ul, no contact force)
        W.selfc = None
if args.cap == 'ul':                                       # (UL's F, no fitted skin; needs --self-contact 1 --sc-sigma > 0)
    W.set_cap_ul()
if args.cap == 'pos':
    W.set_cap_positions()
    W.set_cap_floor(args.a_floor)
if args.cap == 'wall':
    W.set_cap_walls()
if args.cap == 'dir':
    W.set_free_direction()
W.prepare()
n = W.n; X = st.X; mass = m.mass.to(f64); M3 = mass[:, None]
up = torch.tensor(upt, dtype=f64, device=dev); fg = -9.8*M3*up; ref = float(fg.norm())
x = X.clone(); v = torch.tensor(V0, dtype=f64, device=dev).expand_as(x).clone()
if W.cap_ul:
    W.update_ul(x)                                         # (the map at rest, for the blocks below)
Kbb = probe_blocks(W, x, colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n))
W.visc_ref(x)
W.tball[0], W.tball[1] = 2., .5
if args.cap == 'dir':
    CG = JointTrustCG(W, mass, DT, per=2); ST = JointMidpointStep(W, CG, mass, Kbb, fg, DT, 0.)
    print('free direction: %d colours for the curvature probe, %d non-positive (floored)' % (W.dir_colours, W.dir_neg), flush=True)
else:
    CG = TrustCG(W, mass, DT, per=2, product_iters=1); ST = MidpointStep(W, CG, mass, Kbb, fg, DT, 0.)
tol = 1e-3*ref
ST.set_state(x, v, fg/M3)
xs, Fs, ts = [X.cpu().numpy()], [np.tile(np.eye(3), (n, 1, 1))], [0.]
Js = [np.ones(n)]
h = float(np.median(g['masses']))**(1/3); CL = [6068, 6607, 6954, 3323, 3209, 1541]; nb_seen, sp = 0, []
def self_report():
    """the self contacts born since the last record: rest distance in cell sizes, birth overlap / (r_i + r_k), the caps'
    facing n.m (the weaker of n.m_i and -n.m_k); the closure's failing cluster (6068 ...) in a pair"""
    global nb_seen
    B_ = W.self_births[nb_seen:]; nb_seen = len(W.self_births)
    P_ = W._pairs.cpu().numpy(); sp.append(np.c_[np.full(len(P_), len(ts)-1), P_])
    out = '; self contact pairs %d (born %d, left %d, cut failures %d); cluster cells in a pair %d' % (
        W.self_stats['pairs'], W.self_stats['born'], W.self_stats['left'], W.self_stats['cut_fail'], int(np.isin(CL, P_).sum()))
    if W.cap_ul:
        out += '; UL walls %d, radii iterations %d, radii failures %d, J floored %d' % (W.ul_stats['walls'], W.ul_stats['radii_it'], W.ul_stats['radii_fail'], W.ul_stats['floored'])
    if B_:
        B_ = torch.cat(B_).numpy(); rr = cells['radii']
        rd = B_[:, 4]/h; d0r = B_[:, 3]/(rr[B_[:, 1].astype(int)]+rr[B_[:, 2].astype(int)]); fc = np.minimum(B_[:, 5], -B_[:, 6])
        out += ('\n    born since the last record %d: rest distance / cell size min %.1f, median %.1f (below 3: %d); birth overlap / (r_i + r_k) '
                'median %.3f, max %.3f; facing n.m min %.3f, median %.3f' % (len(B_), rd.min(), np.median(rd), int((rd < 3).sum()),
                                                                           np.median(d0r), d0r.max(), fc.min(), np.median(fc)))
    return out
def save():
    sv = dict(x=np.array(xs), F=np.array(Fs), t=np.array(ts), J=np.array(Js), source_index=cells['source_index'])
    if W.selfc is not None:
        sv.update(self_births=torch.cat(W.self_births).numpy() if W.self_births else np.zeros((0, 7)), self_pairs=np.concatenate(sp))
    np.savez((args.save or OUT)/('lion_frames_beta%g%s.npz' % (args.beta, args.tag)), **sv)
def floor_report():
    """the frame's smallest fit area a over the caps, and the caps below the floor (q = n_t along the reference)"""
    G_ = wp.to_torch(W.B['G'], requires_grad=False); nt_ = torch.linalg.cross(torch.einsum('bij,bj->bi', G_, st.t1), torch.einsum('bij,bj->bi', G_, st.t2))[st.cap]
    q_ = (nt_*W._pos['nr'][st.cap]).sum(1)
    return '; min a %.4f, below the floor %d, of them q < 0: %d' % (float(nt_.norm(dim=1).min()), int((q_ < args.a_floor).sum()), int((q_ < 0).sum()))
t0 = time.time(); nsteps = int(round(args.T/DT))
for s in range(1, nsteps+1):
    prods, its, g2 = ST.step(tol, lambda g2_: .5*tol)
    if not math.isfinite(g2) or math.sqrt(g2) > tol or (args.cap == "dir" and not ST.dd <= ST.dd_tol):
        print('step %d: the solve did not converge (|grad| / weight %s after %d Newton iterations); stopping' % (s, math.sqrt(g2)/ref if math.isfinite(g2) else g2, its), flush=True)
        zf_ = wp.to_torch(ST.a['z'], requires_grad=False).clone(); W.grad(zf_)     # (the failing midpoint: its F_B, J)
        gf_ = W.tgx.norm(dim=1); of_ = torch.argsort(gf_, descending=True)[:6]
        print('failing iterate: largest |dE/dx| at bodies %s' % ' '.join('%d %.2f,' % (c_, gf_[c_]) for c_ in of_.tolist()), flush=True)
        np.savez((args.save or OUT)/('lion_fail_beta%g%s.npz' % (args.beta, args.tag)), z=zf_.cpu().numpy(), x=ST.tx.cpu().numpy(), F=W.tF.cpu().numpy(), J=torch.exp(W.tlj).cpu().numpy(), step=s, t=s*DT)
        break
    nrb = W._pos['nr'].clone() if args.a_floor > 0 else None
    ST.end()
    if args.watch and s >= args.watch_from:
        G_ = wp.to_torch(W.B['G'], requires_grad=False); Fc_ = wp.to_torch(W.B['Fc'], requires_grad=False); nr_ = W._pos['nr']
        for c_ in [int(v_) for v_ in args.watch.split(',')]:
            nt_ = torch.linalg.cross(G_[c_]@st.t1[c_], G_[c_]@st.t2[c_]); q_ = float(nt_@nrb[c_]); perp_ = float((nt_-q_*nrb[c_]).norm())
            cof_ = torch.linalg.det(Fc_[c_])*torch.linalg.inv(Fc_[c_]).T@st.mA[c_]; cof_ = cof_/cof_.norm()
            print('  step %d cell %d: a %.4f q %+.4f sideways %.4f; normal turned %5.1f deg this step; normal vs F_B normal %5.1f deg; det F_B %+.3f; Newton %d' % (
                s, c_, float(nt_.norm()), q_, perp_, float(torch.rad2deg(torch.arccos((nr_[c_]@nrb[c_]).clamp(-1, 1)))), float(torch.rad2deg(torch.arccos((nr_[c_]@cof_).clamp(-1, 1)))), float(torch.linalg.det(Fc_[c_])), its), flush=True)
    if s % args.every == 0 or (s*DT <= args.fine_until+1e-9 and s % 4 == 0):
        xe = ST.tx.clone(); W.grad(xe)
        xs.append(xe.cpu().numpy()); Fs.append(W.tF.cpu().numpy()); ts.append(s*DT); Js.append(torch.exp(W.tlj).cpu().numpy())
        com = (M3*xe).sum(0)/mass.sum()
        dirs = ''
        if getattr(W, "cap_dir", False):
            ang = torch.rad2deg(torch.acos((W._dw*W.st.mA).sum(1).clamp(-1, 1)))[W.st.cap]
            dirs = '; d off m: max %.1f deg, median %.2f deg; direction decrease / its tolerance %.2f' % (float(ang.max()), float(ang.median()), ST.dd/ST.dd_tol)
        print('t %.4f s (%.0f s): com up %.5f m; min J %.3f; products %d, Newton %d%s%s%s' % (s*DT, time.time()-t0, float(com@up), math.exp(float(W.tlj.min())), prods, its,
              self_report() if W.selfc is not None else ('; UL walls %d, radii iterations %d, radii failures %d, J floored %d' % tuple(W.ul_stats[k_] for k_ in ('walls', 'radii_it', 'radii_fail', 'floored')) if W.cap_ul else ''), dirs, floor_report() if args.a_floor > 0 else ''), flush=True)
        if s % 20 == 0:
            save()
save()
print('done: %d frames, %.0f s' % (len(ts), time.time()-t0), flush=True)
