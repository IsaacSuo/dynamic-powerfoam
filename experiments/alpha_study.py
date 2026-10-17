"""What does alpha do in the bonded model? (the share of the shear modulus carried by the bond springs, kn = ks = 2 alpha mu A/L;
the averaged-F Neo-Hookean body term carries (1 - alpha) mu and lam). Continuum as reference, no training.

Variants: global alpha in --alphas, and a local alpha tied to how well each body's averaged F is determined:
q_a = lambda_min / lambda_max of the neighbour moment matrix M_a, r_a = min(1, q_a / median q), body term (1 - alpha0) r_a mu,
springs alpha_a = 1 - (1 - alpha0) r_a (bond stiffness by the geometric mean over its two ends) - implemented as per-body theta
offsets (theta columns 0-3 scale the springs, 5 the body mu), so the same model code runs both.

Measures (model as in springgaus_train without --cosserat; true rotational inertia):
  patch    burger geometry, homogeneous F0, interior-body force / (|P| A)
  omax     burger omega_max (time step) and the share of the fastest mode on bodies with q < 0.1
  ball     free lattice ball (R 3.5 cm, nu 0.3, E* 5e4): lowest non-rigid modes; Lamb 0T2 = 47.5 Hz; modes below 0.9 x 47.5 Hz
           are spurious
  plates   cantilever plates 1..4 layers, tip deflection / Hex8 FEM
  oblique  ball impact (1.0 sideways, 1.5 down m/s), Coulomb ground mu 1, nu 0.45, no damping: vx, spin, peak height vs MPM
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import warp as wp

from experiments.foam_dem_energy import quat_mul, quat_small
from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.nu_signal_sphere import lattice_ball
from experiments.pacnerf_dem_forward import build
from experiments.pacnerf_modal_check import dem_stiffness
from experiments.plate_bending_check import lattice_plate, fem_tip_deflection

RHO = 1000.


def local_theta(model, alpha_glob, alpha0, ncol=7):
    """Per-body theta offsets realising the local alpha on a model built with global alpha_glob."""
    M = torch.zeros(model.n, 3, 3, dtype=torch.float64, device=model.X.device).index_add(
        0, model.ls_a, (model.ls_w[:, None, None]*model.ls_dX[:, :, None]*model.ls_dX[:, None, :]).double())
    ev = torch.linalg.eigvalsh(M)
    q = (ev[:, 0]/ev[:, 2].clamp(min=1e-300)).clamp(min=0)
    r = (q/q.median()).clamp(max=1.)
    a = 1-(1-alpha0)*r
    th = torch.zeros(model.n, ncol, dtype=model.X.dtype, device=model.X.device)
    a = a.to(model.X.dtype); r = r.to(model.X.dtype)
    th[:, :4] = torch.log(a/alpha_glob)[:, None]                    # springs (bond k uses exp of the mean of its ends' theta)
    th[:, 5] = torch.log(((1-alpha0)*r).clamp(min=1e-12)/(1-alpha_glob))   # body mu
    return th, q, a


def variant_theta(model, v, ncol=7):
    if v['local']:
        th, q, a = local_theta(model, v['alpha_build'], v['alpha0'], ncol)
        return th
    return torch.zeros(model.n, ncol, dtype=model.X.dtype, device=model.X.device)


def make(g, E, nu, v, dtype, dev):
    return build(g, E, nu, dev, 'noslip', v['alpha_build'], .3, dtype)


def patch(v, g, dev):
    m = make(g, 2e4, .3, v, torch.float64, dev)
    th = variant_theta(m, v)
    from scipy.spatial.transform import Rotation as Rot
    F0 = torch.tensor([[1.2, .3, 0.], [0., .9, .1], [.05, 0., 1.1]], dtype=torch.float64, device=dev)
    U_, S_, Vh = torch.linalg.svd(F0)
    qp = Rot.from_matrix((U_@Vh).cpu().numpy()).as_quat(); qp = torch.tensor([qp[3], qp[0], qp[1], qp[2]], dtype=torch.float64, device=dev)
    x0, q0, _, _ = m.initial_state()
    f, _ = m.forces(th, x0@F0.T, qp.expand(m.n, 4).contiguous())
    m1 = torch.zeros(m.n, 3, dtype=torch.float64, device=dev).index_add(0, m.ls_a, m.ls_w[:, None]*m.ls_dX)
    tot = torch.zeros(m.n, dtype=torch.float64, device=dev).index_add(0, m.ls_a, m.ls_w*m.ls_dX.norm(dim=1))
    M = torch.zeros(m.n, 3, 3, dtype=torch.float64, device=dev).index_add(0, m.ls_a, m.ls_w[:, None, None]*m.ls_dX[:, :, None]*m.ls_dX[:, None, :])
    ev = torch.linalg.eigvalsh(M); thin = ev[:, 0]/ev[:, 2] < .1
    bad = (m1.norm(dim=1)/tot > .05) | thin
    near = torch.zeros(m.n, dtype=torch.bool, device=dev); near[m.ls_a[bad[m.ls_b]]] = True
    inner = ~bad & ~near
    P = (2e4/2.6/RHO)*(F0-torch.linalg.inv(F0).T)
    fi = f[inner].norm(dim=1)/(float(P.norm())*float(m.cell_volume.mean()**(2/3)))
    return dict(patch_median=float(fi.median()), patch_max=float(fi.max()))


def omax(v, g, dev):
    m = make(g, 2e4, .3, v, torch.float64, dev)
    th = variant_theta(m, v)
    w = m.max_frequency(th, iterations=150)
    x0, q0, _, _ = m.initial_state()
    mvec = torch.cat((m.mass[:, None].expand(-1, 3), m.inertia[:, None].expand(-1, 3)), 1)

    def force(z):
        f, tau = m.forces(th, x0+z[:, :3], quat_mul(quat_small(z[:, 3:]), q0)); return torch.cat((f, tau), 1)
    y = torch.randn(m.n, 6, dtype=torch.float64, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    for _ in range(150):
        y = y/y.norm(); _, Ky = torch.func.jvp(force, (torch.zeros_like(y),), (y/mvec.sqrt(),)); y = -Ky/mvec.sqrt()
    y = y/y.norm()
    M = torch.zeros(m.n, 3, 3, dtype=torch.float64, device=dev).index_add(0, m.ls_a, m.ls_w[:, None, None]*m.ls_dX[:, :, None]*m.ls_dX[:, None, :])
    ev = torch.linalg.eigvalsh(M); thin = ev[:, 0]/ev[:, 2] < .1
    share_thin = float((y[thin]**2).sum())
    return dict(omega_max=w, substeps_120fps=int(np.ceil(120**-1*w/(2*.5))), fastest_share_thin=share_thin,
                fastest_share_rot=float((y[:, 3:]**2).sum()))


def ball_modes(v, dev, n_modes=14):
    R, a, Estar, nu = .035, .0035, 5e4, .3
    g = lattice_ball(R, a, prune=True)
    m = make(g, Estar*(1-nu*nu), nu, v, torch.float64, dev)
    th = variant_theta(m, v)
    K = dem_stiffness_any(m, th)
    mvec = np.concatenate((np.repeat(m.mass.cpu().numpy()[:, None], 3, 1), np.repeat(m.inertia.cpu().numpy()[:, None], 3, 1)), 1).reshape(-1)
    Dm = sp.diags(1/np.sqrt(mvec))
    w2 = spla.eigsh((Dm@K@Dm).tocsc(), k=n_modes+6, sigma=-1., which='LM', return_eigenvectors=False)
    f = np.sort(np.sqrt(np.clip(w2, 0, None))/(2*np.pi))[6:]
    lamb = 2.501*np.sqrt(Estar*(1-nu)/2/RHO)/(2*np.pi*R)
    return dict(ball_modes_hz=f.tolist(), lamb_0T2_hz=lamb, spurious_below=int((f < .9*lamb).sum()))


def dem_stiffness_any(m, th):
    """Sparse stiffness of the bonded model by forward-mode probing on a graph colouring (distance-3 in the bond graph)."""
    n = m.n
    adj = [set() for _ in range(n)]
    for i_, j_ in zip(m.i.cpu().numpy(), m.j.cpu().numpy()):
        adj[i_].add(j_); adj[j_].add(i_)
    reach = []
    for b in range(n):                                   # bodies within 3 hops (coupling range of the body term + bonds)
        s1 = adj[b]; s2 = set().union(*[adj[c] for c in s1]) if s1 else set(); s3 = set().union(*[adj[c] for c in s2]) if s2 else set()
        reach.append(s1 | s2 | s3 | {b})
    color = -np.ones(n, int)
    for b in np.argsort([-len(r) for r in reach]):       # greedy colouring of the square of the 3-hop graph
        used = set()
        for c in reach[b]:
            for d in reach[c]:
                if color[d] >= 0:
                    used.add(color[d])
        k = 0
        while k in used:
            k += 1
        color[b] = k
    x0, q0, _, _ = m.initial_state()

    def force(z):
        f, tau = m.forces(th, x0+z[:, :3], quat_mul(quat_small(z[:, 3:]), q0)); return torch.cat((f, tau), 1)
    rows, cols, vals = [], [], []
    zero = torch.zeros(n, 6, dtype=torch.float64, device=x0.device)
    for c in range(color.max()+1):
        members = np.flatnonzero(color == c)
        owner = -np.ones(n, int)
        for b in members:
            for r in reach[b]:
                owner[r] = b
        sel = torch.as_tensor(color == c, device=x0.device)
        rr = np.flatnonzero(owner >= 0); ss = owner[rr]
        for d in range(6):
            vv = zero.clone(); vv[sel, d] = 1.
            _, Kv = torch.func.jvp(force, (zero,), (vv,))
            Kv = (-Kv).cpu().numpy()
            for e in range(6):
                rows.append(6*rr+e); cols.append(6*ss+d); vals.append(Kv[rr, e])
    K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(6*n, 6*n)).tocsr()
    return .5*(K+K.T)


FEM_CACHE = {}


def plates(v, layers=(1, 2, 3, 4), E=24107., nu=.414, a=.02, nx=16, ny=6):
    out = {}
    for nz in layers:
        g, idx = lattice_plate(nx, ny, nz, a)
        if nz not in FEM_CACHE:
            FEM_CACHE[nz] = fem_tip_deflection(nx, ny, nz, a, 3, E, nu)
        m = make(g, E, nu, v, torch.float64, 'cpu')
        th = variant_theta(m, v)
        K = dem_stiffness(m, th).tocsr()
        free = np.repeat(~g['fixed'], 6); tip = idx[:, 0] == idx[:, 0].max()
        f = np.zeros(6*m.n); f[6*np.flatnonzero(tip)+2] = 1./tip.sum()
        try:
            u = np.zeros(6*m.n); u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
            out[nz] = float(u[6*np.flatnonzero(tip)+2].mean()*RHO/E*E/RHO)/FEM_CACHE[nz]
        except Exception as e:                            # singular (e.g. a one-layer plate without bonds)
            out[nz] = float('nan')
    return dict(plates={str(k): val for k, val in out.items()})


def oblique(v, dev):
    R, Estar, nu, mu_f = .035, 5e4, .45, 1.
    g = lattice_ball(R, .0035, prune=True)
    E = Estar*(1-nu*nu)
    m32 = make(g, E, nu, v, torch.float32, dev); m64 = make(g, E, nu, v, torch.float64, dev)
    for m in (m32, m64):
        m.set_environment(gravity=[0, 0, -9.8], ground_point=[0, 0, 0], ground_normal=[0, 0, 1], contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
    th = variant_theta(m32, v, 9).float(); th[:, 6] = -20.
    fps = 1000.
    sub = int(np.ceil(1/fps/m64.stable_dt(variant_theta(m64, v), safety=.4, include_damping=False)))
    wf = WarpFoam(m32, contact='coulomb', layer_dx=.002, friction0=mu_f, implicit_damping=True, graphs=True)
    x0, q0, _, w0 = m32.initial_state(); r_c = float(m32.contact_radius.max())
    x0 = x0+torch.tensor([0., 0., R+r_c+.0015], device=dev)
    vel = torch.tensor([1., 0., -1.5], device=dev).expand_as(x0).contiguous()
    with torch.no_grad():
        P, _ = warp_rollout(wf, th.contiguous(), (x0.contiguous(), q0.contiguous(), vel, w0.contiguous()), 1/fps/sub, sub, 250)
    X = P.double().cpu().numpy(); mb = m32.mass.double().cpu().numpy(); Mt = mb.sum()
    if not np.isfinite(X).all():
        return dict(oblique='non-finite')
    V = np.gradient(X, axis=0)*fps
    zmin = X[:, :, 2].min(1); t_off = int(np.flatnonzero(zmin <= r_c+.003).max())
    ws, vs = [], []
    for k in range(t_off+10, len(X)-1):
        c = (mb[:, None]*X[k]).sum(0)/Mt; vc = (mb[:, None]*V[k]).sum(0)/Mt; r = X[k]-c
        L = (mb[:, None]*np.cross(r, V[k]-vc)).sum(0); I = (mb*(r*r).sum(1)).sum()*np.eye(3)-(mb[:, None]*r).T@r
        ws.append(np.linalg.solve(I, L)); vs.append(vc)
    peak = max(((mb[:, None]*X[k]).sum(0)/Mt)[2] for k in range(t_off, len(X), 5))
    return dict(oblique_vx=float(np.mean(vs, 0)[0]), oblique_spin=float(np.mean(ws, 0)[1]), oblique_peak_mm=1e3*float(peak), oblique_substeps_per_ms=sub)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alphas', type=float, nargs='*', default=[.05, .1, .25, .5, 1.])
    p.add_argument('--local-alpha0', type=float, nargs='+', default=[.1, .25])
    p.add_argument('--only', nargs='+', default=None, help='measures to run (patch omax ball plates oblique)')
    p.add_argument('--tag', default='')
    p.add_argument('--out', type=Path, default=Path('output/alpha_study'))
    args = p.parse_args()
    wp.init()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    g = dict(np.load('output/springgaus_dem/burger_spacing8k_mf0.1/geometry.npz'))
    variants = [dict(name='alpha %.2f' % a, alpha_build=a, local=False) for a in args.alphas]
    variants += [dict(name='local alpha0 %.2f' % a0, alpha_build=.5, alpha0=a0, local=True) for a0 in args.local_alpha0]
    rows = []
    for v in variants:
        r = dict(variant=v['name'])
        fns = dict(patch=lambda: patch(v, g, dev), omax=lambda: omax(v, g, dev), ball=lambda: ball_modes(v, dev), plates=lambda: plates(v), oblique=lambda: oblique(v, dev))
        for key, fn in fns.items():
            if args.only and key not in args.only:
                continue
            try:
                r.update(fn())
            except Exception as e:
                r.setdefault('errors', []).append(repr(e)[:200])
            (args.out/('alpha_study%s.json' % args.tag)).write_text(json.dumps(rows+[r], indent=1))
        rows.append(r)
        bm = r.get('ball_modes_hz', [])
        print('%-18s patch %.3f/%.3f | omega_max %.0f (substeps %s, fastest on thin %.2f, rot %.2f) | ball lowest %s Hz (Lamb %.1f, spurious %s) | plates %s | oblique vx %s spin %s peak %s | %s' % (
            v['name'], r.get('patch_median', np.nan), r.get('patch_max', np.nan), r.get('omega_max', np.nan), r.get('substeps_120fps'), r.get('fastest_share_thin', np.nan),
            r.get('fastest_share_rot', np.nan), np.round(bm[:4], 1).tolist(), r.get('lamb_0T2_hz', np.nan), r.get('spurious_below'),
            {k: round(val, 2) for k, val in r.get('plates', {}).items()}, '%.3f' % r['oblique_vx'] if 'oblique_vx' in r else r.get('oblique'),
            '%.1f' % r['oblique_spin'] if 'oblique_spin' in r else '-', '%.0f' % r['oblique_peak_mm'] if 'oblique_peak_mm' in r else '-', r.get('errors', '')), flush=True)


if __name__ == '__main__':
    main()
