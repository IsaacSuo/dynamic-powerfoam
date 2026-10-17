"""Checks of the Hertz-Mindlin ground contact (WarpFoam contact='hertz').

A lattice ball (springs + true volume, nu_signal_sphere.lattice_ball) dropped on the ground, no material damping unless
--damping is given. Per substep: kinetic (translation + spin), elastic (model.energy), contact potential (Hertz normal
8/15 E* sqrt(R) d^2.5 + Mindlin spring |F_t|^2 / (2 k_t)), gravity. Reports, for each friction coefficient and substep
multiplier: the total-energy drift through the impact (max |E - E0| / KE_impact and the value after lift-off), the
centre-of-mass restitution vz_out / vz_in, the contact duration, the lift-off spin and horizontal velocity (oblique
drops). Frictionless runs isolate the integration error (the contact is then conservative).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import WarpFoam, _pad_state, _state_dtypes
from experiments.nu_signal_sphere import lattice_ball
from experiments.pacnerf_dem_forward import build


def run(g, E, nu, mu_f, scale, v0, args, dev='cuda'):
    nl = bool(getattr(args, 'nonlinear', False))
    m = build(g, E, nu, dev, 'noslip', .25, .3, torch.float32, nodalvol=True, nonlinear=nl)
    m64 = build(g, E, nu, dev, 'noslip', .25, .3, torch.float64, nodalvol=True, nonlinear=nl)
    for mm in (m, m64):
        mm.set_environment(gravity=[0, 0, -9.8], ground_point=[0, 0, 0], ground_normal=[0, 0, 1],
                           contact_radius=mm.contact_radius.cpu().numpy(), ground_mode='noslip')
    wf = WarpFoam(m, contact='hertz', friction0=max(mu_f, 1e-6), implicit_damping=True)
    th = torch.zeros(m.n, 8, dtype=torch.float32, device=dev)
    th[:, :4] = 0.
    th[:, 6] = float(np.log(args.damping)) if args.damping > 0 else -30.
    sub = int(np.ceil(scale/args.fps/m64.stable_dt(th[:, :7].double(), safety=.5, include_damping=False)))
    dt = 1/args.fps/sub
    x0, q0, _, w0 = m.initial_state()
    x0 = x0+torch.tensor([0., 0., args.R+float(m.contact_radius.max())+.002], device=dev)
    v = torch.zeros_like(x0); v[:, 0] = v0[0]; v[:, 2] = -v0[1]
    mk = lambda t, d: wp.from_torch(t.detach().float().contiguous().clone(), dtype=d)
    dts = _state_dtypes(wf)
    st = tuple(mk(t, d) for t, d in zip(_pad_state(wf, (x0, q0, v, w0)), dts))
    st1 = tuple(wp.zeros(m.n, dtype=d, device=dev) for d in dts)
    buf = wf._buffers(False)
    mass, inert = m.mass.double(), m.inertia.double()
    lam = m.k_v*torch.exp(th[:, 4].double()); mu = m.mu_ref*torch.exp(th[:, 5].double())
    nu_ = .5*lam/(lam+mu); E_ = mu*(3*lam+2*mu)/(lam+mu); Es = E_/(1-nu_**2); Gs = mu/(2-nu_)
    R = m.contact_radius.double()
    gvec = torch.tensor([0., 0., -9.8], dtype=torch.float64, device=dev)

    def energy(s_):
        x, q, vv, ww, hc = (wp.to_torch(a) for a in s_)
        K = .5*((mass*(vv.double()**2).sum(1)).sum()+(inert*(ww.double()**2).sum(1)).sum())
        with torch.no_grad():
            U = float(m.energy(th, x, q))
        d = (R-x[:, 2].double()).clamp(min=0)
        kt = 8*Gs*(R*d).sqrt()
        Uc = (8/15*Es*R.sqrt()*d**2.5).sum()+torch.where(kt > 0, (hc[:, :3].double()**2).sum(1)/(2*kt.clamp(min=1e-30)), torch.zeros_like(kt)).sum()
        Ug = -(mass*(x.double()@gvec)).sum()
        vc = (mass[:, None]*vv.double()).sum(0)/mass.sum()
        r_ = x.double()-(mass[:, None]*x.double()).sum(0)/mass.sum()
        L = (mass[:, None]*torch.cross(r_, vv.double(), dim=1)).sum(0)+(inert[:, None]*ww.double()).sum(0)
        Iyy = (mass*(r_[:, 0]**2+r_[:, 2]**2)).sum()+inert.sum()
        return float(K), U, float(Uc), float(Ug), vc, int((d > 0).sum()), (L, Iyy)
    rows = []
    nsteps = int(args.T*args.fps*sub)
    K0, U0, C0, G0, vc0, _, _ = energy(st)
    xx0 = wp.to_torch(st[0]); h0 = float(xx0[:, 2].max()-xx0[:, 2].min()); w0_ = float(xx0[:, 0].max()-xx0[:, 0].min())
    E0 = K0+U0+C0+G0
    for k in range(nsteps):
        for a_ in buf.values():
            a_.zero_()
        wf.step(wp.from_torch(th), st, st1, buf, dt)
        st, st1 = st1, st
        if k % max(1, sub//16) == 0 or k == nsteps-1:
            K, U, C, G, vc, nc, L = energy(st)
            xx = wp.to_torch(st[0])
            rows.append(dict(t=(k+1)*dt, K=K, U=U, C=C, G=G, E=K+U+C+G, vz=float(vc[2]), vx=float(vc[0]), ncontact=nc, Ly=float(L[0][1]), spin=float(L[0][1]/L[1]),
                             height=float(xx[:, 2].max()-xx[:, 2].min()), width=float(max(xx[:, 0].max()-xx[:, 0].min(), xx[:, 1].max()-xx[:, 1].min()))))
    t = np.array([r['t'] for r in rows]); nc = np.array([r['ncontact'] for r in rows])
    on = np.flatnonzero(nc > 0)
    vin = rows[max(on[0]-1, 0)]['vz'] if len(on) else np.nan
    Kimp = .5*float(mass.sum())*(vin**2+v0[0]**2)
    after = rows[on[-1]+1:] if len(on) and on[-1]+1 < len(rows) else []
    vout = after[0]['vz'] if after else np.nan
    Edev = np.array([r['E']-E0 for r in rows])/Kimp
    # spin of the free body after lift-off from the angular momentum about the centroid (y axis for an x-z drop)
    res = dict(mu_f=mu_f, scale=scale, substeps=sub, restitution=float(-vout/vin) if after else np.nan,
               contact_ms=1e3*float(t[on[-1]]-t[on[0]]) if len(on) else 0., energy_drift_max=float(np.abs(Edev).max()),
               energy_drift_end=float(Edev[-1]), vx_after=float(after[0]['vx']) if after else np.nan,
               Ly_after=float(after[0]['Ly']) if after else np.nan, max_contacts=int(nc.max()),
               spin_after=float(np.mean([r['spin'] for r in after[:8]])) if after else np.nan,
               bounce_mm=1e3*float(after[0]['vz']**2/(2*9.8)) if after else np.nan,
               min_height_ratio=min(r['height'] for r in rows)/h0, max_width_ratio=max(r['width'] for r in rows)/w0_)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--R', type=float, default=.035)
    p.add_argument('--cell', type=float, default=.0035)
    p.add_argument('--Estar', type=float, default=5e4)
    p.add_argument('--nu', type=float, default=.45)
    p.add_argument('--v', type=float, nargs=2, default=[0., 1.5], help='horizontal, downward impact speed (m/s)')
    p.add_argument('--mus', type=float, nargs='+', default=[0., .5])
    p.add_argument('--scales', type=float, nargs='+', default=[1., 2., 4.])
    p.add_argument('--damping', type=float, default=0.)
    p.add_argument('--nonlinear', action='store_true', help='neo-Hookean normal bond springs')
    p.add_argument('--fps', type=float, default=120.)
    p.add_argument('--T', type=float, default=.08)
    p.add_argument('--out', type=Path, default=Path('output/hertz_contact_check'))
    args = p.parse_args()
    wp.init()
    args.out.mkdir(parents=True, exist_ok=True)
    g = lattice_ball(args.R, args.cell)
    E = args.Estar*(1-args.nu**2)
    out = []
    for mu_f in args.mus:
        for sc in args.scales:
            r = run(g, E, args.nu, mu_f, sc, args.v, args)
            out.append(r)
            print('mu_f %.2f substeps x%.0f (%d/frame) | energy drift: max %.4f, after lift-off %+.4f of impact KE | restitution %.4f, contact %.1f ms, vx after %.4f, spin after %.2f rad/s, bounce %.1f mm, max contacts %d | min height %.3f, max width %.3f of rest' % (
                mu_f, sc, r['substeps'], r['energy_drift_max'], r['energy_drift_end'], r['restitution'], r['contact_ms'], r['vx_after'], r['spin_after'], r['bounce_mm'], r['max_contacts'],
                r['min_height_ratio'], r['max_width_ratio']), flush=True)
            (args.out/'results.json').write_text(json.dumps(dict(config={k: str(v) for k, v in vars(args).items()}, results=out), indent=1))


if __name__ == '__main__' and '--gradcheck' not in __import__('sys').argv:
    main()


def gradcheck():
    """Adjoint (Warp tape through the rollout, checkpointed) vs central differences of a trajectory loss on the ball,
    for theta columns 4 (lambda), 5 (mu), 6 (damping), 7 (friction), at several substep multipliers."""
    from experiments.foam_dem_warp import rollout
    p = argparse.ArgumentParser()
    p.add_argument('--gradcheck', action='store_true')
    p.add_argument('--R', type=float, default=.035)
    p.add_argument('--cell', type=float, default=.0035)
    p.add_argument('--Estar', type=float, default=5e4)
    p.add_argument('--nu', type=float, default=.45)
    p.add_argument('--v', type=float, nargs=2, default=[1., 1.5])
    p.add_argument('--mu', type=float, default=.5)
    p.add_argument('--scales', type=float, nargs='+', default=[1., 2., 4.])
    p.add_argument('--frames', type=int, default=6)
    p.add_argument('--eps', type=float, default=1e-2)
    p.add_argument('--visco', action='store_true', help='Zener bonds (g0 1, tau0 2 ms); also checks theta columns 8, 9')
    args = p.parse_args()
    wp.init()
    dev = 'cuda'
    g = lattice_ball(args.R, args.cell)
    E = args.Estar*(1-args.nu**2)
    m = build(g, E, args.nu, dev, 'noslip', .25, .3, torch.float32, nodalvol=True)
    m64 = build(g, E, args.nu, dev, 'noslip', .25, .3, torch.float64, nodalvol=True)
    for mm in (m, m64):
        mm.set_environment(gravity=[0, 0, -9.8], ground_point=[0, 0, 0], ground_normal=[0, 0, 1],
                           contact_radius=mm.contact_radius.cpu().numpy(), ground_mode='noslip')
    wf = WarpFoam(m, contact='hertz', friction0=args.mu, implicit_damping=True, visco=args.visco, visco_g0=1., visco_tau0=.002)
    nc = 10 if args.visco else 8
    x0, q0, _, w0 = m.initial_state()
    x0 = x0+torch.tensor([0., 0., args.R+float(m.contact_radius.max())+.002], device=dev)
    v = torch.zeros_like(x0); v[:, 0] = args.v[0]; v[:, 2] = -args.v[1]
    W = torch.randn(args.frames+1, m.n, 3, device=dev, generator=torch.Generator(dev).manual_seed(0))
    shared0 = torch.zeros(nc, device=dev)
    def loss_of(shared, sub, grad):
        th = shared.expand(m.n, nc)
        pos, _ = rollout(wf, th, (x0, q0, v, w0), 1/120/sub, sub, args.frames, 25 if grad else None)
        return (W*pos).mean()
    base_sub = int(np.ceil(1/120/m64.stable_dt(torch.zeros(m.n, 7, dtype=torch.float64, device=dev), safety=.5, include_damping=False)))
    for sc in args.scales:
        sub = int(np.ceil(sc*base_sub*(np.sqrt(2.) if args.visco else 1.)))
        sh = shared0.clone().requires_grad_(True)
        L = loss_of(sh, sub, True); L.backward()
        ga = sh.grad.detach().clone()
        out = []
        for c in ((4, 5, 6, 7, 8, 9) if args.visco else (4, 5, 6, 7)):
            e_ = torch.zeros(nc, device=dev); e_[c] = args.eps
            with torch.no_grad():
                fd = (float(loss_of(shared0+e_, sub, False))-float(loss_of(shared0-e_, sub, False)))/(2*args.eps)
            out.append((c, float(ga[c]), fd))
        print('substeps x%.0f (%d/frame): ' % (sc, sub)+' | '.join('col %d adjoint %+.4e FD %+.4e (rel %.1e)' % (c, a, f, abs(a-f)/max(abs(f), 1e-12)) for c, a, f in out), flush=True)


if __name__ == '__main__' and '--gradcheck' in __import__('sys').argv:
    gradcheck()
    raise SystemExit
