"""Where is the angular momentum lost in the oblique bounce? (IDENTIFIABILITY.md section 11 follow-up)

Same ball and impact as mpm_oblique.py / model_oblique.py (nu 0.45, mu_f 1, no damping). Our clean model (translations only,
so no hidden body spins) and the MPM are sampled every 0.25 ms. From the total momentum P and the angular momentum about
the centroid L_c: ground force F = dP/dt - M g and ground torque about the centroid tau = dL_c/dt. The torque is split into
the friction part (tangential force x centroid height, which spins the ball forward) and the rest, written as an
effective forward offset dx_N of the normal force (tau_rest = -dx_N F_n; dx_N > 0 = ahead of the centroid, braking the
spin). Also: |F_t| / F_n against mu_f, the contact-patch position, and the angular momentum about the moving contact point."""
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import WarpFoam, rollout as warp_rollout
from experiments.mpm_oblique import ball_particles, simulate as mpm_simulate
from experiments.nu_signal_sphere import lattice_ball
from experiments.pacnerf_dem_forward import build

R, Estar, nu, mu_f, v0 = .035, 5e4, .45, 1., (1.0, -1.5)
fps = 4000.
out = Path('output/oblique_diag'); out.mkdir(parents=True, exist_ok=True)


def analyse(name, t, c, P, L, M, patch_x, g=9.8):
    """t (T,), centroid c (T,3) [x forward, z up], momentum P (T,3), angular momentum about the centroid L (T,3), patch x."""
    dt = np.gradient(t)
    F = np.gradient(P, axis=0)/dt[:, None]-M*np.array([0., 0., -g])
    tau = np.gradient(L, axis=0)/dt[:, None]
    Fn, Ft = F[:, 2], F[:, 0]
    h = c[:, 2]
    tau_fric = -h*Ft                                              # (-h z) x (Ft x) = -h Ft y
    rest = tau[:, 1]-tau_fric
    contact = Fn > .02*Fn.max()
    with np.errstate(divide='ignore', invalid='ignore'):
        dxN = np.where(contact, -rest/Fn, np.nan)
    J_n, J_t = np.trapezoid(np.where(contact, Fn, 0), t), np.trapezoid(np.where(contact, Ft, 0), t)
    print('%s: contact %.1f ms | normal impulse %.4f, tangential %.4f (ratio %.2f, mu_f %.1f) | angular impulse about centroid: total %.5f = friction %.5f + normal offset %.5f' % (
        name, 1e3*contact.sum()*np.median(dt), J_n, J_t, abs(J_t)/J_n, mu_f, np.trapezoid(np.where(contact, tau[:, 1], 0), t),
        np.trapezoid(np.where(contact, tau_fric, 0), t), np.trapezoid(np.where(contact, rest, 0), t)), flush=True)
    idx = np.flatnonzero(contact)
    for k in idx[::max(1, len(idx)//8)]:
        print('   t %5.1f ms  Fn %7.3f  Ft %+7.3f (|Ft|/Fn %.2f)  tau_y %+8.5f  friction part %+8.5f  normal offset dx_N %+6.2f mm  patch-centroid %+6.2f mm' % (
            1e3*t[k], Fn[k], Ft[k], abs(Ft[k])/max(Fn[k], 1e-12), tau[k, 1], tau_fric[k], 1e3*dxN[k], 1e3*(patch_x[k]-c[k, 0])), flush=True)
    return dict(J_n=J_n, J_t=J_t)


def ours():
    wp.init()
    dev = 'cuda'
    g = lattice_ball(R, .0035, prune=True)
    E = Estar*(1-nu*nu)
    m32 = build(g, E, nu, dev, 'noslip', .25, .3, torch.float32, shell=True, clean=True)
    m64 = build(g, E, nu, dev, 'noslip', .25, .3, torch.float64, shell=True, clean=True)
    for m in (m32, m64):
        m.set_environment(gravity=[0, 0, -9.8], ground_point=[0, 0, 0], ground_normal=[0, 0, 1], contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
    th = torch.zeros(m32.n, 9, dtype=torch.float32, device=dev); th[:, 6] = -20.
    sub = max(1, int(np.ceil(1/fps/m64.stable_dt(th[:, :7].double(), safety=.4, include_damping=False))))
    wf = WarpFoam(m32, contact='coulomb', layer_dx=.002, friction0=mu_f, implicit_damping=True, graphs=True)
    x0, q0, _, w0 = m32.initial_state(); r_c = float(m32.contact_radius.max())
    x0 = x0+torch.tensor([0., 0., R+r_c+.0015], device=dev)
    v = torch.tensor([v0[0], 0., v0[1]], device=dev).expand_as(x0).contiguous()
    frames = int(.06*fps)
    with torch.no_grad():
        pos, _ = warp_rollout(wf, th, (x0.contiguous(), q0.contiguous(), v, w0.contiguous()), 1/fps/sub, sub, frames)
    X = pos.double().cpu().numpy(); m = m32.mass.double().cpu().numpy(); Mt = m.sum()
    t = np.arange(len(X))/fps
    V = np.gradient(X, axis=0)*fps
    c = (m[None, :, None]*X).sum(1)/Mt; vc = (m[None, :, None]*V).sum(1)/Mt
    P = Mt*vc
    L = (m[None, :, None]*np.cross(X-c[:, None], V-vc[:, None])).sum(1)
    zmin = X[:, :, 2]
    patch = np.array([X[k][zmin[k] <= zmin[k].min()+.0035, 0].mean() for k in range(len(X))])
    return analyse('ours (clean model, Coulomb ground)', t, c, P, L, Mt, patch)


def mpm():
    dx = .0025
    x0 = ball_particles(R, dx/2); x0[:, 1] += R+dx
    E = Estar*(1-nu*nu)
    Mmod = E*(1-nu)/((1+nu)*(1-2*nu)); cp = np.sqrt(Mmod/1000.)
    dt = .3*dx/(cp+3.)
    every = max(1, int(round(1/fps/dt)))
    s = mpm_simulate(x0, [v0[0], v0[1], 0.], E, nu, mu_f, dx, dt, int(np.ceil(.06/dt)), every)
    t = np.array([q['t'] for q in s]); M = s[0]['M']
    sw = lambda a: np.stack((a[:, 0], -a[:, 2], a[:, 1]), 1)          # MPM (x, y up, z) -> (x, y, z up), proper rotation
    c = sw(np.array([q['c'] for q in s])); P = M*sw(np.array([q['v'] for q in s])); L = sw(np.array([q['L'] for q in s]))
    return analyse('MPM (Coulomb grid floor)', t, c, P, L, M, np.array([q['patch_x'] for q in s]))


if __name__ == '__main__':
    r1 = ours()
    r2 = mpm()
    (out/'summary.json').write_text(json.dumps(dict(ours=r1, mpm=r2), indent=1))
