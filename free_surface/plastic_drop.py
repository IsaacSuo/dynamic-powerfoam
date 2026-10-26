"""PAC-NeRF plasticine scene 0 forward against its ground-truth MPM (2026-10-26): the same shape as elastic scene 0
(the frame-0 particles identical), so the 8k foam (output/round01_pruned_exact); the scene's E, nu, yield stress and
start velocity (data/pacnerf/gt/plasticine.json), gravity, the ground at y = -0.05 (PAC-NeRF's y = 0; ours = theirs +
conversion offset (0.5, -0.05, 0.5)). PAC-NeRF's MPM: von Mises on the Hencky strain (the same return as plastic.py),
a sticky ground (all grid velocity zeroed below it): here the friction glue at a large mu. Per frame (1/24 s) against
the MPM particles (output/pacnerf_sim/simulation_data/plasticine/0, extracted from data/pacnerf/simulation_data.zip):
centre of mass, extents, symmetric Chamfer distance (the bodies' sites and the skin points against 30k particles)
  python free_surface/plastic_drop.py [--ys inf: elastic]"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from free_surface.joint_step import MidpointIntegrator
from free_surface.plastic import Plasticity
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser

OFFSET = np.array([.5, -.05, .5])


def load_ply(p):
    v = PlyData.read(str(p))['vertex']
    return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)+OFFSET


def chamfer(a, b, chunk=1024):
    """mean nearest distance a -> b and b -> a (torch, m; float32 distances)"""
    a, b = a.float(), b.float()

    def one(p, q):
        return torch.cat([torch.cdist(p[i:i+chunk], q).min(1).values for i in range(0, len(p), chunk)]).mean()
    return float(one(a, b)), float(one(b, a))


def main():
    gt = json.loads(Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/gt/plasticine.json').read_text())['0']
    a_ = argparse.ArgumentParser()
    a_.add_argument('--out', default='output/round01_pruned_exact')
    a_.add_argument('--sim', type=Path, default=Path('output/pacnerf_sim/simulation_data/plasticine/0'))
    a_.add_argument('--frames', type=int, default=15)
    a_.add_argument('--dt', type=float, default=1/1200)
    a_.add_argument('--tol', type=float, default=1e-3)
    a_.add_argument('--beta', type=float, default=.001, help='Kelvin-Voigt time (s)')
    a_.add_argument('--mu', type=float, default=10., help='friction: large for the MPM\'s sticky ground')
    a_.add_argument('--E', type=float, default=gt['E'])
    a_.add_argument('--nu', type=float, default=gt['nu'])
    a_.add_argument('--ys', type=float, default=gt['ys'], help='yield stress (Pa); inf: elastic')
    a_.add_argument('--tag', default='plastic_drop')
    a_.add_argument('--criterion', choices=('virial', 'FB'), default='FB')
    a_.add_argument('--nested', action='store_true', help='the return\'s passes outside the step (step_fb), not inside its solve')
    a_.add_argument('--log-steps', action='store_true', help='each contact step: per pass products, contact passes, eps change, yielding')
    a_.add_argument('--max-steps', type=int, default=0, help='stop after this many steps (0: all frames)')
    pa = a_.parse_args()
    args = parser().parse_args(['--out', pa.out, '--beta', str(pa.beta), '--integrator', 'midpoint', '--dt', str(pa.dt)])
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=-.05, points=False, shift=True))
    st, W, S, mass = b['st'], b['W'], b['S'], b['mass']
    E0, dt = float(args.E), pa.dt
    W.tfpar[0] = pa.mu
    W.set_material(E=pa.E, nu=pa.nu, beta=pa.beta)                                # (identify.Problem.set_material: the
    b['prec'] = S.preconditioner(b['blocks']*(pa.E/E0), S.bodies)                   # preconditioner's blocks with E)
    S.contact_setup(b['prec'], 100.)
    b['kblocks'] = b['kblocks']*(pa.E/E0)
    if b['vblocks'] is not None:
        b['vblocks'] = b['vblocks']*(pa.E/E0)
    W.tlam.zero_(); W.tact.zero_(); W.tlamt.zero_(); W.tfst.zero_()
    X, Y = st.X, st.Y
    up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    fg = -9.8*mass[:, None]*up
    v0 = torch.tensor(gt['v'], dtype=f64, device=dev).expand_as(X).clone()
    pl = Plasticity(W, st, pa.ys/1000., pa.criterion)
    pl.merged = not pa.nested                                             # (per unit density, rho 1000 as set_material)
    I = MidpointIntegrator(b, dt, X.clone(), v0, Y.clone(), fg, tol=pa.tol, ref=float(fg.norm()), contact=True)
    every = int(round(1/(24*dt)))
    import warp as wp
    cbody = wp.to_torch(W.ct['body']).long()
    rng = np.random.default_rng(0)
    G = [load_ply(pa.sim/('%d.ply' % k)) for k in range(pa.frames+1)]
    pick = rng.choice(len(G[0]), 30000, replace=False)
    w = (mass/mass.sum())[:, None]

    def ours():
        return torch.cat((I.x, I.y_int))

    def row(k):
        g = G[k]; o = ours()
        gc, oc = g.mean(0), (w*I.x).sum(0).cpu().numpy()
        gl, gh = g.min(0), g.max(0)
        ol, oh = o.min(0).values.cpu().numpy(), o.max(0).values.cpu().numpy()
        c1, c2 = chamfer(o, torch.as_tensor(g[pick], dtype=f64, device=dev))
        return dict(k=k, t=k/24, gc=gc, oc=oc, gl=gl, gh=gh, ol=ol, oh=oh, ch=(c1, c2))
    print('PAC-NeRF plasticine 0: E %.1f, nu %.4f, yield %.1f Pa (yield strain %.4f), v0 %s; ours: %d cells, %d skin points, dt 1/%d, tol %.0e, '
          'beta %.3g s, friction mu %g; MPM %d particles' % (pa.E, pa.nu, pa.ys, pa.ys/pa.E, np.round(gt['v'], 4).tolist(), W.n, W.npt, round(1/dt), pa.tol,
                                                           pa.beta, pa.mu, len(G[0])), flush=True)
    rec = [row(0)]
    t0 = time.time()
    s = 0
    prods, passes = 0, []
    for k in range(1, pa.frames+1):
        for _ in range(every):
            s += 1
            t1 = time.time()
            if math.isinf(pa.ys):
                s_ = I.step(); prods += int(s_['products'])
                u = dict(log=[(int(s_['products']), int(s_.get('al', 0)), 0., 0)], lim=0.)
            else:
                s_, u = pl.step(I)
                prods += u['products']; passes.append(u['passes'])
            if pa.log_steps and int((W.tact > 0).sum()):                           # each contact step: its products and solves
                nsolve = sum(c for _, c, _, _ in u['log']) if not getattr(pl, 'merged', False) or math.isinf(pa.ys) else int(s_.get('al', 0))
                print('  step %d (%.1f s): active %d; products %d, solves %d, plastic passes %d; the plastic change / tol by pass %s; yielding %s' % (
                    s, time.time()-t1, int((W.tact > 0).sum()), int(u['products']) if 'products' in u else int(s_['products']), nsolve,
                    len(u['log']), ' '.join('%.1e' % (ch/max(u['lim'], 1e-300)) for _, _, ch, _ in u['log']), u['log'][-1][3] if u['log'] else '-'), flush=True)
                if nsolve >= 20:                                                   # the contact's passes of a step that did not settle
                    epn = pl.eps.flatten(1).norm(dim=1)
                    for q, a in enumerate(s_.get('al_info', [])):
                        if q < 6 or q >= len(s_['al_info'])-6:
                            cells = (a['joined']+a['left'])[:8]
                            print('    pass %2d: moved %d (joined %s, left %s), |dlam| / kappa %.2e; largest change cell %d: %.2e, lam %.3e, end gap %.2e; '
                                  'the moved cells\' end gap %s, lam - rho gap %s, |eps_p| %s; solve residual %.1e, products %d, rejected %d' % (
                                      q+1, a['moved'], a['joined'][:6], a['left'][:6], a['dl'], a['top'], a['top_dl'], a['top_lam'], a['top_gap'],
                                      ['%.1e' % a['gaps'][c] for c in cells], ['%.1e' % a['lr'][c] for c in cells],
                                      ['%.3f' % float(epn[int(cbody[c])]) for c in cells],
                                      a['residual'], a['products'], a['rejected']), flush=True)
            if pa.max_steps and s >= pa.max_steps:
                break
        if pa.max_steps and s >= pa.max_steps:
            print('stopped at step %d (--max-steps)' % s, flush=True)
            break
        r = row(k)
        rec.append(r)
        em, emax = pl.strain(mass)
        ev = torch.linalg.eigvalsh(em).cpu().numpy()
        pz = np.array(passes[-every:]) if passes else np.zeros(1)
        print('frame %2d (t %.3f, %.0f s, products %d): com y ours %.4f MPM %.4f; height %.4f / %.4f; x extent %.4f / %.4f; z %.4f / %.4f; '
              'Chamfer ours->MPM %.1f, MPM->ours %.1f mm; active %d; plastic: mean eps_p eig %s, max %.3f, W_p %.4f J/kg; passes mean %.1f max %d' % (
                  k, k/24, time.time()-t0, prods, r['oc'][1], r['gc'][1], r['oh'][1]-r['ol'][1], r['gh'][1]-r['gl'][1], r['oh'][0]-r['ol'][0],
                  r['gh'][0]-r['gl'][0], r['oh'][2]-r['ol'][2], r['gh'][2]-r['gl'][2], 1e3*r['ch'][0], 1e3*r['ch'][1], int((W.tact > 0).sum()),
                  np.round(ev, 3).tolist(), emax, pl.dissipated/float(mass.sum()), pz.mean(), pz.max()), flush=True)
    r0 = rec[0]
    print('frame 0 (the shapes alone): Chamfer ours->MPM %.1f, MPM->ours %.1f mm' % (1e3*r0['ch'][0], 1e3*r0['ch'][1]), flush=True)
    np.savez(Path(pa.out)/('%s.npz' % pa.tag), x=I.x.cpu().numpy(), y=I.y_int.cpu().numpy(), eps=pl.eps.cpu().numpy(),
             rec=np.array([[r['t'], *r['oc'], *r['gc'], *r['ol'], *r['oh'], *r['gl'], *r['gh'], *r['ch']] for r in rec]))


if __name__ == '__main__':
    main()
