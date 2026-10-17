"""Adjoint of one Warp Cosserat step (COSSERAT_FOAM.md: enhanced strain, thin closure, micropolar terms) against
torch float64 autograd of EnergyFoam.forces, on the burger DEM at a deformed state. Scalar L = sum(wf . f + wt . tau)
with random weights at zero velocity (no dashpot or contact terms; ground far away); compares dL/dtheta, dL/dx and
dL/dq (tangent part), grouped by body type (thin / enhanced / plain), with EAS and thin closure on or off."""
import argparse

import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import WarpFoam
from experiments.pacnerf_dem_forward import build


def rel(a, b):
    return float((a-b).norm()/b.norm().clamp(min=1e-30))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', default='output/springgaus_dem/burger_spacing8k_mf0.1/geometry.npz')
    p.add_argument('--E', type=float, default=2e4)
    p.add_argument('--nu', type=float, default=.3)
    p.add_argument('--noise', type=float, default=.05, help='position noise, fraction of the median bond length')
    p.add_argument('--squash', type=float, default=1., help='z scale of the deformed state (1: none)')
    p.add_argument('--theta', type=float, default=.29, help='log multiplier on lam and mu')
    p.add_argument('--mid', type=int, default=1, help='bond-level law: two-end gradient variant')
    p.add_argument('--clean', type=int, default=0, help='CLEAN_FOAM.md model (translations only, central dashpots)')
    p.add_argument('--dilatation', type=int, default=0, help='SPRING_DILATATION.md model (springs + volumetric multi-body term)')
    p.add_argument('--truevol', type=int, default=0, help='SPRING_DILATATION.md section 2 (springs + true-volume tetrahedra term)')
    p.add_argument('--variants', nargs='+', default=['full', 'eas off', 'thin off', 'both off'])
    args = p.parse_args()
    wp.init()
    g = dict(np.load(args.geometry))
    dev = 'cuda'
    kw = dict(kinematics='exact', residual=True, shell=True, cosserat=True)
    if args.clean:
        kw = dict(shell=True, clean=True)
    if args.dilatation:
        kw = dict(dilatation=True)
    if args.truevol:
        kw = dict(nodalvol=True)
    m64 = build(g, args.E, args.nu, dev, 'noslip', .25, .3, torch.float64, **kw)
    m32 = build(g, args.E, args.nu, dev, 'noslip', .25, .3, torch.float32, **kw)
    for m in (m64, m32):
        m.bond_level_mid = bool(args.mid)
        m.set_environment(gravity=[0., 0., 9.8], ground_point=[0., 0., 100.], ground_normal=[0., 0., -1.],
                          contact_radius=m.contact_radius.cpu().numpy(), ground_mode='noslip')
    x0, q0, _, _ = m64.initial_state()
    L = float(torch.linalg.vector_norm(m64.X[m64.j]-m64.X[m64.i], dim=1).median())
    torch.manual_seed(0)
    c = x0.mean(0)
    x = c+(x0-c)*torch.tensor([1., 1., args.squash], dtype=x0.dtype, device=dev)+args.noise*L*torch.randn_like(x0)
    q = q0+.05*torch.randn_like(q0); q = q/q.norm(dim=-1, keepdim=True)
    th = torch.zeros(m64.n, 9, dtype=torch.float64, device=dev); th[:, 4:6] = args.theta
    wf_ = torch.randn(m64.n, 3, dtype=torch.float64, device=dev); wt_ = torch.randn(m64.n, 3, dtype=torch.float64, device=dev)
    thin, eas = m64.thin.clone(), m64.eas_on.clone()
    groups = dict(thin=thin, enhanced=eas & ~thin, plain=~eas & ~thin)
    print('bond_level %s mid %s | bodies %d: thin %d, enhanced %d, plain %d; bond length %.4g, noise %.3g L, squash %.2f' % (
        getattr(m64, 'bond_level', False), args.mid, m64.n, int(thin.sum()), int((eas & ~thin).sum()), int(groups['plain'].sum()), L, args.noise, args.squash))
    for name, eas_on, thin_on in (('full', True, True), ('eas off', False, True), ('thin off', True, False), ('both off', False, False)):
        if name not in args.variants:
            continue
        for m in (m64, m32):
            m.eas_on = eas.clone() if eas_on else torch.zeros_like(eas)
            m.thin = thin.clone() if thin_on else torch.zeros_like(thin)
            m.enhanced = not args.dilatation and not args.clean and not args.truevol
        # torch reference
        th_t = th[:, :7].clone().requires_grad_(True); x_t = x.clone().requires_grad_(True); q_t = q.clone().requires_grad_(True)
        f64, t64 = m64.forces(th_t, x_t, q_t)
        Lt = (wf_*f64).sum()+(wt_*t64).sum()
        gth, gx, gq = torch.autograd.grad(Lt, (th_t, x_t, q_t))
        # Warp: one step at zero velocity, tape from the elastic force / torque buffers
        wf = WarpFoam(m32, contact='smooth', layer_dx=.002, implicit_damping=True, graphs=False)
        dtypes = (wp.vec3, wp.vec4, wp.vec3, wp.vec3)
        s = tuple(wp.from_torch(t.float().contiguous().clone(), dtype=d, requires_grad=True)
                  for t, d in zip((x, q, torch.zeros_like(x), torch.zeros_like(x)), dtypes))
        s1 = tuple(wp.zeros(m64.n, dtype=d, device='cuda', requires_grad=True) for d in dtypes)
        buf = wf._buffers(True)
        thw = wp.from_torch(th.float().contiguous(), requires_grad=True)
        tape = wp.Tape()
        with tape:
            wf.step(thw, s, s1, buf, 1e-6)
        tau_key = 'tau3' if 'tau3' in buf else 'tau2' if 'tau2' in buf else 'tau'
        f_key = 'f2' if 'f2' in buf else 'f'                     # true-volume term: final force in f2
        f32, t32 = wp.to_torch(buf[f_key]).double(), wp.to_torch(buf[tau_key]).double()
        tape.backward(grads={buf[f_key]: wp.from_torch(wf_.float().contiguous(), dtype=wp.vec3),
                             buf[tau_key]: wp.from_torch(wt_.float().contiguous(), dtype=wp.vec3)})
        wth = wp.to_torch(thw.grad).double()[:, :7]; wx = wp.to_torch(s[0].grad).double(); wq = wp.to_torch(s[1].grad).double()
        proj = lambda gq_: gq_-(gq_*q).sum(-1, keepdim=True)*q
        print('%-9s forces rel %.1e torques rel %.1e (max |torque| warp %.1e) | grad rel: theta[lam] %.1e theta[mu] %.1e x %.1e q %.1e' % (
            name, rel(f32, f64.detach()), rel(t32, t64.detach()) if float(t64.abs().max()) > 0 else 0., float(t32.abs().max()), rel(wth[:, 4], gth[:, 4]), rel(wth[:, 5], gth[:, 5]), rel(wx, gx), rel(proj(wq), proj(gq))))
        for gname, msk in groups.items():
            if not bool(msk.any()):
                continue
            e_th = rel(wth[msk][:, 4:6], gth[msk][:, 4:6]); e_x = rel(wx[msk], gx[msk]); e_q = rel(proj(wq)[msk], proj(gq)[msk])
            worst = (wth[:, 4:6]-gth[:, 4:6]).norm(dim=1)/gth[:, 4:6].norm(dim=1).clamp(min=1e-30)
            print('          %-8s theta %.1e x %.1e q %.1e | worst body theta rel %.2g' % (gname, e_th, e_x, e_q, float(worst[msk].max())))


if __name__ == '__main__':
    main()
