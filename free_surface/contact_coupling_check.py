"""Why a new contact's multipliers converge slowly (2026-10-24). At a saved state (skin_drop_check.py --save-state N,
the midpoint rule) the step's contact passes are replayed (JointSolver.solve with the active set held, then
lam <- max(0, lam - rho c)). At each pass's solution z: the active cells' compliance S_cd = dg_c . H0^-1 dg_d (g the raw
gap, dg_c from the contact term alone with lam - rho c = -1 on cell c; H0 the joint Hessian without any contact term, one
preconditioned CG solve per active cell). The update's map of the multipliers' error is (I + cs rho S)^-1 (cs 2, the
midpoint rule; each cell alone: 1 / (1 + cs rho_c S_cc)): its largest eigenvalue against the passes' observed reduction,
the slowest mode's cells, the skin points they share. A cheap S~ (the joint preconditioner in place of H0^-1) and the
Newton-type update lam <- mu - (cs S~)^-1 c: its map (I - S~^-1 S) (I + cs rho S)^-1.
  python free_surface/contact_coupling_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step349.npz"""
from pathlib import Path

import numpy as np
import torch

from free_surface.joint_step import MidpointIntegrator
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def main():
    a = parser()
    a.add_argument('--state', type=Path, required=True)
    a.add_argument('--gravity', type=float, default=9.8)
    a.add_argument('--passes', type=int, default=4)
    a.add_argument('--compliance', action='store_true', help='also S, S~ and the maps (one CG solve per active cell and pass)')
    args = a.parse_args()
    assert args.integrator == 'midpoint'
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., shift=True))
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    T = lambda k: torch.as_tensor(z0[k], device=dev)
    z0 = np.load(args.state)
    up = torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    fg = -args.gravity*mass[:, None]*up
    ref = float(fg.norm()); tol, dt = args.tols[0], args.dt
    I = MidpointIntegrator(b, dt, T('x'), T('v'), T('ym1'), fg, tol=tol, ref=ref, contact=True)
    I.x, I.v, I.a, I.xm1, I.ym1, I.xm2, I.ym2 = (T(k) for k in ('x', 'v', 'a', 'xm1', 'ym1', 'xm2', 'ym2'))
    W.tlam.copy_(T('lam')); W.tact.copy_(T('act')); W.tgoff.copy_(T('goff')); W.trho.copy_(T('rho'))
    for k_, r_ in W.tref.items():
        r_.copy_(T('ref_'+k_))
    J, n = I.J, W.n
    x, v = I.x, I.v
    xm = x+.5*dt*v+.25*dt*dt*I.a
    ym = I.predict_skin(S, xm, I.xm1, I.xm2, I.ym1, I.ym2)
    J.set_step(x, x+.5*dt*v, fg)
    z = torch.cat((xm, ym))
    if getattr(prec, 'ck', 0):                                                     # as JointSolver.solve_contact: rho from the
        W.trho.copy_(torch.where(prec.taua > 0, S.kappa/prec.taua, W.trho))       # last preconditioner's contact cells
    cs = float(W.tcpar[0])
    pid, ebody = st.e_pid.cpu().numpy(), st.e_body.cpu().numpy()
    cbody = W.cg['body'].cpu().numpy()
    pts = lambda c: set(pid[ebody == cbody[c]].tolist())                          # the skin points of cell c's entries

    def grad_g(z, c):
        """dg_c / dz (flat): the contact term alone, cell c only, lam - rho c = -1 there"""
        lam0, act0, tw0 = W.tlam.clone(), W.tact.clone(), W.ttw.clone()
        W.ttw.zero_(); W.ttw[6] = 1.
        W.tact.zero_(); W.tact[c] = 1
        W.grad(z[:n], z[n:]); cc = float(W.tgap[c])
        W.tlam.zero_(); W.tlam[c] = float(W.trho[c])*cc-1.
        _, gx, gy = W.grad(z[:n], z[n:])
        W.tlam.copy_(lam0); W.tact.copy_(act0); W.ttw.copy_(tw0)
        return torch.cat((gx, gy)).reshape(-1)

    def compliance(z, cells):
        """S (exact, H0^-1 by preconditioned CG) and S~ (the preconditioner) for the cells; no contact term in H0"""
        act0 = W.tact.clone()
        G = torch.stack([grad_g(z, c) for c in cells])
        W.tact.zero_()                                                             # H0: no contact term
        e0, g0, aux = J.eg(None, z)
        P = J.prec_at(prec, None, z, aux)
        H = J.hvp(None, z, g0)
        Wsol, its = [], []
        for k in range(len(cells)):
            bvec = G[k][None]
            u = torch.zeros_like(bvec); r = bvec.clone(); zz = P(r); p = zz.clone(); rz = float((r*zz).sum()); bn = float(bvec.norm())
            for it in range(1, 3000):
                Hp = H(p); al = rz/float((p*Hp).sum())
                u = u+al*p; r = r-al*Hp
                if float(r.norm()) <= 1e-7*bn:
                    break
                zz = P(r); rz1 = float((r*zz).sum()); p = zz+(rz1/rz)*p; rz = rz1
            Wsol.append(u[0]); its.append(it)
        Wm = torch.stack(Wsol)
        Sx = (G@Wm.T).cpu().numpy(); Sx = .5*(Sx+Sx.T)
        St = (G@P(G).T).cpu().numpy(); St = .5*(St+St.T)
        W.tact.copy_(act0)
        return Sx, St, its, G

    starts = []
    for k in range(1, args.passes+1):
        g0 = J.eg(None, z)[1]
        starts.append(float(g0.norm())/ref)
        act = torch.nonzero(W.tact > 0)[:, 0].cpu().numpy()
        lam_k = W.tlam.clone()
        W.grad(z[:n], z[n:]); c_start = W.tgap.clone()
        z, s_ = J.solve(None, z, prec, ref, tol=tol, maxit=2000)
        c = s_['aux'][2]
        print('pass %d: start residual %.2e ref, %d products, end residual %.2e ref, active %d cells' % (
            k, starts[-1], s_['products'], s_['residual'], len(act)), flush=True)
        for c_ in act:                                                             # per active cell: the multiplier used, the
            mu = float(lam_k[c_]-W.trho[c_]*c[c_])                                 # condition at the start and at the solution,
            print('   cell %d: lam %.3e, rho %.3e; c start %.3e, end %.3e; force lam - rho c start %.3e, end %.3e' % (   # the force
                c_, float(lam_k[c_]), float(W.trho[c_]), float(c_start[c_]), float(c[c_]),
                float(lam_k[c_]-W.trho[c_]*c_start[c_]), mu), flush=True)
        if len(act) and args.compliance:
            Sx, St, its, G = compliance(z, act)
            rho = W.trho[torch.as_tensor(act, device=dev)].cpu().numpy()
            Mx = np.eye(len(act))+cs*rho[:, None]*Sx
            fac = 1/np.linalg.eigvals(Mx).real
            fac1 = 1/(1+cs*rho*np.diag(Sx))
            Nmap = (np.eye(len(act))-np.linalg.solve(St, Sx))@np.linalg.inv(Mx)
            ev, V = np.linalg.eig(np.linalg.inv(Mx))
            j = np.argmax(ev.real); vec = np.abs(V[:, j].real); top = np.argsort(-vec)[:4]
            print('   S: eigenvalues min %.3e max %.3e (diagonal min %.3e); CG per solve %d-%d' % (
                np.linalg.eigvalsh(Sx).min(), np.linalg.eigvalsh(Sx).max(), np.diag(Sx).min(), min(its), max(its)), flush=True)
            print('   the update\'s map: largest factor %.3f (each cell alone: largest %.3e); with S~ (Newton-type): spectral radius %.3f; '
                  'S~ / S on the diagonal median %.2f' % (fac.max(), fac1.max(), np.abs(np.linalg.eigvals(Nmap)).max(), np.median(np.diag(St)/np.diag(Sx))), flush=True)
            cells = [int(act[t]) for t in top]
            print('   slowest mode on cells %s (weights %s); shared skin points between them: %s; |cos| of their dg: %s' % (
                cells, ' '.join('%.2f' % vec[t] for t in top),
                ' '.join('%d-%d:%d' % (cells[i], cells[j_], len(pts(cells[i]) & pts(cells[j_]))) for i in range(len(cells)) for j_ in range(i+1, len(cells))),
                ' '.join('%.3f' % abs(float(G[top[i]]@G[top[j_]])/float(G[top[i]].norm()*G[top[j_]].norm())) for i in range(len(top)) for j_ in range(i+1, len(top)))), flush=True)
        lr = W.tlam-W.trho*c
        W.tlam.copy_(lr.clamp(min=0))
        W.tact.copy_((lr > 0).to(W.tact.dtype))
    print('observed reduction of the passes\' start residuals: %s' % ' '.join('%.3f' % (starts[i+1]/starts[i]) for i in range(len(starts)-1)), flush=True)


if __name__ == '__main__':
    main()
