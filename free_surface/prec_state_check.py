"""Is the first pass slow because the preconditioner's blocks are those at rest (2026-10-24)? At a saved state
(skin_drop_check.py --save-state N, the midpoint rule) the step's first contact pass (JointSolver.solve, active set and
multipliers held) from the integrator's start, 2 x 2: the skin's group blocks at rest (K_yy at rest times 1 + 2 beta / dt,
turned by the bodies' rotations: build_model) or probed at the start state (K_yy + 2 Et''/dt, contact term off; the
active contacts by Sherman-Morrison as always), the bodies' blocks at rest (R (K_B + 2 D_B / dt) R^T) or probed there
(skin held); products, outer iterations, products per e-fold of the residual, |z - z_rest| / |z_rest - z_start|. Also the
drift: per block the generalised eigenvalues of (turned rest block, current block), whose spread is what the rest
blocks cost the CG.
  python free_surface/prec_state_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step900.npz"""
import math
import time
from pathlib import Path

import numpy as np
import torch

from free_surface.joint_step import MidpointIntegrator
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def body_blocks_at(W, st, x, y, h=1e-7):
    """the bodies' 3 x 3 blocks of the loss's Hessian at (x, y) with the skin held, contact term off (body_blocks'
    colouring, one-sided differences), world frame; symmetrised"""
    n = st.n
    i, j = st.i.cpu().numpy(), st.j.cpu().numpy()
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    A = ((A+A.T+sp.eye(n)) > 0).astype(np.int32).tocsr()
    A2 = (A@A).tocsr()
    col = -np.ones(n, dtype=int)
    for a in range(n):
        used = set(col[A2.indices[A2.indptr[a]:A2.indptr[a+1]]].tolist())
        c = 0
        while c in used:
            c += 1
        col[a] = c
    colt = torch.as_tensor(col, device=dev)
    B = torch.zeros(n, 3, 3, dtype=f64, device=dev)
    g0 = W.grad(x, y)[1]
    for c in range(int(col.max())+1):
        m = colt == c
        for k in range(3):
            w = torch.zeros_like(x); w[m, k] = 1.
            B[m, :, k] = ((W.grad(x+h*w, y)[1]-g0)/h)[m]
    return .5*(B+B.transpose(1, 2)), 3*(int(col.max())+1)


def gen_eig(A, B):
    """generalised eigenvalues of B v = l A v per batch (A, B symmetric, A positive definite)"""
    L = torch.linalg.cholesky(A)
    Li = torch.linalg.inv(L)
    return torch.linalg.eigvalsh(Li@B@Li.transpose(-1, -2))


def spread(lam):
    lo, hi = lam.min(-1).values, lam.max(-1).values
    q = lambda t, p: float(torch.quantile(t, p))
    return 'min eigenvalue median %.2f (p10 %.2f, min %.2f); max median %.2f (p90 %.2f, max %.2f); ratio max/min median %.2f (p90 %.1f, max %.1f)' % (
        q(lo, .5), q(lo, .1), float(lo.min()), q(hi, .5), q(hi, .9), float(hi.max()), q(hi/lo, .5), q(hi/lo, .9), float((hi/lo).max()))


def main():
    a = parser()
    a.add_argument('--state', type=Path, required=True)
    a.add_argument('--gravity', type=float, default=9.8)
    args = a.parse_args()
    b = build_model(args, contact=dict(normal=(0., 1., 0.), c0=0., shift=True))
    st, W, S, prec, mass = b['st'], b['W'], b['S'], b['prec'], b['mass']
    z0 = np.load(args.state)
    T = lambda k: torch.as_tensor(z0[k], device=dev)
    fg = -args.gravity*mass[:, None]*torch.tensor([0., 1., 0.], dtype=f64, device=dev)
    ref = float(fg.norm()); tol, dt = args.tols[0], args.dt
    I = MidpointIntegrator(b, dt, T('x'), T('v'), T('ym1'), fg, tol=tol, ref=ref, contact=True)
    I.x, I.v, I.a, I.xm1, I.ym1, I.xm2, I.ym2 = (T(k) for k in ('x', 'v', 'a', 'xm1', 'ym1', 'xm2', 'ym2'))
    def restore():
        W.tlam.copy_(T('lam')); W.tact.copy_(T('act')); W.tgoff.copy_(T('goff')); W.trho.copy_(T('rho'))
        for k_, r_ in W.tref.items():
            r_.copy_(T('ref_'+k_))
        W.total_mode()
    restore()
    J, n = I.J, W.n
    x, v = I.x, I.v
    xs = x+.5*dt*v+.25*dt*dt*I.a                                                   # the integrator's start (MidpointIntegrator.step)
    zs = torch.cat((xs, I.predict_skin(S, xs, I.xm1, I.xm2, I.ym1, I.ym2)))
    J.set_step(x, x+.5*dt*v, fg)
    print('state %s: %d bodies, %d skin points, %d active contacts' % (args.state.name, n, W.npt, int((W.tact > 0).sum())), flush=True)

    # blocks at the start state (contact term off: the active contacts enter by Sherman-Morrison as at rest)
    t0 = time.perf_counter()
    W.total_mode(); W.ttw[6] = 0.
    Bb_cur, nb = body_blocks_at(W, st, zs[:n], zs[n:])
    calls, grad0 = [0], W.grad                                                     # count the skin probe's gradients
    def grad_counted(x_, y_):
        calls[0] += 1
        return grad0(x_, y_)
    W.grad = grad_counted
    t1 = time.perf_counter()
    K_cur = S.probe_K(zs[:n], zs[n:])
    del W.grad
    W.total_mode()
    Bs_cur = S.blocks_probed(K_cur, S.bodies)
    torch.cuda.synchronize()
    print('blocks probed at the start state: bodies %d gradients (%.1f s), skin %d gradients (central differences; %.1f s with the colouring)' % (
        nb, t1-t0, calls[0], time.perf_counter()-t1), flush=True)

    # drift: rest blocks as the solve uses them (turned by the start state's rotations) against the current ones
    W.grad(zs[:n], zs[n:]); R = W.tR.clone()
    Bb_rest = R@(J.Kb+J.wvis/dt*J.Db)@R.transpose(1, 2)
    Mi = J.c_in*mass[:, None, None]*torch.eye(3, dtype=f64, device=dev)
    print('bodies, (inertia + rest block) vs (inertia + current block): ' + spread(gen_eig(Mi+Bb_rest, Mi+Bb_cur)), flush=True)
    print('bodies, inertia %.3e vs the blocks\' largest eigenvalue: rest median %.3e, current median %.3e (per unit mass)' % (
        J.c_in, float(torch.quantile(torch.linalg.eigvalsh(Bb_rest)[:, -1]/mass, .5)), float(torch.quantile(torch.linalg.eigvalsh(Bb_cur)[:, -1]/mass, .5))), flush=True)
    g = S.bodies
    Rg = R[g['owner']]                                                             # each group turned by its owner's rotation
    Sg, G = g['S'], g['G']
    lam = []
    for c0 in range(0, G, 256):                                                    # (chunks: G blocks of 3 S square)
        c1 = min(G, c0+256)
        RR = torch.zeros(c1-c0, 3*Sg, 3*Sg, dtype=f64, device=dev)
        for s_ in range(Sg):
            RR[:, 3*s_:3*s_+3, 3*s_:3*s_+3] = Rg[c0:c1]
        lam.append(gen_eig(RR@b['blocks'][c0:c1]@RR.transpose(1, 2), Bs_cur[c0:c1]))
    print('skin groups (%d, up to %d points), rest (turned) vs current: ' % (G, Sg) + spread(torch.cat(lam)), flush=True)

    # the preconditioners: skin rest = b['prec']; skin current = a Warp preconditioner of the current blocks, not turned
    prec_cur = S.preconditioner(Bs_cur, S.bodies)
    S.contact_setup(prec_cur, S.kappa)
    eye = torch.eye(3, dtype=f64, device=dev).expand(n, 3, 3).contiguous()
    class Unturned:
        def __init__(self, p):
            self.p, self.ck = p, p.ck
        def __call__(self, Rb):
            return self.p(eye)
    skin_cur = Unturned(prec_cur)
    restore()

    def prec_at(make_prec, body_cur):
        def f_(mp, x_, z_, aux):
            fs = mp(aux[1])                                                        # (as JointSolver.prec_at, the active set held)
            fs.wprec.contact_update(S.contact_weights(aux[4]), W.trho*float(W.tcpar[1]*W.tcpar[0]**2)*(W.tact > 0))
            if body_cur:
                Bb = Bb_cur
            else:
                R_ = aux[1]; Bb = R_@(J.Kb+J.wvis/dt*J.Db)@R_.transpose(1, 2)
            Pi = torch.linalg.inv(Mi+Bb)
            def f(r):
                k = r.shape[0]
                r = r.view(k, -1, 3)
                zb = torch.einsum('nij,knj->kni', Pi, r[:, :n])
                zs_ = fs(r[:, n:].reshape(k, -1)).view(k, -1, 3)
                return torch.cat((zb, zs_), 1).reshape(k, -1)
            f.wprec = fs.wprec
            f.Pi = Pi
            return f
        return f_

    r0 = float(J.eg(None, zs)[1].norm())/ref
    print('first pass from the start: residual %.3e of |m g| (bodies + skin), tol %.0e' % (r0, tol), flush=True)
    res = {}
    for skin_c in (False, True):
        for body_c in (False, True):
            restore()
            J.prec_at = prec_at(None, body_c)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            z, s_ = J.solve(None, zs.clone(), skin_cur if skin_c else prec, ref, tol=tol, maxit=2000)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            res[(skin_c, body_c)] = z
            ef = math.log(r0/max(s_['residual'], 1e-300))
            print('skin %-7s bodies %-7s: %4d products, %2d outer, %d rejected, %d negative; residual %.2e; %.1f products per e-fold; %.0f ms' % (
                'current' if skin_c else 'rest', 'current' if body_c else 'rest', s_['products'], s_['outer'], s_['rejected'], s_['negative'],
                s_['residual'], s_['products']/ef, 1e3*(t2-t1)), flush=True)
    z0_ = res[(False, False)]
    for k_, z in res.items():
        print('|z - z(rest, rest)| / |z(rest, rest) - z_start| %.1e  (skin %s, bodies %s)' % (
            float((z-z0_).norm()/(z0_-zs).norm()), 'current' if k_[0] else 'rest', 'current' if k_[1] else 'rest'), flush=True)


if __name__ == '__main__':
    main()
