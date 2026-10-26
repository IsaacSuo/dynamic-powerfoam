"""Objectivity of the skin model under rotations (2026-10-23): the lateral term now pairs R_B^T delta with the rest-frame
surface gradients and projectors (sphere_model.SkinTerms). A non-uniform state (smooth body deformation, rough skin) and
the same state turned by a rigid rotation must have the same energy, term by term. Also printed: the earlier lateral term
(delta in the world frame) on the same states, and the difference new - old at small amplitude (third order: the Hessian at
rest is unchanged)."""
import argparse
import math

import numpy as np
import torch

from experiments.foam_surface_completion import polar_newton
from free_surface.skin_contact_check import setup, f64, dev


def lateral_world(st, theta, x, y):
    """the lateral term with delta in the world frame (before 10-23), from the model's own pieces"""
    F = st.F(x, y)
    Fe = F[st.e_body]
    delta = y[st.e_pid]-x[st.e_body]-torch.einsum('eij,ej->ei', Fe, st.e_d)
    g = st.wb[:, None, None, None]*(F[st.j]-F[st.i])[:, :, :, None]*st.dX[:, None, None, :]
    G = torch.zeros(st.n, 3, 3, 3, dtype=x.dtype, device=x.device).index_add(0, st.i, g).index_add(0, st.j, g)
    Gam = torch.einsum('bijl,blk->bijk', G, st.Mb_inv); Gam = .5*(Gam+Gam.transpose(2, 3))
    delta = delta-.5*torch.einsum('eijk,ej,ek->ei', Gam[st.e_body], st.e_d, st.e_d)
    da, db = delta[st.pr_a], delta[st.pr_b]
    lam = st.model.k_v*torch.exp(theta[0, 4]); sm = torch.exp(theta[:, 5])
    mu_p = (st.mu*sm[st.p_body])[st.pr_part]; r_p = st.p_r[st.pr_part]
    kap = lam*mu_p/(lam+2*mu_p)
    tU = torch.einsum('ni,nij,nj->n', da, st.pr_U, db); tT = torch.einsum('ni,nij,nj->n', da, st.pr_T, db)
    tTs = torch.einsum('ni,nij,nj->n', db, st.pr_T, da)
    return (r_p/3*(.5*mu_p*(tU+tTs)+kap*tT)).sum()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--case', choices=('part', 'foam'), default='foam')
    a.add_argument('--skin', default='boundary_skin.npz')
    a.add_argument('--interp', default='shepard0')
    args = a.parse_args()
    terms, S = setup(args.case, args.skin, args.interp)
    st = terms
    X, Y = st.X, st.Y; n = len(X)
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    size = float((X.max(0).values-X.min(0).values).max()); cen = X.mean(0); rmed = float(S.r.median())
    gen = torch.Generator(device=dev).manual_seed(0)
    kv = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*2*math.pi/size
    amp = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*.03*size
    ph = torch.rand(3, generator=gen, dtype=f64, device=dev)*2*math.pi
    field = lambda Z: torch.sin(Z@kv.T+ph)@amp.T
    noise = torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)*.05*rmed
    x0, y0 = X+field(X), Y+field(Y)+noise
    T0 = {k: float(v) for k, v in st.energy_terms(th, x0, y0).items()}
    L0 = float(lateral_world(st, th, x0, y0))
    tot0 = sum(T0.values())
    print('[%s] state: smooth deformation (3%% of the size) + skin offsets (5%% of the median radius); energy %.4e; terms %s; '
          'lateral, world-frame deviation (before) %.4e' % (args.case, tot0, ' '.join('%s %.4e' % kv_ for kv_ in T0.items()), L0), flush=True)
    for ang in (.3, 1., 2.5):
        ax = torch.nn.functional.normalize(torch.randn(3, generator=gen, dtype=f64, device=dev), dim=0)
        K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=f64, device=dev)
        Q = torch.eye(3, dtype=f64, device=dev)+math.sin(ang)*K+(1-math.cos(ang))*K@K
        t = torch.tensor([.1, -.2, .05], dtype=f64, device=dev)
        x1, y1 = (x0-cen)@Q.T+cen+t, (y0-cen)@Q.T+cen+t
        T1 = {k: float(v) for k, v in st.energy_terms(th, x1, y1).items()}
        L1 = float(lateral_world(st, th, x1, y1))
        rel = {k: abs(T1[k]-T0[k])/tot0 for k in T0}
        print('[%s] rotation %.1f rad: |change| / total energy per term %s; lateral before 10-23: %.4e -> %.4e (change %.1e of the total)' % (
            args.case, ang, ' '.join('%s %.1e' % kv_ for kv_ in rel.items()), L0, L1, abs(L1-L0)/tot0), flush=True)
    # new - old at small amplitude: third order (Hessian at rest unchanged)
    u = field(X)/.03; uy = field(Y)/.03+noise/(.05*rmed)*.01*size
    for eps in (1e-3, 5e-4, 2.5e-4):
        x_, y_ = X+eps*u, Y+eps*uy
        d = float(st.energy_terms(th, x_, y_)['outer_lateral'])-float(lateral_world(st, th, x_, y_))
        print('[%s] amplitude %.2e: lateral new - old %.3e (energy %.3e)' % (args.case, eps, d, float(st.energy_xy(th, x_, y_))), flush=True)


if __name__ == '__main__':
    main()
