"""Re-cutting the Power Foam at a deformed state, offline (2026-10-24, the idea: rebuild the cells from the bodies' motion
every step). The jack is one body per cell, a body at its site, so the deformed sites are the bodies' positions. At a
saved step (skin_drop_check.py --save-state N; bodies at the midpoint, the skin with them for the model's own volume
ratio J = exp(log J) per body, SkinWarp.tlj):
  (a) the radii kept: the exact geometry (gpu_geometry, the union of balls cut by the power diagram) of the moved sites;
      each cell's volume against V0 J (V0 its rest volume), the walls that vanished / appeared, the walls' areas, the
      exposed area, hidden sites;
  (b) the radii solved so that every cell holds V0 J: damped Newton on the weights w = r^2 along a path of the sites
      from rest (10-24: a plain Newton from the rest radii diverged, cells vanished), whose Jacobian is exact and
      symmetric (checked against differences): dV_i/dw_j = -A_ij / (2 L_ij) on a wall, dV_i/dw_i = sum_j A_ij / (2 L_ij) + A_patch,i / (2 r_i)
      (a volume moves only by its boundary's normal velocity: the radical plane at h_i = (L^2 + r_i^2 - r_j^2) / (2 L),
      the sphere at r_i); the structure (regular triangulation) recomputed every iteration; the radii's change, the
      iterations, the time of a structure and of the values.
  python free_surface/retess_check.py --out output/round01_pruned_exact --dt 2.0833333e-4 --tols 1e-4 --beta 0.01 --integrator midpoint --state output/round01_pruned_exact/skin_drop_state_step900.npz"""
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch

from free_surface.gpu_geometry import Geometry, structure_triangulation
from free_surface.skin_contact_check import f64, dev
from free_surface.skin_dynamics_check import build_model, parser


def geometry(p, r):
    t0 = time.perf_counter()
    S = structure_triangulation(p.cpu().numpy(), r.cpu().numpy())
    t1 = time.perf_counter()
    G = Geometry(p, r, S)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return G, S, t1-t0, t2-t1


def walls(G):
    keep = G.wall_A > 0
    P = G.P[keep]
    return {(int(a), int(b)): float(A) for (a, b), A in zip(P.cpu().numpy(), G.wall_A[keep].cpu().numpy())}


def q(x, ps=(0, 1, 5, 50, 95, 99, 100)):
    return ' '.join('p%d %+.2e' % (p_, np.percentile(x, p_)) for p_ in ps)


def main():
    a = parser()
    a.add_argument('--state', type=Path, required=True)
    a.add_argument('--newton', type=int, default=30)
    a.add_argument('--path', type=int, default=50)
    args = a.parse_args()
    out = Path(args.out)
    cells = np.load(out/'dem_exact_single'/'cells.npz')
    p0 = torch.as_tensor(cells['points'], dtype=f64, device=dev); r0 = torch.as_tensor(cells['radii'], dtype=f64, device=dev)
    Vm = cells['masses']
    b = build_model(args)
    W = b['W']
    z0 = np.load(args.state)
    for k_, r_ in W.tref.items():
        r_.copy_(torch.as_tensor(z0['ref_'+k_], device=dev))
    p1 = torch.as_tensor(z0['xm1'], dtype=f64, device=dev)
    W.grad(p1, torch.as_tensor(z0['ym1'], dtype=f64, device=dev))
    J = torch.exp(W.tlj).double()
    print('state %s: J (the model\'s volume ratio per body) %s' % (args.state.name, q((J-1).cpu().numpy())), flush=True)

    G0, S0, ts0, tv0 = geometry(p0, r0)
    V0 = G0.V.detach()
    print('rest: %d sites (hidden %d), walls %d, volume %.6f m^3 (DEM masses %.6f, cells differ max %.1e relative), exposed area %.4f m^2; '
          'structure %.2f s, values %.2f s' % (len(p0), int(S0['hidden'].sum()), int((G0.wall_A > 0).sum()), float(V0.sum()), Vm.sum(),
                                                float(((V0.cpu().numpy()-Vm)/Vm).__abs__().max()), float(G0.patch_A.sum()), ts0, tv0), flush=True)
    surf = (G0.patch_A > 0).cpu().numpy()
    tiny = (V0 < torch.quantile(V0, .5)/100).cpu().numpy()
    print('   cells below 1/100 of the median volume: %d (%d on the surface); smallest %.2e of the median' % (
        int(tiny.sum()), int((tiny & surf).sum()), float(V0.min()/torch.quantile(V0, .5))), flush=True)
    Vt = V0*J                                                                       # the target: the material's volume now

    # (a) the radii kept
    G1, S1, ts1, tv1 = geometry(p1, r0)
    V1 = G1.V.detach()
    e = (V1/Vt-1).cpu().numpy()
    print('(a) radii kept: hidden %d, volume %.6f m^3 against sum V0 J %.6f; V / (V0 J) - 1: %s' % (
        int(S1['hidden'].sum()), float(V1.sum()), float(Vt.sum()), q(e)), flush=True)
    print('    interior cells: %s; surface cells: %s; the tiny cells: %s' % (q(e[~surf], (5, 50, 95)), q(e[surf], (5, 50, 95)), q(e[tiny], (0, 50, 100)) if tiny.any() else '-'), flush=True)
    w0, w1 = walls(G0), walls(G1)
    lost, new = set(w0)-set(w1), set(w1)-set(w0)
    common = sorted(set(w0) & set(w1))
    dA = np.array([w1[k]/w0[k]-1 for k in common])
    A_lost = sum(w0[k] for k in lost); A_new = sum(w1[k] for k in new)
    print('    walls: %d at rest, %d now; %d vanished (%.2f %% of the wall area), %d appeared (%.2f %%); kept walls area change %s; exposed area %.4f m^2; structure %.2f s, values %.2f s' % (
        len(w0), len(w1), len(lost), 100*A_lost/sum(w0.values()), len(new), 100*A_new/sum(w1.values()), q(dA, (1, 50, 99)), float(G1.patch_A.sum()), ts1, tv1), flush=True)

    # (b) the radii solved for V = V0 J: damped Newton on w = r^2 (semi-discrete optimal transport's: the full step
    # halved until every cell keeps a positive volume and the residual falls), along a path of the sites from rest to
    # the state in --path pieces (each from the last radii: what re-cutting every step meets), targets V0 (1 + s (J - 1))
    n = len(p0)

    def hessian(G, r):
        keep = G.wall_A > 0
        i, j = G.i[keep], G.j[keep]
        L = (G.p[j]-G.p[i]).norm(dim=1)
        c = (G.wall_A[keep]/(2*L)).detach()
        d = torch.zeros(n, dtype=f64, device=dev).index_add(0, i, c).index_add(0, j, c)+G.patch_A.detach()/(2*r)
        d = torch.where(d > 0, d, d.max())                                         # (no wall, no patch: hidden; repair() first)
        I, Jj, C = i.cpu().numpy(), j.cpu().numpy(), c.cpu().numpy()
        return sp.coo_matrix((np.r_[-C, -C, d.cpu().numpy()], (np.r_[I, Jj, np.arange(n)], np.r_[Jj, I, np.arange(n)])), shape=(n, n)).tocsc()

    # the analytic Jacobian against central differences at rest (the structure held)
    H0 = hessian(G0, r0)
    gen = torch.Generator(device=dev).manual_seed(0)
    dw = torch.randn(n, generator=gen, dtype=f64, device=dev)*(r0*r0)*1e-6
    Vp = Geometry(p0, torch.sqrt(r0*r0+dw), S0).V.detach(); Vn = Geometry(p0, torch.sqrt(r0*r0-dw), S0).V.detach()
    fd = (Vp-Vn)/2
    an = torch.as_tensor(H0@dw.cpu().numpy(), dtype=f64, device=dev)
    print('(b) Jacobian at rest against central differences (a random w change of 1e-6 relative): relative %.1e' % float((an-fd).norm()/fd.norm()), flush=True)

    from scipy.spatial import cKDTree

    def repair(pk, r, hid):
        """each hidden site's weight raised until it owns its own position: w_i = max_j (w_j - |p_i - p_j|^2) (1 + 1e-9)
        over the sites whose balls could reach it"""
        P_, w = pk.cpu().numpy(), (r*r).cpu().numpy()
        tree = cKDTree(P_)
        for i_ in np.nonzero(hid)[0]:
            nb = tree.query_ball_point(P_[i_], float(np.sqrt(w.max()))+float(np.sqrt(w[i_])))
            nb = [j_ for j_ in nb if j_ != i_]
            need = max(w[j_]-float(((P_[i_]-P_[j_])**2).sum()) for j_ in nb)
            w[i_] = max(w[i_], need*(1+1e-9)+1e-12*w.max())
        return torch.sqrt(torch.as_tensor(w, dtype=f64, device=dev))

    r = r0.clone()
    tS = tV = 0.; nev = 0; its = []; nrep = 0
    for k in range(1, args.path+1):
        sk = k/args.path
        pk = p0+sk*(p1-p0)
        Vt_k = V0*(1+sk*(J-1))
        G, S, ts, tv = geometry(pk, r); tS += ts; tV += tv; nev += 1
        h0 = int(S['hidden'].sum())
        for _ in range(20):                                                        # hidden (or empty) cells: repaired first
            if int(S['hidden'].sum()) == 0 and bool((G.V > 0).all()):
                break
            hid = (S['hidden'].cpu().numpy() > 0) | (G.V.detach().cpu().numpy() <= 0)
            r = repair(pk, r, hid); nrep += int(hid.sum())
            G, S, ts, tv = geometry(pk, r); tS += ts; tV += tv; nev += 1
        if h0:
            print('   piece %d: %d sites hidden at the start (repaired); smallest cell then %.2e of its target' % (
                k, h0, float((G.V.detach()/Vt_k).min())), flush=True)
        res = Vt_k-G.V.detach()
        it = 0
        while float((res/Vt_k).abs().max()) > 1e-9 and it < args.newton:
            it += 1
            dw = torch.as_tensor(spl.spsolve(hessian(G, r), res.cpu().numpy()), dtype=f64, device=dev)
            if float((dw/(r*r)).abs().max()) < 1e-10:                              # the step at the rounding: converged
                break
            w = r*r; st_ = 1.; nr = float(res.norm()); trace = []
            while True:
                wt = w+st_*dw
                if bool((wt > 0).all()):
                    Gt, St, ts, tv = geometry(pk, torch.sqrt(wt)); tS += ts; tV += tv; nev += 1
                    rt = Vt_k-Gt.V.detach()
                    trace.append((st_, int(St['hidden'].sum()), float(Gt.V.min()), float(rt.norm())/nr, int((Gt.wall_A > 0).sum())))
                    if int(St['hidden'].sum()) == 0 and bool((Gt.V > 0).all()) and float(rt.norm()) <= (1-st_/2)*nr:
                        break
                else:
                    trace.append((st_, -1, float('nan'), float('nan'), 0))
                st_ *= .5
                if st_ < 1e-6:
                    print('   stall: piece %d, Newton %d: |res| %.3e, |dw / w| max %.2e; walls at the start %d, hidden %d, min V %.3e; '
                          'trace (step, hidden, min V, |res| ratio, walls): %s' % (k, it, nr, float((dw/w).abs().max()), int((G.wall_A > 0).sum()),
                                                                                    int(S['hidden'].sum()), float(G.V.min()),
                                                                                    '; '.join('%.2g %d %.2e %.4f %d' % t_ for t_ in trace[:12])), flush=True)
                    i_ = int((dw/w).abs().argmax())
                    print('   the largest relative w change at site %d: dw / w %.2e, V %.3e, target %.3e, V0 %.3e, r0 %.3e, patch %.3e, walls %d' % (
                        i_, float(dw[i_]/w[i_]), float(G.V[i_]), float(Vt_k[i_]), float(V0[i_]), float(r0[i_]), float(G.patch_A[i_]),
                        int(((G.i == i_) | (G.j == i_))[G.wall_A > 0].sum())), flush=True)
                    raise RuntimeError('damped Newton stalled at piece %d' % k)
            r, G, S, res = torch.sqrt(wt), Gt, St, rt
        its.append(it)
        if k in (1, 2, 5, args.path//2, args.path) or it > 6:
            print('   piece %d / %d: Newton %d, max |V / target - 1| %.1e, hidden %d, radii r / r0 - 1 %s' % (
                k, args.path, it, float((res/Vt_k).abs().max()), int(S['hidden'].sum()), q((r/r0-1).cpu().numpy(), (0, 50, 100))), flush=True)
    dr = (r/r0-1).cpu().numpy()
    print('(b) along %d pieces: Newton iterations per piece %s (mean %.1f), geometry evaluations %d (%.1f per piece), hidden-site repairs %d' % (
        args.path, np.bincount(its).tolist(), np.mean(its), nev, nev/args.path, nrep), flush=True)
    print('    radii change r / r0 - 1: %s; interior %s, surface %s; the tiny cells %s' % (
        q(dr), q(dr[~surf], (5, 50, 95)), q(dr[surf], (5, 50, 95)), q(dr[tiny], (0, 50, 100)) if tiny.any() else '-'), flush=True)
    w2 = walls(G)
    lost2, new2 = set(w0)-set(w2), set(w2)-set(w0)
    print('    walls against rest: %d vanished (%.2f %% of the area), %d appeared (%.2f %%); exposed area %.4f m^2 (rest %.4f); volume %.6f (target %.6f); '
          'per evaluation: structure %.2f s, values %.2f s' % (len(lost2), 100*sum(w0[k_] for k_ in lost2)/sum(w0.values()), len(new2),
                                                                100*sum(w2[k_] for k_ in new2)/sum(w2.values()), float(G.patch_A.sum()), float(G0.patch_A.sum()),
                                                                float(G.V.sum()), float(Vt.sum()), tS/nev, tV/nev), flush=True)


if __name__ == '__main__':
    main()
