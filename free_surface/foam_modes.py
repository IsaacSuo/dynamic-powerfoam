"""The current model (A: walls, completion springs on the exposed spheres) against the boundary-skin model (C: Pi +
transfinite Shepard skin, lever c, skin free) on the new foam's voxel FEM modes (foam_exact.py), offline, 2026-10-22.
Each FEM mode (M-normalised, phi^T K phi = (2 pi f)^2) interpolated to the bodies (and, for B, to the skin points,
extrapolated outside the voxels) gives:
  A  stiffness u^T K_A u / (2 pi f)^2 and its split (bond normal / shear / bending / torsion, volume, completion);
  B  the skin on the FEM displacement: w^T K_C w / (2 pi f)^2, w = (u_x, u_y);
  C  the skin free: min over the skin, K_yy z = -K_yx u_x by batched conjugate gradients on Hessian-vector products
     (no dense matrix: ~76k unknowns), stiffness (u_x, z)^T K_C (u_x, z) / (2 pi f)^2 and its split per term;
  lumped mass sum m |u_x|^2 (the FEM's mass is 1) and the Rayleigh frequency ratio sqrt(stiffness / mass).
Checks: rest energy and gradient, a rigid rotation, the CG residuals."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.dem_fem_mode_match import interpolate
from experiments.foam_surface_completion import matrix_quat, polar_newton
from experiments.pacnerf_dem_forward import build
from free_surface.boundary_skin import BoundaryTerms
from free_surface.grad_weights import MidpointTerms
from free_surface.part_diag import interpolate_extrapolated
from free_surface.part_modes import PatchSprings
from free_surface.part_smooth import bond_parts

f64, dev = torch.float64, 'cuda'


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--out', type=Path, default=Path('output/round01_exact'))
    a.add_argument('--fem-h', type=float, default=.008)
    a.add_argument('--modes', type=int, default=20)
    a.add_argument('--batch', type=int, default=4)
    a.add_argument('--tol', type=float, default=1e-8)
    a.add_argument('--maxit', type=int, default=5000)
    a.add_argument('--skin', default='boundary_skin.npz')
    a.add_argument('--tag', default='')
    args = a.parse_args()
    t0 = time.time()
    E, nu = 23008.38, .3008
    cells = np.load(args.out/'dem_exact_single'/'cells.npz'); p, r = cells['points'], cells['radii']
    g = dict(np.load(args.out/'dem_exact_single'/'geometry.npz'))
    skin = dict(np.load(args.out/'sphere_skin.npz')); bsk = dict(np.load(args.out/args.skin))
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = BoundaryTerms(m, g, p, r, bsk, lever='c')
    mid = MidpointTerms(m, PatchSprings(g, r, skin, float(m.mu_ref), st.V), g)
    n, npt = st.n, st.npt; n3, v3 = 3*n, 3*npt
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    X, Y = st.X, st.Y
    print('bodies %d, skin points %d, regions %d; setup %.0f s' % (n, npt, len(bsk['part_site']), time.time()-t0), flush=True)
    # checks
    def EC(z):
        return st.energy_xy(th, X+z[:n3].view(-1, 3), Y+z[n3:].view(-1, 3))
    def EA(z):
        return mid.energy(th, X+z.view(-1, 3), smooth_polar=True)[0]
    zC = torch.zeros(n3+v3, dtype=f64, device=dev); zA = torch.zeros(n3, dtype=f64, device=dev)
    gC, gA = torch.func.grad(EC), torch.func.grad(EA)
    hvpC = torch.func.vmap(lambda v: torch.func.jvp(gC, (zC,), (v,))[1])
    hvpA = torch.func.vmap(lambda v: torch.func.jvp(gA, (zA,), (v,))[1])
    c = X.mean(0); size = float((X.max(0).values-X.min(0).values).max())
    w = torch.tensor([.3, -.8, .5], dtype=f64, device=dev)*.05
    rig = torch.cat((torch.cross(w.expand(n, 3), X-c, dim=1).reshape(-1), torch.cross(w.expand(npt, 3), Y-c, dim=1).reshape(-1)))
    cache = np.load(args.out/('fem_modes_h%g.npz' % args.fem_h))
    f_fem, Vf = cache['f'][:args.modes], cache['V'][:, :args.modes]
    fem = dict(nodes=cache['nodes'], node_grid=cache['node_grid'], lo=cache['lo'], h=float(cache['h']), masses=cache['masses'])
    Ux, _ = interpolate(fem, Vf, X.cpu().numpy())
    Uy, outside = interpolate_extrapolated(fem, Vf, Y.cpu().numpy())
    om2 = (2*np.pi*f_fem)**2
    ux = torch.as_tensor(Ux.transpose(2, 0, 1).reshape(args.modes, -1), dtype=f64, device=dev)          # (modes, n3)
    uy = torch.as_tensor(Uy.transpose(2, 0, 1).reshape(args.modes, -1), dtype=f64, device=dev)
    sA_scale = float(hvpA(ux[:1])[0]@ux[0])
    print('checks: rest |grad| A %.1e, C %.1e (of the force of mode 0); rigid rotation |K w| / |K u_0| A %.1e, C %.1e | skin points outside the voxels %d of %d (%.0f s)' % (
        float(gA(zA).norm()), float(gC(zC).norm()), float(hvpA(rig[None, :n3])[0].norm())/float(hvpA(ux[:1])[0].norm()),
        float(hvpC(rig[None])[0].norm())/float(hvpC(torch.cat((ux[:1], uy[:1]), 1))[0].norm()), int(outside.sum()), npt, time.time()-t0), flush=True)
    # block-Jacobi preconditioner for K_yy: the 3 x 3 block of every skin point, probed colour by colour (two skin points
    # couple when their cells are equal or bond neighbours: F_B, the bonds' rotations, the walls' volume and the variance
    # term reach one bond; a colour holds no two coupled points)
    t1 = time.time()
    nbr = [set([c_]) for c_ in range(n)]
    for i_, j_ in g['pairs']:
        nbr[i_].add(j_); nbr[j_].add(i_)
    cells_of = [set() for _ in range(npt)]
    for q_, c_ in zip(bsk['ent_pid'], bsk['ent_site']):
        cells_of[q_].add(int(c_))
    pts_of_cell = [[] for _ in range(n)]
    for q_ in range(npt):
        for c_ in cells_of[q_]:
            pts_of_cell[c_].append(q_)
    # the curvature reference Gamma_B of a cell is fitted from its neighbours' F, so a cell's skin terms reach the skin of
    # cells two bonds apart: colours must separate points whose cells are within two bonds
    nbr2 = [set().union(*(nbr[k_] for k_ in nbr[c_])) for c_ in range(n)]
    colour = -np.ones(npt, int)
    for q_ in range(npt):
        reach = set().union(*(nbr2[c_] for c_ in cells_of[q_]))
        used = {colour[p_] for c_ in reach for p_ in pts_of_cell[c_] if colour[p_] >= 0}
        k_ = 0
        while k_ in used:
            k_ += 1
        colour[q_] = k_
    ncol = int(colour.max())+1
    blocks = torch.zeros(npt, 3, 3, dtype=f64, device=dev)
    col_t = torch.as_tensor(colour, device=dev)
    probes = [(k_, d_) for k_ in range(ncol) for d_ in range(3)]
    for s_ in range(0, len(probes), args.batch):
        chunk = probes[s_:s_+args.batch]
        Vp = torch.zeros(len(chunk), npt, 3, dtype=f64, device=dev)
        for b_, (k_, d_) in enumerate(chunk):
            Vp[b_, col_t == k_, d_] = 1.
        Kv = hvpC(torch.cat((torch.zeros(len(chunk), n3, dtype=f64, device=dev), Vp.reshape(len(chunk), -1)), 1))[:, n3:].view(len(chunk), npt, 3)
        for b_, (k_, d_) in enumerate(chunk):
            sel = col_t == k_
            blocks[sel, :, d_] = Kv[b_, sel]
    blocks = .5*(blocks+blocks.transpose(1, 2))
    ev = torch.linalg.eigvalsh(blocks)
    qs = torch.randperm(npt, generator=torch.Generator().manual_seed(1))[:3].tolist()
    chk = 0.
    for q_ in qs:
        Vq = torch.zeros(3, npt, 3, dtype=f64, device=dev); Vq[[0, 1, 2], q_, [0, 1, 2]] = 1.
        Kq = hvpC(torch.cat((torch.zeros(3, n3, dtype=f64, device=dev), Vq.reshape(3, -1)), 1))[:, n3:].view(3, npt, 3)[:, q_].T
        chk = max(chk, float((Kq-blocks[q_]).abs().max()/blocks[q_].abs().max()))
    Binv = torch.linalg.inv(blocks)
    print('preconditioner: %d colours, %d probes; block eigenvalues min / max over points %.1e (smallest %.1e, largest %.1e); probe check on 3 points %.1e (%.0f s)' % (
        ncol, len(probes), float(ev[:, 0].min()/ev[:, 2].max()), float(ev[:, 0].min()), float(ev[:, 2].max()), chk, time.time()-t1), flush=True)
    if float(ev[:, 0].min()) <= 0:
        bad = torch.nonzero(ev[:, 0] <= 0)[:, 0]
        print('STOP: %d skin points with a diagonal block not positive definite (an SPD K_yy has none): smallest eigenvalues %s' % (
            len(bad), np.round(ev[bad, 0][:10].cpu().numpy(), 4).tolist()), flush=True)
        return
    prec = lambda rv: torch.einsum('qij,bqj->bqi', Binv, rv.view(rv.shape[0], npt, 3)).reshape(rv.shape[0], -1)
    mass = m.mass.to(f64)
    rows = []
    print('mode | f_FEM | stiffness A, B (skin on FEM), C (skin free) | lumped mass | frequency A, C | CG iterations, residual | C split', flush=True)
    for s0 in range(0, args.modes, args.batch):
        t1 = time.time()
        U = ux[s0:s0+args.batch]; Bn = len(U)
        sA = (hvpA(U)*U).sum(1)
        WB = torch.cat((U, uy[s0:s0+args.batch]), 1)
        sB = (hvpC(WB)*WB).sum(1)
        # K_yy z = -K_yx u_x, batched CG
        zx = torch.zeros(Bn, n3, dtype=f64, device=dev)
        Kyy = lambda vy: hvpC(torch.cat((zx, vy), 1))[:, n3:]
        bvec = -hvpC(torch.cat((U, torch.zeros(Bn, v3, dtype=f64, device=dev)), 1))[:, n3:]
        z = torch.zeros_like(bvec); res = bvec.clone(); zr = prec(res); pv = zr.clone(); rz = (res*zr).sum(1); bn = bvec.norm(dim=1)
        it = 0
        while it < args.maxit and float((res.norm(dim=1)/bn).max()) > args.tol:
            Ap = Kyy(pv); al = rz/(pv*Ap).sum(1)
            z = z+al[:, None]*pv; res = res-al[:, None]*Ap
            zr = prec(res); rz1 = (res*zr).sum(1); pv = zr+(rz1/rz)[:, None]*pv; rz = rz1; it += 1
            if it % 200 == 0:
                print('   modes %d..%d: PCG iteration %d, relative residual max %.1e (%.0f s)' % (s0, s0+Bn-1, it, float((res.norm(dim=1)/bn).max()), time.time()-t1), flush=True)
        WC = torch.cat((U, z), 1)
        sC = (hvpC(WC)*WC).sum(1)
        true_res = float(((Kyy(z)-bvec).norm(dim=1)/bn).max())
        for q in range(Bn):
            k = s0+q
            u = U[q].view(-1, 3); ms = float((mass*(u*u).sum(1)).sum())
            eps = 1e-6/float(WC[q].abs().max())
            x, y = X+eps*u, Y+eps*z[q].view(-1, 3)
            T = {k_: float(v_)*2/eps**2/om2[k] for k_, v_ in st.energy_terms(th, x, y).items()}
            q_ = matrix_quat(polar_newton(st.F(x, y)))
            T.update({k_: v_*2/eps**2/om2[k] for k_, v_ in bond_parts(m, th, x, q_).items()}); T.pop('bonds')
            xa = X+eps*u; Fa = mid.F(xa); Ra = polar_newton(Fa)
            TA = {k_: v_*2/eps**2/om2[k] for k_, v_ in bond_parts(m, th, xa, matrix_quat(Ra)).items()}
            TA.update(volume=float(mid.volume_energy(xa, Fa, Ra, m.k_v))*2/eps**2/om2[k], completion=float(mid.completion_energy(Fa, Ra, th))*2/eps**2/om2[k])
            row = dict(mode=k, f_fem=float(f_fem[k]), A=float(sA[q])/om2[k], B=float(sB[q])/om2[k], C=float(sC[q])/om2[k], mass=ms, cg_iterations=it, cg_residual=true_res,
                       split_C=T, split_A=TA)
            row['freq_A'], row['freq_C'] = float(np.sqrt(row['A']/ms)), float(np.sqrt(row['C']/ms))
            rows.append(row)
            print('%2d | %.3f | %.4f %.4f %.4f | %.4f | %.4f %.4f | %d %.1e | %s | A: %s (%.0f s)' % (
                k, row['f_fem'], row['A'], row['B'], row['C'], ms, row['freq_A'], row['freq_C'], it, true_res,
                ' '.join('%s %.3f' % (k_, v_) for k_, v_ in T.items()), ' '.join('%s %.3f' % (k_, v_) for k_, v_ in TA.items()), time.time()-t1), flush=True)
    R = {k_: float(np.median([r_[k_] for r_ in rows])) for k_ in ('A', 'B', 'C', 'mass', 'freq_A', 'freq_C')}
    print('median: stiffness A %.4f, B %.4f, C %.4f | lumped mass %.4f | frequency A %.4f, C %.4f (%.0f s)' % (R['A'], R['B'], R['C'], R['mass'], R['freq_A'], R['freq_C'], time.time()-t0), flush=True)
    (args.out/('foam_modes_h%g%s.json' % (args.fem_h, args.tag))).write_text(json.dumps(dict(rows=rows, median=R), indent=1))


if __name__ == '__main__':
    main()
