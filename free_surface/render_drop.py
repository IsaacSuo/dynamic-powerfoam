"""Render the jack's drop with its Power Foam (2026-10-25), PowerSim's way (Nguyen & Bhattad 2026, eq. 6-7): every cell
is a body (one body per cell, its site at the body), so per record the site goes to the body's position, the dipole
plane and the radiance axes turn with the polar rotation R of the body's F_B (q = q0 (x) conj(quat(R)), a = R a0), the
radius scales by det(F_B)^(1/3), and the adjacency is rebuilt as the renderer's own Cech complex (BVH). The foam is the
run the jack was cut from (material cells only: dem_exact_single/cells.npz source_index); its code (configs,
data_loader, powerfoam) is the run's worktree's. Positions: the drop's shift (record 0 minus rest) is taken off, so the
object stays where the foam sits. Writes a grid per radius rule (rows: views; columns: the rest foam, then the
records) on white, and the rest check (the update at rest against the untouched foam). --solved 1: also the radii
solved for every cell's volume V0 det F_B (recut.Recut), against the scaled ones (PowerSim's: no volume holds).
  python free_surface/render_drop.py --npz output/round01_pruned_exact/skin_drop_check_beta0.001_render.npz"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIN = '/home/fangsuo/py/dynamic-powerfoam/.claude/worktrees/thin-surface'
RUN = THIN+'/output/pacnerf_elastic0_f0_round01_8k_pruned'
sys.path.insert(0, THIN)                                                           # the run's configs / data_loader / powerfoam
import configargparse                                                              # noqa: E402
import warp as wp                                                                  # noqa: E402
from PIL import Image, ImageDraw                                                   # noqa: E402

from configs import Params, add_group                                              # noqa: E402
from data_loader import DataHandler                                                # noqa: E402
from powerfoam.scene import PowerfoamScene                                         # noqa: E402

NAMES = ('points', 'radii', 'quaternions', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height')


def load(run):
    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument('-c', '--config', is_config_file=True)
    args = get_params(parser.parse_args(['-c', str(Path(run)/'config.yaml')]))
    wp.init()
    data = DataHandler(args)
    data.reload('all', downsample=args.downsample[-1])
    model = PowerfoamScene(args)
    model.initialize_from_dataset(data, device='cuda')
    model.load_pt(str(Path(run)/'model.pt'))
    return model, data


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def quat_from_matrix(m):
    t = m[..., 0, 0]+m[..., 1, 1]+m[..., 2, 2]
    c = torch.stack((
        torch.stack((1+t, m[..., 2, 1]-m[..., 1, 2], m[..., 0, 2]-m[..., 2, 0], m[..., 1, 0]-m[..., 0, 1]), -1),
        torch.stack((m[..., 2, 1]-m[..., 1, 2], 1+m[..., 0, 0]-m[..., 1, 1]-m[..., 2, 2], m[..., 0, 1]+m[..., 1, 0], m[..., 0, 2]+m[..., 2, 0]), -1),
        torch.stack((m[..., 0, 2]-m[..., 2, 0], m[..., 0, 1]+m[..., 1, 0], 1-m[..., 0, 0]+m[..., 1, 1]-m[..., 2, 2], m[..., 1, 2]+m[..., 2, 1]), -1),
        torch.stack((m[..., 1, 0]-m[..., 0, 1], m[..., 0, 2]+m[..., 2, 0], m[..., 1, 2]+m[..., 2, 1], 1-m[..., 0, 0]-m[..., 1, 1]+m[..., 2, 2]), -1)), -2)
    key = torch.stack((t, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]), -1).argmax(-1)
    q = torch.gather(c, -2, key[..., None, None].expand(*key.shape, 1, 4))[..., 0, :]
    return q/q.norm(dim=-1, keepdim=True)


def inv_softplus(y, beta=100.):
    return y+torch.log(-torch.expm1(-beta*y))/beta


class Update:
    """the rest state of the kept cells; apply(x, F) sets the PowerSim update and rebuilds the adjacency"""

    def __init__(self, model):
        self.m = model
        self.p0 = model.points.data.clone()
        self.q0 = model.quaternions.data/model.quaternions.data.norm(dim=-1, keepdim=True)
        self.r0 = model.get_radii().detach().clone()
        self.a0 = model.texel_sv_axis.data.clone()
        self.th0 = model.texel_height.data.clone()
        with torch.no_grad():
            self.frame0 = torch.stack((model.get_normals(), *model.get_tangents()), 1)   # rows: n, t, b (R(q0)'s rows)

    def apply(self, x, F, radii=None, plane='R'):
        """plane 'R': PowerSim's (the plane turns with R, its distance h = texel_height r scales with the radius);
        'F': the plane carried by F as the material's plane is: normal along F^-T n0, distance h0 / |F^-T n0| (the
        texel frame's tangent along F t0, orthogonalised)"""
        m = self.m
        F = F.to(self.p0.dtype)
        U, S, Vh = torch.linalg.svd(F)
        R = U@Vh
        a = (R[:, None]@self.a0.view(len(R), -1, 3)[..., None])[..., 0].reshape(self.a0.shape)
        r = self.r0*torch.det(F).clamp(min=1e-3)**(1/3) if radii is None else radii.to(self.r0.dtype)
        th = self.th0
        if plane == 'R':
            q = quat_mul(self.q0, quat_from_matrix(R)*torch.tensor([1., -1., -1., -1.], dtype=self.q0.dtype, device=self.q0.device))
        else:
            n, t0 = self.frame0[:, 0], self.frame0[:, 1]
            g = (torch.linalg.inv(F).transpose(1, 2)@n[..., None])[..., 0]
            s = g.norm(dim=1)
            n = g/s[:, None]
            t = (F@t0[..., None])[..., 0]
            t = t-(t*n).sum(1, keepdim=True)*n
            t = t/t.norm(dim=1, keepdim=True)
            q = quat_from_matrix(torch.stack((n, t, torch.cross(n, t, dim=1)), 1))
            th = self.th0*(self.r0/(s*r))[:, None]
        m.points = torch.nn.Parameter(x.to(self.p0.dtype).contiguous())
        m.quaternions = torch.nn.Parameter(q.contiguous())
        m.texel_sv_axis = torch.nn.Parameter(a.contiguous())
        m.texel_height = torch.nn.Parameter(th.contiguous())
        m.radii = torch.nn.Parameter(inv_softplus(r).contiguous())
        m.rebuild_adjacency()


def solved_radii(X, F, r0, cache):
    """the radii along all records with every cell's volume V0 det F_B (recut.Recut: the union of balls cut by the power
    diagram, V0 its volume at rest), each record solved from the last record's radii; cached. Also the volumes the
    scaled radii r0 det(F_B)^(1/3) give, against the same targets."""
    from free_surface.recut import Recut
    if cache.exists():
        z = np.load(cache)
        return torch.as_tensor(z['r'], device='cuda'), z['err_scaled']
    R = Recut(len(r0))
    V0 = R.geometry(X[0], r0)[0].V.detach()
    J = torch.det(F)
    r, out, its, err = r0.clone(), [r0.clone()], [], [np.zeros(3)]
    for k in range(1, len(X)):
        Vt = V0*J[k]
        Gs = R.geometry(X[k], r0*J[k]**(1/3))[0]
        err.append(torch.quantile((Gs.V.detach()/Vt-1).float(), torch.tensor([.05, .5, .95], device='cuda')).cpu().numpy())
        r, G, S, it, nrep = R.solve_radii(X[k], r, Vt)
        out.append(r.clone()); its.append(it)
        if nrep:
            print('   record %d: %d hidden sites repaired' % (k, nrep), flush=True)
        e = float((G.V.detach()/Vt-1).abs().max())
        if e > 1e-9:
            print('   record %d: stopped at the volumes\' evaluation floor, max |V / target - 1| %.1e' % (k, e), flush=True)
    r = torch.stack(out)
    print('solved radii along %d records: Newton iterations %s; stops at the evaluation floor %d; last max |V / target - 1| %.1e' % (
        len(X)-1, np.bincount(its).tolist(), R.stats.get('floor', 0), float((G.V.detach()/Vt-1).abs().max())), flush=True)
    np.savez(cache, r=r.cpu().numpy(), its=np.array(its), err_scaled=np.array(err))
    return r, np.array(err)


class Blend:
    """the old pipeline's blend field (experiments/foam_deform_render.py, mode 'blend'): every cell mixes the affine maps
    x_k + F_k (p0 - X_k) of its K nearest bodies with Gaussian weights of the rest distance (width x the median
    nearest-body spacing); the cell's F is the same mix of the F_k. Here the cells are the bodies, so each cell's own
    body is one of its K."""

    def __init__(self, X0, K=8, width=1.):
        from scipy.spatial import cKDTree
        X = X0.cpu().numpy()
        tree = cKDTree(X)
        spacing = float(np.median(tree.query(X, k=2)[0][:, 1]))
        d, idx = tree.query(X, k=K)
        w = np.exp(-.5*(d/(width*spacing))**2)
        self.idx = torch.as_tensor(idx, device=X0.device)
        self.w = torch.as_tensor(w/w.sum(1, keepdims=True), dtype=X0.dtype, device=X0.device)
        self.lever = X0[:, None]-X0[self.idx]

    def __call__(self, x, F):
        w = self.w[..., None]
        p = (w*(x[self.idx]+(F[self.idx]@self.lever[..., None])[..., 0])).sum(1)
        return p, (w[..., None]*F[self.idx]).sum(1)


def plane_stats(up, F, t, ks, pairs):
    """per record, over the cells whose plane cuts their ball somewhere (|texel_height| < 1): the angle between the
    plane's normal under R and under F, and the plane distance under F over that under R (h0 / |F^-T n0| against
    h0 J^(1/3)); over the bonded pairs the angle of R_i^T R_j; the cells with J > 1"""
    cut = (up.th0.abs() < 1).any(1)
    n0 = up.frame0[:, 0].double()
    i, j = pairs[:, 0], pairs[:, 1]
    qs = lambda x, p: torch.quantile(x.float(), torch.tensor(p, device=x.device)).cpu().numpy()
    print('cells whose plane cuts the ball: %d of %d' % (int(cut.sum()), len(cut)), flush=True)
    for k in ks:
        Fk = F[k]
        U, S, Vh = torch.linalg.svd(Fk); R = U@Vh
        J = torch.det(Fk)
        g = (torch.linalg.inv(Fk).transpose(1, 2)@n0[..., None])[..., 0]
        s = g.norm(dim=1)
        nR = (R@n0[..., None])[..., 0]
        ang = torch.rad2deg(torch.atan2(torch.cross(nR, g/s[:, None], dim=1).norm(dim=1), (nR*g).sum(1)/s))[cut]
        dist = (1/(s*J**(1/3)))[cut]
        Q = quat_from_matrix(R[i].transpose(1, 2)@R[j])
        rel = torch.rad2deg(2*torch.atan2(Q[:, 1:].norm(dim=1), Q[:, 0].abs()))
        print('t %.3f s: plane normal R vs F angle p50 %.1f p95 %.1f max %.1f deg; plane distance F / R p5 %.3f p50 %.3f p95 %.3f; '
              'neighbours\' relative rotation p50 %.2f p95 %.2f p99 %.2f max %.1f deg; J p0.1 %.3f p99.9 %.3f, cells with J > 1.05: %d (%d of them plane cells, '
              'radius / median %s)' % (
                  t[k], *qs(ang, [.5, .95, 1.]), *qs(dist, [.05, .5, .95]), *qs(rel, [.5, .95, .99, 1.]), *qs(J, [.001, .999]),
                  int((J > 1.05).sum()), int((cut & (J > 1.05)).sum()),
                  np.round((up.r0[J > 1.05]/up.r0.median()).cpu().numpy()[:8], 2).tolist()), flush=True)


def render(model, cam):
    with torch.no_grad():
        q = .5*torch.ones(cam.height, cam.width, 1, device=model.device)
        rgb, alpha = model.forward(cam, depth_quantiles=q)[:2]
    return rgb.clamp(0, 1), alpha.clamp(0, 1)


def on_white(rgb, alpha):
    alpha = alpha.reshape(*rgb.shape[:2], 1)
    return (rgb+(1-alpha)).clamp(0, 1).cpu().numpy()


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--npz', type=Path, default=None)
    a.add_argument('--uniform', type=int, default=0, help='the rest foam under uniform F (similarity, mild and strong anisotropic) instead of the drop')
    a.add_argument('--cells', type=Path, default=Path('output/round01_pruned_exact/dem_exact_single/cells.npz'))
    a.add_argument('--times', type=float, nargs='+', default=[0., .05, .0667, .1, .15, .2, .25, .3, .35, .42, .5])
    a.add_argument('--views', type=int, nargs='+', default=[0, 3])
    a.add_argument('--out', type=Path, default=None)
    a.add_argument('--solved', type=int, default=0, help='also the radii solved for V0 det F_B (about 1 min, cached)')
    a.add_argument('--planes', nargs='+', default=['R'], choices=['R', 'F'], help='R: PowerSim\'s plane update; F: the plane carried by F (Update.apply)')
    a.add_argument('--field', default='affine', choices=['affine', 'blend'], help='affine: each cell follows its own body; blend: the old pipeline\'s blend of the K nearest bodies')
    a.add_argument('--blend', type=float, nargs=2, default=[8, 1.], help='K and width (x the median spacing); the burger runs used 24 2')
    a.add_argument('--renderer', default='powerfoam', choices=['powerfoam', 'lagrange'], help='lagrange: cells carried by their F (lagrange_render.py)')
    args = a.parse_args()
    if not args.uniform:
        out = args.out or args.npz.parent/('render_'+args.npz.stem)
        out.mkdir(parents=True, exist_ok=True)
    model, data = load(RUN)
    src = torch.as_tensor(np.load(args.cells)['source_index'], device=model.device)
    with torch.no_grad():
        for name in NAMES:
            setattr(model, name, torch.nn.Parameter(getattr(model, name).data[src].clone()))
        model.rebuild_adjacency()
    up = Update(model)
    lr = None
    if args.renderer == 'lagrange':
        from free_surface.lagrange_render import LagrangeRender
        lr = LagrangeRender(model)
        cams_ = [data.cameras[v] for v in args.views]
        ref = [render(model, c) for c in cams_]
        eye3 = torch.eye(3, device='cuda').expand(len(src), 3, 3)
        mine = [lr.render(c, up.p0, eye3) for c in cams_]
        d_rgb = max(float((a_[0]-b_[0]).abs().max()) for a_, b_ in zip(ref, mine)); d_a = max(float((a_[1]-b_[1]).abs().max()) for a_, b_ in zip(ref, mine))
        m_rgb = max(float((a_[0]-b_[0]).abs().mean()) for a_, b_ in zip(ref, mine))
        print('lagrange renderer at rest against Power Foam\'s: max |rgb| %.1e (mean %.1e), max |alpha| %.1e' % (d_rgb, m_rgb, d_a), flush=True)
    draw = (lambda x_, F_, c_: lr.render(c_, x_, F_)) if lr is not None else None
    if args.uniform:                                                               # the rest foam under uniform F about its centroid
        cams = [data.cameras[v] for v in args.views]
        rest = [render(model, c) for c in cams]
        c0 = up.p0.mean(0)
        cases = [('0.9 I', (.9, .9, .9)), ('1.04 / 0.9 / 1.04', (1.04, .9, 1.04)), ('1.1 / 0.65 / 1.1', (1.1, .65, 1.1))]
        cols = []
        for name, dg in cases:
            Fu = torch.diag(torch.tensor(dg, dtype=up.p0.dtype, device='cuda'))
            if draw is not None:
                cols.append((name, [draw(c0+(up.p0-c0)@Fu.T, Fu.expand(len(src), 3, 3), c) for c in cams]))
                continue
            up.apply(c0+(up.p0-c0)@Fu.T, Fu.expand(len(src), 3, 3), plane='F')
            cols.append((name, [render(model, c) for c in cams]))
        H, Wd = cams[0].height, cams[0].width
        grid = np.ones((len(cams)*H+24, (len(cols)+1)*Wd, 3))
        for v in range(len(cams)):
            grid[24+v*H:24+(v+1)*H, :Wd] = on_white(*rest[v])
            for j, (_, imgs) in enumerate(cols):
                grid[24+v*H:24+(v+1)*H, (j+1)*Wd:(j+2)*Wd] = on_white(*imgs[v])
        im = Image.fromarray((grid*255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((4, 4), 'rest foam', fill=(0, 0, 0))
        for j, (name, _) in enumerate(cols):
            d.text(((j+1)*Wd+4, 4), 'F = diag(%s), y vertical' % name, fill=(0, 0, 0))
        out = args.out or Path('output/round01_pruned_exact/render_uniform')
        out.mkdir(parents=True, exist_ok=True)
        im.save(out/('uniform_grid%s.png' % ('_lagrange' if draw is not None else '')))
        print('wrote %s' % (out/('uniform_grid%s.png' % ('_lagrange' if draw is not None else ''))), flush=True)
        return
    z = np.load(args.npz)
    t, X, F = z['rec'][:, 0], torch.as_tensor(z['X'], dtype=torch.float64, device='cuda'), torch.as_tensor(z['F'], dtype=torch.float64, device='cuda')
    shift = X[0]-up.p0.double()
    print('cells %d; the drop\'s shift %s (spread %.1e)' % (len(src), shift.mean(0).cpu().numpy().round(5).tolist(), float((shift-shift.mean(0)).abs().max())), flush=True)
    cams = [data.cameras[v] for v in args.views]
    rest = [render(model, c) for c in cams]
    up.apply(up.p0, torch.eye(3, dtype=torch.float64, device='cuda').expand(len(src), 3, 3))
    again = [render(model, c) for c in cams]
    print('rest check (the update with F = I at rest against the untouched foam): max |rgb| %.1e, max |alpha| %.1e' % (
        max(float((r1[0]-r2[0]).abs().max()) for r1, r2 in zip(rest, again)), max(float((r1[1]-r2[1]).abs().max()) for r1, r2 in zip(rest, again))), flush=True)
    Xf = X-shift.mean(0)
    ks = [int(np.argmin(np.abs(t-tt))) for tt in args.times]
    radii = dict(scaled=None)
    if args.solved:
        rs, err = solved_radii(Xf, F, torch.as_tensor(np.load(args.cells)['radii'], dtype=torch.float64, device='cuda'), out/'solved_radii.npz')
        radii['solved'] = rs
        sc = up.r0.double()*torch.det(F)**(1/3)
        for k in ks[1:]:
            q = torch.quantile((rs[k]/sc[k]-1).float(), torch.tensor([0., .05, .5, .95, 1.], device='cuda')).cpu().numpy()
            print('t %.3f s: the scaled radii\'s volumes / target - 1 p5 %+.3f p50 %+.3f p95 %+.3f; solved / scaled radii - 1 min %+.3f p5 %+.3f p50 %+.3f p95 %+.3f max %+.3f' % (
                t[k], *err[k], *q), flush=True)
    if 'F' in args.planes:
        plane_stats(up, F, t, ks, torch.as_tensor(np.load(args.cells)['pairs'], device='cuda').long())
        up.apply(up.p0, torch.eye(3, dtype=torch.float64, device='cuda').expand(len(src), 3, 3), plane='F')
        again = [render(model, c) for c in cams]
        print('rest check of the plane carried by F: max |rgb| %.1e, max |alpha| %.1e' % (
            max(float((r1[0]-r2[0]).abs().max()) for r1, r2 in zip(rest, again)), max(float((r1[1]-r2[1]).abs().max()) for r1, r2 in zip(rest, again))), flush=True)
    H, Wd = cams[0].height, cams[0].width
    bl = Blend(Xf[0], K=int(args.blend[0]), width=args.blend[1]) if args.field == 'blend' else None
    for name, rr, pl in [(n_, r_, p_) for n_, r_ in radii.items() for p_ in args.planes]:
        cols = []
        for k in ks:
            xk, Fk = bl(Xf[k], F[k]) if bl is not None else (Xf[k], F[k])
            if draw is not None:
                cols.append((t[k], [draw(xk, Fk, c) for c in cams]))
                continue
            up.apply(xk.to(up.p0.dtype), Fk, None if rr is None else rr[k], plane=pl)
            cols.append((t[k], [render(model, c) for c in cams]))
        name = (name if pl == 'R' else name+'_planeF')+('_blend%d_%g' % (bl.idx.shape[1], args.blend[1]) if bl is not None else '')+('_lagrange' if draw is not None else '')
        grid = np.ones((len(cams)*H+24, (len(cols)+1)*Wd, 3))
        for v in range(len(cams)):
            grid[24+v*H:24+(v+1)*H, :Wd] = on_white(*rest[v])
            for j, (tk, imgs) in enumerate(cols):
                grid[24+v*H:24+(v+1)*H, (j+1)*Wd:(j+2)*Wd] = on_white(*imgs[v])
        im = Image.fromarray((grid*255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.text((4, 4), 'rest foam', fill=(0, 0, 0))
        for j, (tk, _) in enumerate(cols):
            d.text(((j+1)*Wd+4, 4), 't %.3f s (%s radii)' % (tk, name), fill=(0, 0, 0))
        im.save(out/('drop_grid_%s.png' % name))
        print('wrote %s (%d views x %d times, %dx%d each)' % (out/('drop_grid_%s.png' % name), len(cams), len(cols), Wd, H), flush=True)


if __name__ == '__main__':
    main()
