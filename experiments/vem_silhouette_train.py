"""Drive the reconstructed Power Foam with VEM and train (E, nu, beta) from video only.

Same protocol and metrics as pacnerf_silhouette_train.py (DEM): 6 training views (0 2 4 6 8 10),
frames 1-10 train, 11-13 held out; silhouette L2 (optionally + PowerSim photometric term);
report silhouette IoU / PSNR per split and the 3D trajectory nRMSE on the DEM bodies (never used
in training). --iterations 0 only evaluates the starting parameters (step 4: rendering).

Cell update: a material cell's own polyhedron gives its deformation, F_c = volume-weighted mean
of its pyramids' F (the VEM projection of the cell); its site moves with the cell's vertex-mean
anchor, p = a_c + F_c (p0 - A_c); frame rotation polar(F_c) (PowerFoam stores normals as rows:
q0 (x) conj(quat(R))), radiance axes R a, radius r0 det(F_c)^(1/3). Air-shell cells (2 rings, not
in the mechanics) blend the maps of their K nearest material cells with Gaussian weights.
Positions keep their graph (alpha -> sites -> vertices -> rollout -> parameters); frames and
radii are detached, as in the DEM runs.
Parameters: E = E0 exp(z0), nu = nu0 + z1 (clamped to [0, 0.45]), beta = beta0 exp(z2).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from experiments.foam_deform_render import inv_softplus, polar_rotation, quat_from_matrix
from experiments.foam_dynamic_render import air_shell, keep_cells, load, quat_mul, render
from experiments.pacnerf_dem_forward import cell_targets
from experiments.pacnerf_to_powerfoam import object_mask
from experiments.vem_dynamics import VEMSolid
from powerfoam.metrics import ssim_eval


class VEMFoam:
    """Per-frame update of the kept Power Foam cells from VEM vertex positions."""

    def __init__(self, foam, solid, n_material, offset, K=8, width=1., mode='cell', scale=True):
        self.foam, self.m, self.nm = foam, solid, n_material
        self.mode, self.scale = mode, scale
        dev, dt = foam.device, foam.points.dtype
        self.offset = offset.to(dt)
        self.p0 = foam.points.data.clone()
        self.q0 = foam.quaternions.data/foam.quaternions.data.norm(dim=-1, keepdim=True)
        self.r0 = foam.get_radii().detach().clone()
        self.axis0 = foam.texel_sv_axis.data.clone()
        m = solid
        self.cell_of_sub = torch.as_tensor(m.sub_cell_np, device=dev)
        self.cell_vol = torch.zeros(n_material, dtype=m.dtype, device=dev).index_add_(0, self.cell_of_sub, m.s_vol)
        A, _ = self.anchors_F(m.X)
        self.A = A.to(dt)
        # air shell: blend K nearest material cells (rest positions in mechanics frame)
        sites = (self.p0-self.offset).double().cpu().numpy()
        tree = cKDTree(sites[:n_material])
        spacing = float(np.median(tree.query(sites[:n_material], k=2)[0][:, 1]))
        d, idx = tree.query(sites[n_material:] if mode == 'cell' else sites, k=K)
        w = np.exp(-.5*(d/(width*spacing))**2); w /= w.sum(1, keepdims=True)
        self.air_idx = torch.as_tensor(idx, device=dev)
        self.air_w = torch.as_tensor(w, dtype=dt, device=dev)

    def anchors_F(self, x):
        m = self.m
        xbar, _, _ = m.kinematics(x)
        Fs = m.sub_F(x, xbar)
        Fc = torch.zeros(self.nm, 3, 3, dtype=x.dtype, device=x.device).index_add_(0, self.cell_of_sub, m.s_vol[:, None, None]*Fs)
        Fc = Fc/self.cell_vol[:, None, None]
        a = m._points(x)[len(x)+m.nf:]                    # cell apexes = vertex means
        return a, Fc

    def place(self, x):
        f = self.foam
        dt = self.p0.dtype
        a, F = self.anchors_F(x)
        a, F = a.to(dt), F.to(dt)
        idx, w = self.air_idx, self.air_w[..., None]
        Fd = F.detach()
        if self.mode == 'cell':                          # material: own cell map; air: blend
            lever = self.p0[:self.nm]-self.offset-self.A
            p_mat = a+(F@lever[..., None])[..., 0]
            lev_air = (self.p0[self.nm:]-self.offset)[:, None, :]-self.A[idx]
            p_air = (w*(a[idx]+(F[idx]@lev_air[..., None])[..., 0])).sum(1)
            F_all = torch.cat((Fd, (w[..., None]*Fd[idx]).sum(1)))
            p = torch.cat((p_mat, p_air))+self.offset
        else:                                            # every kept cell blends its K nearest cell maps
            lev = (self.p0-self.offset)[:, None, :]-self.A[idx]
            p = (w*(a[idx]+(F[idx]@lev[..., None])[..., 0])).sum(1)+self.offset
            F_all = (w[..., None]*Fd[idx]).sum(1)
        R = polar_rotation(F_all)
        scale = torch.det(F_all).clamp(min=1e-3)**(1/3) if self.scale else torch.ones(len(p), dtype=p.dtype, device=p.device)
        qR = quat_from_matrix(R)
        q = quat_mul(self.q0, qR*torch.tensor([1., -1., -1., -1.], dtype=qR.dtype, device=qR.device))
        f._parameters.pop('points', None); f.points = p
        f._parameters.pop('quaternions', None); f.quaternions = q
        axis = self.axis0.view(len(p), -1, 3)
        f._parameters.pop('texel_sv_axis', None)
        f.texel_sv_axis = (R[:, None]@axis[..., None])[..., 0].reshape(self.axis0.shape)
        f._parameters.pop('radii', None); f.radii = inv_softplus(self.r0*scale)
        with torch.no_grad():
            f.aabb_tree.update(p.detach(), f.get_radii())
            f.adjacency, f.adjacency_offsets = f.aabb_tree.build_cech_complex()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--run', default='output/pacnerf_elastic0_f0_v1')
    a.add_argument('--dem', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0'))
    a.add_argument('--geometry', type=Path, default=Path('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz'))
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--frames', type=int, default=13)
    a.add_argument('--train-frames', type=int, default=10)
    a.add_argument('--train-views', type=int, nargs='+', default=[0, 2, 4, 6, 8, 10])
    a.add_argument('--iterations', type=int, default=40)
    a.add_argument('--lr', type=float, default=.05)
    a.add_argument('--E0', type=float, default=None, help='starting E (default: GT)')
    a.add_argument('--nu0', type=float, default=None, help='starting nu (default: GT)')
    a.add_argument('--beta0', type=float, default=4.)
    a.add_argument('--rgb-weight', type=float, default=0.)
    a.add_argument('--ssim-lambda', type=float, default=.2)
    a.add_argument('--render-mode', default='cell', choices=('cell', 'blend'))
    a.add_argument('--blend-k', type=int, default=8)
    a.add_argument('--blend-width', type=float, default=1.)
    a.add_argument('--no-scale', action='store_true', help='keep radii (no det(F)^(1/3) scaling)')
    a.add_argument('--save-frames', type=int, nargs='*', default=[], help='views whose renders are saved at evaluation')
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    bodies = dict(np.load(args.dem/'geometry.npz')); cells = dict(np.load(args.dem/'cells.npz'))
    offset = torch.tensor(json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/conversion.json').read_text())['offset'],
                          dtype=torch.float32, device='cuda')
    foam, data = load(args.run)
    n_mat = len(cells['source_index'])
    keep, _ = air_shell(foam, cells['source_index'], np.arange(n_mat), rings=2)
    keep_cells(foam, keep)
    for name in ('radii', 'density', 'texel_sites', 'texel_sv_axis', 'texel_sv_rgb', 'texel_height'):
        getattr(foam, name).requires_grad_(False)
    m = VEMSolid(args.geometry, 'cuda', torch.float32, stab='pyrlin')
    m.ground_mode = 'noslip'
    d = np.load(args.geometry)
    flat = np.zeros(len(d['sub_volume'])); np.maximum.at(flat, d['sub_id'], np.linalg.norm(d['sub_g'], axis=1))
    keep_sub = (d['sub_volume'] >= 1e-6*d['volume'][d['sub_elem']]) & (flat*np.cbrt(np.abs(d['sub_volume'])) <= 10.)
    m.sub_cell_np = d['sub_cell'][keep_sub]
    assert len(m.sub_cell_np) == m.ns
    vf = VEMFoam(foam, m, n_mat, offset, K=args.blend_k, width=args.blend_width, mode=args.render_mode, scale=not args.no_scale)
    P = torch.as_tensor(bodies['points'], dtype=torch.float32, device='cuda'); elem = torch.arange(len(P), device='cuda')

    order = [int(Path(f['file_path']).stem.split('_')[1]) for f in
             json.loads(Path('data/pacnerf_powerfoam/elastic_0_f0/transforms_all.json').read_text())['frames']]
    src = Path('data/pacnerf/data')/args.scene/'data'
    masks, caps, bgs = {}, {}, {}
    for c, v in enumerate(order):
        bg = np.asarray(Image.open(src/f'r_{v}_-1.png').convert('RGB'), float)/255
        bgs[c] = torch.tensor(bg, dtype=torch.float32, device='cuda')
        for k in range(args.frames+1):
            img = np.asarray(Image.open(src/f'r_{v}_{k}.png').convert('RGB'), float)/255
            masks[k, c] = torch.tensor(object_mask(img, bg), dtype=torch.float32, device='cuda')
            caps[k, c] = torch.tensor(img, dtype=torch.float32, device='cuda')
    train_cams = [order.index(v) for v in args.train_views]
    test_cams = [c for c in range(len(order)) if c not in train_cams]
    targets, has, _ = cell_targets(cells, bodies['body_of_cell'], bodies['points'],
                                   Path('data/pacnerf/simulation_data')/args.scene, args.frames)
    E0 = args.E0 or gt['E']; nu0 = args.nu0 if args.nu0 is not None else gt['nu']; b0 = args.beta0
    z = torch.zeros(3, dtype=torch.float32, device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=args.lr)
    v0 = torch.tensor(gt['v'], dtype=torch.float32, device='cuda').expand(m.nv, 3).clone()

    def params(zz):
        return (E0*torch.exp(zz[0]), (nu0+zz[1]).clamp(0., .45), b0*torch.exp(zz[2]))

    def readable(zz):
        E, nu, b = (float(t) for t in params(zz.detach()))
        return dict(E=E, nu=nu, beta=b, E_over_gt=E/gt['E'])

    def substeps_for(zz):
        p = tuple(t.detach() for t in params(zz))
        return int(np.ceil(m.max_frequency(p)/24))           # dt = 1 / omega_max (half the limit)

    def rollout(frames, sub, grad=True):
        p = params(z if grad else z.detach())
        xs, _ = m.rollout(p, m.X, v0, 1/24/sub, sub, frames, compile=True)
        return xs

    def view_terms(cam, c):
        rgb, alpha = render(foam, cam)
        alpha = alpha.reshape(*alpha.shape[:2])
        comp = rgb.clamp(0, 1)+(1-alpha.clamp(0, 1))[..., None]*bgs[c]
        return alpha, comp

    def evaluate(tag, sub):
        with torch.no_grad():
            xs = rollout(args.frames, sub, grad=False)
            pos = torch.stack([m.probe(x, P, elem) for x in xs])
            iou = np.zeros((args.frames+1, len(order))); psnr = np.zeros_like(iou)
            for k in range(args.frames+1):
                vf.place(xs[k])
                for c, cam in enumerate(data.cameras):
                    alpha, comp = view_terms(cam, c)
                    mm, g = alpha > .5, masks[k, c] > .5
                    iou[k, c] = float((mm & g).sum()/(mm | g).sum())
                    psnr[k, c] = float(-10*torch.log10(((comp-caps[k, c])**2).mean()))
                    if order[c] in args.save_frames:
                        im = torch.cat((comp, caps[k, c]), 1).clamp(0, 1).cpu().numpy()
                        Image.fromarray((im*255).astype(np.uint8)).save(args.out/f'{tag}_view{order[c]}_frame{k:02d}.png')
        tr, te = slice(1, args.train_frames+1), slice(args.train_frames+1, args.frames+1)
        Pn = pos.double().cpu().numpy(); Pn = Pn-Pn[:1]+targets[:1]

        def nrmse(lo, hi):
            pp, tt = Pn[lo:hi+1, has], targets[lo:hi+1, has]
            return float(np.sqrt(np.mean((pp-tt)**2)/np.mean((tt-targets[:1, has])**2)))
        np.savez_compressed(args.out/f'{tag}_metrics.npz', iou=iou, psnr=psnr, positions=Pn)
        return dict(iou_trainviews_trainframes=float(iou[tr][:, train_cams].mean()),
                    iou_testviews_trainframes=float(iou[tr][:, test_cams].mean()),
                    iou_trainviews_testframes=float(iou[te][:, train_cams].mean()),
                    iou_testviews_testframes=float(iou[te][:, test_cams].mean()),
                    psnr_testviews_trainframes=float(psnr[tr][:, test_cams].mean()),
                    psnr_testviews_testframes=float(psnr[te][:, test_cams].mean()),
                    iou_rest_allviews=float(iou[0].mean()), psnr_rest_allviews=float(psnr[0].mean()),
                    traj_nrmse_frames_1_10=nrmse(1, args.train_frames), traj_nrmse_frames_11_13=nrmse(args.train_frames+1, args.frames),
                    traj_nrmse_frames_1_13=nrmse(1, args.frames), parameters=readable(z), substeps=sub)

    sub = substeps_for(z)
    t0 = time.perf_counter()
    report = dict(config={k: str(v) for k, v in vars(args).items()}, initial=evaluate('initial', sub))
    report['initial_eval_seconds'] = time.perf_counter()-t0
    print('initial', json.dumps(report['initial']), flush=True)
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    if args.iterations == 0:
        return
    history, best = [], (np.inf, z.detach().clone())
    start = time.perf_counter()
    for it in range(args.iterations):
        if it and it % 5 == 0:
            sub = substeps_for(z)
        optimizer.zero_grad()
        xs = rollout(args.train_frames, sub)
        loss = 0.
        for k in range(1, args.train_frames+1):
            vf.place(xs[k])
            for c in train_cams:
                alpha, comp = view_terms(data.cameras[c], c)
                loss = loss+((alpha-masks[k, c])**2).mean()
                if args.rgb_weight > 0:
                    l = args.ssim_lambda
                    loss = loss+args.rgb_weight*((1-l)*((comp-caps[k, c])**2).mean()+l*(1-ssim_eval(comp, caps[k, c])))
        loss = loss/(args.train_frames*len(train_cams))
        value = float(loss.detach())
        loss.backward()
        if not np.isfinite(value) or not torch.isfinite(z.grad).all():
            print('non-finite at iteration', it, flush=True); break
        if value < best[0]:
            best = (value, z.detach().clone())
        optimizer.step()
        history.append(dict(iteration=it, silhouette_mse=value, grad=z.grad.tolist(), parameters=readable(z),
                            substeps=sub, seconds=time.perf_counter()-start))
        print(json.dumps(history[-1]), flush=True)
        (args.out/'history.json').write_text(json.dumps(history, indent=1))
    with torch.no_grad():
        z.copy_(best[1])
    report['best'] = evaluate('best', substeps_for(z))
    report['train_seconds'] = time.perf_counter()-start
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print('best', json.dumps(report['best']), flush=True)


if __name__ == '__main__':
    main()
