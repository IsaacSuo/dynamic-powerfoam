"""Regenerate the Vid2Sim GSO ground truth in 3D: a Kaolin-free port of Vid2Sim's data generator
(references/vid2sim_repo/Vid2Sim: simulators/lbs_simulator.py + dataset_toolkits/process_objaverse_dataset.py,
Kaolin 0.16 kaolin.physics.simplicits, sources in references/vid2sim_repo/kaolin_physics).

Generator, as in the original: GSO mesh normalised (centred, AABB diagonal = 1); points = mesh vertices; 500 cubature
points (farthest-point sampling from vertex 0); skinning weights = SimplicitsMLP (3 -> 64 x9 -> 10, ELU) initialised
from the released template, last layer refined 10k iterations data-free (Neo-Hookean energy of random handle
transforms 0.1 N(0,1) on 1000 random vertices x 0.1, + 1e6 x orthogonality), plus one constant rigid handle;
reduced coordinates z = 11 affine handles (11 x 3 x 4); lumped mass rho / 500 per cubature point ("volume" 1, not
the real volume: it cancels in the motion); Neo-Hookean 1/2 mu (I_C - 3) + 1/2 lam (J - 1)^2 - mu (J - 1) at the
cubature points, weight 1/500, F = I + dF/dz z with the finite-difference Jacobian of the LBS field; gravity
energy rho x . a with a = +9.8 along the floor axis (force -a); floor as a quadratic penalty 1e5 (h - z)^2 on the
cubature points below z = -0.7 (no friction); backward Euler dt = 0.05 (one step per frame), 10 Newton iterations
with the original's Armijo line search. 24 frames. Output: every vertex at every frame
(data/vid2sim_gt/<obj>/trajectory.npz) and a check against the released masks (projected surface samples, IoU).
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import ndimage

SRC = Path('data/vid2sim/ext')
MESHES = Path('data/vid2sim/gso_meshes')
TEMPLATE = Path('data/vid2sim/vid2sim_checkpoints/ckpt_lbs_template.pth')
GSO_ID = {'sofa': '3D_Dollhouse_Sofa'}
dev = torch.device('cuda')


def seed_everything(seed):
    if isinstance(seed, str):
        seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def load_mesh(path):
    """load_points_from_mesh(size=.5): centred, scaled by 1 / AABB diagonal."""
    mesh = trimesh.load_mesh(path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    v = mesh.vertices
    bmax, bmin = v.max(0), v.min(0)
    mesh.vertices = .5*(v-(bmax+bmin)/2)/(np.linalg.norm(bmax-bmin)*.5)
    return mesh


def fps(x, n):
    """torch_geometric fps with random_start=False: farthest-point sampling starting at point 0."""
    d = torch.full((len(x),), float('inf'), device=x.device, dtype=x.dtype)
    idx = torch.zeros(n, dtype=torch.long, device=x.device)
    cur = 0
    for k in range(n):
        idx[k] = cur
        d = torch.minimum(d, ((x-x[cur])**2).sum(1))
        cur = int(d.argmax())
    return idx


class SimplicitsMLP(torch.nn.Module):
    def __init__(self, dim=3, width=64, handles=10, layers=8):
        super().__init__()
        L = [torch.nn.Linear(dim, width), torch.nn.ELU()]
        for _ in range(layers):
            L += [torch.nn.Linear(width, width), torch.nn.ELU()]
        L.append(torch.nn.Linear(width, handles))
        self.net = torch.nn.Sequential(*L)

    def forward(self, x):
        return self.net(x.clone())


def lbs(x0, tfms, w):
    """standard_lbs: x_i = x0_i + sum_h w_ih T_h [x0_i; 1]; x0 (N, 3), tfms (B, H, 3, 4), w (N, H) -> (N, B, 3)."""
    x03 = torch.cat((x0, torch.ones_like(x0[:, :1])), 1)
    return torch.einsum('nh,bhrc,nc->nbr', w, tfms, x03)+x0[:, None]


def finite_diff_jac(fcn, x, eps=1e-7):
    """kaolin finite_diff_jac: central differences with delta = sqrt(eps); last axis = derivative direction."""
    delta = float(np.sqrt(np.float32(eps)))
    h = delta*torch.eye(3, device=x.device, dtype=x.dtype)
    xs = torch.cat([x+h[0], x+h[1], x+h[2], x-h[0], x-h[1], x-h[2]], 0)
    j = fcn(xs)
    j = j.reshape(2, 3, -1, *j.shape[1:])
    j = j[0]-j[1]
    return j.permute(*range(1, j.dim()), 0)/(2.*delta)


def nh_energy(mu, lam, F):
    """kaolin neohookean_energy, F (..., 3, 3)."""
    I1 = (F*F).sum((-2, -1))
    J = torch.det(F)
    return .5*mu*(I1-3)+.5*lam*(J-1)**2-mu*(J-1)


def lbs_matrix(x0, w):
    """kaolin lbs_matrix: x = B z + x0 with z the row-major flattening of (H, 3, 4); B (3N, 12H)."""
    N, H = w.shape
    x03 = torch.cat((x0, torch.ones_like(x0[:, :1])), 1)
    B = torch.zeros(3*N, 12*H, device=x0.device, dtype=x0.dtype)
    hcol = torch.arange(H, device=x0.device)*12
    for r in range(3):
        for c in range(4):
            B[r::3, hcol+r*4+c] = w*x03[:, c:c+1]
    return B


def refine_lbs(model, points, yms, prs, iters, seed, all_layers=False):
    """train_step_lbs x iters on the last layer only (produce_data=True), Adam 1e-3, cosine to 0."""
    seed_everything(seed)
    opt = torch.optim.Adam([{'params': list(model.parameters() if all_layers else model.net[-1].parameters()), 'lr': 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0)
    mu, lam = yms/(2*(1+prs)), yms*prs/((1+prs)*(1-2*prs))
    eye = torch.eye(10, device=points.device)
    model.train()
    t0 = time.perf_counter()
    for i in range(iters):
        tfms = .1*torch.randn(10, 10, 3, 4, dtype=points.dtype, device=points.device)
        sample = points[torch.randperm(points.shape[0], device=points.device)[:1000]]
        weights = model(sample)
        F = finite_diff_jac(lambda X: lbs(X, tfms, model(X)), sample)              # (N, B, 3, 3)
        le = 1e-1*(1./sample.shape[0])*nh_energy(mu, lam, F).sum()                  # appx_vol = 1
        lo = 1e6*torch.nn.functional.mse_loss(weights.T@weights, eye)
        loss = le+lo
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if i % 2000 == 0 or i == iters-1:
            print('  refine %5d  le %.4g  lo %.4g  (%.0f s)' % (i, float(le), float(lo), time.perf_counter()-t0), flush=True)
    model.eval()


class Generator:
    def __init__(self, points, cub_idx, model_rigid, yms, prs, rho=1000., floor=-.7, axis=2, penalty=1e5, dt=.05):
        self.dt, self.floor, self.axis, self.penalty = dt, floor, axis, penalty
        self.X, self.Xc = points, points[cub_idx]
        n = len(cub_idx)
        self.mu, self.lam = yms/(2*(1+prs)), yms*prs/((1+prs)*(1-2*prs))
        self.w_int = 1./n                                                           # particle_volume / num_cubature
        with torch.no_grad():
            w_full, w_cub = model_rigid(self.X), model_rigid(self.Xc)
        self.B_full, self.Bc = lbs_matrix(self.X, w_full), lbs_matrix(self.Xc, w_cub)
        mass = torch.full((n,), rho/n, device=dev)                                  # lumped_mass_matrix(rho, volume 1)
        self.BMB = self.Bc.T@(mass.repeat_interleave(3)[:, None]*self.Bc)
        self.grav_g = self.Bc.T@(self.w_int*rho*torch.tensor([0., 0., 9.8], device=dev).expand(n, 3).reshape(-1))
        self.rho, self.n = rho, n
        nz = 12*w_cub.shape[1]

        def F_of_z(z):
            tf = z.reshape(1, -1, 3, 4)
            return finite_diff_jac(lambda Xs: lbs(Xs, tf, model_rigid(Xs)), self.Xc).reshape(-1)
        self.dFdz = torch.autograd.functional.jacobian(F_of_z, torch.zeros(nz, device=dev), vectorize=True)   # (9n, nz)
        self.J = self.dFdz.reshape(n, 9, nz)
        self.I9 = torch.eye(3, device=dev).reshape(9).repeat(n)
        self.eye9 = torch.eye(9, device=dev)

    def material(self, z):
        F = (self.dFdz@z+self.I9).reshape(-1, 3, 3)
        mu, lam = self.mu, self.lam
        J = torch.det(F)[:, None, None]
        FinvT = torch.inverse(F).transpose(1, 2)
        E = self.w_int*nh_energy(mu, lam, F).sum()
        P = mu*F+(J*(lam*(J-1)-mu))*FinvT                                            # unbatched_neohookean_gradient
        G = self.dFdz.T@(self.w_int*P.reshape(-1))
        g = (J*(lam*(2*J-1)-mu)).reshape(-1, 1, 1)
        dg = g-lam*J.reshape(-1, 1, 1)**2
        ff = torch.bmm(FinvT.reshape(-1, 9, 1), FinvT.reshape(-1, 1, 9))
        H = mu*self.eye9+g*ff-dg*ff.reshape(-1, 3, 3, 3, 3).transpose(2, 4).reshape(-1, 9, 9)   # unbatched_neohookean_hessian
        Hz = torch.bmm(self.J.transpose(1, 2), torch.bmm(self.w_int*H, self.J)).sum(0)
        return E, G, Hz

    def floor_terms(self, z):
        x = (self.Bc@z).reshape(-1, 3)+self.Xc
        zc = x[:, self.axis]
        d = torch.clamp(self.floor-zc, min=0.)
        E = self.penalty*(d*d).sum()
        g = torch.zeros_like(x); g[:, self.axis] = -2*self.penalty*d
        G = self.Bc.T@g.reshape(-1)
        hdiag = torch.zeros_like(x); hdiag[:, self.axis] = 2*self.penalty*(zc < self.floor).to(x.dtype)
        Hz = self.Bc.T@(hdiag.reshape(-1)[:, None]*self.Bc)
        return E, G, Hz

    def gravity(self, z):
        x = (self.Bc@z).reshape(-1, 3)+self.Xc
        return self.w_int*self.rho*9.8*x[:, self.axis].sum(), self.grav_g

    def newton_EGH(self, z, z_prev, z_dot):
        dt = self.dt
        Em, Gm, Hm = self.material(z)
        Ef, Gf, Hf = self.floor_terms(z)
        Eg, Gg = self.gravity(z)
        E = .5*z@(self.BMB@z)-z@(self.BMB@z_prev)-dt*z@(self.BMB@z_dot)+dt*dt*(Em+Ef+Eg)
        G = self.BMB@z-self.BMB@z_prev-dt*self.BMB@z_dot+dt*dt*(Gm+Gf+Gg)
        H = self.BMB+dt*dt*(Hm+Hf)
        return E, G, H

    def energy(self, z, z_prev, z_dot):
        dt = self.dt
        Em = self.material(z)[0]; Ef = self.floor_terms(z)[0]; Eg = self.gravity(z)[0]
        return .5*z@(self.BMB@z)-z@(self.BMB@z_prev)-dt*z@(self.BMB@z_dot)+dt*dt*(Em+Ef+Eg)

    def step(self, z, z_dot, iters=10):
        """newtons_method + line_search_1 of the original (alpha 1e-3, beta .3, 10 steps, initial step 10)."""
        z_prev = z.clone()
        with torch.no_grad():
            for _ in range(iters):
                _, G, H = self.newton_EGH(z, z_prev, z_dot)
                if torch.det(H) == 0:
                    H = H+1e-4*torch.eye(H.shape[0], device=dev)
                p = -torch.linalg.solve(H, G)
                if torch.abs(G@p) < 1e-3:
                    break
                t, f, gTd, can_break = 10., self.energy(z, z_prev, z_dot), G@p, False
                for _ in range(10):
                    f_new = self.energy(z+t*p, z_prev, z_dot)
                    if f_new <= f+1e-3*t*gTd:
                        if can_break:
                            break
                        can_break = True; t = t/.3
                    else:
                        t *= .3
                z = z+t*p
        return z, (z-z_prev)/self.dt


def project_iou(verts, faces, scene, frame, n_samples=200000):
    meta = json.loads((SRC/scene/'transforms_simulation.json').read_text())
    mesh = trimesh.Trimesh(verts, faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    mask0 = np.asarray(Image.open(SRC/scene/'data'/'a_0_0.png'))[..., 3] > 127
    H, W = mask0.shape
    f = .5*W/np.tan(.5*meta['camera_angle_x'])
    ious = []
    for fr in meta['frames'][:12]:
        w2c = np.linalg.inv(np.array(fr['transform_matrix']))
        cam = pts@w2c[:3, :3].T+w2c[:3, 3]; z = -cam[:, 2]
        u = (f*cam[:, 0]/z+W/2).astype(int); v = (-f*cam[:, 1]/z+H/2).astype(int)
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        img = np.zeros((H, W), bool); img[v[ok], u[ok]] = True
        img = ndimage.binary_closing(img, iterations=2)
        vid = int(fr['file_path'].split('_')[1])
        m = np.asarray(Image.open(SRC/scene/'data'/f'a_{vid}_{frame}.png'))[..., 3] > 127
        ious.append(float((img & m).sum()/max((img | m).sum(), 1)))
    return ious


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--obj', default='sofa')
    a.add_argument('--gso', default=None, help='GSO model id (default: GSO_ID[obj])')
    a.add_argument('--seed', default=None, help='int, or a string hashed like the original (default: obj name)')
    a.add_argument('--refine-iters', type=int, default=10000)
    a.add_argument('--lbs', type=Path, default=None, help='reuse a refined LBS model (skip the refinement)')
    a.add_argument('--all-layers', type=int, default=0, help='refine all layers (produce_data=False) instead of the last one')
    a.add_argument('--newton-iters', type=int, default=10)
    a.add_argument('--E-scale', type=float, default=1., help='multiply the released E (to probe what the released motion corresponds to)')
    a.add_argument('--nu-override', type=float, default=None)
    a.add_argument('--frames', type=int, default=23)
    a.add_argument('--tag', default='')
    a.add_argument('--out', type=Path, default=Path('data/vid2sim_gt'))
    args = a.parse_args()
    import yaml
    gt = yaml.safe_load(open(SRC/args.obj/'gt_phys_params.yaml'))
    yms, prs = float(gt['yms'])*args.E_scale, float(gt['prs']) if args.nu_override is None else args.nu_override
    gso = args.gso or GSO_ID[args.obj]
    seed = args.seed if args.seed is not None else args.obj
    seed = int(seed) if isinstance(seed, str) and seed.lstrip('-').isdigit() else seed
    out = args.out/(args.obj+args.tag); out.mkdir(parents=True, exist_ok=True)
    print('object %s  GSO %s  E %.0f nu %.3f  seed %s' % (args.obj, gso, yms, prs, seed), flush=True)

    mesh = load_mesh(MESHES/gso/'meshes'/'model.obj')
    points = torch.tensor(mesh.vertices, dtype=torch.float32, device=dev)
    cub_idx = fps(points, min(points.shape[0], 500))
    print('vertices %d, cubature %d' % (points.shape[0], len(cub_idx)), flush=True)

    model = SimplicitsMLP().to(dev)
    model.load_state_dict(torch.load(TEMPLATE, map_location=dev, weights_only=True))
    if args.lbs is not None:
        model.load_state_dict(torch.load(args.lbs, map_location=dev, weights_only=True)); model.eval()
    else:
        refine_lbs(model, points, yms, prs, args.refine_iters, seed, all_layers=bool(args.all_layers))
    model_rigid = lambda x: torch.cat((model(x), torch.ones((x.shape[0], 1), device=dev)), 1)

    t0 = time.perf_counter()
    gen = Generator(points, cub_idx, model_rigid, yms, prs)
    print('setup %.1f s (dF/dz %s)' % (time.perf_counter()-t0, tuple(gen.dFdz.shape)), flush=True)
    z = torch.zeros(gen.BMB.shape[0], device=dev); z_dot = torch.zeros_like(z)
    traj, zs = [points.cpu().numpy()], [z.cpu().numpy()]
    with torch.no_grad():
        for k in range(args.frames):
            z, z_dot = gen.step(z, z_dot, iters=args.newton_iters)
            traj.append(((gen.B_full@z).reshape(-1, 3)+points).cpu().numpy()); zs.append(z.cpu().numpy())
    traj = np.array(traj)
    print('simulated %d frames in %.1f s; z range %.3f' % (args.frames, time.perf_counter()-t0, np.abs(zs[-1]).max()), flush=True)
    low = traj[..., 2].min(1)
    print('lowest vertex per frame:', np.round(low, 3).tolist(), flush=True)
    np.savez_compressed(out/'trajectory.npz', vertices=traj.astype(np.float32), faces=np.asarray(mesh.faces),
                        cubature=cub_idx.cpu().numpy(), z=np.array(zs), yms=yms, prs=prs, seed=str(seed), gso=gso)
    torch.save(model.state_dict(), out/'lbs_model.pth')
    ious = [project_iou(traj[t], mesh.faces, args.obj, t) for t in range(args.frames+1)]
    per_frame = [float(np.mean(x)) for x in ious]
    print('IoU vs released masks per frame (12 views mean):', np.round(per_frame, 3).tolist(), flush=True)
    print('IoU mean frames 0-5 (free fall) %.3f, 6-23 %.3f, min %.3f' % (np.mean(per_frame[:6]), np.mean(per_frame[6:]), np.min(per_frame)), flush=True)
    (out/'check.json').write_text(json.dumps(dict(obj=args.obj, gso=gso, seed=str(seed), yms=yms, prs=prs,
                                                  iou_per_frame=per_frame, iou_per_frame_view=ious), indent=1))


if __name__ == '__main__':
    main()
