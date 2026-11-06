"""Material identification of the Vid2Sim lion with the old DEM (fitted skin, one-way cap, Hertz ground, no self contact,
as dem_contact_lion.py's dem_old1 run on 10-31: the smoothed start and the carried region on), pure elasticity.

Unknowns (ln E, nu, ln beta), solved for as (ln mu, ln K/mu, ln beta) (11-06, see phi_of); fixed from the dataset: the foam's rest state (inverse_data/lion/foam, the
fourier_lion_8k_round_clean cells), start velocity (0, 0, -g dt / 2) with Vid2Sim's dt 0.05 (vid2sim_to_springgaus.py),
gravity -z, the ground z = -0.2 in the foam's frame (-0.7 in the world: x_world = x_foam - offset), frictionless (mu 0).
No self contact: Vid2Sim's ground truth has none (user, 11-06). beta is bounded below by the scene's beta_min
(lion 0.05: its one-way cap needs it, user 10-29; pacnerf_e0 0.003: the consistent mass's ringing, SCENES), in every
forward and in LM (--beta-min only raises it).

Observation: per view and frame (20 fps, frame k at t = k / 20 s, 60 steps of 1/1200 s) the silhouette of the union of the
cells' spheres (centre x, radius r0 det(F_B)^(1/3), r0 the foam's radii; at rest this union is exactly the material
region) by an anti-aliased ray-sphere test (Warp), reduced to identify.moments' features: centroid, second-moment widths,
sqrt(area), in pixels. Ground truth: the alpha masks a_<view>_<frame> (inverse_data/lion/masks, vid2sim_to_springgaus.py).
Protocol (10-18, Vid2Sim benchmark): fit frames 1..15 in all views but the held one (5); report the held view and the
future frames 16..23 (IoU of the silhouettes at 0.5), log10 E and nu errors against gt_phys_params (E 10648.6, nu 0.391).

Commands (run from the repository root; python with warp 1.10, torch 2.9, as requirements-sim.txt):
  python -u free_surface/lion_inverse.py check                    frame 0: our silhouette vs the masks per view (no simulation)
  python -u free_surface/lion_inverse.py run --theta lnE nu lnbeta --frames K --out f.npz [--tol 1e-3] [--save-frames]
                                                                  one forward (one process, one material): features per frame
  python -u free_surface/lion_inverse.py lm --out DIR [--obs real|synth --truth ...] [--start ...] [--jobs N]
                                                                  Levenberg-Marquardt, the Jacobian by central differences of
                                                                  whole runs, all of an iteration's runs in parallel processes
  python -u free_surface/lion_inverse.py eval --theta ... --out DIR   frames 0..23, IoU per view and frame, the errors
A run whose step does not converge exits 1 (its trial is a rejection in lm). No timeout anywhere."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DT = 1/1200
SCENE_NAME = 'lion'
SCENES = dict(                                                                      # (--scene; 11-06: PAC-NeRF for the inversion, Vid2Sim forward only)
    lion=dict(data='lion', cam='vid2sim', fps=20, nframes=24, up=2, c0=-.2, v0=(0., 0., -.245), mu=0.,
              gt=dict(E=10648.614517785627, nu=.3912951704721983), beta_min=.05,            # (the lion's one-way cap needs it, user 10-29)
              photos='/home/fangsuo/py/dynamic-powerfoam/data/vid2sim/ext/lion/data'),
    pacnerf_e0=dict(data='pacnerf_e0', cam='nerf', fps=24, nframes=14, up=1, c0=-.05,            # (the ground at PAC-NeRF's y = 0,
                    v0=(-0.4078950600549248, -.5, -0.06629882732047176), mu=1e6,                 # the scene's -0.05; glued, user 11-06)
                    gt=dict(E=23008.377696598774, nu=.30079413488868173), beta_min=.003,  # (consistent mass: below it the sites far inside
                    photos='/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/data/elastic/0/data'))
# (pacnerf_e0's beta_min, 11-06, user: "设置个最低值": with the consistent mass the sites far inside their material carry
# little inertia of their own, and their cell-scale modes ring after the impact undamped by the midpoint rule. Cells whose
# det F jumps by > 0.3 between frames after frame 8: beta 1e-4 3024, 1e-3 1224, 3e-3 217, the lumped mass at 3e-3 219.)


def set_scene(name):
    global SCENE, DATA, FPS, SPF, NFRAMES, UP, C0, V0, MU, GT, FOAM, BETA_MIN
    SCENE = SCENES[name]; DATA = ROOT/'inverse_data'/SCENE['data']; BETA_MIN = SCENE['beta_min']
    FPS, NFRAMES, UP, C0, V0, MU, GT = SCENE['fps'], SCENE['nframes'], SCENE['up'], SCENE['c0'], SCENE['v0'], SCENE['mu'], SCENE['gt']
    SPF = int(round(1/(FPS*DT))); FOAM = DATA/'foam'


set_scene('lion')
PARAMS = ('lnmu', 'lnk', 'lnbeta')                                                 # LM's unknowns (11-06): shear modulus, k = K / mu, beta
NU_RANGE = (.01, .49)                                                              # (11-06: at nu 0 the forward failed, Vid2Sim's real stage)
_k = lambda nu: 2*(1+nu)/(3*(1-2*nu))
LNK_RANGE = (math.log(_k(NU_RANGE[0])), math.log(_k(NU_RANGE[1])))                # (nu 0.01 .. 0.49: k 0.69 .. 49.7)


def phi_of(th):
    """(ln E, nu, ln beta) -> (ln mu, ln k, ln beta), mu = E / 2 (1 + nu) in Pa, k = K / mu = 2 (1 + nu) / 3 (1 - 2 nu).
    11-06: in nu the silhouettes' response is strongly nonlinear (k grows from 2.2 at nu 0.3 to 50 at 0.49), and the
    first Gauss-Newton step from nu 0.3 ran to the bound 0.49 on a synthetic truth of 0.391"""
    E, nu = math.exp(th[0]), float(th[1])
    return np.array([math.log(E/(2*(1+nu))), math.log(2*(1+nu)/(3*(1-2*nu))), float(th[2])])


def theta_of(ph):
    """(ln mu, ln k, ln beta) -> (ln E, nu, ln beta): E = 9 k mu / (3 k + 1), nu = (3 k - 2) / 2 (3 k + 1)"""
    mu, k = math.exp(ph[0]), math.exp(ph[1])
    return np.array([math.log(9*k*mu/(3*k+1)), (3*k-2)/(2*(3*k+1)), float(ph[2])])


# ---------------------------------------------------------------- the observation
def cameras():
    """meta (width, height, camera_names), per view K, R, t (OpenCV, world to camera), the foam frame's offset (x_world = x_foam - offset)"""
    if SCENE['cam'] == 'vid2sim':
        meta = json.loads((DATA/'cameras.json').read_text())
        off = np.array(json.loads((DATA/'conversion.json').read_text())['offset'], float)
        cams = [meta['cameras'][nm] for nm in meta['camera_names']]
        return meta, cams, off
    tr = json.loads((DATA/'transforms_all.json').read_text())                       # (NeRF / OpenGL c2w in the foam's own frame)
    W, H = int(tr['w']), int(tr['h']); focal = .5*W/math.tan(.5*tr['camera_angle_x'])
    fr = sorted(tr['frames'], key=lambda f_: int(Path(f_['file_path']).name.split('_')[1]))
    cams, names = [], []
    for f_ in fr:
        c2w = np.array(f_['transform_matrix'], float)
        Rw2c = (c2w[:3, :3]@np.diag([1., -1., -1.])).T
        cams.append(dict(K=[[focal, 0., (W-1)/2], [0., focal, (H-1)/2], [0., 0., 1.]], R=Rw2c.tolist(), t=(-Rw2c@c2w[:3, 3]).tolist()))
        names.append(Path(f_['file_path']).name.split('_')[1])
    return dict(width=W, height=H, camera_names=names), cams, np.zeros(3)


def gt_masks():
    """(views, frames, H, W) alpha in [0, 1], float32 (PAC-NeRF: frame 0 only for now, the Power Foam scene's masks)"""
    from PIL import Image
    meta, _, _ = cameras()
    if SCENE['cam'] == 'nerf':
        return np.stack([np.asarray(Image.open(DATA/'masks'/('r_%s_0.png' % nm)), np.float32)[None]/255. for nm in meta['camera_names']])
    return np.stack([np.stack([np.asarray(Image.open(DATA/'masks'/nm/('%02d.png' % f)), np.float32)/255. for f in range(NFRAMES)])
                     for nm in meta['camera_names']])


def moments_np(a):
    """identify.moments for one map: centroid x, y, widths x, y, sqrt(area) (pixel centres at integer coordinates)"""
    H, W = a.shape
    ys, xs = np.arange(H, dtype=np.float64)[:, None], np.arange(W, dtype=np.float64)[None, :]
    m0 = max(float(a.sum()), 1e-6)
    cx, cy = float((a*xs).sum())/m0, float((a*ys).sum())/m0
    sx = math.sqrt(max(float((a*(xs-cx)**2).sum())/m0, 1e-12)); sy = math.sqrt(max(float((a*(ys-cy)**2).sum())/m0, 1e-12))
    return np.array([cx, cy, sx, sy, math.sqrt(m0)])


def gt_features():
    path = DATA/'gt_features.npz'
    if path.exists():
        return np.load(path)['f']
    A = gt_masks()
    f = np.stack([np.stack([moments_np(A[v, k].astype(np.float64)) for v in range(A.shape[0])]) for k in range(A.shape[1])])
    if A.shape[1] == NFRAMES:                                                        # (frames, views, 5); cached only when complete
        np.savez(path, f=f)
    return f


class Silhouette:
    """the union of spheres seen by each camera: per sphere its pixels' coverage clamp(1/2 + (r - d) f / t, 0, 1) (d the
    centre's distance to the pixel's ray, t its depth along the ray: one pixel of anti-aliasing), the maximum over spheres"""

    def __init__(self):
        import torch
        import warp as wp
        self.torch, self.wp = torch, wp
        meta, cams, off = cameras()
        self.W, self.H = meta['width'], meta['height']
        self.R = torch.tensor(np.array([c['R'] for c in cams]), dtype=torch.float64, device='cuda')
        self.t = torch.tensor(np.array([c['t'] for c in cams]), dtype=torch.float64, device='cuda')
        self.K = np.array([c['K'] for c in cams])
        self.off = torch.tensor(off, dtype=torch.float64, device='cuda')
        self.nv = len(cams)
        self.img = wp.zeros((self.nv, self.H, self.W), dtype=wp.float32, device='cuda')
        self.timg = wp.to_torch(self.img)

        @wp.kernel
        def k_sil(pc: wp.array2d(dtype=wp.vec3f), r: wp.array(dtype=wp.float32), fx: wp.array(dtype=wp.float32),
                  cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32), Wd: int, Hd: int, img: wp.array3d(dtype=wp.float32)):
            v, i = wp.tid()
            p = pc[v, i]
            z = p[2]
            if z <= 1e-6:
                return
            f = fx[v]
            ux = f*p[0]/z+cx[v]
            uy = f*p[1]/z+cy[v]
            R = int(wp.ceil(1.1*f*r[i]/z+2.0))
            x0 = wp.max(int(wp.floor(ux))-R, 0)
            x1 = wp.min(int(wp.floor(ux))+R+1, Wd-1)
            y0 = wp.max(int(wp.floor(uy))-R, 0)
            y1 = wp.min(int(wp.floor(uy))+R+1, Hd-1)
            for y in range(y0, y1+1):
                for x in range(x0, x1+1):
                    d = wp.normalize(wp.vec3f((wp.float32(x)-cx[v])/f, (wp.float32(y)-cy[v])/f, 1.0))
                    t = wp.dot(p, d)
                    dist = wp.length(p-t*d)
                    a = wp.clamp(0.5+(r[i]-dist)*f/t, 0.0, 1.0)
                    if a > 0.0:
                        wp.atomic_max(img, v, y, x, a)
        self.k_sil = k_sil
        self.fx = wp.array(self.K[:, 0, 0].astype(np.float32), dtype=wp.float32, device='cuda')
        self.cx = wp.array(self.K[:, 0, 2].astype(np.float32), dtype=wp.float32, device='cuda')
        self.cy = wp.array(self.K[:, 1, 2].astype(np.float32), dtype=wp.float32, device='cuda')
        H, W = self.H, self.W
        self.ys = torch.arange(H, dtype=torch.float64, device='cuda')[:, None]
        self.xs = torch.arange(W, dtype=torch.float64, device='cuda')[None, :]

    def render(self, x, r):
        """x (n, 3) foam frame, r (n,): (views, H, W) coverage (torch float32, a view into the Warp image)"""
        torch, wp = self.torch, self.wp
        xw = x.to(torch.float64)-self.off
        pc = torch.einsum('vij,nj->vni', self.R, xw)+self.t[:, None, :]
        self.img.zero_()
        wp.launch(self.k_sil, (self.nv, x.shape[0]), inputs=[wp.from_torch(pc.to(torch.float32).contiguous(), dtype=wp.vec3f),
                                                              wp.from_torch(r.to(torch.float32).contiguous()), self.fx, self.cx, self.cy,
                                                              self.W, self.H], outputs=[self.img])
        return self.timg

    def features(self, x, r):
        """(views, 5): identify.moments of each view's coverage"""
        a = self.render(x, r).to(self.torch.float64)
        m0 = a.sum((1, 2)).clamp(min=1e-6)
        cx = (a*self.xs).sum((1, 2))/m0; cy = (a*self.ys).sum((1, 2))/m0
        sx = ((a*(self.xs-cx[:, None, None])**2).sum((1, 2))/m0).clamp(min=1e-12).sqrt()
        sy = ((a*(self.ys-cy[:, None, None])**2).sum((1, 2))/m0).clamp(min=1e-12).sqrt()
        return self.torch.stack((cx, cy, sx, sy, m0.sqrt()), 1)


FOAM = DATA/'foam'                                                                 # (--foam: another foam of the same lion, frame and cameras)


def load_foam():
    """cells, geometry and the ground balls' cells: patch_site.npz if the foam has one, else None (the exposed spheres,
    ClosureTerms.c_body: on inverse_data/lion/foam the same 4024 cells in the same order as its patch_site, 11-06)"""
    cells = dict(np.load(FOAM/'dem_exact_single'/'cells.npz')); g = dict(np.load(FOAM/'dem_exact_single'/'geometry.npz'))
    site = np.load(FOAM/'patch_site.npz')['patch_site'] if (FOAM/'patch_site.npz').exists() else None
    return cells, g, site


# ---------------------------------------------------------------- one forward
CHAIN = 0                                                                          # (--chain 1: ClosureWarp.set_chain, the rigid chain cells)
DRAG = 0.                                                                          # (--drag s > 0: the ground's tangential force without memory, k_t x s)
GLUE = 'site'                                                                     # (--glue-point: where the ground's glue holds a cell, ClosureWarp.set_glue_point)
SHAPE = 'rigid'                                                                    # (--ball-shape: the ground's normal contact ball, ClosureWarp.set_ball_shape)
CAPC = 0.                                                                          # (--cap-contact alpha > 0: the ground on the exposed caps, penalty alpha (lam + 2 mu) / h, no Hertz balls)
CAPT = 0                                                                           # (--cap-traction 1: the touching caps' traction into the closure, ClosureWarp.cap_traction)
CLIP = 0.                                                                          # (--cap-clip w > 0: the touching caps' normal column from the ground, ClosureWarp.set_cap_clip; implies the cap contact)
CLIPP = 0.                                                                         # (--cap-clip-pen: the cap contact's normal penalty with the clip, x this)
CLIPS = 0                                                                          # (--cap-clip-stick 1: with --cap-clip, the column's tangential part from the no-slip too)
BAND = 0.                                                                          # (--cap-band delta > 0 (m): the no-slip without a held force, fading over delta, ClosureWarp.set_cap_band; implies the cap contact)
BANDK = 20.                                                                        # (--cap-band-kappa: its hard stop's stand-in, x the cell's inertia)


def chain_colouring(W, i_, j_, colouring):
    """probe_blocks' colouring over the sites and set_chain's rotation entries: a rotation entry coupled to its cell and to
    every site and rotation its cell's bonds reach"""
    n = W.n; ck = W._ck_t.cpu().numpy()
    dof = lambda c: [c]+([n+int(ck[c])] if ck[c] >= 0 else [])
    ei, ej = [], []
    for a, b in zip(i_.tolist(), j_.tolist()):
        for u in dof(a):
            for v in dof(b):
                ei.append(u); ej.append(v)
    for c in W.chain.tolist():
        ei.append(c); ej.append(n+int(ck[c]))
    return colouring(np.array(ei), np.array(ej), W.ndof)


def forward(theta, frames, tol=1e-3, save_frames=False, verbose=True, on_step=None):
    """one run at theta = (lnE, nu, lnbeta) to frame `frames`: dict(f (frames + 1, views, 5), x, r per frame if asked,
    stats); raises RuntimeError at a step that does not converge"""
    import torch
    import warp as wp
    from experiments.pacnerf_dem_forward import build
    from free_surface.closure_drop import colouring, probe_blocks
    from free_surface.closure_warp import ClosureWarp, MidpointStep, TrustCG
    from free_surface.sphere_closure import ClosureTerms
    f64, dev = torch.float64, 'cuda'
    E, nu, beta = math.exp(theta[0]), float(theta[1]), math.exp(theta[2])
    if beta < BETA_MIN*(1-1e-6):                                                    # (logs typed to 6 decimals)
        raise ValueError('beta %.5g below this scene\'s minimum %.5g (SCENES beta_min)' % (beta, BETA_MIN))
    t0 = time.time()
    cells, g, site_np = load_foam()
    m = build(g, E, nu, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    st = ClosureTerms(m, g, cells['points'], cells['radii'])
    W = ClosureWarp(st)
    W.set_viscosity(beta*W.mu, beta*(W.lam+2*W.mu/3), DT)
    W.visc_weight = 2.; W.total_mode()
    boc = torch.as_tensor(np.asarray(g['body_of_cell']), device=dev).long()
    site = st.c_body.clone() if site_np is None else torch.as_tensor(site_np, device=dev).long()
    cbody = boc[site]; Rb = torch.as_tensor(cells['radii'], dtype=f64, device=dev)[site]
    upt = tuple(1. if k == UP else 0. for k in range(3))
    capc = CAPC if CAPC else (30. if (CLIP or BAND) else 0.)                        # (the clip, the band: on the cap contact's points
    W.set_ball(cbody, Rb, upt, C0 if not capc else -10., MU if not capc else 0.)    # (the scene's friction; no self contact; --cap-contact: the balls' plane far away)
    W.set_glue_point(GLUE); W.set_ball_shape(SHAPE); W.cap_traction = bool(CAPT)
    W.set_cap_positions(); W.set_cap_floor(0.)
    if capc:
        from free_surface.cap_contact import cap_quadrature
        cl, off, wq = cap_quadrature(cells['points'], cells['radii'], st.c_body.cpu().numpy(), cache=str(FOAM/'dem_exact_single'/'cap_quadrature.npz'))
        o_ = np.argsort(cl, kind='stable'); cl, off, wq = cl[o_], off[o_], wq[o_]    # (by cell: the clip's per-cell lists)
        h = float((st.V.sum()/st.n)**(1/3))
        W.set_cap_contact(cl, off, wq, upt, C0, capc*(W.lam+2*W.mu)/h, 1., MU)
        if CLIP:
            W.set_cap_clip(cl, off, upt, C0, CLIP, pen=CLIPP, stick=bool(CLIPS))
        if BAND:                                                                    # (kstop: kappa x the cell's midpoint inertia 4 V / dt^2 over the
            A_ = np.bincount(cl, weights=wq, minlength=st.n)                          # contact's z-stiffness factor cs^2 wt = 2, split by area)
            Vn = st.V.cpu().numpy()
            W.set_cap_band(BANDK*(4*Vn[cl]/DT**2)/2.*wq/A_[cl], BAND)
    nr = W.set_chain() if CHAIN else 0
    W.cm_cache = str(FOAM/'dem_exact_single'/'cell_moments.npz')                    # (the consistent mass's cell moments, by geometry)
    W.prepare()
    n = W.n; X = st.X; mass = m.mass.to(f64); M3 = mass[:, None]
    up = torch.tensor(upt, dtype=f64, device=dev); fg = -9.8*M3*up; ref = float(fg.norm())   # (the tolerance's scale: the lumped weight)
    if W.cmass is not None:                                                         # (the consistent weight M g: each cell's at its material)
        fg = -9.8*torch.as_tensor(W.cmass['rowsum'], dtype=f64, device=dev)[:, None]*up
    if nr:
        mass_e = torch.cat([mass, W.rot_inertia(mass)]); Xe = W.X_ext
        Kbb = probe_blocks(W, Xe, chain_colouring(W, st.i.cpu().numpy(), st.j.cpu().numpy(), colouring))
        print('  chain cells %d (bonds touching them %d), rotational inertia / m r^2 0.4' % (nr, W.chain_bonds), flush=True)
    else:
        mass_e, Xe = mass, X
        Kbb = probe_blocks(W, X, colouring(st.i.cpu().numpy(), st.j.cpu().numpy(), n))
    W.visc_ref(Xe)
    W.tball[0], W.tball[1] = 2., .5
    fg_e = W.ext(fg)
    CG = TrustCG(W, mass_e, DT, per=2, product_iters=1); ST = MidpointStep(W, CG, mass_e, Kbb, fg_e, DT, MU)
    ST.smooth_pred = ST.carry_delta = True                                           # (dem_old1's: the same solution, faster)
    if DRAG > 0:
        ST.glue_memory, ST.glue_scale = False, DRAG
    tl = tol*ref
    ST.set_state(Xe, W.ext(torch.tensor(V0, dtype=f64, device=dev).expand_as(X).clone()), W.ext(-9.8*up.expand_as(X).clone()))   # (free fall: a = g)
    sil = Silhouette()
    r0 = torch.as_tensor(cells['radii'], dtype=f64, device=dev)
    fs, xs, rs = [sil.features(X, r0).cpu().numpy()], [X.cpu().numpy().astype(np.float32)], [r0.cpu().numpy().astype(np.float32)]
    tb = time.time()-t0; t1 = time.time(); prods = 0
    for s in range(1, frames*SPF+1):
        p_, its, g2 = ST.step(tl, lambda g2_: .5*tl)
        if not (math.isfinite(g2) and math.sqrt(g2) <= tl):
            raise RuntimeError('step %d (t %.4f s): the solve did not converge (|grad| / weight %s after %d Newton iterations)' % (
                s, s*DT, math.sqrt(g2)/ref if math.isfinite(g2) else g2, its))
        ST.end(); prods += p_
        if on_step is not None:                                                     # (a probe after every step: on_step(s, W, ST, cbody, Rb))
            on_step(s, W, ST, cbody, Rb)
        if s % SPF:
            continue
        xe = ST.tx.clone(); x = xe[:n]
        W.grad(xe); Fb = W.tF.clone()
        r = r0*torch.linalg.det(Fb).abs().clamp(min=1e-6)**(1/3)
        fs.append(sil.features(x, r).cpu().numpy())
        if save_frames:
            xs.append(x.cpu().numpy().astype(np.float32)); rs.append(r.cpu().numpy().astype(np.float32))
        if verbose:
            print('  frame %d (t %.2f s, %.0f s): com height %.4f m, balls touching %d, min J %.3f, products so far %d' % (
                s//SPF, s*DT, time.time()-t1, float((M3*x).sum(0)[UP]/mass.sum()), int((W.tbdelta > 0).sum()), math.exp(float(W.tlj.min())), prods), flush=True)
            if CLIP:                                                                # (the clip at the frame: caps clipped, their s / s_ps, the cap points' depth)
                cq = W.tcq; cb_ = cq['cb'].long()
                xq = x[cb_]+(W.tFs[cb_]@cq['rq'][:, :, None])[:, :, 0]
                dep = -(xq[:, UP]-C0)
                on = (W.tsps-W.tsclip > 1e-3) & (W.tsps > 0)
                rat = (W.tsclip[on]/W.tsps[on]) if bool(on.any()) else torch.ones(1, dtype=f64, device=dev)
                print('     clip: caps clipped %d, s / s_ps median %.3f min %.3f; cap points below the ground %d, deepest %.2f mm; glue bound > 0 on %d points' % (
                    int(on.sum()), float(rat.median()), float(rat.min()), int((dep > 0).sum()), 1e3*float(dep.max()), int((cq['bound'] > 0).sum())), flush=True)
                if CLIPS and bool(on.any()):                                        # (the stick: each clipped cap's lowest point from its anchor, along the plane)
                    low = torch.full((n,), -1e9, dtype=f64, device=dev).scatter_reduce(0, cb_, dep, 'amax')
                    il = torch.nonzero(on[cb_] & (dep >= low[cb_]-1e-15))[:, 0]
                    gap = xq[il]-cq['xqn'][il]; gap[:, UP] = 0
                    print('     stick: lowest points of clipped caps %d, along-plane distance from the step-start anchor median %.3f mm, max %.3f mm' % (
                        len(il), 1e3*float(gap.norm(dim=1).median()), 1e3*float(gap.norm(dim=1).max())), flush=True)
    out = dict(f=np.stack(fs), stats=dict(build=tb, seconds=time.time()-t1, products=prods))
    if save_frames:
        out.update(x=np.stack(xs), r=np.stack(rs))
    return out


def cmd_run(a):
    th = np.array(a.theta, float)
    print('run: E %.2f, nu %.4f, beta %.5f; frames 0..%d; tol %g' % (math.exp(th[0]), th[1], math.exp(th[2]), a.frames, a.tol), flush=True)
    try:
        o = forward(th, a.frames, a.tol, a.save_frames)
    except RuntimeError as e:
        print('FAILED: %s' % e, flush=True)
        np.savez(a.out, failed=str(e), theta=th)
        sys.exit(1)
    np.savez(a.out, theta=th, tol=a.tol, f=o['f'], **{k: o[k] for k in ('x', 'r') if k in o}, stats=json.dumps(o['stats']))
    print('done: build %.0f s, run %.0f s, products %d -> %s' % (o['stats']['build'], o['stats']['seconds'], o['stats']['products'], a.out), flush=True)


# ---------------------------------------------------------------- frame 0 against the masks
def cmd_check(a):
    import torch
    cells, g, _ = load_foam()
    sil = Silhouette()
    x = torch.as_tensor(cells['points'], dtype=torch.float64, device='cuda'); r = torch.as_tensor(cells['radii'], dtype=torch.float64, device='cuda')
    img = sil.render(x, r).cpu().numpy() > .5
    A = gt_masks()[:, 0] > .5
    fo = sil.features(x, r).cpu().numpy(); fg_ = gt_features()[0]
    meta, _, _ = cameras()
    for v, nm in enumerate(meta['camera_names']):
        iou = (img[v] & A[v]).sum()/(img[v] | A[v]).sum()
        print('%s: IoU %.4f; features ours - GT (px): centroid %+.2f %+.2f, widths %+.2f %+.2f, sqrt area %+.2f' % (nm, iou, *(fo[v]-fg_[v])), flush=True)
    sx, sy = np.abs(fo-fg_).mean(0)[:2]
    print('mean |centroid| difference %.2f / %.2f px' % (sx, sy), flush=True)


# ---------------------------------------------------------------- parallel runs
def run_batch(thetas, frames, tol, out, jobs, tag):
    """each theta in its own process (`run`), at most `jobs` at once: list of feature arrays (None for a failed run);
    tol one value or one per theta"""
    tols = list(tol) if isinstance(tol, (list, tuple)) else [tol]*len(thetas)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''))
    paths = [out/('%s_%d.npz' % (tag, k)) for k in range(len(thetas))]
    todo, live, t0 = list(range(len(thetas))), {}, time.time()
    while todo or live:
        while todo and len(live) < jobs:
            k = todo.pop(0); th = thetas[k]
            if paths[k].exists():
                paths[k].unlink()
            log = open(out/('%s_%d.log' % (tag, k)), 'w')
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'run', '--theta', *['%.17g' % v for v in th], '--frames', str(frames),
                   '--tol', '%g' % tols[k], '--out', str(paths[k]), '--foam', str(FOAM), '--chain', str(CHAIN), '--scene', SCENE_NAME, '--drag', '%g' % DRAG, '--glue-point', GLUE, '--ball-shape', SHAPE, '--cap-traction', str(CAPT), '--cap-contact', '%g' % CAPC,
                   '--cap-clip', '%g' % CLIP, '--cap-clip-pen', '%g' % CLIPP,
                   '--cap-band', '%g' % BAND, '--cap-band-kappa', '%g' % BANDK, '--cap-clip-stick', str(CLIPS)]
            live[k] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env), log)
        time.sleep(2)
        for k in list(live):
            p, log = live[k]
            if p.poll() is not None:
                log.close(); del live[k]
    res = []
    for k, pth in enumerate(paths):
        z = np.load(pth) if pth.exists() else None
        res.append(None if z is None or 'failed' in z.files else z['f'])
    print('   %d runs (%s) in %.0f s, %d failed' % (len(thetas), tag, time.time()-t0, sum(r_ is None for r_ in res)), flush=True)
    return res


# ---------------------------------------------------------------- Levenberg-Marquardt
def cmd_lm(a):
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    K, held = a.frames, list(a.held)
    fit = [v for v in range(12) if v not in held]
    if a.beta_min is not None and a.beta_min < BETA_MIN:
        print('--beta-min %.4g below this scene\'s minimum: %.4g' % (a.beta_min, BETA_MIN), flush=True)
    lnb_min = math.log(max(a.beta_min or BETA_MIN, BETA_MIN))
    if a.obs == 'real':
        obs = gt_features()
    else:
        print('synthetic observations: one run at the truth %s' % a.truth, flush=True)
        obs = run_batch([np.array(a.truth)], K, a.tol, out/'runs', 1, 'truth')[0]
        if obs is None:
            raise SystemExit('the truth run failed')
    rel = (lambda f_: f_-f_[:1]) if a.relative else (lambda f_: f_)              # (each side's own frame 0 taken off: the
    obs = rel(obs)                                                                   # static shape / registration bias, 1-4.5 px)
    flat = lambda f_: (rel(f_)[1:K+1][:, fit]-obs[1:K+1][:, fit]).reshape(-1)
    held_c = lambda f_: float(((rel(f_)[1:K+1][:, held]-obs[1:K+1][:, held])**2).sum())
    path = out/'lm.json'
    hist = json.loads(path.read_text()) if a.resume and path.exists() else []
    if hist and 'phi' not in hist[-1]:
        raise SystemExit('%s is from the version with nu as the unknown (before 11-06): delete %s and start again' % (path, out))
    th = np.array([e['phi'] for e in hist if e['accepted']][-1]) if hist else phi_of(np.array(a.start, float))   # (th: phi from here on)
    it0 = max([e['it'] for e in hist], default=0)
    lo_b, hi_b = np.array([-math.inf, LNK_RANGE[0], lnb_min]), np.array([math.inf, LNK_RANGE[1], math.inf])
    clamp = lambda t_: np.minimum(np.maximum(t_, lo_b), hi_b)
    th = clamp(th)                                                                  # (a start below beta's minimum: at it)
    eps = np.array(a.eps, float)

    def log(it, t_, c_, f_, acc, info):
        q = theta_of(t_)
        e = dict(it=it, phi=t_.tolist(), theta=q.tolist(), E=math.exp(q[0]), nu=q[1], beta=math.exp(q[2]), mu=math.exp(t_[0]), k=math.exp(t_[1]),
                 cost=c_, accepted=acc, rms=math.sqrt(c_/(K*len(fit)*5)) if math.isfinite(c_) else None, held=held_c(f_) if f_ is not None else None,
                 log10E_err=abs(q[0]-math.log(GT['E']))/math.log(10), nu_err=abs(q[1]-GT['nu']), **info)
        hist.append(e); path.write_text(json.dumps(hist, indent=1))
        print('it %d %s: E %.1f (log10 error %.3f), nu %.4f, beta %.4f (mu %.1f, K/mu %.3f); cost %.4e (RMS %.3f px; held view %s)' % (
            it, 'accepted' if acc else 'rejected', e['E'], e['log10E_err'], e['nu'], e['beta'], e['mu'], e['k'], c_,
            e['rms'] if e['rms'] else float('nan'), '%.4e' % e['held'] if e['held'] is not None else '-'), flush=True)

    def jac(t_):
        """central differences in phi (steps --eps), all 2p runs at once; a parameter at its bound: the pair moved inside"""
        ths, steps = [], []
        for j in range(3):
            tp, tm = t_.copy(), t_.copy(); tp[j] += eps[j]; tm[j] -= eps[j]
            if tm[j] < lo_b[j]:
                tm[j], tp[j] = lo_b[j], lo_b[j]+2*eps[j]
            if tp[j] > hi_b[j]:
                tp[j], tm[j] = hi_b[j], hi_b[j]-2*eps[j]
            ths += [tp, tm]; steps.append(tp[j]-tm[j])
        fs = run_batch([theta_of(t__) for t__ in ths], K, a.tol, out/'runs', a.jobs, 'jac')
        if any(f_ is None for f_ in fs):
            raise SystemExit('a difference run failed: %s' % [k for k, f_ in enumerate(fs) if f_ is None])
        J_ = np.stack([(flat(fs[2*j])-flat(fs[2*j+1]))/steps[j] for j in range(3)], 1)
        np.savez(out/'jac.npz', phi=t_, theta=theta_of(t_), J=J_)
        return J_

    ipath = out/'identifiability.json'
    ident_hist = json.loads(ipath.read_text()) if a.resume and ipath.exists() else []

    def ident(t_, J_, r_, d_):
        """how tightly the data hold each parameter at t_ (11-06, user: nu could still climb slowly): the Gauss-Newton
        covariance s^2 (J^T J)^-1 in phi with s^2 the residual's variance per entry (and, as a floor, the forward's own
        error's), carried to (log10 E, nu, ln beta) by the map's Jacobian; E-nu correlation; J's conditioning. A large
        sigma(nu) or |corr(E, nu)| near 1 means the silhouettes do not fix nu (a valley), whatever value the LM reports"""
        N = len(r_); s2, n2 = float(r_@r_)/max(N-3, 1), float(d_@d_)/N
        A = np.linalg.inv(J_.T@J_)
        M = np.stack([(theta_of(t_+h_)-theta_of(t_-h_))/2e-6 for h_ in np.eye(3)*1e-6], 1)
        M[0] /= math.log(10)                                                          # (rows log10 E, nu, ln beta)
        out_ = {}
        for nm, sc in (('residual', s2), ('forward_error', n2)):
            C = sc*M@A@M.T; sd = np.sqrt(np.diag(C))
            out_[nm] = dict(sigma_log10E=sd[0], sigma_nu=sd[1], sigma_lnbeta=sd[2], corr_E_nu=C[0, 1]/max(sd[0]*sd[1], 1e-300))
        Cp = A/np.sqrt(np.outer(np.diag(A), np.diag(A)))
        sv = np.linalg.svd(J_/np.linalg.norm(J_, axis=0), compute_uv=False)
        rec = dict(phi=t_.tolist(), corr_phi=Cp.tolist(), singular_normalised=sv.tolist(), condition=float(sv[0]/sv[-1]), **out_)
        ident_hist.append(rec); ipath.write_text(json.dumps(ident_hist, indent=1))
        r0 = out_['residual']
        print('   identifiability at E %.1f, nu %.4f: sigma(log10 E) %.4f, sigma(nu) %.4f, sigma(ln beta) %.4f, corr(E, nu) %+.3f (from the residual; '
              'from the forward\'s error alone: %.2e, %.2e); corr(ln mu, ln K/mu) %+.3f; normalised J singular values %s (condition %.1f)' % (
                  math.exp(theta_of(t_)[0]), theta_of(t_)[1], r0['sigma_log10E'], r0['sigma_nu'], r0['sigma_lnbeta'], r0['corr_E_nu'],
                  out_['forward_error']['sigma_log10E'], out_['forward_error']['sigma_nu'], Cp[0, 1], np.round(sv, 4).tolist(), sv[0]/sv[-1]), flush=True)

    # the start: the run at tol and at --ref-tol (the forward's own error), together
    f, f_ref = run_batch([theta_of(th)]*2, K, [a.tol, a.ref_tol], out/'runs', a.jobs, 'start')
    if f is None or f_ref is None:
        raise SystemExit('the start run failed')
    dvec = flat(f)-flat(f_ref)
    print('the forward\'s error at the start (tol %g against %g): |d| %.3e px' % (a.tol, a.ref_tol, float(np.linalg.norm(dvec))), flush=True)
    r = flat(f); c, radius = float(r@r), math.inf
    if not hist:
        log(0, th, c, f, True, dict(radius=radius))
    J = jac(th); ident(th, J, r, dvec)
    for it in range(it0+1, it0+a.iters+1):
        D = np.diag(J.T@J); dn = lambda s_: float(np.sqrt((D*s_*s_).sum()))

        def step(mu):
            if mu == 0:
                return np.linalg.lstsq(J, -r, rcond=None)[0]
            return np.linalg.lstsq(np.vstack((J, np.diag(np.sqrt(mu*D)))), np.concatenate((-r, np.zeros(3))), rcond=None)[0]
        res = float(np.linalg.norm(J@np.linalg.lstsq(J, dvec, rcond=None)[0]))
        ok = lambda s_: dn(s_) <= radius and float(np.abs(s_).max()) <= a.max_step   # (the region and, from the first step on,
        while True:                                                                  # at most --max-step in any parameter)
            mu, s_ = 0., step(0.)
            if not ok(s_):
                lo, hi = 0., 1.
                while not ok(step(hi)):
                    lo, hi = hi, 2*hi
                for _ in range(60):
                    m_ = .5*(lo+hi)
                    lo, hi = (m_, hi) if not ok(step(m_)) else (lo, m_)
                mu, s_ = hi, step(hi)
            tt = clamp(th+s_); s_ = tt-th
            pred, slope = float(np.linalg.norm(J@s_)), float(2*r@(J@s_))
            if pred <= res:
                print('LM: the predicted change |J s| %.3e is not above the forward\'s error in J\'s range %.3e: stop' % (pred, res), flush=True)
                return finish(a, th)
            if slope >= 0:
                print('LM: the step (at a bound) does not descend: stop', flush=True)
                return finish(a, th)
            f_t = run_batch([theta_of(tt)], K, a.tol, out/'runs', 1, 'trial')[0]
            c_t = float(flat(f_t)@flat(f_t)) if f_t is not None else math.inf
            kap = c_t-c-slope
            al = -slope/(2*kap) if kap > 0 else math.inf
            if not math.isfinite(c_t):
                al = .25                                                             # (a failed trial: a quarter of the step)
            radius = al*dn(s_)
            info = dict(damping=mu, step=dn(s_), pred=pred, resolution=res, pred_cost=c+slope+float((J@s_)@(J@s_)), alpha=al, radius=radius)
            if c_t < c:
                th, f, r, c = tt, f_t, flat(f_t), c_t
                log(it, th, c, f, True, info)
                J = jac(th); ident(th, J, r, dvec)
                break
            log(it, tt, c_t, f_t, False, info)
    return finish(a, th)


def finish(a, ph):
    th = theta_of(ph)
    print('final theta: E %.2f, nu %.4f, beta %.5f; log10 E error %.3f, nu error %.4f' % (
        math.exp(th[0]), th[1], math.exp(th[2]), abs(th[0]-math.log(GT['E']))/math.log(10), abs(th[1]-GT['nu'])), flush=True)
    (a.out/'final_theta.json').write_text(json.dumps(dict(theta=th.tolist(), phi=ph.tolist(), E=math.exp(th[0]), nu=th[1], beta=math.exp(th[2]))))
    return th


# ---------------------------------------------------------------- evaluation
def cmd_eval(a):
    a.out.mkdir(parents=True, exist_ok=True)
    th = np.array(a.theta, float)
    o = forward(th, NFRAMES-1, a.tol, save_frames=True, verbose=True)
    import torch
    sil = Silhouette()
    A = gt_masks() > .5
    iou = np.zeros((A.shape[0], NFRAMES))
    for k in range(NFRAMES):
        img = sil.render(torch.as_tensor(o['x'][k], device='cuda'), torch.as_tensor(o['r'][k], device='cuda')).cpu().numpy() > .5
        for v in range(A.shape[0]):
            iou[v, k] = (img[v] & A[v, k]).sum()/max((img[v] | A[v, k]).sum(), 1)
    held = list(a.held); fit = [v for v in range(A.shape[0]) if v not in held]
    rep = dict(theta=th.tolist(), E=math.exp(th[0]), nu=th[1], beta=math.exp(th[2]),
               log10E_err=abs(th[0]-math.log(GT['E']))/math.log(10), nu_err=abs(th[1]-GT['nu']),
               iou_fit_views_train=float(iou[fit][:, 1:16].mean()), iou_held_view_train=float(iou[held][:, 1:16].mean()),
               iou_future_all_views=float(iou[:, 16:].mean()), iou_future_held_view=float(iou[held][:, 16:].mean()),
               iou=iou.tolist(), stats=o['stats'])
    (a.out/'eval.json').write_text(json.dumps(rep, indent=1))
    np.savez(a.out/'eval_frames.npz', x=o['x'], r=o['r'], f=o['f'])
    print('eval: E %.2f (log10 error %.3f), nu %.4f (error %.4f), beta %.5f | IoU: fit views frames 1-15 %.4f, held view %.4f; future frames 16-23 '
          'all views %.4f, held view %.4f' % (rep['E'], rep['log10E_err'], rep['nu'], rep['nu_err'], rep['beta'], rep['iou_fit_views_train'],
                                             rep['iou_held_view_train'], rep['iou_future_all_views'], rep['iou_future_held_view']), flush=True)


def cmd_show(a):
    """one forward at --theta to --frames, then the photos with our silhouette's outline (red) for --views: per frame one
    row of the views (show_<frame>.png), a sheet of all frames (rows) and views (columns), and a GIF"""
    import torch
    from PIL import Image
    a.out.mkdir(parents=True, exist_ok=True)
    th = np.array(a.theta, float)
    K = a.frames if a.frames is not None else NFRAMES-1
    print('show: E %.2f, nu %.4f, beta %.5f; frames 0..%d; views %s' % (math.exp(th[0]), th[1], math.exp(th[2]), K, a.views), flush=True)
    try:
        o = forward(th, K, a.tol, save_frames=True, verbose=True)
    except RuntimeError as e:
        print('FAILED: %s' % e, flush=True); sys.exit(1)
    np.savez(a.out/'show_frames.npz', x=o['x'], r=o['r'], f=o['f'], theta=th)
    sil = Silhouette(); meta, _, _ = cameras(); names = meta['camera_names']
    photo = lambda v, k: np.asarray(Image.open(Path(SCENE['photos'])/('r_%s_%d.png' % (names[v], k))).convert('RGB'))
    rows = []
    for k in range(K+1):
        m = sil.render(torch.as_tensor(o['x'][k], device='cuda'), torch.as_tensor(o['r'][k], device='cuda')).cpu().numpy() > .5
        row = []
        for v in a.views:
            im = photo(v, k).copy(); mk = m[v]
            inner = mk.copy(); inner[1:-1, 1:-1] = mk[1:-1, 1:-1] & mk[:-2, 1:-1] & mk[2:, 1:-1] & mk[1:-1, :-2] & mk[1:-1, 2:]
            edge = mk & ~inner
            edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)                  # (2 px wide)
            im[edge] = (255, 0, 0)
            row.append(im)
        row = np.concatenate(row, 1)
        Image.fromarray(row).save(a.out/('show_%02d.png' % k)); rows.append(row)
    Image.fromarray(np.concatenate(rows, 0)).resize((rows[0].shape[1]//2, len(rows)*rows[0].shape[0]//2)).save(a.out/'show_sheet.png')
    ims = [Image.fromarray(r_) for r_ in rows]
    ims[0].save(a.out/'show.gif', save_all=True, append_images=ims[1:], duration=1000//FPS*3, loop=0)
    print('wrote %s: show_00..%02d.png (views %s side by side, red: our silhouette), show_sheet.png, show.gif; run %.0f s' % (
        a.out, K, a.views, o['stats']['seconds']), flush=True)


def main():
    global FOAM, CHAIN, SCENE_NAME, DRAG, GLUE, SHAPE, CAPT, CAPC, CLIP, CLIPP, BAND, BANDK, CLIPS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('check')
    q.add_argument('--foam', type=Path, default=None)
    q.add_argument('--scene', default='lion', choices=sorted(SCENES))
    q.add_argument('--drag', type=float, default=0., help='> 0: the ground without memory, a drag k_t x this (0: the glue as before)')
    q.add_argument('--glue-point', default='site', choices=('site', 'material'), help='where the ground\'s glue holds a touching cell (ClosureWarp.set_glue_point)')
    q.add_argument('--ball-shape', default='rigid', choices=('rigid', 'deformed'), help='the ground\'s normal contact ball (ClosureWarp.set_ball_shape)')
    q.add_argument('--cap-traction', type=int, default=0, help='1: the touching caps\' traction (the ground\'s force over the cap area, one step late) into the closure')
    q.add_argument('--cap-contact', type=float, default=0., help='alpha > 0: the ground on the exposed caps (penalty alpha (lam + 2 mu) / h per area), not the Hertz balls')
    q.add_argument('--cap-clip', type=float, default=0., help='w > 0: the touching caps\' normal column from the ground (softmin width w; ClosureWarp.set_cap_clip)')
    q.add_argument('--cap-clip-pen', type=float, default=0., help='with --cap-clip: the cap contact\'s normal penalty x this (0: the clip alone)')
    q.add_argument('--cap-clip-stick', type=int, default=0, help='1: with --cap-clip, the column\'s tangential part from the contact\'s no-slip too')
    q.add_argument('--cap-band', type=float, default=0., help='delta > 0 (m): the tangential no-slip without a held force, fading over delta (ClosureWarp.set_cap_band)')
    q.add_argument('--cap-band-kappa', type=float, default=20., help='with --cap-band: the hard stop\'s stand-in, x the cell\'s inertia')
    q.add_argument('--chain', type=int, default=0, help='1: rigid chain cells with their own rotation (ClosureWarp.set_chain)')
    q = sub.add_parser('run')
    q.add_argument('--foam', type=Path, default=None)
    q.add_argument('--scene', default='lion', choices=sorted(SCENES))
    q.add_argument('--drag', type=float, default=0., help='> 0: the ground without memory, a drag k_t x this (0: the glue as before)')
    q.add_argument('--glue-point', default='site', choices=('site', 'material'), help='where the ground\'s glue holds a touching cell (ClosureWarp.set_glue_point)')
    q.add_argument('--ball-shape', default='rigid', choices=('rigid', 'deformed'), help='the ground\'s normal contact ball (ClosureWarp.set_ball_shape)')
    q.add_argument('--cap-traction', type=int, default=0, help='1: the touching caps\' traction (the ground\'s force over the cap area, one step late) into the closure')
    q.add_argument('--cap-contact', type=float, default=0., help='alpha > 0: the ground on the exposed caps (penalty alpha (lam + 2 mu) / h per area), not the Hertz balls')
    q.add_argument('--cap-clip', type=float, default=0., help='w > 0: the touching caps\' normal column from the ground (softmin width w; ClosureWarp.set_cap_clip)')
    q.add_argument('--cap-clip-pen', type=float, default=0., help='with --cap-clip: the cap contact\'s normal penalty x this (0: the clip alone)')
    q.add_argument('--cap-clip-stick', type=int, default=0, help='1: with --cap-clip, the column\'s tangential part from the contact\'s no-slip too')
    q.add_argument('--cap-band', type=float, default=0., help='delta > 0 (m): the tangential no-slip without a held force, fading over delta (ClosureWarp.set_cap_band)')
    q.add_argument('--cap-band-kappa', type=float, default=20., help='with --cap-band: the hard stop\'s stand-in, x the cell\'s inertia')
    q.add_argument('--chain', type=int, default=0, help='1: rigid chain cells with their own rotation (ClosureWarp.set_chain)')
    q.add_argument('--theta', type=float, nargs=3, required=True)
    q.add_argument('--frames', type=int, default=15)
    q.add_argument('--tol', type=float, default=1e-3)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--save-frames', action='store_true')
    q = sub.add_parser('lm')
    q.add_argument('--foam', type=Path, default=None)
    q.add_argument('--scene', default='lion', choices=sorted(SCENES))
    q.add_argument('--drag', type=float, default=0., help='> 0: the ground without memory, a drag k_t x this (0: the glue as before)')
    q.add_argument('--glue-point', default='site', choices=('site', 'material'), help='where the ground\'s glue holds a touching cell (ClosureWarp.set_glue_point)')
    q.add_argument('--ball-shape', default='rigid', choices=('rigid', 'deformed'), help='the ground\'s normal contact ball (ClosureWarp.set_ball_shape)')
    q.add_argument('--cap-traction', type=int, default=0, help='1: the touching caps\' traction (the ground\'s force over the cap area, one step late) into the closure')
    q.add_argument('--cap-contact', type=float, default=0., help='alpha > 0: the ground on the exposed caps (penalty alpha (lam + 2 mu) / h per area), not the Hertz balls')
    q.add_argument('--cap-clip', type=float, default=0., help='w > 0: the touching caps\' normal column from the ground (softmin width w; ClosureWarp.set_cap_clip)')
    q.add_argument('--cap-clip-pen', type=float, default=0., help='with --cap-clip: the cap contact\'s normal penalty x this (0: the clip alone)')
    q.add_argument('--cap-clip-stick', type=int, default=0, help='1: with --cap-clip, the column\'s tangential part from the contact\'s no-slip too')
    q.add_argument('--cap-band', type=float, default=0., help='delta > 0 (m): the tangential no-slip without a held force, fading over delta (ClosureWarp.set_cap_band)')
    q.add_argument('--cap-band-kappa', type=float, default=20., help='with --cap-band: the hard stop\'s stand-in, x the cell\'s inertia')
    q.add_argument('--chain', type=int, default=0, help='1: rigid chain cells with their own rotation (ClosureWarp.set_chain)')
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--obs', choices=('real', 'synth'), default='real')
    q.add_argument('--truth', type=float, nargs=3, default=[math.log(GT['E']), GT['nu'], math.log(.05)], help='--obs synth: theta of the observations')
    q.add_argument('--start', type=float, nargs=3, default=[math.log(GT['E']*3), .3, math.log(.1)])
    q.add_argument('--frames', type=int, default=15)
    q.add_argument('--held', type=int, nargs='*', default=[5])
    q.add_argument('--tol', type=float, default=1e-3)
    q.add_argument('--ref-tol', type=float, default=2e-3)
    q.add_argument('--eps', type=float, nargs=3, default=[.01, .05, .01], help='central differences\' steps in (ln mu, ln k, ln beta)')
    q.add_argument('--max-step', type=float, default=.3, help='at most this change of any of ln mu, ln k, ln beta per step (the first one too)')
    q.add_argument('--beta-min', type=float, default=None, help='beta\'s lower bound (default and floor: the scene\'s beta_min)')
    q.add_argument('--iters', type=int, default=15)
    q.add_argument('--jobs', type=int, default=7, help='runs at once (an iteration has 6 difference runs)')
    q.add_argument('--resume', action='store_true')
    q.add_argument('--relative', type=int, default=1, help='1: features minus their own frame 0 (both sides)')
    q = sub.add_parser('show')
    q.add_argument('--foam', type=Path, default=None)
    q.add_argument('--scene', default='lion', choices=sorted(SCENES))
    q.add_argument('--drag', type=float, default=0., help='> 0: the ground without memory, a drag k_t x this (0: the glue as before)')
    q.add_argument('--glue-point', default='site', choices=('site', 'material'), help='where the ground\'s glue holds a touching cell (ClosureWarp.set_glue_point)')
    q.add_argument('--ball-shape', default='rigid', choices=('rigid', 'deformed'), help='the ground\'s normal contact ball (ClosureWarp.set_ball_shape)')
    q.add_argument('--cap-traction', type=int, default=0, help='1: the touching caps\' traction (the ground\'s force over the cap area, one step late) into the closure')
    q.add_argument('--cap-contact', type=float, default=0., help='alpha > 0: the ground on the exposed caps (penalty alpha (lam + 2 mu) / h per area), not the Hertz balls')
    q.add_argument('--cap-clip', type=float, default=0., help='w > 0: the touching caps\' normal column from the ground (softmin width w; ClosureWarp.set_cap_clip)')
    q.add_argument('--cap-clip-pen', type=float, default=0., help='with --cap-clip: the cap contact\'s normal penalty x this (0: the clip alone)')
    q.add_argument('--cap-clip-stick', type=int, default=0, help='1: with --cap-clip, the column\'s tangential part from the contact\'s no-slip too')
    q.add_argument('--cap-band', type=float, default=0., help='delta > 0 (m): the tangential no-slip without a held force, fading over delta (ClosureWarp.set_cap_band)')
    q.add_argument('--cap-band-kappa', type=float, default=20., help='with --cap-band: the hard stop\'s stand-in, x the cell\'s inertia')
    q.add_argument('--chain', type=int, default=0)
    q.add_argument('--theta', type=float, nargs=3, required=True)
    q.add_argument('--frames', type=int, default=None)
    q.add_argument('--views', type=int, nargs='+', default=[0, 3, 6, 10])
    q.add_argument('--tol', type=float, default=1e-3)
    q.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('eval')
    q.add_argument('--foam', type=Path, default=None)
    q.add_argument('--scene', default='lion', choices=sorted(SCENES))
    q.add_argument('--drag', type=float, default=0., help='> 0: the ground without memory, a drag k_t x this (0: the glue as before)')
    q.add_argument('--glue-point', default='site', choices=('site', 'material'), help='where the ground\'s glue holds a touching cell (ClosureWarp.set_glue_point)')
    q.add_argument('--ball-shape', default='rigid', choices=('rigid', 'deformed'), help='the ground\'s normal contact ball (ClosureWarp.set_ball_shape)')
    q.add_argument('--cap-traction', type=int, default=0, help='1: the touching caps\' traction (the ground\'s force over the cap area, one step late) into the closure')
    q.add_argument('--cap-contact', type=float, default=0., help='alpha > 0: the ground on the exposed caps (penalty alpha (lam + 2 mu) / h per area), not the Hertz balls')
    q.add_argument('--cap-clip', type=float, default=0., help='w > 0: the touching caps\' normal column from the ground (softmin width w; ClosureWarp.set_cap_clip)')
    q.add_argument('--cap-clip-pen', type=float, default=0., help='with --cap-clip: the cap contact\'s normal penalty x this (0: the clip alone)')
    q.add_argument('--cap-clip-stick', type=int, default=0, help='1: with --cap-clip, the column\'s tangential part from the contact\'s no-slip too')
    q.add_argument('--cap-band', type=float, default=0., help='delta > 0 (m): the tangential no-slip without a held force, fading over delta (ClosureWarp.set_cap_band)')
    q.add_argument('--cap-band-kappa', type=float, default=20., help='with --cap-band: the hard stop\'s stand-in, x the cell\'s inertia')
    q.add_argument('--chain', type=int, default=0, help='1: rigid chain cells with their own rotation (ClosureWarp.set_chain)')
    q.add_argument('--theta', type=float, nargs=3, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--held', type=int, nargs='*', default=[5])
    q.add_argument('--tol', type=float, default=1e-3)
    a = p.parse_args()
    SCENE_NAME = a.scene; set_scene(SCENE_NAME)
    if a.foam is None:
        a.foam = FOAM
    FOAM = Path(a.foam).resolve()
    CHAIN = int(getattr(a, 'chain', 0)); DRAG = float(getattr(a, 'drag', 0.)); GLUE = getattr(a, 'glue_point', 'site'); SHAPE = getattr(a, 'ball_shape', 'rigid'); CAPT = int(getattr(a, 'cap_traction', 0)); CAPC = float(getattr(a, 'cap_contact', 0.))
    CLIP = float(getattr(a, 'cap_clip', 0.)); CLIPP = float(getattr(a, 'cap_clip_pen', 0.))
    BAND = float(getattr(a, 'cap_band', 0.)); BANDK = float(getattr(a, 'cap_band_kappa', 20.))
    CLIPS = int(getattr(a, 'cap_clip_stick', 0))
    print('foam: %s%s' % (FOAM, ', chain cells rigid with their own rotation' if CHAIN else ''), flush=True)
    dict(check=cmd_check, run=cmd_run, lm=cmd_lm, eval=cmd_eval, show=cmd_show)[a.cmd](a)


if __name__ == '__main__':
    main()
