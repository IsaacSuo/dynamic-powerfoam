"""PAC-NeRF's own MPM, forward only: lib/engine/mpm_simulator.py (references/pac_nerf_code), elasticity branch, re-written
without the chunked tape but with the same arithmetic per substep:
  F <- (I + dt C) F;  P2G (quadratic B-spline, APIC: affine = -dt p_vol 4/dx^2 tau + m C, tau = mu F F^T + (lam ln J - mu) I);
  grid: v = p/m + dt g where m > 1e-8 dx^3, sticky ground (BC ground [[0,0,0],[0,1,0]]): v = 0 at nodes with y_I = I_y dx <= 1e-6;
  G2P: v = sum w v_I, C = 4/dx sum w v_I (I - fx)^T, x += dt v
p_vol = (dx/2)^3, p_mass = rho p_vol (rho 1000; both cancel in the motion), float32 as the scene's config.
Start: the GT frame 0 particles (simulation_data/elastic/0/0.ply, world frame, ground y = 0) with v0, E, nu of
gt/elastic.json scene 0; F = I, C = 0.
--split 2 replaces every particle by 8 at +-h/4 (h = the GT particles' mean spacing, (volume/N)^(1/3) = 5.24 mm), for fine
grids (GT: dx 0.02, ~56 particles per cell); the saved frames are then each parent's mean.
Saves x (frames, N, 3) float32 at every frame (t = k/24) to --out."""
import argparse
import json
import time
import numpy as np
import taichi as ti
from plyfile import PlyData
D = '/home/fangsuo/py/dynamic-powerfoam/data/pacnerf'
ap = argparse.ArgumentParser()
ap.add_argument('--dx', type=float, required=True)
ap.add_argument('--dt', type=float, default=1/4800)
ap.add_argument('--split', type=int, default=1)
ap.add_argument('--frames', type=int, default=15)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_fraction=.5)
gt = json.load(open(D+'/gt/elastic.json'))['0']
E, nu, v0 = gt['E'], gt['nu'], np.array(gt['v'])
mu_, lam_ = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
X0 = np.stack([PlyData.read(D+'/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(np.float64)
N0 = len(X0)
if a.split == 2:
    o = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing='ij')).reshape(3, -1).T*5.24e-3/4   # +-h/4, h = (volume/N)^(1/3) = 5.24 mm
    X0 = (X0[:, None, :]+o[None]).reshape(-1, 3)
N = len(X0)
dx, inv_dx, dt = a.dx, 1/a.dx, a.dt
p_vol = (dx*.5)**3; p_mass = 1000.*p_vol
lo = np.floor(np.array([-.55, -.06, -.45])/dx).astype(int)-3
hi = np.ceil(np.array([.45, .9, .45])/dx).astype(int)+3
shape = tuple(int(s) for s in hi-lo)
x = ti.Vector.field(3, ti.f32, N); v = ti.Vector.field(3, ti.f32, N)
C = ti.Matrix.field(3, 3, ti.f32, N); F = ti.Matrix.field(3, 3, ti.f32, N)
gm = ti.field(ti.f32, shape, offset=tuple(int(s) for s in lo)); gv = ti.Vector.field(3, ti.f32, shape, offset=tuple(int(s) for s in lo))
bad = ti.field(ti.i32, ())
x.from_numpy(X0.astype(np.float32)); v.from_numpy(np.tile(v0, (N, 1)).astype(np.float32))
C.from_numpy(np.zeros((N, 3, 3), np.float32)); F.from_numpy(np.tile(np.eye(3), (N, 1, 1)).astype(np.float32))
LO, HI = [int(s) for s in lo], [int(s) for s in hi]


@ti.kernel
def substep():
    for I in ti.grouped(gm):
        gm[I] = 0.; gv[I] = ti.Vector([0., 0., 0.])
    for p in x:
        Fn = (ti.Matrix.identity(ti.f32, 3)+dt*C[p])@F[p]
        F[p] = Fn
        base = ti.floor(x[p]*inv_dx-.5).cast(int)
        fx = x[p]*inv_dx-base.cast(ti.f32)
        w = [.5*(1.5-fx)**2, .75-(fx-1)**2, .5*(fx-.5)**2]
        J = Fn.determinant()
        stress = mu_*(Fn@Fn.transpose())+(lam_*ti.log(J)-mu_)*ti.Matrix.identity(ti.f32, 3)
        stress = (-dt*p_vol*4*inv_dx**2)*stress
        affine = stress+p_mass*C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            off = ti.Vector([i, j, k])
            dpos = (off.cast(ti.f32)-fx)*dx
            weight = w[i][0]*w[j][1]*w[k][2]
            gv[base+off] += weight*(p_mass*v[p]+affine@dpos)
            gm[base+off] += weight*p_mass
    for I in ti.grouped(gm):
        if gm[I] > 1e-8*dx**3:
            vo = gv[I]/gm[I]+dt*ti.Vector([0., -9.8, 0.])
            if I[1]*dx <= 1e-6:
                vo = ti.Vector([0., 0., 0.])
            gv[I] = vo
        else:
            gv[I] = ti.Vector([0., 0., 0.])
    for p in x:
        base = ti.floor(x[p]*inv_dx-.5).cast(int)
        fx = x[p]*inv_dx-base.cast(ti.f32)
        w = [.5*(1.5-fx)**2, .75-(fx-1.)**2, .5*(fx-.5)**2]
        nv = ti.Vector([0., 0., 0.]); nC = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = ti.Vector([i, j, k]).cast(ti.f32)-fx
            gvi = gv[base+ti.Vector([i, j, k])]
            weight = w[i][0]*w[j][1]*w[k][2]
            nv += weight*gvi
            nC += 4*weight*gvi.outer_product(dpos)*inv_dx
        v[p] = nv; x[p] += dt*nv; C[p] = nC
        b = ti.floor(x[p]*inv_dx-.5).cast(int)
        for d in ti.static(range(3)):
            if b[d] < LO[d]+1 or b[d]+3 > HI[d]-1:
                bad[None] = 1


nsub = round((1/24)/dt)
print('N %d (split %d), dx %.5f, dt %.3e (%d substeps per frame), grid %s, E %.1f nu %.4f v0 %s' % (N, a.split, dx, dt, nsub, shape, E, nu, v0), flush=True)
out = np.zeros((a.frames+1, N0, 3), np.float32)
out[0] = X0.reshape(N0, -1, 3).mean(1)
t0 = time.time()
for f in range(1, a.frames+1):
    for s in range(nsub):
        substep()
    xs = x.to_numpy()
    if bad[None] or not np.isfinite(xs).all():
        raise SystemExit('left the grid or NaN at frame %d' % f)
    out[f] = xs.reshape(N0, -1, 3).mean(1)
    print('frame %2d  %.1f s  lowest %.2f mm  com %s' % (f, time.time()-t0, 1e3*xs[:, 1].min(), np.round(xs.mean(0), 4)), flush=True)
np.savez(a.out, x=out, dx=dx, dt=dt, split=a.split)
print('saved', a.out)
