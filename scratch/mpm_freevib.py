"""free vibration on PAC-NeRF's MPM (mpm_pacnerf_repro.py's substep: no ground, no gravity): the GT body at frame 0 with
freevib_prep.py's v0 (FEM modes 0-8, the rigid part off); records the motion's least-squares projection on the FEM's
modes 0-11, q (samples, 12), at --rate Hz for --T s (split 2: each parent's mean)"""
import argparse
import time
from pathlib import Path
import numpy as np
import taichi as ti
from plyfile import PlyData
D = '/home/fangsuo/py/dynamic-powerfoam/data/pacnerf'
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--dx', type=float, required=True)
ap.add_argument('--dt', type=float, default=1/4800)
ap.add_argument('--split', type=int, default=1)
ap.add_argument('--T', type=float, default=2.)
ap.add_argument('--rate', type=float, default=400.)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_fraction=.5)
import json
gt = json.load(open(D+'/gt/elastic.json'))['0']
E, nu = gt['E'], gt['nu']
mu_, lam_ = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
pr = np.load(ROOT/'out_inverse'/'freevib'/'prep.npz')
X0 = np.stack([PlyData.read(D+'/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(np.float64)
V0 = pr['v0P']; phi = pr['phiP'].astype(np.float64); N0 = len(X0)
G = np.einsum('nak,naj->kj', phi, phi); Gi = np.linalg.inv(G)
if a.split == 2:
    o = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing='ij')).reshape(3, -1).T*5.24e-3/4
    X0 = (X0[:, None, :]+o[None]).reshape(-1, 3); V0 = np.repeat(V0, 8, axis=0)
N = len(X0)
dx, inv_dx, dt = a.dx, 1/a.dx, a.dt
p_vol = (dx*.5)**3; p_mass = 1000.*p_vol
lo = np.floor(np.array([-.55, -.06, -.45])/dx).astype(int)-3
hi = np.ceil(np.array([.45, .9, .45])/dx).astype(int)+3
shape = tuple(int(s) for s in hi-lo); off_ = tuple(int(s) for s in lo)
x = ti.Vector.field(3, ti.f32, N); v = ti.Vector.field(3, ti.f32, N)
C = ti.Matrix.field(3, 3, ti.f32, N); F = ti.Matrix.field(3, 3, ti.f32, N)
gm = ti.field(ti.f32, shape, offset=off_); gv = ti.Vector.field(3, ti.f32, shape, offset=off_)
x.from_numpy(X0.astype(np.float32)); v.from_numpy(V0.astype(np.float32))
C.from_numpy(np.zeros((N, 3, 3), np.float32)); F.from_numpy(np.tile(np.eye(3), (N, 1, 1)).astype(np.float32))


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
            gv[I] = gv[I]/gm[I]
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


nsub = round(1/(a.rate*dt)); ns = round(a.T*a.rate)
print('N %d (split %d), dx %.5f, dt %.3e, %d substeps per sample' % (N, a.split, dx, dt, nsub), flush=True)
Xp0 = X0.reshape(N0, -1, 3).mean(1)
q = np.zeros((ns+1, phi.shape[2])); t0 = time.time()
for s in range(1, ns+1):
    for _ in range(nsub):
        substep()
    u = x.to_numpy().astype(np.float64).reshape(N0, -1, 3).mean(1)-Xp0
    q[s] = Gi@np.einsum('nak,na->k', phi, u)
    if s % round(a.rate/4) == 0:
        print('t %.2f s (%.0f s): q0..2 %s' % (s/a.rate, time.time()-t0, np.round(q[s, :3], 6)), flush=True)
np.savez(a.out, q=q, rate=a.rate, dx=dx, dt=dt, split=a.split)
print('saved', a.out)
