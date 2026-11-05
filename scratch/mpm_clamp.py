"""the clamped test on PAC-NeRF's MPM (mpm_pacnerf_repro.py's substep, no ground): the GT body at frame 0, the clamp set
of clamp_prep.py held: its particles keep x, v = 0, C = 0, and a grid node whose mass is more than half theirs gets v = 0.
Loads: --gravity 1 (from rest, gravity on: the sag about the static state) or --kick 1 (gravity off, the free material
starts with v0's horizontal part). Records the groups' mean positions (all, tips 0-3, core) at --rate Hz for --T s."""
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
ap.add_argument('--sets', required=True)
ap.add_argument('--gravity', type=int, default=0)
ap.add_argument('--kick', type=int, default=0)
ap.add_argument('--kscale', type=float, default=1.)
ap.add_argument('--T', type=float, default=1.)
ap.add_argument('--rate', type=float, default=240.)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_fraction=.5)
gt = json.load(open(D+'/gt/elastic.json'))['0']
E, nu, v0 = gt['E'], gt['nu'], np.array(gt['v'])
mu_, lam_ = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
X0 = np.stack([PlyData.read(D+'/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(np.float64)
Sz = np.load(a.sets); clamp = Sz['clamp_m']; groups = Sz['groups_m']
N0 = len(X0)
if a.split == 2:
    o = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing='ij')).reshape(3, -1).T*5.24e-3/4
    X0 = (X0[:, None, :]+o[None]).reshape(-1, 3); clamp = np.repeat(clamp, 8); groups = np.repeat(groups, 8, axis=1)
N = len(X0)
dx, inv_dx, dt = a.dx, 1/a.dx, a.dt
p_vol = (dx*.5)**3; p_mass = 1000.*p_vol
grav = -9.8 if a.gravity else 0.
V0 = np.zeros((N, 3))
if a.kick:
    V0[:] = [a.kscale*v0[0], 0., a.kscale*v0[2]]
V0[clamp] = 0.
lo = np.floor(np.array([-.55, -.06, -.45])/dx).astype(int)-3
hi = np.ceil(np.array([.45, .9, .45])/dx).astype(int)+3
shape = tuple(int(s) for s in hi-lo)
x = ti.Vector.field(3, ti.f32, N); v = ti.Vector.field(3, ti.f32, N)
C = ti.Matrix.field(3, 3, ti.f32, N); F = ti.Matrix.field(3, 3, ti.f32, N); cf = ti.field(ti.i32, N)
off_ = tuple(int(s) for s in lo)
gm = ti.field(ti.f32, shape, offset=off_); gmc = ti.field(ti.f32, shape, offset=off_); gv = ti.Vector.field(3, ti.f32, shape, offset=off_)
x.from_numpy(X0.astype(np.float32)); v.from_numpy(V0.astype(np.float32)); cf.from_numpy(clamp.astype(np.int32))
C.from_numpy(np.zeros((N, 3, 3), np.float32)); F.from_numpy(np.tile(np.eye(3), (N, 1, 1)).astype(np.float32))


@ti.kernel
def substep():
    for I in ti.grouped(gm):
        gm[I] = 0.; gmc[I] = 0.; gv[I] = ti.Vector([0., 0., 0.])
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
            if cf[p]:
                gmc[base+off] += weight*p_mass
    for I in ti.grouped(gm):
        if gm[I] > 1e-8*dx**3:
            vo = gv[I]/gm[I]+dt*ti.Vector([0., grav, 0.])
            if gmc[I] > .5*gm[I]:
                vo = ti.Vector([0., 0., 0.])
            gv[I] = vo
        else:
            gv[I] = ti.Vector([0., 0., 0.])
    for p in x:
        if cf[p]:
            v[p] = ti.Vector([0., 0., 0.]); C[p] = ti.Matrix.zero(ti.f32, 3, 3)
        else:
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
W = groups.astype(np.float64); W /= W.sum(1, keepdims=True)
print('N %d (split %d), clamped %d, dx %.5f, dt %.3e, %d substeps per sample, gravity %d kick %d' % (N, a.split, clamp.sum(), dx, dt, nsub, a.gravity, a.kick), flush=True)
rec = np.zeros((ns+1, len(W), 3)); rec[0] = W@X0
t0 = time.time()
for s in range(1, ns+1):
    for _ in range(nsub):
        substep()
    xs = x.to_numpy().astype(np.float64)
    if not np.isfinite(xs).all():
        raise SystemExit('NaN at sample %d' % s)
    rec[s] = W@xs
    if s % round(a.rate/8) == 0:
        print('t %.3f s (%.1f s): com %s' % (s/a.rate, time.time()-t0, np.round(rec[s, 0], 4)), flush=True)
np.savez(a.out, rec=rec, rate=a.rate, dx=dx, dt=dt, split=a.split, gravity=a.gravity, kick=a.kick, hc=float(Sz['hc']))
print('saved', a.out)
