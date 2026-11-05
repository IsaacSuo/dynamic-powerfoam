"""the ground's action on PAC-NeRF's MPM (mpm_pacnerf_repro.py's substep, the GT run: sticky ground y = 0, gravity), per
1/1200 s: at every substep each sticky node's momentum change when its velocity is zeroed, imp_I = -m_I v_I (v_I after
gravity), is the ground's impulse on the body there; summed over the window: the force, the torque about the particles'
centre of mass (APIC carries the nodes' angular momentum exactly: the torque is sum (x_I - c) x imp_I), the centre of
pressure (by the normal impulse, all nodes and pushing nodes only), the number of nodes; per frame 5..9 the impulse per
floor node (x_I, z_I) summed over the frame. Per unit mass (divided by the particles' total mass)."""
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
ap.add_argument('--frames', type=int, default=12)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_fraction=.5)
gt = json.load(open(D+'/gt/elastic.json'))['0']
E, nu, v0 = gt['E'], gt['nu'], np.array(gt['v'])
mu_, lam_ = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
X0 = np.stack([PlyData.read(D+'/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(np.float64)
if a.split == 2:
    o = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing='ij')).reshape(3, -1).T*5.24e-3/4
    X0 = (X0[:, None, :]+o[None]).reshape(-1, 3)
N = len(X0)
dx, inv_dx, dt = a.dx, 1/a.dx, a.dt
p_vol = (dx*.5)**3; p_mass = 1000.*p_vol; Mtot = N*p_mass
lo = np.floor(np.array([-.55, -.06, -.45])/dx).astype(int)-3
hi = np.ceil(np.array([.45, .9, .45])/dx).astype(int)+3
shape = tuple(int(s) for s in hi-lo); off_ = tuple(int(s) for s in lo)
x = ti.Vector.field(3, ti.f32, N); v = ti.Vector.field(3, ti.f32, N)
C = ti.Matrix.field(3, 3, ti.f32, N); F = ti.Matrix.field(3, 3, ti.f32, N)
gm = ti.field(ti.f32, shape, offset=off_); gv = ti.Vector.field(3, ti.f32, shape, offset=off_)
com = ti.Vector.field(3, ti.f64, ())
acc = ti.field(ti.f64, 16)                                       # F(3), tau(3), sum fy, fy x, fy z, sum fy+, fy+ x, fy+ z, nodes
mp = ti.Vector.field(3, ti.f64, (shape[0], shape[2]), offset=(off_[0], off_[2]))
rec_map = ti.field(ti.i32, ())
x.from_numpy(X0.astype(np.float32)); v.from_numpy(np.tile(v0, (N, 1)).astype(np.float32))
C.from_numpy(np.zeros((N, 3, 3), np.float32)); F.from_numpy(np.tile(np.eye(3), (N, 1, 1)).astype(np.float32))


@ti.kernel
def centre():
    com[None] = ti.Vector([0., 0., 0.], dt=ti.f64)
    for p in x:
        com[None] += x[p].cast(ti.f64)/N


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
                imp = (-gm[I]*vo).cast(ti.f64)
                xI = I.cast(ti.f64)*dx
                r = xI-com[None]
                tq = r.cross(imp)
                for k in ti.static(range(3)):
                    acc[k] += imp[k]; acc[3+k] += tq[k]
                acc[6] += imp[1]; acc[7] += imp[1]*xI[0]; acc[8] += imp[1]*xI[2]
                if imp[1] > 0:
                    acc[9] += imp[1]; acc[10] += imp[1]*xI[0]; acc[11] += imp[1]*xI[2]
                acc[12] += 1.
                if rec_map[None]:
                    mp[I[0], I[2]] += imp
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


nsub = round((1/24)/dt); per = round((1/1200)/dt)
print('N %d (split %d), dx %.5f, dt %.3e: %d substeps per frame, %d per record' % (N, a.split, dx, dt, nsub, per), flush=True)
rows, maps = [], {}
t0 = time.time()
for f_ in range(a.frames):
    rec_map[None] = 1 if 5 <= f_ <= 9 else 0
    mp.fill(0)
    for s in range(nsub):
        if s % per == 0:
            acc.fill(0); cs = []
        centre(); cs.append(com.to_numpy())
        substep()
        if s % per == per-1:
            A = acc.to_numpy(); T = per*dt
            rows.append(np.r_[(f_*nsub+s+1)*dt, np.mean(cs, 0), A[:6]/T/Mtot, A[6:12]/T/Mtot, A[12]/per])
    if rec_map[None]:
        M_ = mp.to_numpy(); ix, iz = np.nonzero(np.abs(M_).sum(-1) > 0)
        maps['map%d' % f_] = np.c_[(ix+off_[0])*dx, (iz+off_[2])*dx, M_[ix, iz]/Mtot]
    print('frame %d -> %d (%.0f s)' % (f_, f_+1, time.time()-t0), flush=True)
np.savez(a.out, rows=np.array(rows), dx=dx, **maps)
print('saved', a.out)
