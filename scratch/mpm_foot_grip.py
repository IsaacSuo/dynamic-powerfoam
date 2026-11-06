"""the ground's grip per foot in PAC-NeRF's MPM (mpm_foot_loops.py's run, sticky ground y = 0): every 1/1200 s per floor
node (x_I, z_I) its impulse over the window (-m_I v_I, as mpm_ground_force.py) and whether material reaches it (its mass
> 0: material within the B-spline's reach of the wall). Per foot (nodes ahead of / behind the centre of mass along e_f):
the forward force T, the normal force N, the area reached (nodes with mass) and pressed (normal impulse > 0) x dx^2, and
the change of T since the last window split by nodes: held (reached in both windows: their own change), new (reached now
only: their force), lost (reached before only: their force before). Per unit mass. args: --dx --split --frames --out"""
import argparse
import json
import time
import numpy as np
import taichi as ti
from plyfile import PlyData
D = '/home/fangsuo/py/dynamic-powerfoam/data/pacnerf'
H = .05
EF = np.array([-0.4078950600549248, 0., -0.06629882732047176]); EF /= np.linalg.norm(EF)
ap = argparse.ArgumentParser()
ap.add_argument('--dx', type=float, required=True)
ap.add_argument('--dt', type=float, default=1/4800)
ap.add_argument('--split', type=int, default=1)
ap.add_argument('--frames', type=int, default=11)
ap.add_argument('--out', required=True)
a = ap.parse_args()
ti.init(arch=ti.cuda, default_fp=ti.f32, device_memory_fraction=.5)
gt = json.load(open(D+'/gt/elastic.json'))['0']
E, nu, v0 = gt['E'], gt['nu'], np.array(gt['v'])
mu_, lam_ = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
X0 = np.stack([PlyData.read(D+'/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(np.float64)
c0 = X0.mean(0)
lab0 = np.where(X0[:, 1] < X0[:, 1].min()+H, np.where((X0-c0)@EF > 0, 1, 2), 0)
print('foot material: front %d, back %d of %d particles (lowest %.4f m)' % ((lab0 == 1).sum(), (lab0 == 2).sum(), len(X0), X0[:, 1].min()), flush=True)
if a.split == 2:
    o = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], indexing='ij')).reshape(3, -1).T*5.24e-3/4
    X0 = (X0[:, None, :]+o[None]).reshape(-1, 3)
    lab0 = np.repeat(lab0, 8)
N = len(X0)
dx, inv_dx, dt = a.dx, 1/a.dx, a.dt
p_vol = (dx*.5)**3; p_mass = 1000.*p_vol; Mtot = N*p_mass
lo = np.floor(np.array([-.55, -.06, -.45])/dx).astype(int)-3
hi = np.ceil(np.array([.45, .9, .45])/dx).astype(int)+3
shape = tuple(int(s) for s in hi-lo); off_ = tuple(int(s) for s in lo)
x = ti.Vector.field(3, ti.f32, N); v = ti.Vector.field(3, ti.f32, N)
C = ti.Matrix.field(3, 3, ti.f32, N); F = ti.Matrix.field(3, 3, ti.f32, N)
lab = ti.field(ti.i32, N)
gm = ti.field(ti.f32, shape, offset=off_); gv = ti.Vector.field(3, ti.f32, shape, offset=off_)
com = ti.Vector.field(3, ti.f64, ())
ef = ti.Vector(list(EF))
acc = ti.field(ti.f64, 4)                                        # front fy, back fy, front fx (along e_f), back fx
hs = ti.field(ti.f64, 4)                                         # front sum y, back sum y, counts
mp = ti.Vector.field(3, ti.f64, (shape[0], shape[2]), offset=(off_[0], off_[2]))   # the floor nodes' impulse in the window
mm = ti.field(ti.f64, (shape[0], shape[2]), offset=(off_[0], off_[2]))             # and their mass (summed over its substeps)
x.from_numpy(X0.astype(np.float32)); v.from_numpy(np.tile(v0, (N, 1)).astype(np.float32))
C.from_numpy(np.zeros((N, 3, 3), np.float32)); F.from_numpy(np.tile(np.eye(3), (N, 1, 1)).astype(np.float32))
lab.from_numpy(lab0.astype(np.int32))


@ti.kernel
def centre():
    com[None] = ti.Vector([0., 0., 0.], dt=ti.f64)
    for p in x:
        com[None] += x[p].cast(ti.f64)/N


@ti.kernel
def heights():
    for k in range(4):
        hs[k] = 0.
    for p in x:
        if lab[p] > 0:
            hs[lab[p]-1] += ti.cast(x[p][1], ti.f64)
            hs[lab[p]+1] += 1.


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
                fw = r[0]*ef[0]+r[2]*ef[2]
                it = imp[0]*ef[0]+imp[2]*ef[2]
                mp[I[0], I[2]] += imp; mm[I[0], I[2]] += ti.cast(gm[I], ti.f64)
                if fw > 0:
                    acc[0] += imp[1]; acc[2] += it
                else:
                    acc[1] += imp[1]; acc[3] += it
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
rows, grip = [], []
GX, GZ = np.meshgrid((np.arange(shape[0])+off_[0])*dx, (np.arange(shape[2])+off_[2])*dx, indexing='ij')
prev = dict(act=np.zeros(GX.shape, bool), fwd=np.zeros(GX.shape, bool), tf=np.zeros(GX.shape))
t0 = time.time()
for f_ in range(a.frames):
    for s in range(nsub):
        if s % per == 0:
            acc.fill(0); mp.fill(0); mm.fill(0)
        centre()
        substep()
        if s % per == per-1:
            A = acc.to_numpy(); T = per*dt
            centre(); heights(); h = hs.to_numpy()
            rows.append(np.r_[(f_*nsub+s+1)*dt, com.to_numpy(), A/T/Mtot, h[0]/h[2], h[1]/h[3]])
            P = mp.to_numpy()/T/Mtot; act = mm.to_numpy() > 0; c_ = com.to_numpy()
            fwd = (GX-c_[0])*EF[0]+(GZ-c_[2])*EF[2] > 0
            tf = P[..., 0]*EF[0]+P[..., 2]*EF[2]; ny = P[..., 1]
            row = [(f_*nsub+s+1)*dt]
            for ft_ in (fwd, ~fwd):
                m_ = act & ft_
                held = m_ & prev['act']; new = m_ & ~prev['act']; lost = prev['act'] & ~act & (prev['fwd'] if ft_ is fwd else ~prev['fwd'])
                row += [tf[m_].sum(), ny[m_].sum(), m_.sum()*dx*dx, (m_ & (ny > 0)).sum()*dx*dx,
                        (tf[held]-prev['tf'][held]).sum(), tf[new].sum(), -prev['tf'][lost].sum()]
            grip.append(row); prev = dict(act=act, fwd=fwd, tf=tf)
    print('frame %d -> %d (%.0f s)' % (f_, f_+1, time.time()-t0), flush=True)
np.savez(a.out, rows=np.array(rows), cols='t, com (3), Nf, Nb, Tf, Tb (per unit mass), hf, hb (m over the ground)', dx=dx, grip=np.array(grip),
         gcols='t; per foot (front, back): T, N, area reached, area pressed (m^2), dT held, T new, -T lost (per unit mass)')
print('saved', a.out)
