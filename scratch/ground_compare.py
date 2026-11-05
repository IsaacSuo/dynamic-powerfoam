"""where and how the ground acts, MPM (mpm_ground_force.py) against our DEM (dem_ground_force.py), per unit mass, in
windows of 1/120 s through the impact: e_f the forward direction (v0's horizontal part), a_t = up x e_f the forward-tipping
axis. Columns: the normal force F_y, the tangential F_f (< 0 brakes), the tipping torque about the centre of mass
tau_t = tau . a_t, split for points at the ground as tau_t = -H F_f - (cop_f - c_f) F_y (friction part: H the centre's
height above the points; normal part: the centre of pressure's forward offset), cop_f - c_f (mm). For ours: the torque the
model applies (forces at the sites), and the torque the same forces would give at the cells' deformed lowest points
('mat'); then the angular impulse from frame 4 (int tau_t dt) for each."""
import sys
import numpy as np
ef = np.array([-0.4078950600549248, 0., -0.06629882732047176]); ef /= np.linalg.norm(ef)
up = np.array([0., 1., 0.]); at = np.cross(up, ef)
OFF = np.array([.5, -.05, .5])
mfiles, dfile = sys.argv[1:-1], sys.argv[-1]
W = 10                                                                             # (1/1200 s samples per window)


def bins(t, cols):
    k = np.floor(t*120+1e-9).astype(int)
    out = {}
    for kk in np.unique(k):
        q = k == kk
        out[kk] = [c[q].mean(0) for c in cols]
    return out


runs = []
for p in mfiles:
    z = np.load(p); r = z['rows']
    t, c, F, tau = r[:, 0], r[:, 1:4], r[:, 4:7], r[:, 7:10]
    sy, syx, syz = r[:, 10], r[:, 11], r[:, 12]
    cop = np.c_[syx/np.where(sy != 0, sy, np.nan), np.zeros(len(t)), syz/np.where(sy != 0, sy, np.nan)]
    copf = np.nan_to_num(((cop-c)*np.array([1, 0, 1]))@ef)
    H = c[:, 1]
    runs.append(('MPM dx %.4f' % float(z['dx']), t, F@ef, F[:, 1], tau@at, -(H)*(F@ef), -copf*F[:, 1], 1e3*copf, None))
z = np.load(dfile)
t, c, F = z['t'], z['com'], z['F']
G0 = -.05
tau_s, tau_m = (z['tau_model'] if 'tau_model' in z else z['tau_site'])@at, z['tau_mat']@at
cops = (((z['cop_model'] if 'cop_model' in z else z['cop_site'])-c)*np.array([1, 0, 1]))@ef; copm = ((z['cop_mat']-c)*np.array([1, 0, 1]))@ef
Hm = c[:, 1]-G0
runs.append(('ours (sites)', t, F@ef, F[:, 1], tau_s, None, None, 1e3*np.where(z['Fn'] > 0, cops, 0), tau_m))
M = float(z['M'])
print('per unit mass; windows of 1/120 s (frame = 5 windows); F in m/s^2, tau in m^2/s^2, cop offset in mm (+ ahead of the centre)')
for name, t_, Ff, Fy, tt_, fr_, nr_, cp, tm in runs:
    sc = 1/M if name.startswith('ours') else 1.
    print('== %s' % name)
    print(' frame | F_y | F_f | tau_t%s | cop_f - c_f (mm) | int tau_t from frame 4%s' % (' (friction part, normal part)' if fr_ is not None else ', tau_t at the deformed lowest points', '' if tm is None else ' (sites, lowest points)'))
    k = np.floor(t_*120+1e-9).astype(int)
    cum = cumm = 0.
    for kk in range(int(4*5), int(10.4*5)):
        q = k == kk
        if not q.any():
            continue
        dtw = q.sum()*(t_[1]-t_[0])
        a_ = tt_[q].mean()*sc; cum += a_*dtw
        line = ' %5.1f | %+7.2f | %+6.2f | %+.4f' % (kk/5, Fy[q].mean()*sc, Ff[q].mean()*sc, a_)
        if fr_ is not None:
            line += ' (%+.4f, %+.4f)' % (fr_[q].mean(), nr_[q].mean())
        else:
            b_ = tm[q].mean()*sc; cumm += b_*dtw; line += ', %+.4f' % b_
        line += ' | %+6.1f | %+.5f' % (cp[q].mean(), cum)
        if tm is not None:
            line += ', %+.5f' % cumm
        print(line)
