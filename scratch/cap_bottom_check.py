"""step 0 of the cap contact: is the jack's bottom made of exposed caps that face down, and does it sit where the GT
body's bottom is? At rest (scene frame, y up; the jack lands with little rotation, so its rest bottom is what touches).
  exposed cap points: each sphere's surface points outside every other sphere (the union of the spheres is the body,
  its surface the caps), Fibonacci-sampled (area weight 4 pi r^2 / N), near the bottom only
  per slab of thickness h above the lowest material point: the body's horizontal section there (the union of spheres on a
  2 mm grid) against the GT particles' (2D occupancy, each particle a disc of the mean spacing); the exposed caps' area
  projected on the ground (sum w max(-u_y, 0)) inside the slab against the section; the caps' downward share
  and the lowest points: GT particles, the foam's union, the contact balls' bottoms"""
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
OFF = np.array([.5, -.05, .5])
c_ = np.load(ROOT/'inverse_data/pacnerf_e0/foam/dem_exact_single/cells.npz'); p, r = c_['points'].astype(float), c_['radii'].astype(float)
site = np.load(ROOT/'inverse_data/pacnerf_e0/foam/patch_site.npz')['patch_site']
P = np.stack([PlyData.read('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data/elastic/0/0.ply')['vertex'][c] for c in 'xyz'], 1).astype(float)+OFF
ylo_f = (p[:, 1]-r).min(); ylo_g = P[:, 1].min(); ylo_b = (p[site, 1]-r[site]).min()
print('lowest point (mm, scene frame): GT particles %.1f, foam union %.1f, contact balls %.1f' % (1e3*ylo_g, 1e3*ylo_f, 1e3*ylo_b))
N = 4000; k = np.arange(N)+.5; ph = np.arccos(1-2*k/N); th = np.pi*(1+5**.5)*k
U = np.c_[np.cos(th)*np.sin(ph), np.cos(ph), np.sin(th)*np.sin(ph)]          # (y the second axis)
tree = cKDTree(p); rmax = r.max()
cand = np.nonzero(p[:, 1]-r < ylo_f+.03)[0]
pts, wts, nrm, own = [], [], [], []
for i in cand:
    u = U[U[:, 1] < .3]; x = p[i]+r[i]*u
    q = x[:, 1] < ylo_f+.03
    u, x = u[q], x[q]
    if not len(x):
        continue
    nb = [j for j in tree.query_ball_point(p[i], r[i]+rmax) if j != i]
    out = np.ones(len(x), bool)
    if nb:
        d2 = ((x[:, None, :]-p[nb][None])**2).sum(-1)-r[nb][None]**2
        out = (d2 >= 0).all(1)
    pts.append(x[out]); nrm.append(u[out]); wts.append(np.full(out.sum(), 4*np.pi*r[i]**2/N)); own.append(np.full(out.sum(), i))
pts, nrm, wts, own = map(np.concatenate, (pts, nrm, wts, own))
print('exposed cap samples near the bottom: %d from %d cells (%d of them contact-ball cells)' % (len(pts), len(np.unique(own)), np.isin(np.unique(own), site).sum()))
gs = 2e-3
xs = np.arange(p[:, 0].min()-.05, p[:, 0].max()+.05, gs); zs = np.arange(p[:, 2].min()-.05, p[:, 2].max()+.05, gs)
X2, Z2 = np.meshgrid(xs, zs, indexing='ij')
gt2 = cKDTree(P[:, [0, 2]])
sp = (np.ptp(P, 0).prod()/len(P))**(1/3)
for h in (.005, .01, .02):
    y = ylo_f+h
    near = np.nonzero(np.abs(p[:, 1]-y) < r)[0]
    rr = np.sqrt(np.maximum(r[near]**2-(p[near, 1]-y)**2, 0))
    t2 = cKDTree(p[near][:, [0, 2]])
    G = np.c_[X2.ravel(), Z2.ravel()]
    inside = np.zeros(len(G), bool)
    for j, lst in enumerate(t2.query_ball_point(G, rr.max())):
        if lst:
            inside[j] = (((G[j]-p[near][lst][:, [0, 2]])**2).sum(1) < rr[lst]**2).any()
    a_f = inside.sum()*gs*gs
    yg = ylo_g+h; sl = P[:, 1] < yg
    occ = np.zeros(len(G), bool)
    if sl.any():
        d, _ = cKDTree(P[sl][:, [0, 2]]).query(G); occ = d < .5*sp
    a_g = occ.sum()*gs*gs
    q = pts[:, 1] < y
    proj = (wts[q]*np.maximum(-nrm[q, 1], 0)).sum()
    down = (wts[q]*(nrm[q, 1] < -.7)).sum()/max(wts[q].sum(), 1e-30)
    print('slab %2.0f mm: section area foam %.2f cm^2, GT %.2f cm^2 | exposed caps in the slab: projected %.2f cm^2 (%.2f of the foam section), %.0f %% of their area facing down (n_y < -0.7)'
          % (1e3*h, 1e4*a_f, 1e4*a_g, 1e4*proj, proj/max(a_f, 1e-30), 100*down))
