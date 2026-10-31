"""A page to look at a merged landing against PAC-NeRF's MPM (2026-11-01): per frame, an oblique view of each (ours the cells' spheres, the MPM's its particles, one depth-buffer shading for both) and a vertical slice through each one's centre of mass: ours the exact power
cells of that frame, the MPM's the nearest particle of a thin slab, both coloured by horizontal speed over the frame, the MPM's outline over ours. Spreads over each one's own rest beside them. One html with its images inside.
  python free_surface/merged_view.py --foam output/fourier8k_clean --frames output/spring/frames_k1.npz --out output/spring/view/plastic_landing.html"""
import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--frames', type=Path, required=True)
a_.add_argument('--out', type=Path, required=True)
a_.add_argument('--template', type=Path, default=Path(__file__).with_name('merged_view.html'))
a_.add_argument('--res3d', type=float, default=.002)
a_.add_argument('--resslice', type=float, default=.0015)
args = a_.parse_args()
Z = zipfile.ZipFile('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf/simulation_data.zip')
OFF = np.array([.5, -.05, .5]); Y0 = -.05; DT = 1/24; NF = 16


def mpm(k):
    v = PlyData.read(io.BytesIO(Z.read('simulation_data/plasticine/0/%d.ply' % k)))['vertex']
    return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)+OFF


g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
V0, X0, r0 = g['masses'], g['points'], cl['radii']
f = np.load(args.frames)


def ours(k):
    if k == 0:
        return X0, r0
    i = int(np.argmin(np.abs(f['t']-k/24)))
    return f['x'][i], f['r'][i]


G = [mpm(k) for k in range(NF)]
O = [ours(k) for k in range(NF)]


def png(rgba):
    b = io.BytesIO(); Image.fromarray((np.clip(rgba, 0, 1)*255).astype(np.uint8), 'RGBA').save(b, 'PNG', optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()


# the oblique view
az, el = np.radians(35.), np.radians(16.)
Rt = np.array([np.cos(az), 0., np.sin(az)])
Fw = np.array([-np.sin(az)*np.cos(el), -np.sin(el), np.cos(az)*np.cos(el)])
Up = np.cross(Fw, Rt); Up = Up*np.sign(Up[1])
C = np.array([.42, .19, .48]); W3, H3 = 1.16, .62; res = args.res3d
nx, ny = int(W3/res), int(H3/res)
CLAY = np.array([.80, .62, .45])
light = np.array([-.45, .75, -.5]); light = light/np.linalg.norm(light)


def ground_layer():
    """a faint 10 cm grid on the ground, and its depth"""
    u = (np.arange(nx)+.5)*res-W3/2; v = H3/2-(np.arange(ny)+.5)*res
    U, Vv = np.meshgrid(u, v)
    O_ = C[None, None]+U[..., None]*Rt+Vv[..., None]*Up
    s = (Y0-O_[..., 1])/Fw[1]
    P = O_+s[..., None]*Fw
    d = np.maximum(np.abs(((P[..., 0]+.05) % .1)-.05), np.abs(((P[..., 2]+.05) % .1)-.05))
    line = np.minimum(np.abs(((P[..., 0]+.05) % .1)-.05), np.abs(((P[..., 2]+.05) % .1)-.05)) < .0011
    ok = (s > -.2) & (np.abs(P[..., 0]-C[0]) < .62) & (np.abs(P[..., 2]-C[2]) < .62)
    return s, line & ok, ok


Gs, Gl, Gok = ground_layer()


def shade(zb):
    has = np.isfinite(zb)
    z = np.where(has, zb, 0.)
    dx = np.zeros_like(z); dy = np.zeros_like(z)
    dx[:, 1:-1] = (z[:, 2:]-z[:, :-2])/(2*res); dy[1:-1, :] = (z[2:, :]-z[:-2, :])/(2*res)
    bad = ~(has & np.roll(has, 1, 1) & np.roll(has, -1, 1) & np.roll(has, 1, 0) & np.roll(has, -1, 0))
    dx[bad] = 0.; dy[bad] = 0.
    dx, dy = np.clip(dx, -3, 3), np.clip(dy, -3, 3)
    N = dx[..., None]*Rt-dy[..., None]*Up-Fw[None, None]*1.                         # (the rows run down the screen: dz/d(up) = -dy)
    N = N/np.linalg.norm(N, axis=-1, keepdims=True)
    lam = np.clip((N*light).sum(-1), 0, 1)
    sh = .42+.62*lam
    im = np.zeros((ny, nx, 4))
    im[Gl] = np.array([.5, .5, .5, .45])
    im[has] = np.concatenate([np.clip(sh[has, None]*CLAY, 0, 1), np.ones((has.sum(), 1))], 1)
    # the ground's line under the material
    im[Gl & ~has] = np.array([.5, .5, .5, .45])
    return im


def view_spheres(x, r):
    zb = np.full((ny, nx), np.inf)
    u = ((x-C)@Rt+W3/2)/res-.5; v = (H3/2-(x-C)@Up)/res-.5; dpt = (x-C)@Fw
    for i in range(len(x)):
        p = r[i]/res; a0, a1 = int(max(u[i]-p, 0)), int(min(u[i]+p+1, nx-1)); b0, b1 = int(max(v[i]-p, 0)), int(min(v[i]+p+1, ny-1))
        if a1 <= a0 or b1 <= b0:
            continue
        ua, vb = np.arange(a0, a1+1), np.arange(b0, b1+1)
        d2 = ((ua-u[i])[None, :]**2+(vb-v[i])[:, None]**2)*res*res
        q = d2 < r[i]*r[i]
        zz = dpt[i]-np.sqrt(np.maximum(r[i]*r[i]-d2, 0.))
        # clipped by the ground
        Y = x[i, 1]+((ua-u[i])[None, :]*Rt[1]-(vb-v[i])[:, None]*Up[1])*res+(zz-dpt[i])*Fw[1]
        q = q & (Y >= Y0)
        sub = zb[b0:b1+1, a0:a1+1]
        sub[q] = np.minimum(sub[q], zz[q])
    return zb


def view_points(x, r_=.0065):
    zb = np.full((ny, nx), np.inf)
    u = ((x-C)@Rt+W3/2)/res-.5; v = (H3/2-(x-C)@Up)/res-.5; dpt = (x-C)@Fw
    rp = int(np.ceil(r_/res))
    oa, ob = np.meshgrid(np.arange(-rp, rp+1), np.arange(-rp, rp+1))
    q = (oa*oa+ob*ob)*res*res < r_*r_
    oa, ob = oa[q], ob[q]; dz = -np.sqrt(np.maximum(r_*r_-(oa*oa+ob*ob)*res*res, 0.))
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    A = (ui[:, None]+oa[None]); B = (vi[:, None]+ob[None]); D = dpt[:, None]+dz[None]
    ok = (A >= 0) & (A < nx) & (B >= 0) & (B < ny)
    np.minimum.at(zb, (B[ok], A[ok]), D[ok])
    return zb


# the slice
CX, Wsl = .43, 1.04; Y1 = .475; rs = args.resslice
sx, sy = int(Wsl/rs), int((Y1-Y0+.004)/rs)
px = CX-Wsl/2+(np.arange(sx)+.5)*rs; py = Y1-(np.arange(sy)+.5)*rs
VMAX = 2.
stops = np.array([[.05, .03, .25], [.20, .15, .52], [.09, .42, .55], [.24, .66, .42], [.80, .80, .25], [.99, .91, .60]])


def cmap(v):
    t = np.clip(v/VMAX, 0, 1)*(len(stops)-1); i = np.minimum(t.astype(int), len(stops)-2); w = (t-i)[..., None]
    return stops[i]*(1-w)+stops[i+1]*w


def uh(a, b):
    return np.linalg.norm((b-a)[:, [0, 2]], axis=1)/DT


def speed(seq, k):
    return uh(seq[k], seq[k+1]) if k < NF-1 else uh(seq[k-1], seq[k])


def mpm_slice(k):
    x = G[k]; zc = x[:, 2].mean()
    q = np.abs(x[:, 2]-zc) < .006
    pts = x[q][:, :2]; sp = speed(G, k)[q]
    Pxx, Pyy = np.meshgrid(px, py)
    d, i = cKDTree(pts).query(np.stack([Pxx.ravel(), Pyy.ravel()], 1), distance_upper_bound=.0075, workers=-1)
    has = np.isfinite(d).reshape(sy, sx) & (Pyy >= Y0)
    im = np.zeros((sy, sx, 4))
    ii = i.reshape(sy, sx)
    im[has] = np.concatenate([cmap(sp[ii[has]]), np.ones((has.sum(), 1))], 1)
    return im, has


def our_slice(k, mask):
    x, r = O[k]; zc = (V0*x[:, 2]).sum()/V0.sum()
    ou = O[k+1][0] if k < NF-1 else None
    sp = uh(x, O[k+1][0]) if k < NF-1 else uh(O[k-1][0], x)
    c = np.where(np.abs(x[:, 2]-zc) < r)[0]
    Pxx, Pyy = np.meshgrid(px, py)
    best = np.full((sy, sx), np.inf); who = np.full((sy, sx), -1)
    for i in c:
        a0, a1 = np.searchsorted(px, [x[i, 0]-r[i], x[i, 0]+r[i]]); b0, b1 = np.searchsorted(-py, [-(x[i, 1]+r[i]), -(x[i, 1]-r[i])])
        if a1 <= a0 or b1 <= b0:
            continue
        pw = (Pxx[b0:b1, a0:a1]-x[i, 0])**2+(Pyy[b0:b1, a0:a1]-x[i, 1])**2+(zc-x[i, 2])**2-r[i]*r[i]
        sub, sw = best[b0:b1, a0:a1], who[b0:b1, a0:a1]
        q = (pw < 0) & (pw < sub)
        sub[q] = pw[q]; sw[q] = i
    who[Pyy < Y0] = -1
    has = who >= 0
    im = np.zeros((sy, sx, 4))
    im[has] = np.concatenate([cmap(sp[who[has]]), np.ones((has.sum(), 1))], 1)
    e = np.zeros_like(has)
    e[:, :-1] |= who[:, :-1] != who[:, 1:]; e[:-1, :] |= who[:-1, :] != who[1:, :]
    e &= has
    im[e] = np.array([.07, .07, .09, 1.])
    # the MPM's outline
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
    im[edge] = np.array([1., .36, .10, 1.])
    return im


def spreads(x, w=None):
    w = np.ones(len(x)) if w is None else w
    w = w/w.sum(); m = (w[:, None]*x).sum(0)
    return np.sqrt((w[:, None]*(x-m)**2).sum(0)), m


s0g = spreads(G[0])[0]; s0o = spreads(X0, V0)[0]
data = dict(frames=[])
for k in range(NF):
    a = view_points(G[k]); b = view_spheres(*O[k])
    ms, mask = mpm_slice(k)
    os_ = our_slice(k, mask)
    sg, mg = spreads(G[k]); so, mo = spreads(O[k][0], V0)
    data['frames'].append(dict(t=k/24, m3=png(shade(a)), o3=png(shade(b)), ms=png(ms), os=png(os_), ty=[sg[1]/s0g[1], so[1]/s0o[1]], tx=[sg[0]/s0g[0], so[0]/s0o[0]], tz=[sg[2]/s0g[2], so[2]/s0o[2]], cy=[mg[1]-Y0, mo[1]-Y0]))
    print('frame', k, 'done', flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
html = args.template.read_text().replace('/*DATA*/null', json.dumps(data))
args.out.write_text(html)
print('wrote', args.out, '%.1f MB' % (len(html)/1e6), flush=True)
