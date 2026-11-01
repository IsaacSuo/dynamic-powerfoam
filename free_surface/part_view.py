"""A page to look at a part's landing, run beside run, as merged_view.py: per frame an oblique view (each cell's sphere, radius r0 exp(L / 3)) and two vertical slices (the power cells through the centre of mass; through the hind leg), coloured by horizontal speed, the cells whose rest neighbours stand inside out in red, nearly so (det under .2) in orange. One html with its images inside.
  python free_surface/part_view.py --foam output/part/lion_hind --dir output/part/hind --group 弹性 hRing=RingUL hCut=全程切分 hFade2=淡出 --group 屈服 hCutP=全程切分 hFadeP=淡出 --out output/part/view/hind_folds.html"""
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--dir', type=Path, required=True)
a_.add_argument('--group', nargs='+', action='append', required=True, help='a name, then tag=label ...')
a_.add_argument('--out', type=Path, required=True)
a_.add_argument('--every', type=int, default=2)
a_.add_argument('--res', type=float, default=.002)
a_.add_argument('--ground', type=float, default=-.2)
a_.add_argument('--vmax', type=float, default=1.)
a_.add_argument('--t0', type=float, default=.22, help='the frames from here on: the fall before it is a rigid body')
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz'); cl = np.load(args.foam/'dem_exact_single'/'cells.npz')
V0, X0, r0, P = g['masses'], g['points'], cl['radii'], g['pairs']
n = len(V0)
nb = [[] for _ in range(n)]
for i, j in P:
    nb[i].append(j); nb[j].append(i)
nb = [np.array(q) for q in nb]
LEG = (np.abs(X0[:, 1]-.39) < .04) & (X0[:, 0] > .72)
runs = {}
for grp in args.group:
    for s in grp[1:]:
        tag = s.split('=')[0]
        if tag not in runs:
            z = np.load(args.dir/('frames_%s.npz' % tag))
            runs[tag] = dict(t=z['t'], x=z['x'], L=z['L'])
T = max((r['t'] for r in runs.values()), key=len)[::args.every]
T = T[T >= args.t0]
print('%d frames, up to t %.3f' % (len(T), T[-1]), flush=True)


def at(r, t):
    return int(np.argmin(np.abs(r['t']-t)))


def dets(x):
    d = np.full(n, 1.)
    for c in range(n):
        if len(nb[c]) >= 6:
            d[c] = np.linalg.det(np.linalg.lstsq(X0[nb[c]]-X0[c], x[nb[c]]-x[c], rcond=None)[0].T)
    return d


def png(rgba):
    b = io.BytesIO(); Image.fromarray((np.clip(rgba, 0, 1)*255).astype(np.uint8), 'RGBA').save(b, 'PNG', optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()


az, el = np.radians(35.), np.radians(16.)
Rt = np.array([np.cos(az), np.sin(az), 0.])
Fw = np.array([-np.sin(az)*np.cos(el), np.cos(az)*np.cos(el), -np.sin(el)])
Up = np.cross(Fw, Rt); Up = Up*np.sign(Up[2])
light = np.array([-.45, -.5, .75]); light = light/np.linalg.norm(light)
CLAY, RED, ORANGE = np.array([.80, .62, .45]), np.array([.85, .12, .12]), np.array([.96, .55, .10])
res = args.res
allx = np.concatenate([r['x'][at(r, t)] for r in runs.values() for t in T[::4]])
pad = float(r0.max())*.6
u_, v_ = allx@Rt, allx@Up
u0, u1, v0, v1 = u_.min()-pad, u_.max()+pad, v_.min()-pad, v_.max()+pad
nx, ny = int((u1-u0)/res), int((v1-v0)/res)
xmin, xmax, zmax = allx[:, 0].min()-pad, allx[:, 0].max()+pad, allx[:, 2].max()+pad
sx, sy = int((xmax-xmin)/res), int((zmax-args.ground)/res)+2
px, pz = xmin+(np.arange(sx)+.5)*res, zmax-(np.arange(sy)+.5)*res
stops = np.array([[.05, .03, .25], [.20, .15, .52], [.09, .42, .55], [.24, .66, .42], [.80, .80, .25], [.99, .91, .60]])


def cmap(v):
    t = np.clip(v/args.vmax, 0, 1)*(len(stops)-1); i = np.minimum(t.astype(int), len(stops)-2); w = (t-i)[..., None]
    return stops[i]*(1-w)+stops[i+1]*w


def oblique(x, r, D):
    zb, who = np.full((ny, nx), np.inf), np.full((ny, nx), -1)
    u, v, dpt = (x@Rt-u0)/res-.5, (v1-x@Up)/res-.5, x@Fw
    for i in range(n):
        p = r[i]/res
        a0, a1, b0, b1 = int(max(u[i]-p, 0)), int(min(u[i]+p+1, nx-1)), int(max(v[i]-p, 0)), int(min(v[i]+p+1, ny-1))
        if a1 <= a0 or b1 <= b0:
            continue
        ua, vb = np.arange(a0, a1+1), np.arange(b0, b1+1)
        d2 = ((ua-u[i])[None, :]**2+(vb-v[i])[:, None]**2)*res*res
        q = d2 < r[i]*r[i]
        zz = dpt[i]-np.sqrt(np.maximum(r[i]*r[i]-d2, 0.))
        Z = x[i, 2]+((ua-u[i])[None, :]*Rt[2]-(vb-v[i])[:, None]*Up[2])*res+(zz-dpt[i])*Fw[2]
        q = q & (Z >= args.ground)
        sub, sw = zb[b0:b1+1, a0:a1+1], who[b0:b1+1, a0:a1+1]
        c = q & (zz < sub)
        sub[c] = zz[c]; sw[c] = i
    has = who >= 0
    z = np.where(has, zb, 0.)
    dx, dy = np.zeros_like(z), np.zeros_like(z)
    dx[:, 1:-1] = (z[:, 2:]-z[:, :-2])/(2*res); dy[1:-1, :] = (z[2:, :]-z[:-2, :])/(2*res)
    bad = ~(has & np.roll(has, 1, 1) & np.roll(has, -1, 1) & np.roll(has, 1, 0) & np.roll(has, -1, 0))
    dx[bad] = 0.; dy[bad] = 0.
    dx, dy = np.clip(dx, -3, 3), np.clip(dy, -3, 3)
    N = dx[..., None]*Rt-dy[..., None]*Up-Fw[None, None]
    N = N/np.linalg.norm(N, axis=-1, keepdims=True)
    sh = .42+.62*np.clip((N*light).sum(-1), 0, 1)
    col = np.tile(CLAY, (n, 1)); col[D < .2] = ORANGE; col[D < 0] = RED
    im = np.zeros((ny, nx, 4))
    im[has] = np.concatenate([np.clip(sh[has, None]*col[who[has]], 0, 1), np.ones((has.sum(), 1))], 1)
    return im


def slice_at(x, r, sp, yc, D):
    Xx, Zz = np.meshgrid(px, pz)
    best, who = np.full((sy, sx), np.inf), np.full((sy, sx), -1)
    for i in np.where(np.abs(x[:, 1]-yc) < r)[0]:
        a0, a1 = np.searchsorted(px, [x[i, 0]-r[i], x[i, 0]+r[i]]); b0, b1 = np.searchsorted(-pz, [-(x[i, 2]+r[i]), -(x[i, 2]-r[i])])
        if a1 <= a0 or b1 <= b0:
            continue
        pw = (Xx[b0:b1, a0:a1]-x[i, 0])**2+(Zz[b0:b1, a0:a1]-x[i, 2])**2+(yc-x[i, 1])**2-r[i]*r[i]
        sub, sw = best[b0:b1, a0:a1], who[b0:b1, a0:a1]
        q = (pw < 0) & (pw < sub)
        sub[q] = pw[q]; sw[q] = i
    who[Zz < args.ground] = -1
    has = who >= 0
    col = cmap(sp); col[D < .2] = ORANGE; col[D < 0] = RED
    im = np.zeros((sy, sx, 4))
    im[has] = np.concatenate([col[who[has]], np.ones((has.sum(), 1))], 1)
    e = np.zeros_like(has)
    e[:, :-1] |= who[:, :-1] != who[:, 1:]; e[:-1, :] |= who[:-1, :] != who[1:, :]
    e &= has
    im[e] = np.array([.07, .07, .09, 1.])
    return im


data = dict(groups=[dict(name=g_[0], runs=[dict(tag=s.split('=')[0], label=s.split('=')[1]) for s in g_[1:]]) for g_ in args.group], t=T.tolist(), vmax=args.vmax, frames=[], stop={})
for tag, r in runs.items():
    data['stop'][tag] = float(r['t'][-1])
for k, t in enumerate(T):
    fr = {}
    for tag, r in runs.items():
        i = at(r, t)
        x, rr = r['x'][i], r0*np.exp(r['L'][i]/3)
        dtf = r['t'][1]-r['t'][0]
        sp = np.linalg.norm((r['x'][min(i+1, len(r['t'])-1)]-r['x'][max(i-1, 0)])[:, :2], axis=1)/(2*dtf)
        D = dets(x)
        yc = float((V0*x[:, 1]).sum()/V0.sum()); yl = float(np.median(x[LEG, 1]))
        fr[tag] = dict(o=png(oblique(x, rr, D)), s=png(slice_at(x, rr, sp, yc, D)), l=png(slice_at(x, rr, sp, yl, D)), out=int((D < 0).sum()), near=int((D < .2).sum()))
    data['frames'].append(fr)
    print('frame', k, 't %.3f' % t, {a: b['out'] for a, b in fr.items()}, flush=True)
html = Path(__file__).with_name('part_view.html').read_text().replace('/*DATA*/null', json.dumps(data))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(html)
print('wrote', args.out, '%.1f MB' % (len(html)/1e6), flush=True)
