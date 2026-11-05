"""the free-vibration runs' own frequencies, per near-degenerate cluster of the FEM's modes (0-2, 3-5, 6-8: the jack's
symmetry), by two estimates:
  Hankel DMD: the cluster's 3 projections with d delays (d samples), SVD-truncated to rank 2 x 3 (+2 for leakage), its
    eigenvalues -> f; the 3 largest-amplitude frequencies within 0.7-1.4 of the cluster's mean FEM frequency
  one damped cosine per projection (c + A e^(-zeta w t) cos(w t + phi), clamp_fit.py's): an amplitude-weighted mean over
    the cluster's (slowly beating) eigenfrequencies
reported as f / f_FEM (the cluster's FEM frequencies in the same order); ours' Rayleigh-Ritz ratios for reference"""
import sys
import numpy as np
from scipy.optimize import curve_fit
f_fem = np.load('out_inverse/freevib/prep.npz')['f']
RR = {'lumped': [1.064, 1.069, 1.065, 1.045, 1.042, 1.047, 1.055, 1.057, 1.063], 'consistent': [.994, .999, .998, .961, .960, .966, .974, .976, .984]}
model = lambda t, c, A, z, f, ph: c+A*np.exp(-z*2*np.pi*f*t)*np.cos(2*np.pi*f*t+ph)
CL = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]


def hankel_dmd(Y, rate, d=40, r=8):
    H = np.concatenate([Y[i:len(Y)-d+i+1] for i in range(d)], 1).T
    H0, H1 = H[:, :-1], H[:, 1:]
    U, S, Vh = np.linalg.svd(H0, full_matrices=False)
    U, S, Vh = U[:, :r], S[:r], Vh[:r]
    At = U.T@H1@Vh.T/S
    lam, Wv = np.linalg.eig(At)
    Phi = H1@Vh.T/S@Wv
    b = np.linalg.lstsq(Phi, H0[:, 0].astype(complex), rcond=None)[0]
    s = np.log(lam.astype(complex))*rate
    return np.abs(s.imag)/(2*np.pi), -s.real/np.abs(s), np.abs(b)*np.linalg.norm(Phi, axis=0), s.imag


def fit1(t, y):
    yy = y-y.mean()
    F = np.abs(np.fft.rfft(yy*np.hanning(len(yy)), 16*len(yy))); f0 = np.fft.rfftfreq(16*len(yy), t[1]-t[0])[np.argmax(F)]
    try:
        p, _ = curve_fit(model, t, y, p0=[y.mean(), np.abs(yy).max(), .01, f0, 0.], maxfev=20000)
        return abs(p[3]), p[2]
    except RuntimeError:
        return np.nan, np.nan


res = {}
for p in sys.argv[1:]:
    z = np.load(p); q, rate = z['q'], float(z['rate']); k0 = int(.05*rate); t = np.arange(len(q))[k0:]/rate
    name = p.split('/')[-1][:-4]; res[name] = []
    for cl in CL:
        fm = f_fem[list(cl)].mean()
        f, zt, amp, im = hankel_dmd(q[k0:, list(cl)], rate)
        ok = (im > 0) & (f > .7*fm) & (f < 1.4*fm)
        o = np.argsort(-amp*ok)[:3]
        fd = np.sort(f[o][ok[o]])
        ff = [fit1(t, q[k0:, j]) for j in cl]
        res[name].append((fd, ff))
print('cluster (FEM f, Hz) | run: DMD f / f_FEM (sorted) | one-cosine fits f / f_FEM per mode (zeta) | ours RR lumped / consistent')
for ci, cl in enumerate(CL):
    fe = f_fem[list(cl)]
    print('modes %s (%s)' % (list(cl), ' '.join('%.3f' % x for x in fe)))
    for name, rr in res.items():
        fd, ff = rr[ci]
        print('   %-22s DMD %-24s | fits %s' % (name, ' '.join('%.3f' % (x/fe.mean()) for x in fd),
              ' '.join('%.3f (%.3f)' % (x[0]/fe[k], x[1]) for k, x in enumerate(ff))))
    print('   %-22s %s / %s' % ('our Rayleigh-Ritz', ' '.join('%.3f' % RR['lumped'][j] for j in cl), ' '.join('%.3f' % RR['consistent'][j] for j in cl)))
print('medians of the one-cosine fits over modes 0-8 (f / f_FEM):')
for name, rr in res.items():
    print('   %-22s %.3f' % (name, np.nanmedian([x[0]/f_fem[CL[ci][k]] for ci in range(3) for k, x in enumerate(rr[ci][1])])))
