"""clamped runs (mpm_clamp.py, dem_clamp.py) side by side: each group's displacement from t = 0 (centre of mass, the two
upper tips, the two lower tips), projected on its main direction in the first run (largest-variance axis, sign: mean
positive); per run: the mean over t >= 0.25 s (the static state the motion oscillates about), the oscillation's RMS about
it, and its frequency (zero crossings of the motion about the mean, t >= 0.05 s, averaged; and the zero-padded FFT peak)"""
import sys
import numpy as np
runs = [(p.split('/')[-1][:-4], np.load(p)) for p in sys.argv[1:]]
S0 = np.load('out_inverse/clamp/sets_hc04.npz'); up, lo = list(S0['upper']), list(S0['lower'])
gi = [('com', 0), ('upper %d' % up[0], 1+up[0]), ('upper %d' % up[1], 1+up[1]), ('lower %d' % lo[0], 1+lo[0]), ('lower %d' % lo[1], 1+lo[1])]


def disp(z, i):
    r = z['rec'][:, i]; return r-r[0]


axes = []
for _, i in gi:
    d = disp(runs[0][1], i); _, _, Vt = np.linalg.svd(d-d.mean(0)); a = Vt[0]; axes.append(a if (d@a).mean() > 0 else -a)
for (gname, i), a in zip(gi, axes):
    print('%s (main direction %s)' % (gname, np.round(a, 2)))
    for name, z in runs:
        rate = float(z['rate']); s = disp(z, i)@a*1e3; t = np.arange(len(s))/rate
        k = t >= .25; m = s[k].mean(); y = s-m
        q = (t >= .05)
        zc = t[1:][q[1:] & (np.sign(y[1:]) != np.sign(y[:-1]))]
        f_zc = 1/(2*np.diff(zc).mean()) if len(zc) > 2 else np.nan
        yy = (y[q]-y[q].mean())*np.hanning(q.sum()); F = np.abs(np.fft.rfft(yy, 16*len(yy))); fr = np.fft.rfftfreq(16*len(yy), 1/rate)
        print('  %-18s mean %+7.2f mm | RMS about it %6.2f | f zero-crossings %5.2f Hz, FFT peak %5.2f Hz' % (name, m, np.sqrt((y[k]**2).mean()), f_zc, fr[np.argmax(F)]))
