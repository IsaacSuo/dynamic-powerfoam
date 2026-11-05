"""clamped kick runs: each group's displacement (com, the two upper tips) on its main direction in the first run, fitted over
t in [0.05, T] by an offset + one damped cosine, c + A exp(-zeta 2 pi f t) cos(2 pi f t + phi) (start: the FFT peak);
prints f (Hz), zeta, A (mm), c (mm) and the fit's RMS residual (mm)"""
import sys
import numpy as np
from scipy.optimize import curve_fit
runs = [(p.split('/')[-1][:-4], np.load(p)) for p in sys.argv[1:]]
S0 = np.load('out_inverse/clamp/sets_hc04.npz'); up = list(S0['upper'])
gi = [('com', 0), ('upper %d' % up[0], 1+up[0]), ('upper %d' % up[1], 1+up[1])]
model = lambda t, c, A, z, f, ph: c+A*np.exp(-z*2*np.pi*f*t)*np.cos(2*np.pi*f*t+ph)
for gname, i in gi:
    d0 = runs[0][1]['rec'][:, i]-runs[0][1]['rec'][0, i]; _, _, Vt = np.linalg.svd(d0-d0.mean(0)); a = Vt[0]
    print(gname)
    for name, z in runs:
        rate = float(z['rate']); s = (z['rec'][:, i]-z['rec'][0, i])@a*1e3; t = np.arange(len(s))/rate
        q = t >= .05; y = s[q]-s[q].mean()
        F = np.abs(np.fft.rfft(y*np.hanning(len(y)), 16*len(y))); f0 = np.fft.rfftfreq(16*len(y), 1/rate)[np.argmax(F)]
        try:
            p, _ = curve_fit(model, t[q], s[q], p0=[s[q].mean(), np.abs(y).max(), .02, f0, 0.], maxfev=20000)
            if p[1] < 0:
                p[1], p[4] = -p[1], p[4]+np.pi
            res = np.sqrt(((model(t[q], *p)-s[q])**2).mean())
            print('  %-20s f %.3f Hz | zeta %+.3f | A %6.1f mm | offset %+6.1f mm | residual %5.1f mm' % (name, p[3], p[2], p[1], p[0], res))
        except RuntimeError:
            print('  %-20s fit failed (FFT peak %.2f Hz)' % (name, f0))
