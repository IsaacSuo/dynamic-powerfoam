"""mpm_foot_loops.py against dem_foot_loops.py: per foot the ground's normal force against the foot's squash (its material's
mean height at first touch minus now), the loading and unloading branches; and in time the front - back normal force and
the forward force. Prints per foot: first touch, the deepest squash (when, N there), and N at half that squash on the way
down and on the way up; a table at half frames 5 .. 9. Plot: argv[3] (png). args: mpm.npz dem.npz out.png"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FPS = 24
runs = {'MPM (dx 7.8 mm)': np.load(sys.argv[1])['rows'], 'ours (cap contact)': np.load(sys.argv[2])['rows']}
COL = {'MPM (dx 7.8 mm)': '#2a78d6', 'ours (cap contact)': '#eb6834'}
I = dict(t=0, Nf=4, Nb=5, Tf=6, Tb=7, hf=8, hb=9)


def foot(R, k):
    t, N, h = R[:, 0], R[:, I['N'+k]], R[:, I['h'+k]]
    on = np.nonzero(N > .01*N.max())[0]
    i0 = on[0]; sq = 1e3*(h[i0]-h)
    im = int(np.argmax(np.where(np.arange(len(t)) >= i0, sq, -1e9)))
    half = .5*sq[im]
    down = i0+int(np.argmin(np.abs(sq[i0:im+1]-half)))
    after = np.nonzero((np.arange(len(t)) > im) & (N > .01*N.max()))[0]
    up = (im+int(np.argmin(np.abs(sq[im:after[-1]+1]-half)))) if len(after) else None
    return dict(t=t, N=N, sq=sq, i0=i0, im=im, down=down, up=up, last=after[-1] if len(after) else im)


for nm, R in runs.items():
    print('== %s' % nm)
    for k, lab in (('f', 'front'), ('b', 'back')):
        d = foot(R, k)
        t = d['t']
        print('  %s foot: touch frame %.2f, deepest squash %.1f mm at frame %.2f (N %.1f), last contact frame %.2f; at half the squash (%.1f mm): N %.1f going down (frame %.2f), %s going up' % (
            lab, t[d['i0']]*FPS, d['sq'][d['im']], t[d['im']]*FPS, d['N'][d['im']], t[d['last']]*FPS, .5*d['sq'][d['im']], d['N'][d['down']], t[d['down']]*FPS,
            ('%.1f (frame %.2f)' % (d['N'][d['up']], t[d['up']]*FPS)) if d['up'] is not None else '-'))
    print('  frame | N front, N back, front - back | forward force front, back, total | squash front, back (mm)')
    df, db = foot(R, 'f'), foot(R, 'b')
    for f in np.arange(5., 9.01, .5):
        i = int(np.argmin(np.abs(R[:, 0]*FPS-f)))
        print('  %4.1f  | %6.1f %6.1f %+6.1f | %+6.2f %+6.2f %+6.2f | %5.1f %5.1f' % (
            f, R[i, I['Nf']], R[i, I['Nb']], R[i, I['Nf']]-R[i, I['Nb']], R[i, I['Tf']], R[i, I['Tb']], R[i, I['Tf']]+R[i, I['Tb']], df['sq'][i], db['sq'][i]))

fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
for nm, R in runs.items():
    for j, (k, lab) in enumerate((('f', 'front foot'), ('b', 'back foot'))):
        d = foot(R, k); a_ = ax[0, j]
        sl = slice(d['i0'], d['last']+1)
        a_.plot(d['sq'][sl], d['N'][sl], color=COL[nm], lw=2, label=nm)
        for f in (5.5, 6, 6.5, 7, 7.5, 8):
            i = int(np.argmin(np.abs(d['t']*FPS-f)))
            if d['i0'] <= i <= d['last']:
                a_.plot(d['sq'][i], d['N'][i], 'o', ms=6, color=COL[nm], mec='#fcfcfb', mew=1.5)
                a_.annotate('%g' % f, (d['sq'][i], d['N'][i]), textcoords='offset points', xytext=(5, 4), fontsize=8, color='#52514e')
        a_.set_title('%s: normal force vs squash (dots: frames)' % lab, fontsize=10, color='#0b0b0b')
        a_.set_xlabel('squash (mm): mean height of its material at touch - now', fontsize=9, color='#52514e')
        a_.set_ylabel('normal force per unit mass (m/s^2)', fontsize=9, color='#52514e')
    t = R[:, 0]*FPS
    ax[1, 0].plot(t, R[:, I['Nf']]-R[:, I['Nb']], color=COL[nm], lw=2, label=nm)
    ax[1, 1].plot(t, R[:, I['Tf']]+R[:, I['Tb']], color=COL[nm], lw=2, label=nm)
ax[1, 0].set_title('front - back normal force (m/s^2)', fontsize=10, color='#0b0b0b')
ax[1, 1].set_title('forward force of the ground, both feet (m/s^2; < 0 brakes)', fontsize=10, color='#0b0b0b')
for a_ in ax[1]:
    a_.set_xlim(4.5, 11); a_.set_xlabel('frame', fontsize=9, color='#52514e'); a_.axhline(0, color='#52514e', lw=.8)
for a_ in ax.ravel():
    a_.grid(color='#e6e5e0', lw=.6); a_.legend(fontsize=8, frameon=False)
    for s_ in a_.spines.values():
        s_.set_color('#c3c2b7')
    a_.tick_params(colors='#52514e', labelsize=8)
fig.patch.set_facecolor('#fcfcfb')
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=130)
print('wrote', sys.argv[3])
