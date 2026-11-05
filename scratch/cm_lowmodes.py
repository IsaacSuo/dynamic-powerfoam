"""the consistent mass's weak directions: the smallest eigenvectors of D^-1/2 Mh D^-1/2 (Mh, moments from
consistent_mass_check.py): how many below 0.1 / 0.3, each one's participation (cells carrying it), whether those cells are
cap cells, their site-centroid offset, sample count, volume against the median"""
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
O = ROOT/'out_inverse'/'consistent_mass'
Mh = sp.load_npz(O/'Mh.npz'); mo = np.load(O/'moments.npz')
V = np.load(ROOT/'inverse_data/pacnerf_e0/foam/dem_exact_single/geometry.npz')['masses']
n = Mh.shape[0]
off = np.linalg.norm(mo['d'], axis=1); cnt = mo['cnt']
aC = mo['aC']; ncap = np.bincount(aC, minlength=n)              # stencil size (cap cells have the 2-ring)
Dh = 1/np.sqrt(Mh.diagonal())
A = torch.as_tensor(Mh.toarray()*Dh[:, None]*Dh[None, :], device='cuda')
w, U = torch.linalg.eigh(A); w, U = w.cpu().numpy(), U.cpu().numpy()
print('eigenvalues of D^-1/2 M D^-1/2: below 0.05 %d, below 0.1 %d, below 0.3 %d, above 3 %d (of %d)' % ((w < .05).sum(), (w < .1).sum(), (w < .3).sum(), (w > 3).sum(), n))
cap = ncap > np.median(ncap)*0 + 40
for k in list(range(6))+[int(np.searchsorted(w, .3))]:
    u = U[:, k]**2; top = np.argsort(-u)[:5]; pr = 1/(u**2).sum()
    print('  lambda %.4f: participation %.1f cells; top cells (share, stencil, offset mm, samples, V/median): %s' % (
        w[k], pr, '; '.join('%d (%.2f, %d, %.1f, %d, %.2f)' % (c, u[c], ncap[c], 1e3*off[c], cnt[c], V[c]/np.median(V)) for c in top)))
