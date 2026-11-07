"""(kept here 11-06 from the 10-28 job tmp /home/fangsuo/.claude/jobs/0d6c02d1/tmp/convert_foam.py, unchanged below; it made
foams/fourier8k_clean from thin-surface fourier3d/output/fourier_pacnerf_elastic0_8k_free_clean/model.pt, and on 11-06
foams/fourier8k_split_keep_clean from ..._8k_split_keep_clean/model.pt). Run as the 10-28 run_ul.sh did: from the
updated-lagrange worktree (free_surface.ul_model's ULModel, cell_moments_grid) with the main repo on PYTHONPATH
(experiments.pacnerf_foam_geometry.visual_hull):
  cd .claude/worktrees/updated-lagrange && PYTHONPATH=<repo>:<repo>/.claude/worktrees/updated-lagrange \
    python -u <this file> <model.pt> <repo>/data/pacnerf_powerfoam/elastic_0_f0 <k1 worktree>/foams/<name>
then copy plasticine_scenes.json beside dem_exact_single (scene 0 needs no transform)."""
"""a PowerFoam model.pt (no planes: cells = power cells x balls) -> output/<name>/dem_exact_single (points, masses = the
material-only cut's volumes, radii; source_index) and ul_moments_0.002.npz; material = centres inside every view's mask,
then the largest connected component of the material-only cut (as experiments/pacnerf_foam_geometry.py)
usage: convert_foam.py <model.pt> <scene dir> <out dir>"""
import sys, numpy as np, torch
from pathlib import Path
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from experiments.pacnerf_foam_geometry import visual_hull
from free_surface.ul_model import ULModel, cell_moments_grid
model, scene, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
st = torch.load(model, map_location='cpu', weights_only=True)
p = st['points'].double().numpy(); rr = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()
idx = np.flatnonzero(visual_hull(p, scene))
for it in range(5):                                                                 # material-only cut until it is one component
    mdl = ULModel(p[idx], rr[idx], np.ones(len(idx)), np.zeros((len(idx), 3)), np.zeros((len(idx), 3, 3)), 23., .3)
    geo = mdl.geometry(mdl.X0, mdl.r0)
    V = geo['V'].cpu().numpy(); ii, jj = geo['i'].cpu().numpy(), geo['j'].cpu().numpy()
    keep = V > 1e-12
    nc, lab = connected_components(sparse.coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(len(idx), len(idx))), directed=False)
    big = np.bincount(lab[keep]).argmax(); ok = keep & (lab == big)
    print('pass %d: %d sites, %d kept' % (it, len(idx), ok.sum()), flush=True)
    if ok.all():
        break
    idx = idx[ok]
(out/'dem_exact_single').mkdir(parents=True, exist_ok=True)
X, r = p[idx], rr[idx]
np.savez(out/'dem_exact_single'/'geometry.npz', points=X, masses=V, pairs=np.stack([ii, jj], 1), areas=geo['A'].cpu().numpy(), contacts=geo['c'].cpu().numpy())
np.savez(out/'dem_exact_single'/'cells.npz', points=X, radii=r, source_index=idx)
dlt, J0, cnt = cell_moments_grid(X, r, V, .002)
np.savez(out/'ul_moments_0.002.npz', dlt=dlt, J0=J0, cnt=cnt)
print('%d cells written to %s; volume %.4e m^3; moments on the 2 mm grid, cells without a sample %d' % (len(idx), out, V.sum(), int((cnt == 0).sum())))
