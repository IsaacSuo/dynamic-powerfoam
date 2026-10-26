"""Is the skinless model's deficit on the part's FEM modes 1 and 7 the outer fibres? (2026-10-26; FEM data only, no model)
Inference from sphere_closure_split.log: in stretching / bending the material just under the exposed sphere strains more
than the cell's average, which a cell-average F_B cannot see. Per FEM mode, from the Gauss points of the voxel FEM
(sphere_closure_split.fem_gauss), each point to its cell (lowest power distance):
  cap cone   the points of a cell with an exposed sphere whose ray from the site leaves through the exposed part
             (projected onto the sphere, no other ball's power below 0): the outer layer the model's o1 stands for
  true       their continuum energy;  at the cap's mean strain  V_cap W(mean of eps over the cap cone);
             at the cell's mean strain  V_cap W(mean of eps over the cell), the best a cell-average F_B can do
  the rest of the exposed cells and the inner cells: true against V W(cell mean)
All as fractions of the mode's energy (omega^2 / 2). The model's deficit on the exposed cells (split log): modes 1 / 7
14-20 %, modes 0 / 2 1-9 %.
  PYTHONPATH=<main repo>:<worktree> python free_surface/sphere_closure_fiber.py"""
import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build
from free_surface.sphere_closure_split import fem_gauss

f64, dev = torch.float64, 'cuda'


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument('--part', type=Path, default=Path('output/test_part'))
    a_.add_argument('--fem-h', type=float, default=.0015)
    a_.add_argument('--modes', type=int, nargs='*', default=[1, 7, 0, 2])
    args = a_.parse_args()
    stt = torch.load(args.part/'run'/'model.pt', map_location='cpu', weights_only=True)
    p = stt['points'].double().numpy(); r = torch.nn.functional.softplus(stt['radii'].double(), beta=100).numpy()
    c_ = dict(np.load(args.part/('fem_modes_h%g.npz' % args.fem_h)))
    g = dict(np.load(args.part/'dem_exact_single'/'geometry.npz'))
    m = build(g, 23008.38, .3008, dev, 'noslip', .25, .3, f64, kinematics='exact', nodalvol=True, nonlinear=False, ogden_alpha=2.)
    mu, lam = float(m.mu_ref), float(m.k_v)
    pt, rt = torch.as_tensor(p, dtype=f64, device=dev), torch.as_tensor(r, dtype=f64, device=dev)
    n = len(pt)
    W = lambda e: mu*(e*e).sum((-2, -1))+.5*lam*e.diagonal(dim1=-2, dim2=-1).sum(-1)**2
    for k in args.modes:
        X, eps, vol, cell = fem_gauss(c_, k, pt, rt)
        if k == args.modes[0]:                          # the cap cones (the same points for every mode)
            d = X-pt[cell]; Xp = pt[cell]+rt[cell, None]*d/d.norm(dim=1, keepdim=True).clamp(min=1e-300)
            cone = []
            for s in range(0, len(X), 4096):
                pw = torch.cdist(Xp[s:s+4096], pt)**2-rt**2
                pw[torch.arange(len(pw), device=dev), cell[s:s+4096]] = float('inf')
                cone.append(pw.min(1).values >= 0)
            cone = torch.cat(cone)
            Vc = torch.zeros(n, dtype=f64, device=dev).index_add(0, cell, vol)
            Vcap = torch.zeros(n, dtype=f64, device=dev).index_add(0, cell[cone], vol[cone])
            exposed = Vcap > 0
            print('Gauss points %d; cells with a cap cone %d; cap cones hold %.3f of the volume (of the exposed cells %.3f)' % (
                len(X), int(exposed.sum()), float(Vcap.sum()/Vc.sum()), float(Vcap.sum()/Vc[exposed].sum())), flush=True)
        mean = lambda msk: torch.zeros(n, 3, 3, dtype=f64, device=dev).index_add(0, cell[msk], vol[msk, None, None]*eps[msk]) \
            / torch.zeros(n, dtype=f64, device=dev).index_add(0, cell[msk], vol[msk]).clamp(min=1e-300)[:, None, None]
        all_ = torch.ones_like(cone)
        e_cell, e_cap = mean(all_), mean(cone)
        w = W(eps)*vol
        tot = float(w.sum()); om2 = (2*np.pi*float(c_['f'][k]))**2
        ex_pt = exposed[cell]
        cap_true = float(w[cone].sum())
        cap_capmean = float((Vcap*W(e_cap))[exposed].sum())
        cap_cellmean = float((Vcap*W(e_cell))[exposed].sum())
        rest = ex_pt & ~cone
        Vrest = torch.zeros(n, dtype=f64, device=dev).index_add(0, cell[rest], vol[rest])
        rest_true, rest_mean = float(w[rest].sum()), float((Vrest*W(e_cell))[exposed].sum())
        in_true, in_mean = float(w[~ex_pt].sum()), float((Vc*W(e_cell))[~exposed].sum())
        print('mode %d (%.2f Hz; Gauss total / (omega^2 / 2) %.4f), fractions of the mode\'s energy:\n'
              '   cap cones: true %.3f, at the cap\'s mean strain %.3f, at the cell\'s mean strain %.3f  ->  lost to the cell mean %.3f '
              '(the cap\'s mean differing from the cell\'s %.3f, the strain varying inside the cap %.3f)\n'
              '   rest of the exposed cells: true %.3f, at the cell\'s mean strain %.3f (lost %.3f); inner cells: true %.3f, at the cell\'s mean %.3f (lost %.3f)' % (
                  k, float(c_['f'][k]), tot/(.5*om2), cap_true/tot, cap_capmean/tot, cap_cellmean/tot, (cap_true-cap_cellmean)/tot,
                  (cap_capmean-cap_cellmean)/tot, (cap_true-cap_capmean)/tot, rest_true/tot, rest_mean/tot, (rest_true-rest_mean)/tot,
                  in_true/tot, in_mean/tot, (in_true-in_mean)/tot), flush=True)


if __name__ == '__main__':
    main()
