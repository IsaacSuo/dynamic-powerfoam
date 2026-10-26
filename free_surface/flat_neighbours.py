"""Who cuts the material flat when the hull ghosts are removed (2026-10-22)? Real sites only (ball_surface_look.py);
for the flat surface faces (material cell against a non-material real site): the other site's place (inside / outside
the visual hull), its density (the foam's own, softplus beta 100), its radius against the material cell's, and how far it
sits from the material cell's site; area-weighted."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_foam_geometry import visual_hull
from free_surface.ball_surface_look import geometry


def main():
    run = Path('output/pacnerf_elastic0_f0_transparent_cutft'); scene = Path('/home/fangsuo/py/dynamic-powerfoam/data/pacnerf_powerfoam/elastic_0_f0')
    st = torch.load(run/'model.pt', map_location='cpu', weights_only=True)
    n = int(st['num_sites'])
    pts = st['points'].double().numpy()[:n]
    radii = torch.nn.functional.softplus(st['radii'].double(), beta=100).numpy()[:n]
    dens = torch.nn.functional.softplus(st['density'].double(), beta=100).numpy()[:n]
    hull = visual_hull(pts, scene)
    material, volume, cone, pairs, area, flat = geometry(pts, radii, hull)
    m_side = np.where(material[pairs[:, 0]], pairs[:, 0], pairs[:, 1]); o_side = np.where(material[pairs[:, 0]], pairs[:, 1], pairs[:, 0])
    m_side, o_side, A = m_side[flat], o_side[flat], area[flat]
    out = ~hull[o_side]
    dmat = np.median(dens[material])
    print('flat surface faces %d, area %.4f m^2; the other site outside the hull: %.1f %% of the area (%d sites), inside the hull but not material %.1f %%' % (
        len(A), A.sum(), 100*A[out].sum()/A.sum(), len(np.unique(o_side[out])), 100*A[~out].sum()/A.sum()), flush=True)
    def wq(x, w, q):
        o = np.argsort(x); cw = np.cumsum(w[o])/w.sum()
        return np.interp(q, cw, x[o])
    print('density: material cells median %.3g; the other sites of the flat faces, area-weighted 10/50/90 %% / material median: %s' % (
        dmat, np.round(wq(dens[o_side]/dmat, A, [.1, .5, .9]), 4).tolist()), flush=True)
    d = np.linalg.norm(pts[o_side]-pts[m_side], axis=1)
    print('other site radius / material site radius median %.2f; distance between the two sites / material radius median %.2f, 90 %% %.2f' % (
        np.median(radii[o_side]/radii[m_side]), np.median(d/radii[m_side]), np.quantile(d/radii[m_side], .9)), flush=True)
    for name, mk in (('outside the hull', out), ('inside the hull, not material', ~out)):
        if mk.any():
            print('   %-30s faces %6d, area %.4f m^2, other-site density / material median: median %.3g, max %.3g' % (
                name, int(mk.sum()), A[mk].sum(), np.median(dens[o_side[mk]])/dmat, dens[o_side[mk]].max()/dmat), flush=True)


if __name__ == '__main__':
    main()
