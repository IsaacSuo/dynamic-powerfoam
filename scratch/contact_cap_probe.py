"""does the free-surface closure undo the ground's compression of the touching caps? lion_inverse.forward of PAC-NeRF
elastic scene 0 (consistent mass, beta 0.003, glue at the material point), at frames 5.5 / 6 / 6.5 / 7: for the touching
cells (ball force > 0, all of them cap cells), weighted by their normal force, the stretch of the material fibre along
the cap's rest mean normal m and along its tangents t1, t2, and det, from
  G    the one-way cap's 2-ring least-squares fit of the site positions (what the positions say), and
  F_B  the closure's F (S_B + comp_f(G) Mcap) Mt^-1 (what the energy sees; comp_f puts the plane-stress normal column)
and the share of the cap in the closure (|Mcap| / |Mt|)"""
import sys
from pathlib import Path
import numpy as np
import torch
import warp as wp
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
li.set_scene('pacnerf_e0'); li.GLUE = 'material'
at = {int(round(f*li.SPF)): f for f in (5.5, 6., 6.5, 7.)}


class Stop(Exception):
    pass


def probe(s, W, ST, cbody, Rb):
    if s not in at:
        return
    st = W.st
    on = W.tbfn > 0; B = cbody[on]; w = W.tbfn[on]; w = w/w.sum()
    G = wp.to_torch(W.B['G'])[B]; F = W.tF[B]
    m, t1, t2 = st.mA[B], st.t1[B], st.t2[B]
    sh = (st.Mcap[B].norm(dim=(1, 2))/(st.Mb[B]+st.Mcap[B]).norm(dim=(1, 2)))
    st_ = lambda M_, v: (torch.einsum('nij,nj->ni', M_, v)).norm(dim=1)
    avg = lambda q: float((w*q).sum())
    print('frame %.1f: %d touching (caps %d); cap share of the closure %.2f' % (at[s], int(on.sum()), int(st.cap[B].sum()), avg(sh)))
    for nm, M_ in (('G (positions)', G), ('F_B (closure)', F)):
        print('   %-14s normal stretch %.3f | tangential %.3f, %.3f | det %.3f' % (nm, avg(st_(M_, m)), avg(st_(M_, t1)), avg(st_(M_, t2)), avg(torch.linalg.det(M_))), flush=True)
    up = torch.tensor([0., 1., 0.], dtype=F.dtype, device=F.device).expand_as(m)
    print('   vertical fibre stretch: G %.3f, F_B %.3f' % (avg(st_(G, up)), avg(st_(F, up))))
    free = st.cap.clone(); free[B] = False
    hgt = ST.tz[:W.n][:, 1]-li.C0; far = free & (hgt > .15)
    Gf, Ff = wp.to_torch(W.B['G'])[far], W.tF[far]
    print('   control, cap cells off the ground (height > 15 cm, %d): det G %.3f, det F_B %.3f (median); |G - F_B| / |G| median %.3f' % (
        int(far.sum()), float(torch.linalg.det(Gf).median()), float(torch.linalg.det(Ff).median()), float(((Gf-Ff).norm(dim=(1, 2))/Gf.norm(dim=(1, 2))).median())), flush=True)
    if s == max(at):
        raise Stop


try:
    li.forward(np.array([10.04361, .30079413, -5.809143]), 8, verbose=False, on_step=probe)
except Stop:
    pass
