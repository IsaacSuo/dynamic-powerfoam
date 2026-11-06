"""step 3, statically: would charging the loaded caps push the feet apart? lion_inverse.forward of PAC-NeRF elastic scene 0
(consistent mass, beta 0.003, the cap contact alpha 30) at frames 5.5 / 6 / 6.5 / 7; at the step's accepted midpoint the
model's elastic forces (the contact switched off for these evaluations) twice: the caps completed as traction-free (the
model now), and with the traction the cap contact puts on each cap (its points' forces over the cap's area; the patch test's
completion). The difference dF = -(dE/dx with - without) on the foot sites (within 30 mm of the ground), per foot (front /
back of the centre of mass along the forward direction e_f): its outward horizontal part (from the foot's centre), its
forward part, its vertical part; against the foot's contact force and the elastic force on those sites now"""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import free_surface.lion_inverse as li
li.set_scene('pacnerf_e0'); li.CAPC = 30.
at = {int(round(f*li.SPF)): f for f in (5.5, 6., 6.5, 7.)}
ef = torch.tensor([-0.4078950600549248, 0., -0.06629882732047176], dtype=torch.float64, device='cuda'); ef /= ef.norm()


class Stop(Exception):
    pass


def probe(s, W, ST, cbody, Rb):
    if s not in at:
        return
    n, st = W.n, W.st; z = ST.tz.clone(); cq = W.tcq
    up = torch.tensor([0., 1., 0.], dtype=torch.float64, device='cuda')
    cb = cq['cb'].long(); fq = cq['fnq'][:, None]*up+cq['ffq']
    fcell = torch.zeros(n, 3, dtype=torch.float64, device='cuda').index_add_(0, cb, fq)
    tr = torch.where(st.cap[:, None], fcell/W.cap_area.clamp(min=1e-30)[:, None], torch.zeros_like(fcell))
    lim = .5/float(W.ttpar[0])                                                      # (the traction's strain kept <= 0.5: the linear completion's range)
    tn = tr.norm(dim=1, keepdim=True); ncap = int((tn[:, 0] > lim).sum())
    tr = tr*torch.clamp(lim/tn.clamp(min=1e-30), max=1.)
    c0 = float(cq['par'][2]); cq['par'][2] = -10.                                   # (the contact off: elastic forces only)
    W.ttrac.zero_(); g0 = W.grad(z)[1][:n].clone(); det0 = torch.linalg.det(W.tF).clone()
    W.ttrac.copy_(tr); g1 = W.grad(z)[1][:n].clone(); det1 = torch.linalg.det(W.tF).clone()
    W.ttrac.zero_(); cq['par'][2] = c0
    dF = -(g1-g0)
    zs = z[:n]; m = W.st.V; com = (m[:, None]*zs).sum(0)/m.sum()
    h = zs[:, 1]-li.C0; foot = h < .03
    fwd = (zs-com)@ef
    touch = fcell[:, 1] > 0
    print('frame %.1f: touching cells %d (traction capped on %d); their det F_B %.3f -> %.3f with the traction (median; min %.3f)' % (at[s], int(touch.sum()), ncap, float(det0[touch].median()), float(det1[touch].median()), float(det1[touch].min())))
    for nm, sel in (('front foot', foot & (fwd > 0)), ('back foot', foot & (fwd < 0))):
        if not bool(sel.any()):
            continue
        w = m[sel]/m[sel].sum(); cen = (w[:, None]*zs[sel]).sum(0)
        o = zs[sel]-cen; o[:, 1] = 0; o = o/o.norm(dim=1, keepdim=True).clamp(min=1e-12)
        d = dF[sel]
        out = float((d*o).sum(1).sum()); fw = float((d@ef).sum()); vt = float(d[:, 1].sum())
        contact = float(fcell[sel][:, 1].sum()); el = float(g0[sel].norm(dim=1).sum()); dn = float(d.norm(dim=1).sum())
        po = (d*o).sum(1); share = float((po > 0).double().mean())
        rel = (d.norm(dim=1)/g0[sel].norm(dim=1).clamp(min=1e-30))
        print('   %-10s %3d sites: sum dF outward %+.4f, forward %+.4f, up %+.4f | sites pushed outward %.0f %%, median |dF| / |elastic now| %.2f | contact force up %.4f, sum |elastic force now| %.4f' % (
            nm, int(sel.sum()), out, fw, vt, 100*share, float(rel.median()), contact, el), flush=True)
    print('   whole body: sum dF %s (internal: ~0)' % np.round(dF.sum(0).cpu().numpy(), 6))
    if s == max(at):
        raise Stop


try:
    li.forward(np.array([10.04361, .30079413, -5.809143]), 8, verbose=False, on_step=probe)
except Stop:
    pass
