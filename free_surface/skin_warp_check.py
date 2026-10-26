"""W1 check (2026-10-23): skin_warp.SkinWarp against the torch model (sphere_model.SkinTerms via BoundaryTerms) on the new
foam: the six energy terms and dE/dx, dE/dy at rest, at a non-uniform state (smooth deformation, rough skin) and that state
turned by pi; timing of one energy and one gradient (Warp after warm-up, torch for reference)."""
import math
import time

import torch

from free_surface.skin_contact_check import setup, f64, dev
from free_surface.skin_warp import SkinWarp


def main():
    terms, S = setup('foam', 'boundary_skin.npz', 'shepard0')
    st = terms
    X, Y = st.X, st.Y; n = len(X)
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    W = SkinWarp(st)
    gen = torch.Generator(device=dev).manual_seed(0)
    size = float((X.max(0).values-X.min(0).values).max()); cen = X.mean(0); rmed = float(S.r.median())
    kv = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*2*math.pi/size
    amp = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*.03*size
    ph = torch.rand(3, generator=gen, dtype=f64, device=dev)*2*math.pi
    field = lambda Z: torch.sin(Z@kv.T+ph)@amp.T
    x1, y1 = X+field(X), Y+field(Y)+torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)*.05*rmed
    ax = torch.nn.functional.normalize(torch.randn(3, generator=gen, dtype=f64, device=dev), dim=0)
    K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=f64, device=dev)
    Q = torch.eye(3, dtype=f64, device=dev)+2*K@K                                   # half turn
    names = ('bonds', 'volume', 'outer_stretch', 'outer_cross', 'outer_skin', 'outer_lateral')
    for nm, (x, y) in (('rest', (X, Y)), ('deformed', (x1, y1)), ('deformed, half turn', ((x1-cen)@Q.T+cen, (y1-cen)@Q.T+cen))):
        Tt = st.energy_terms(th, x, y)
        tw, gx, gy = W.energy_grad(x, y)
        xr, yr = x.clone().requires_grad_(True), y.clone().requires_grad_(True)
        st.energy_xy(th, xr, yr).backward()
        tot = sum(float(v) for v in Tt.values())
        line = '%-20s energy %.6e; term (warp - torch) / total: %s; |grad_x diff| / |grad_x| %.1e, |grad_y diff| / |grad_y| %.1e' % (
            nm, tot, ' '.join('%s %.1e' % (k, (float(tw[i])-float(Tt[k].detach()))/max(tot, 1e-300)) for i, k in enumerate(names)),
            float((gx-xr.grad).norm()/xr.grad.norm().clamp(min=1e-300)), float((gy-yr.grad).norm()/yr.grad.norm().clamp(min=1e-300)))
        print(line+' (|grad_x| torch %.1e, warp %.1e; |grad_y| torch %.1e, warp %.1e)' % (
            float(xr.grad.norm()), float(gx.norm()), float(yr.grad.norm()), float(gy.norm())), flush=True)
    for nm, (x, y) in (('deformed', (x1, y1)), ('deformed, half turn', ((x1-cen)@Q.T+cen, (y1-cen)@Q.T+cen))):
        W.hand_adjoints = False; _, ax, ay = W.energy_grad(x, y)
        W.hand_adjoints = True; _, hx, hy = W.energy_grad(x, y)
        print('%-20s hand-written vs generated adjoints (bonds, walls): |grad_x diff| / |grad_x| %.1e, |grad_y diff| / |grad_y| %.1e' % (
            nm, float((hx-ax).norm()/ax.norm()), float((hy-ay).norm()/ay.norm())), flush=True)
    W.prepare()                                                                      # the CUDA graph path
    _, gx0, gy0 = W.energy_grad(x1, y1); e1, gx1, gy1 = W.grad(x1, y1)
    print('graph vs no graph (deformed): energy %.1e, |grad_x diff| / |grad_x| %.1e, |grad_y diff| / |grad_y| %.1e' % (
        abs(float(e1)-float(W.energy(x1, y1).sum()))/abs(float(e1)), float((gx1-gx0).norm()/gx0.norm()), float((gy1-gy0).norm()/gy0.norm())), flush=True)
    def tm(f, reps=5):
        f(); torch.cuda.synchronize(); t = time.time()
        for _ in range(reps):
            f()
        torch.cuda.synchronize(); return (time.time()-t)/reps
    def torch_grad():
        xr, yr = x1.clone().requires_grad_(True), y1.clone().requires_grad_(True)
        st.energy_xy(th, xr, yr).backward()
    print('timing: warp energy %.4f s, warp energy + gradient %.4f s; torch energy %.4f s, torch energy + gradient %.4f s' % (
        tm(lambda: W.energy(x1, y1)), tm(lambda: W.energy_grad(x1, y1)), tm(lambda: st.energy_xy(th, x1, y1)), tm(torch_grad)), flush=True)

if __name__ == '__main__':
    main()
