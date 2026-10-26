"""Rotations near a half turn (2026-10-23): sphere_model.matrix_quat (Shepperd) against the earlier
foam_surface_completion.matrix_quat (divides by w = sqrt(1 + tr) / 2).
  quaternions  |quat_matrix(q) - R| and |q| - 1 at angles up to pi, both functions
  energy       the skin model (new foam, skin C) at a non-uniform state (smooth deformation + rough skin), turned rigidly
               by angles up to pi: change of the energy relative to the unturned one, with the new and the earlier function
  derivatives  at pi - 1e-7 and pi: d E / d s along a random direction of (x, y) against central differences, and the
               Hessian-vector product turned with the state, H(Q x) (Q v) against Q H(x) v"""
import math

import numpy as np
import torch

import free_surface.sphere_model as sm
from experiments.foam_dem_energy import quat_matrix
from experiments.foam_surface_completion import matrix_quat as matrix_quat_old
from free_surface.skin_contact_check import setup, f64, dev


def rot(axis, ang):
    k = axis/axis.norm()
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=f64, device=dev)
    return torch.eye(3, dtype=f64, device=dev)+math.sin(ang)*K+(1-math.cos(ang))*K@K


def main():
    gen = torch.Generator(device=dev).manual_seed(0)
    angles = (0., 1e-8, .5, 2., 3., math.pi-1e-4, math.pi-1e-7, math.pi)
    axes = torch.nn.functional.normalize(torch.randn(200, 3, generator=gen, dtype=f64, device=dev), dim=1)
    for ang in angles:
        R = torch.stack([rot(a, ang) for a in axes])
        line = 'angle %.8f:' % ang
        for nm, f in (('new', sm.matrix_quat), ('earlier', matrix_quat_old)):
            q = f(R)
            line += ' %s |quat_matrix(q) - R| max %.1e, ||q| - 1| max %.1e;' % (nm, float((quat_matrix(q)-R).abs().max()), float((q.norm(dim=1)-1).abs().max()))
        print(line, flush=True)
    terms, S = setup('foam', 'boundary_skin.npz', 'shepard0')
    st = terms
    X, Y = st.X, st.Y; n = len(X)
    th = torch.zeros(n, 9, dtype=f64, device=dev)
    size = float((X.max(0).values-X.min(0).values).max()); cen = X.mean(0); rmed = float(S.r.median())
    kv = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*2*math.pi/size
    amp = torch.randn(3, 3, generator=gen, dtype=f64, device=dev)*.03*size
    ph = torch.rand(3, generator=gen, dtype=f64, device=dev)*2*math.pi
    field = lambda Z: torch.sin(Z@kv.T+ph)@amp.T
    x0, y0 = X+field(X), Y+field(Y)+torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)*.05*rmed
    axis = torch.nn.functional.normalize(torch.randn(3, generator=gen, dtype=f64, device=dev), dim=0)
    turn = lambda Z, Q: (Z-cen)@Q.T+cen
    E0 = float(st.energy_xy(th, x0, y0))
    for ang in angles[2:]:
        Q = rot(axis, ang)
        x1, y1 = turn(x0, Q), turn(y0, Q)
        e_new = float(st.energy_xy(th, x1, y1))
        sm.matrix_quat, keep = matrix_quat_old, sm.matrix_quat
        e_old = float(st.energy_xy(th, x1, y1))
        sm.matrix_quat = keep
        print('turned %.8f rad: |E - E_0| / E_0 new %.1e, earlier %.1e' % (ang, abs(e_new-E0)/E0, abs(e_old-E0)/E0), flush=True)
    dx = torch.randn(X.shape, generator=gen, dtype=f64, device=dev); dy = torch.randn(Y.shape, generator=gen, dtype=f64, device=dev)
    for ang in (math.pi-1e-7, math.pi):
        Q = rot(axis, ang)
        x1, y1 = turn(x0, Q), turn(y0, Q)
        f = lambda s: st.energy_xy(th, x1+s*dx, y1+s*dy)
        d_ad = float(torch.func.jvp(f, (torch.zeros((), dtype=f64, device=dev),), (torch.ones((), dtype=f64, device=dev),))[1])
        h = 1e-6*rmed
        d_fd = (float(f(torch.tensor(h, dtype=f64, device=dev)))-float(f(torch.tensor(-h, dtype=f64, device=dev))))/(2*h)
        gxy = torch.func.grad(lambda z: st.energy_xy(th, z[:3*n].view(-1, 3), z[3*n:].view(-1, 3)))
        z0 = torch.cat((x0.reshape(-1), y0.reshape(-1))); z1 = torch.cat((x1.reshape(-1), y1.reshape(-1)))
        v0 = torch.cat((dx.reshape(-1), dy.reshape(-1)))
        Qv = torch.cat(((dx@Q.T).reshape(-1), (dy@Q.T).reshape(-1)))
        H0v = torch.func.jvp(gxy, (z0,), (v0,))[1]
        H1v = torch.func.jvp(gxy, (z1,), (Qv,))[1]
        QH0v = torch.cat(((H0v[:3*n].view(-1, 3)@Q.T).reshape(-1), (H0v[3*n:].view(-1, 3)@Q.T).reshape(-1)))
        print('at %.8f rad: d E / d s autodiff %.6e, central difference %.6e (relative %.1e); |H(Qx) Qv - Q H(x) v| / |H v| %.1e' % (
            ang, d_ad, d_fd, abs(d_ad-d_fd)/abs(d_fd), float((H1v-QH0v).norm()/H0v.norm())), flush=True)


if __name__ == '__main__':
    main()
