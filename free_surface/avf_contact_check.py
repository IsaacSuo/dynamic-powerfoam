"""the contacts' average vector field (ring_yield.k_ry_ground, k_ry_contact) alone, on made-up states: the potential Phi against scipy's quad of its definition, the force
against Phi's central differences, the work identity F . D = E(end) - E(start) that makes the contact energy-exact under the midpoint rule, the stored energy.
The ground: lambda from .003 to 1.6 at both ends (in contact, touching, leaving, outside); the pairs: radial, sliding past with both ends outside, deep, near
head-on, random. 11-03: every check at 1e-9 or below (the nearly-still pair's Phi at 2e-6 of 3e-9, roundoff)"""
import numpy as np
import warp as wp
from scipy.integrate import quad

from free_surface.ring_yield import GL8, k_ry_contact, k_ry_ground

wp.init()
dev = 'cuda'
arr = lambda v, dt: wp.array(v, dtype=dt, device=dev)
BPAR = np.array([2., .5, 0., 0., 0.])                                                  # (cs = 1/theta = 2, wt = theta: the end point 2 x - x_n)


def ground():
    rho, K, c0 = .007, 2.3, -.2
    rng = np.random.default_rng(3)
    cases = [(.9, .5), (.5, .9), (.3, .2), (.2, .3), (.05, .01), (.01, .05), (.003, .02), (1.3, .6), (1., .99), (1.2, .1), (.6, 1.4), (.1, 1.0001), (1.5, 1.2), (.7, .7000001), (.4, .4)]
    cases += [tuple(np.exp(rng.uniform(np.log(.004), np.log(1.6), 2))) for _ in range(40)]
    m = len(cases)
    E = lambda u: K*np.log(min(u/rho, 1.))**2
    hs = np.array([c[0] for c in cases])*rho

    def run(he):
        xn = np.zeros((m, 3)); x = np.zeros((m, 3)); xn[:, 2] = c0+hs; x[:, 2] = c0+.5*(hs+he)
        fn, Eg, terms = (wp.zeros(m, dtype=wp.float64, device=dev), wp.zeros(m, dtype=wp.float64, device=dev), wp.zeros((9, 256), dtype=wp.float64, device=dev))
        wp.launch(k_ry_ground, m, inputs=[arr(x, wp.vec3d), arr(xn, wp.vec3d), arr(np.arange(m), wp.int32), arr(np.full(m, rho), wp.float64), arr(np.full(m, K), wp.float64),
                                          arr(np.array([0., 0., 1., c0]), wp.float64), arr(BPAR, wp.float64), arr(GL8, wp.float64)], outputs=[fn, Eg, terms])
        return fn.numpy(), Eg.numpy(), terms.numpy()[6, :m]/BPAR[1]
    he = np.array([c[1] for c in cases])*rho
    fn, Eg, phi = run(he)
    eps = 1e-7*rho
    _, _, pp = run(he+eps); _, _, pm = run(he-eps)
    w = dict(phi=0., force=0., work=0., E=0.)
    for k in range(m):
        a_, b_ = hs[k], he[k]
        f = lambda u: (E(u)-E(a_))/(u-a_) if abs(u-a_) > 1e-15 else 0.
        ref = quad(f, a_, b_, points=[rho] if min(a_, b_) < rho < max(a_, b_) else None, epsabs=0, epsrel=1e-13, limit=200)[0] if b_ != a_ else 0.
        dE = E(b_)-E(a_)
        w['phi'] = max(w['phi'], abs(phi[k]-ref)/max(abs(ref), 1e-12*K))
        w['force'] = max(w['force'], abs(-(pp[k]-pm[k])/(2*eps)-fn[k])/max(abs(fn[k]), 1e-9*K/rho))
        w['work'] = max(w['work'], abs(-fn[k]*(b_-a_)-dE)/max(abs(dE), 1e-12*K))
        w['E'] = max(w['E'], abs(Eg[k]-E(b_))/max(E(b_), 1e-12*K))
    print('the ground, %d cells: Phi against quad %.1e, the push against -dPhi/dh %.1e, the work identity %.1e, E stored %.1e' % (m, w['phi'], w['force'], w['work'], w['E']), flush=True)
    return w


def pairs():
    R, K = .01, 3.7
    rng = np.random.default_rng(5)
    P = [([.9*R, 0, 0], [.5*R, 0, 0]), ([.5*R, 0, 0], [.9*R, 0, 0]), ([1.3*R, 0, 0], [.6*R, 0, 0]), ([.6*R, 0, 0], [1.3*R, 0, 0]), ([1.2*R, .5*R, 0], [-1.2*R, .5*R, 0]),
         ([.8*R, .3*R, .1*R], [-.7*R, .35*R, .05*R]), ([.05*R, .01*R, 0], [.02*R, .012*R, .003*R]), ([.9*R, 1e-3*R, 0], [.1*R, -1e-3*R, 0])]
    while len(P) < 60:
        ds = rng.normal(size=3); ds *= rng.uniform(.03, 1.3)*R/np.linalg.norm(ds)
        D = rng.normal(size=3); D *= rng.uniform(.01, .7)*R/np.linalg.norm(D)
        tu = np.clip(-(ds@D)/(D@D), 0, 1)
        if np.linalg.norm(ds+tu*D) > .01*R and np.linalg.norm(ds+D) > .01*R:
            P.append((list(ds), list(ds+D)))
    m = len(P)
    ds_all, de_all = np.array([p[0] for p in P]), np.array([p[1] for p in P])
    E = lambda L: K*np.log(L/R)**2 if L < R else 0.

    def run(de):
        xn = np.zeros((2*m, 3)); x = np.zeros((2*m, 3)); xn[1::2] = ds_all; x[1::2] = .5*(ds_all+de)
        fn, nv, Ec = wp.zeros(m, dtype=wp.float64, device=dev), wp.zeros(m, dtype=wp.vec3d, device=dev), wp.zeros(m, dtype=wp.float64, device=dev)
        terms = wp.zeros((9, 256), dtype=wp.float64, device=dev)
        wp.launch(k_ry_contact, m, inputs=[arr(x, wp.vec3d), arr(xn, wp.vec3d), arr(2*np.arange(m), wp.int32), arr(2*np.arange(m)+1, wp.int32), arr(np.full(m, R), wp.float64),
                                           arr(np.full(m, K), wp.float64), arr(BPAR, wp.float64), arr(GL8, wp.float64)], outputs=[fn, nv, Ec, terms])
        return -fn.numpy()[:, None]*nv.numpy(), Ec.numpy(), terms.numpy()[7, :m]/BPAR[1]
    F, Ec, phi = run(de_all)
    eps = 1e-7*R
    G = np.zeros((m, 3))
    for c in range(3):
        e = np.zeros(3); e[c] = eps
        G[:, c] = (run(de_all+e)[2]-run(de_all-e)[2])/(2*eps)
    w = dict(phi=0., force=0., work=0., E=0.)
    for c in range(m):
        ds, de = ds_all[c], de_all[c]; D = de-ds; Es = E(np.linalg.norm(ds))
        a2 = D@D; tu = -(ds@D)/a2; Lp = np.linalg.norm(ds+tu*D)
        pts = [tu] if 0 < tu < 1 else []
        if Lp < R:
            hw = np.sqrt(R*R-Lp*Lp)/np.sqrt(a2); pts += [p for p in (tu-hw, tu+hw) if 0 < p < 1]
        ref = quad(lambda t: (E(np.linalg.norm(ds+t*D))-Es)/t if t > 0 else 0., 0, 1, points=sorted(pts) or None, epsabs=0, epsrel=1e-12, limit=400)[0]
        dE = E(np.linalg.norm(de))-Es
        w['phi'] = max(w['phi'], abs(phi[c]-ref)/max(abs(ref), 1e-9*K))
        w['force'] = max(w['force'], np.linalg.norm(G[c]-F[c])/max(np.linalg.norm(F[c]), 1e-6*K/R))
        w['work'] = max(w['work'], abs(F[c]@D-dE)/max(abs(dE), 1e-9*K))
        w['E'] = max(w['E'], abs(Ec[c]-E(np.linalg.norm(de)))/max(E(np.linalg.norm(de)), 1e-12))
    print('the pairs, %d: Phi against quad %.1e, F against dPhi/dd_e %.1e, the work identity %.1e, E stored %.1e' % (m, w['phi'], w['force'], w['work'], w['E']), flush=True)
    return w


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    ground(); pairs()
