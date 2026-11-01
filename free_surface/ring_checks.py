"""The ring spring's claims one at a time, in order of how much rests on them (2026-11-01), no solve, no model yet: motions are prescribed. Half springs per cell: q (rest D until the cell yields), w = mu A / L. F = A M^-1, A = sum w d (x) q, M = sum w q (x) q; R, U its polar; r = d - F q.
Return, step's end: U's log deviator held at the limit (a |dev log U| <= e_y, a = det^1/3), P = exp((1 - beta) dev log U), q_new = P q + (1 - beta^2) R^T r: then F' = R U_e exactly when r = 0, and r' ~ beta^2 r.
  1. does changing only q give perfect plasticity? uniform motion on the real foam against one point integrated by itself: stress capped, plastic strain, a rigid spin laid over, the step's size, a reversal
  2. does a uniform plastic state balance? the springs' force on each site, by depth from the cap cells
  3. do shape and remainder separate? a uniform flow with site noise: false yield, the remainder's work (its reference is the rigid-plastic limit, right only for om << g: see 6)
  4. is the force a gradient at frozen R? asymmetry and curvature, elastic and flowing
  5. the elastic end's modes at rest, no bending or torsion, each cell's rotation relaxed, against the voxel FEM
  6. the remainder's law against the elastic-plastic truth, per site tet: excess dissipation over the uniform flow, three laws (two_scale_toy.py derives them): A limit on |e|, remainder by beta; B limit on |e|, remainder by beta^2; C limit on the cell's whole distortion energy, both by beta
  7. the force w (d - R q) with R read from F: not a gradient unless M is round. The whole spectrum at rest
  8. compression: a uniform state with the surface sites pinned, negative modes
  10. free flight from a deformed, spinning start: energy and angular momentum under the three forms of R
  11. R read from F with its torque's force: modes; uniform strain with the surface held: how far each form relaxes
  12. the free flight with a yield limit, form D, the law C with kappa 3/2 (spring remainders carry twice the FE's energy, so the cell's limit counts them 3/2): energy + dissipation, angular momentum, Newton
  python free_surface/ring_checks.py --foam <dir with dem_exact_single> [--test 1 2 3]"""
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp

a_ = argparse.ArgumentParser()
a_.add_argument('--foam', type=Path, required=True)
a_.add_argument('--test', type=int, nargs='+', default=[1, 2])
a_.add_argument('--ey', type=float, default=.0095)
a_.add_argument('--periods', type=float, default=6.)
a_.add_argument('--forms', default='ABC')
a_.add_argument('--rule', default=None)                                              # (test 6: one law only, 'C1.5', against the truth already logged)
args = a_.parse_args()
g = np.load(args.foam/'dem_exact_single'/'geometry.npz')
X, V, A, pairs = g['points'], g['masses'], g['areas'], g['pairs'].astype(np.int64); n = len(X)
i, j = pairs[:, 0], pairs[:, 1]; m = len(i)
D = X[j]-X[i]; L = np.linalg.norm(D, axis=1); nh = D/L[:, None]
mu = 1.                                                                              # (strains; stress in units of 2 mu)
w = mu*A/L
Ey = args.ey
a_full = np.zeros((n, 3)); np.add.at(a_full, i, -A[:, None]*nh); np.add.at(a_full, j, A[:, None]*nh)
closed = np.linalg.norm(a_full, axis=1) < 1e-9*np.median(A)
Ad = sp.coo_matrix((np.ones(2*m, dtype=bool), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
depth = np.full(n, -1); depth[~closed] = 0; front = depth == 0; k_ = 0
while (depth < 0).any() and front.any():
    k_ += 1
    nx_ = ((Ad@front.astype(np.int64)) > 0) & (depth < 0)
    depth[nx_] = k_; front = nx_
print('%s: %d cells, %d springs; cap cells %d; depth from them up to %d; yield strain %.2f %%' % (args.foam.name, n, m, int((~closed).sum()), k_, 100*Ey), flush=True)


def read(x, qi, qj):
    """F, its polar and singular form, the springs' d"""
    d = x[j]-x[i]
    Am = np.zeros((n, 3, 3)); M = np.zeros((n, 3, 3))
    np.add.at(Am, i, w[:, None, None]*d[:, :, None]*qi[:, None, :]); np.add.at(Am, j, w[:, None, None]*d[:, :, None]*qj[:, None, :])
    np.add.at(M, i, w[:, None, None]*qi[:, :, None]*qi[:, None, :]); np.add.at(M, j, w[:, None, None]*qj[:, :, None]*qj[:, None, :])
    F = Am@np.linalg.inv(M)
    Us, s, Vt = np.linalg.svd(F)
    sg = np.sign(np.linalg.det(Us@Vt)); s = s.copy(); s[:, 2] *= sg; Us = Us.copy(); Us[:, :, 2] *= sg[:, None]
    return d, F, Us@Vt, s, Vt


def ret(d, F, R, s, Vt, qi, qj, rule=2):
    """the return: new q's, beta per cell, the plastic strain increment's norm"""
    ls = np.log(np.maximum(s, 1e-12)); lm = ls.mean(1, keepdims=True); dv = ls-lm; nr = np.linalg.norm(dv, axis=1)
    a = np.exp(lm[:, 0])
    ri = d-np.einsum('fab,fb->fa', F[i], qi); rj = d-np.einsum('fab,fb->fa', F[j], qj)
    if isinstance(rule, str):                                                          # ('C' + kappa: the limit on |e|^2 + kappa sum w |r|^2 / (2 mu V))
        Wr = np.zeros(n); np.add.at(Wr, i, w*(ri*ri).sum(1)); np.add.at(Wr, j, w*(rj*rj).sum(1))
        kappa = float(rule[1:]) if len(rule) > 1 else 1.
        be = np.minimum(1., Ey/np.maximum(np.sqrt((a*nr)**2+kappa*Wr/(2*mu*V)), 1e-300)); br = be
    else:
        be = np.minimum(1., Ey/np.maximum(a*nr, 1e-300)); br = be**rule
    P = (Vt.transpose(0, 2, 1)*np.exp((1-be)[:, None]*dv)[:, None, :])@Vt
    qi2 = np.einsum('fab,fb->fa', P[i], qi)+(1-br[i])[:, None]*np.einsum('fba,fb->fa', R[i], ri)
    qj2 = np.einsum('fab,fb->fa', P[j], qj)+(1-br[j])[:, None]*np.einsum('fba,fb->fa', R[j], rj)
    return qi2, qj2, be, (1-be)*nr, ri, rj


def one_point(Fs):
    """the same law on one point: F_e,tr = dF F_e; the log deviator held; returns U_e's strain and the plastic strain's integral along the path"""
    Fe = np.eye(3); Fp = np.eye(3); ep = 0.; out = []
    for k in range(1, len(Fs)):
        dF = Fs[k]@np.linalg.inv(Fs[k-1])
        Ft = dF@Fe
        Us, s, Vt = np.linalg.svd(Ft)
        ls = np.log(s); lm = ls.mean(); dv = ls-lm; nr = np.linalg.norm(dv); a_p = np.exp(lm)
        be = min(1., Ey/max(a_p*nr, 1e-300))
        Fe = (Us*np.exp(lm+be*dv))@Vt
        ep += (1-be)*nr
        out.append((a_p*be*nr, ep, float(np.linalg.norm((Vt.T*np.exp(lm+be*dv))@Vt-np.eye(3)))))
    return np.array(out), Fe


def path(kind, N, spin=0.):
    """a uniform path: F(s) for N steps"""
    Fs = []
    for k in range(N+1):
        t = k/N
        if kind == 'shear':
            G = np.array([[1, 1.*t, 0], [0, 1, 0], [0, 0, 1.]])
        elif kind == 'reverse':
            g_ = 1.*np.sin(np.pi*t)                                                  # (out to a shear of 1 and back)
            G = np.array([[1, g_, 0], [0, 1, 0], [0, 0, 1.]])
        elif kind == 'squash':
            G = np.diag([np.exp(.4*t), np.exp(.4*t), np.exp(-.8*t)])
        th = spin*t
        Q = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        Fs.append(Q@G)
    return Fs


Xc = X-X.mean(0)
if 1 in args.test or 2 in args.test:
    for kind, N, spin in (('shear', 200, 0.), ('shear', 200, 6*np.pi), ('shear', 50, 0.), ('shear', 800, 0.), ('reverse', 400, 0.), ('reverse', 400, 6*np.pi), ('squash', 200, 0.)):
        Fs = path(kind, N, spin)
        ref, Fe_ref = one_point(Fs)
        qi, qj = D.copy(), D.copy()
        ep = np.zeros(n); worst_e = 0.; worst_ref = 0.; trace = []
        for k in range(1, N+1):
            x = Xc@Fs[k].T
            d, F, R, s, Vt = read(x, qi, qj)
            qi, qj, be, dep, ri, rj = ret(d, F, R, s, Vt, qi, qj)
            ep += dep
            d, F, R, s, Vt = read(x, qi, qj)
            ls = np.log(np.maximum(s, 1e-12)); lm = ls.mean(1, keepdims=True); e_ = np.exp(lm[:, 0])*np.linalg.norm(ls-lm, axis=1)
            worst_e = max(worst_e, float(e_.max()))
            worst_ref = max(worst_ref, float(np.abs(e_-ref[k-1, 0]).max()))
        print('%-8s %4d steps, spin %4.1f turns: strain read, largest over the run / limit %.6f; against one point: strain %.1e; plastic strain integral one point %.4f, cells median %.4f, spread %.1e; r left max %.1e of L; q volume ratio %.8f' % (
            kind, N, spin/(2*np.pi), worst_e/Ey, worst_ref, ref[-1, 1], float(np.median(ep)), float(ep.max()-ep.min()), float(max(np.abs(ri/L[:, None]).max(), np.abs(rj/L[:, None]).max())), float(np.abs(np.linalg.det(np.linalg.inv(np.eye(3)) )))), flush=True)
        if 2 in args.test:
            # the force on the sites at the end's state: each half's w m on its two ends, m = d - R q, the rotation that of its own cell
            d, F, R, s, Vt = read(x, qi, qj)
            mi = d-np.einsum('fab,fb->fa', R[i], qi); mj = d-np.einsum('fab,fb->fa', R[j], qj)
            f = np.zeros((n, 3))
            np.add.at(f, j, w[:, None]*mi); np.add.at(f, i, -w[:, None]*mi); np.add.at(f, j, w[:, None]*mj); np.add.at(f, i, -w[:, None]*mj)
            fn = np.linalg.norm(f, axis=1)
            ref_f = float(2*mu*Ey*np.median(A))                                      # (the yield stress on a median wall)
            print('    force / (yield stress x median wall), by depth: %s' % ', '.join('%d: %.1e' % (k__, fn[depth == k__].max()/ref_f) for k__ in range(min(k_+1, 4))), flush=True)
if 3 in args.test:
    # site noise at no strain: what does the ring's fit read, against the gradient the accepted model read yield on?
    import sys
    sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
    from free_surface.tl_check import build
    Dg = build(args.foam)
    h = float(np.median(V))**(1/3)
    rng = np.random.default_rng(0)
    print('site noise, no strain: the strain read, median / 90 %%, by amplitude (of h); the ring\'s fit against the gradient of the accepted model', flush=True)
    for a_n in (.001, .002, .005, .01, .02):
        u = a_n*h*rng.normal(size=X.shape)
        d, F, R, s, Vt = read(Xc+u, D, D)
        ls = np.log(np.maximum(s, 1e-12)); lm = ls.mean(1, keepdims=True); e_ = np.exp(lm[:, 0])*np.linalg.norm(ls-lm, axis=1)
        Gg = (Dg['Bg']@u.ravel()).reshape(n, 3, 3); Sg = .5*(Gg+Gg.transpose(0, 2, 1)); dg = Sg-(Sg.trace(axis1=1, axis2=2)/3)[:, None, None]*np.eye(3); eg = np.linalg.norm(dg, axis=(1, 2))
        print('  %.3f h: ring %.3f %% / %.3f %%, over the limit %d; gradient %.3f %% / %.3f %%, over the limit %d' % (a_n, 100*np.median(e_), 100*np.quantile(e_, .9), int((e_ > Ey).sum()), 100*np.median(eg), 100*np.quantile(eg, .9), int((eg > Ey).sum())), flush=True)
    # flow with a prescribed fluctuation: a shear rate and a transverse wave moving through it. The continuum's dissipation s_y <|D|> is the truth for a resolved wave; what the cells' return and the remainders' return dissipate, by the rule
    print('steady flow with a wave: dissipation over the late part / the continuum\'s; cells, remainders, total; rule beta, then beta^2', flush=True)
    N, dg_, T = 1000, .0005, 200
    qhat = np.array([.267, .535, .802]); ahat = np.cross(qhat, [1., 0, 0]); ahat /= np.linalg.norm(ahat)
    for lam in (2.5, 4., 8.):
        q_ = 2*np.pi/(lam*h)*qhat
        for tau in (.2, .5):
            om = 2*np.pi/T
            Aeta = tau*dg_*np.sqrt(2)/(np.linalg.norm(q_)*om)                         # (the wave's rate of shear, against the mean's dg_ per step, rms tau)
            out = []
            for rule in (1, 2):
                qi, qj = D.copy(), D.copy()
                Wc = Wr = Wt = 0.
                for k in range(1, N+1):
                    F_ = np.array([[1, dg_*k, 0], [0, 1, 0], [0, 0, 1.]])
                    ph = Xc@q_+om*k
                    x = Xc@F_.T+Aeta*np.sin(ph)[:, None]*ahat
                    d, F, R, s, Vt = read(x, qi, qj)
                    qi, qj, be, dep, ri, rj = ret(d, F, R, s, Vt, qi, qj, rule)
                    if k > 400:
                        # the truth for this step: the rate field's deviatoric norm at each cell, s_y |D| V
                        Gm = np.zeros((3, 3)); Gm[0, 1] = dg_
                        Gw = (-Aeta*om*np.sin(ph))[:, None, None]*ahat[None, :, None]*q_[None, None, :]
                        Dr = Gm[None]+Gw; Ds = .5*(Dr+Dr.transpose(0, 2, 1)); dvv = Ds-(Ds.trace(axis1=1, axis2=2)/3)[:, None, None]*np.eye(3)
                        Wt += float((V*2*Ey*np.linalg.norm(dvv, axis=(1, 2))).sum())
                        Wc += float((V*2*Ey*dep).sum())
                        br = be**rule
                        Wr += float((.5*w*(1-br[i]**2)*(ri*ri).sum(1)).sum()+(.5*w*(1-br[j]**2)*(rj*rj).sum(1)).sum())
                out.append('%.3f, %.3f, %.3f' % (Wc/Wt, Wr/Wt, (Wc+Wr)/Wt))
            print('  wave %.1f h, tau %.1f: %s | %s' % (lam, tau, out[0], out[1]), flush=True)
if 4 in args.test:
    # is the force a gradient? R frozen, beta live: f on the sites at x; H u by differences; u.H v against v.H u, and the curvature, elastic and in flow
    def force(x, qi, qj, Rf, live=True, befix=None):
        d = x[j]-x[i]
        Am = np.zeros((n, 3, 3)); M = np.zeros((n, 3, 3))
        np.add.at(Am, i, w[:, None, None]*d[:, :, None]*qi[:, None, :]); np.add.at(Am, j, w[:, None, None]*d[:, :, None]*qj[:, None, :])
        np.add.at(M, i, w[:, None, None]*qi[:, :, None]*qi[:, None, :]); np.add.at(M, j, w[:, None, None]*qj[:, :, None]*qj[:, None, :])
        F = Am@np.linalg.inv(M)
        H_ = Rf.transpose(0, 2, 1)@F-np.eye(3)
        Sy = .5*(H_+H_.transpose(0, 2, 1)); Om = .5*(H_-H_.transpose(0, 2, 1)); v_ = Sy.trace(axis1=1, axis2=2)/3
        et = Sy-v_[:, None, None]*np.eye(3); nr_ = np.linalg.norm(et, axis=(1, 2))
        be_ = np.minimum(1., Ey/np.maximum(nr_, 1e-300)) if live else befix
        Fp_ = (be_[:, None, None]*et+v_[:, None, None]*np.eye(3)+Om)                  # (the reading's shape part after the return; the remainder scaled by beta too)
        f_ = np.zeros((n, 3))
        for s_, q_ in ((i, qi), (j, qj)):
            rm = np.einsum('fba,fb->fa', Rf[s_], d)-q_-np.einsum('fab,fb->fa', H_[s_], q_)
            mr = np.einsum('fab,fb->fa', Fp_[s_], q_)+be_[s_][:, None]*rm
            wf = w[:, None]*np.einsum('fab,fb->fa', Rf[s_], mr)
            np.add.at(f_, j, wf); np.add.at(f_, i, -wf)
        return f_, be_

    h = float(np.median(V))**(1/3); rng = np.random.default_rng(0)
    for name, gam_ in (('elastic (shear 0.5 e_y)', .5*Ey*np.sqrt(2)), ('flowing (shear 0.1)', .1)):
        Fs = [np.array([[1, gam_*k/100, 0], [0, 1, 0], [0, 0, 1.]]) for k in range(101)]
        qi, qj = D.copy(), D.copy()
        for k in range(1, 101):
            x = Xc@Fs[k].T
            d, F, R, s, Vt = read(x, qi, qj)
            qi, qj, *_ = ret(d, F, R, s, Vt, qi, qj, 1)
        for amp in (0., 3e-4, 3e-3):
            x = Xc@Fs[-1].T+.0005*gam_*Xc@np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0.]]).T+amp*h*rng.normal(size=X.shape)
            d, F, R, s, Vt = read(x, qi, qj)
            f0, be0 = force(x, qi, qj, R)
            rs, cs = [], []
            for tr_ in range(6):
                u = rng.normal(size=X.shape); v_ = rng.normal(size=X.shape); e = 1e-7*h
                Hu = (force(x+e*u, qi, qj, R)[0]-force(x-e*u, qi, qj, R)[0])/(2*e)
                Hv = (force(x+e*v_, qi, qj, R)[0]-force(x-e*v_, qi, qj, R)[0])/(2*e)
                rs.append(abs((v_*Hu).sum()-(u*Hv).sum())/max(abs((v_*Hu).sum()), abs((u*Hv).sum())))
                cs.append(float((u*Hu).sum())/float(np.linalg.norm(u)**2))
            print('%-24s noise %.1e h: on the limit %d cells; asymmetry median %.2e, max %.2e; f here is dE/dx: u.f\'.u / |u|^2 median %.3e (positive = stable)' % (name, amp, int((be0 < 1).sum()), np.median(rs), np.max(rs), np.median(cs)), flush=True)
if 6 in args.test:
    from scipy.spatial import Delaunay
    h = float(np.median(V))**(1/3)
    Dgr = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0.]])
    dg_ = 1e-4; g_rate = dg_/np.sqrt(2)/Ey
    inner = depth >= 1
    tets = Delaunay(X).simplices; tets = tets[(depth[tets] >= 1).all(1)]
    Dm = (X[tets[:, 1:]]-X[tets[:, :1]]).transpose(0, 2, 1); Vt_ = np.abs(np.linalg.det(Dm))/6; Bm = np.linalg.inv(Dm)
    print('inner cells %d, volume %.3e; tets among them %d, volume %.3e; step %.1e, limit %.4f: relaxation rate g %.4f per step' % (int(inner.sum()), V[inner].sum(), len(tets), Vt_.sum(), dg_, Ey, g_rate), flush=True)

    def integrate(Ffun, Wg, N, skip):
        """the truth: each point its own elastic-plastic path; the energy let go per step, over the window"""
        Fe = np.tile(np.eye(3), (len(Wg), 1, 1)); Fp = np.tile(np.eye(3), (len(Wg), 1, 1)); W = 0.
        for k in range(1, N+1):
            F = Ffun(k)
            Ft = F@np.linalg.inv(Fp)@Fe; Fp = F
            Us, s, Vh = np.linalg.svd(Ft)
            ls = np.log(s); lm = ls.mean(1, keepdims=True); dv = ls-lm; nr = np.linalg.norm(dv, axis=1); a = np.exp(lm[:, 0])
            be = np.minimum(1., Ey/np.maximum(a*nr, 1e-300))
            Fe = (Us*np.exp(lm+be[:, None]*dv)[:, None, :])@Vh
            if k > skip:
                W += float((Wg*mu*(a*nr)**2*(1-be**2)).sum())
        return W/(N-skip)

    def model(pos, rule, N, skip):
        """the springs' energy let go by the return, on the inner cells' own halves; the remainder's share at the end"""
        qi, qj = D.copy(), D.copy(); W = 0.
        for k in range(1, N+1):
            x = pos(k)
            d, F, R, s, Vt = read(x, qi, qj)
            qi2, qj2, be, dep, ri, rj = ret(d, F, R, s, Vt, qi, qj, rule)
            if k > skip:
                for own, s_, q0, q1 in ((inner[i], i, qi, qi2), (inner[j], j, qj, qj2)):
                    m0 = d-np.einsum('fab,fb->fa', R[s_], q0); m1 = d-np.einsum('fab,fb->fa', R[s_], q1)
                    W += float((.5*w*((m0*m0).sum(1)-(m1*m1).sum(1)))[own].sum())
            qi, qj = qi2, qj2
        Wr = np.zeros(n); np.add.at(Wr, i, w*(ri*ri).sum(1)); np.add.at(Wr, j, w*(rj*rj).sum(1))
        ls = np.log(np.maximum(s, 1e-12)); lm = ls.mean(1, keepdims=True); e2 = (np.exp(lm[:, 0])*np.linalg.norm(ls-lm, axis=1))**2; r2 = Wr/(2*mu*V)
        return W/(N-skip), float(np.median((r2/(e2+r2))[inner]))

    Fu = lambda k: np.tile(np.eye(3)+dg_*k*Dgr, (1, 1, 1))
    pos_u = lambda k: Xc@(np.eye(3)+dg_*k*Dgr).T
    N0, S0 = 400, 200
    B_tet = integrate(lambda k: np.tile(np.eye(3)+dg_*k*Dgr, (len(tets), 1, 1)), Vt_, N0, S0)/Vt_.sum()
    B_site = integrate(lambda k: np.tile(np.eye(3)+dg_*k*Dgr, (n, 1, 1)), V*inner, N0, S0)/V[inner].sum()
    B_mod = model(pos_u, 1, N0, S0)[0]
    print('uniform flow, energy let go per step per volume: tets %.4e, sites %.4e; the springs on the inner cells %.4e per unit of their volume %.4e' % (B_tet, B_site, B_mod/V[inner].sum(), B_mod), flush=True)
    qhat = np.array([.267, .535, .802]); ahat = np.cross(qhat, [1., 0, 0]); ahat /= np.linalg.norm(ahat)
    rng = np.random.default_rng(0)
    xi_ = rng.normal(size=X.shape); phi_ = rng.uniform(0, 2*np.pi, n)

    def compare(label, pos, Fsite, N, skip, x_):
        if args.rule is not None:                                                      # (one law against the truth already logged)
            W_, s_ = model(pos, args.rule, N, skip)
            print('  %s, om / g %.2f: law %s excess %.4f; remainder share of the distortion energy, median %.3f' % (label, x_, args.rule, W_/B_mod-1, s_), flush=True)
            return
        Ftet = lambda k: (lambda x_k: (x_k[tets[:, 1:]]-x_k[tets[:, :1]]).transpose(0, 2, 1)@Bm)(pos(k))
        Xt = integrate(Ftet, Vt_, N, skip)/Vt_.sum()/B_tet-1
        Xs = integrate(Fsite, V*inner, N, skip)/V[inner].sum()/B_site-1 if Fsite is not None else float('nan')
        out = []; sh = []
        for rule in (1, 2, 'C'):
            W_, s_ = model(pos, rule, N, skip)
            out.append(W_/B_mod-1); sh.append(s_)
        kb = 4*(1+x_**2)/(x_**2+4)
        print('  %s, om / g %.2f: excess, truth on tets %.4f, on sites %.4f; models A %.4f, B %.4f, C %.4f; over the tets A %.2f, B %.2f, C %.2f (the remainder alone would give 2, %.2f, 1); remainder share of the distortion energy, median A %.3f, B %.3f, C %.3f' % (label, x_, Xt, Xs, *out, *[o/Xt for o in out], kb, *sh), flush=True)

    for lam in (2.5, 4., 8.):
        q_ = 2*np.pi/(lam*h)*qhat
        for x_ in (.2, 1., 5.):
            om = x_*g_rate; T = 2*np.pi/om
            tau = .5
            Aeta = tau*dg_*np.sqrt(2)/(np.linalg.norm(q_)*om)
            win = int(round(max(1, np.ceil(360/T))*T)); skip = 200; N = skip+win
            pos = lambda k: Xc@(np.eye(3)+dg_*k*Dgr).T+Aeta*np.sin(Xc@q_+om*k)[:, None]*ahat
            Fs = lambda k: (np.eye(3)+dg_*k*Dgr)[None]+(Aeta*np.cos(Xc@q_+om*k))[:, None, None]*ahat[None, :, None]*q_[None, None, :]
            compare('wave %.1f h, tau %.1f' % (lam, tau), pos, Fs, N, skip, x_)
    # rough: every site its own direction and phase, scaled so the tets' rate of shear has rms tau times the mean's
    G0 = ((xi_*np.cos(phi_)[:, None])[tets[:, 1:]]-(xi_*np.cos(phi_)[:, None])[tets[:, :1]]).transpose(0, 2, 1)@Bm
    S0_ = .5*(G0+G0.transpose(0, 2, 1)); d0 = S0_-(S0_.trace(axis1=1, axis2=2)/3)[:, None, None]*np.eye(3)
    R0 = np.sqrt((Vt_*(d0*d0).sum((1, 2))).sum()/Vt_.sum())*np.sqrt(2)                  # (rms over phases: cos and sin parts)
    for x_ in (.2, 1., 5.):
        for tau in (.5, 1.):
            om = x_*g_rate; T = 2*np.pi/om
            amp = tau*(dg_/np.sqrt(2))/(om*R0/np.sqrt(2))
            win = int(round(max(1, np.ceil(360/T))*T)); skip = 200; N = skip+win
            pos = lambda k: Xc@(np.eye(3)+dg_*k*Dgr).T+amp*xi_*np.sin(om*k+phi_)[:, None]
            compare('rough, %.2e h, tau %.1f' % (amp/h, tau), pos, None, N, skip, x_)
if set(args.test) & {5, 7, 8, 10, 11, 12}:
    # modes at rest of the elastic end: springs with each cell's own rotation relaxed, gg's lam term and variance, the outer layer on gg's F; no bending, no torsion. Against the voxel FEM
    import sys, time
    sys.path.insert(0, '/home/fangsuo/py/dynamic-powerfoam')
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse.linalg import LinearOperator, cg, eigsh, gmres, splu
    from experiments.dem_fem_mode_match import interpolate
    from free_surface.tl_check import build
    E_m, nu_m = 23.00838, .3008
    mu_m, lam_m = E_m/(2*(1+nu_m)), E_m*nu_m/((1+nu_m)*(1-2*nu_m))
    Dg = build(args.foam); Bg = Dg['Bg']; Mc = .5*(Dg['Mc']+Dg['Mc'].transpose(0, 2, 1))
    wm = mu_m*A/L
    e_ = np.arange(m)
    rw, cl_, vl = [], [], []
    for a in range(3):
        rw += [3*e_+a, 3*e_+a]; cl_ += [3*j+a, 3*i+a]; vl += [np.ones(m), -np.ones(m)]
    C1 = sp.coo_matrix((np.concatenate(vl), (np.concatenate(rw), np.concatenate(cl_))), shape=(3*m, 3*n)).tocsr()
    Ks = (C1.T@sp.diags(np.repeat(2*wm, 3))@C1).tocsr()                                # (two halves)
    # a cell's rotation relaxed: delta m = du - w x D; E_i = sum 1/2 w |du - w x D|^2, min over w: minus C^T H^-1 C per cell
    H3 = np.zeros((n, 3, 3)); ro, co, va = [], [], []
    Dx = np.zeros((m, 3, 3)); Dx[:, 0, 1], Dx[:, 0, 2], Dx[:, 1, 0], Dx[:, 1, 2], Dx[:, 2, 0], Dx[:, 2, 1] = -D[:, 2], D[:, 1], D[:, 2], -D[:, 0], -D[:, 1], D[:, 0]
    q2 = wm[:, None, None]*(Dx.transpose(0, 2, 1)@Dx)
    np.add.at(H3, i, q2); np.add.at(H3, j, q2)
    Hi = np.linalg.pinv(H3)
    # the coupling: dE_i = - w (du . (W x D)) = w W . (D x du): row of cell i, 3 by 3 on du_e
    Cc = wm[:, None, None]*Dx                                                          # (sign and orientation: D x du = Dx du)
    rows, cols, vals = [], [], []
    for s_ in (i, j):
        for a in range(3):
            for b in range(3):
                rows += [3*s_+a, 3*s_+a]; cols += [3*j+b, 3*i+b]; vals += [Cc[:, a, b], -Cc[:, a, b]]
    Cr = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3*n, 3*n)).tocsr()
    Hinv = sp.block_diag([Hi[s_] for s_ in range(n)], format='csr')
    Kr = (Ks-Cr.T@Hinv@Cr).tocsr()
    I9 = np.eye(3).ravel()
    Dl = sp.block_diag([V[s_]*lam_m*np.outer(I9, I9) for s_ in range(n)], format='csr')
    Kl = (Bg.T@Dl@Bg).tocsr()
    tr_ = sp.coo_matrix((np.ones(3*n), (np.repeat(np.arange(n), 3), (9*np.arange(n)[:, None]+np.array([0, 4, 8])).ravel())), shape=(n, 9*n)).tocsr()
    dT = sp.coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[e_, e_], np.r_[i, j])), shape=(m, n)).tocsr()
    Kv = ((dT@tr_@Bg).T@sp.diags(2*lam_m*A*L/24)@(dT@tr_@Bg)).tocsr()
    # the outer layer: mu tr(e Mc e), e = sym G
    blocks = np.zeros((n, 9, 9))
    Sym = np.zeros((9, 9))
    for a in range(3):
        for b in range(3):
            Sym[3*a+b, 3*a+b] += .5; Sym[3*a+b, 3*b+a] += .5
    for a in range(3):
        blocks[:, 3*a:3*a+3, 3*a:3*a+3] = 2*mu_m*Mc
    blocks = Sym[None]@blocks@Sym[None]
    Do = sp.block_diag([blocks[s_] for s_ in range(n)], format='csr')
    Ko = (Bg.T@Do@Bg).tocsr()
    Md = sp.diags(np.repeat(V, 3))
    if (args.foam/'fem_modes_h0.0015.npz').exists():
        c_ = np.load(args.foam/'fem_modes_h0.0015.npz'); nm = min(26, c_['V'].shape[1]); f_fem, V_fem = c_['f'][:nm], c_['V'][:, :nm]
        F_ = dict(nodes=c_['nodes'], node_grid=c_['node_grid'], lo=c_['lo'], h=float(c_['h']), masses=c_['masses'])
        Ux, _ = interpolate(F_, V_fem, X); UF = Ux.reshape(-1, nm)
        w1 = (2*np.pi*f_fem[0])**2
    else:
        w1 = mu_m/float(V.sum())**(2/3)                                                # (no FEM here: a scale only, the shear wave's over the body)

    def report(name, K):
        t0 = time.time()
        sigma = -1e-3*w1
        lu = splu((K-sigma*Md).tocsc())
        OP = LinearOperator(K.shape, matvec=lu.solve, dtype=np.float64)
        lam_, vec = eigsh(K, k=70, M=Md, sigma=sigma, OPinv=OP, which='LM', tol=1e-8)
        o = np.argsort(lam_); lam_, vec = lam_[o], vec[:, o]
        nz = int((np.abs(lam_) < 1e-6*w1).sum())
        fr = np.sqrt(np.clip(lam_, 0, None))/(2*np.pi)
        Z = vec[:, nz:].reshape(n, 3, -1)
        num = np.einsum('n,nkd,nkf->df', V, Z, Ux)**2
        den = np.outer(np.einsum('n,nkd,nkd->d', V, Z, Z), np.einsum('n,nkf,nkf->f', V, Ux, Ux))
        Mac = num/den
        ri_, ci_ = linear_sum_assignment(-Mac)
        good = [fr[nz+a__]/f_fem[b__] for a__, b__ in zip(ri_, ci_) if Mac[a__, b__] >= .8]
        sf = np.array([UF[:, k]@(K@UF[:, k])/(UF[:, k]@(Md@UF[:, k]))/(2*np.pi*f_fem[k])**2 for k in range(nm)])
        print('  %-40s zeros %d, negative %d; first modes / FEM %s; along the FEM modes, frequency^2 / FEM median %.3f (%.3f..%.3f); paired %d of %d, frequency / FEM median %s; below half the FEM first %d (%.0f s)' % (
            name, nz, int((lam_ < -1e-6*w1).sum()), np.array2string(fr[nz:nz+6]/f_fem[:6], precision=3), np.median(sf), sf.min(), sf.max(), len(good), nm, '%.3f' % np.median(good) if good else '-', int(((lam_ > 1e-6*w1) & (fr < .5*f_fem[0])).sum()), time.time()-t0), flush=True)

    if 5 in args.test:
        print('modes at rest on %s:' % args.foam.name, flush=True)
        report('springs, rotation relaxed, lam, variance', (Kr+Kl+Kv).tocsr())
        report('the same + outer layer', (Kr+Kl+Kv+Ko).tocsr())
    if set(args.test) & {7, 8, 10, 11, 12}:
        # the force w (d - R q) with R read from F, not the energy's own: omega = 1/2 sum w [M^-1 D]x du, f = L u - C^T omega. Not a gradient unless M is round: the torque skew(e M) has no work to go to. The whole spectrum, dense
        Mr = np.zeros((n, 3, 3)); q2 = wm[:, None, None]*D[:, :, None]*D[:, None, :]
        np.add.at(Mr, i, q2); np.add.at(Mr, j, q2)
        Mri = np.linalg.inv(Mr)
        rows, cols, vals = [], [], []
        for s_ in (i, j):
            v_ = np.einsum('fab,fb->fa', Mri[s_], D)
            Vx = np.zeros((m, 3, 3)); Vx[:, 0, 1], Vx[:, 0, 2], Vx[:, 1, 0], Vx[:, 1, 2], Vx[:, 2, 0], Vx[:, 2, 1] = -v_[:, 2], v_[:, 1], v_[:, 2], -v_[:, 0], -v_[:, 1], v_[:, 0]
            Tb = .5*wm[:, None, None]*Vx
            for a in range(3):
                for b in range(3):
                    rows += [3*s_+a, 3*s_+a]; cols += [3*j+b, 3*i+b]; vals += [Tb[:, a, b], -Tb[:, a, b]]
        TF = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3*n, 3*n)).tocsr()
        Om = np.array([.3, -.5, .8]); ur = np.cross(Om, Xc).ravel()
        print('checks: a rigid spin reads the spin: err %.1e; its force: read from F %.1e, relaxed %.1e (of the springs\' %.1e)' % (np.abs((TF@ur).reshape(n, 3)-Om).max(), np.abs(Ks@ur-Cr.T@(TF@ur)).max(), np.abs(Kr@ur).max(), np.abs(Ks).max()*np.abs(ur).max()), flush=True)
    if 11 in args.test:
        # R read from F with the force of its torque, the energy's whole gradient (as the old DEM by Warp's adjoint): E = 1/2 sum w |du - omega_F x D|^2, K = L - C^T T - T^T C + T^T H T. Uniform strain with the surface sites held: how far do the inside sites relax, in each form?
        H3b = sp.block_diag([H3[s_] for s_ in range(n)], format='csr')
        Kc1 = (Ks-Cr.T@TF-TF.T@Cr+TF.T@H3b@TF).tocsr()
        # one rotation per cell, the minimum of the whole cell: its half springs and its outer layer mu |(F_g - R) Mc^1/2|^2 together, A_cell = sum w d (x) q + F_g 2 mu Mc. The outer layer as virtual springs: (C_o u)_a = eps_abc (G Mo)_cb, H_o = tr Mo - Mo
        Mo = 2*mu_m*Mc
        Ho = np.trace(Mo, axis1=1, axis2=2)[:, None, None]*np.eye(3)-Mo
        Hf_ = np.linalg.pinv(H3+Ho)
        eps = np.zeros((3, 3, 3)); eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1; eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
        rows, cols, vals = [], [], []
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    if eps[a, b, c] != 0:
                        for d_ in range(3):
                            rows.append(3*np.arange(n)+a); cols.append(9*np.arange(n)+3*c+d_); vals.append(eps[a, b, c]*Mo[:, d_, b])
        Co = (sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3*n, 9*n)).tocsr()@Bg).tocsr()
        Ctot = (Cr+Co).tocsr()
        print('whole cell: a rigid spin: C u - H omega %.1e (of %.1e); eigenvalues of H / V (a full ring reads 4 with w = mu A / L, mu = 1 in these units): smallest, cap cells, 1 / 50 %% %s; before the outer layer %s' % (np.abs(Ctot@ur-((H3+Ho)@Om).ravel()).max(), np.abs(((H3+Ho)@Om).ravel()).max(), np.array2string(np.quantile(np.linalg.eigvalsh(H3+Ho)[~closed, 0]/(mu_m*V[~closed]), [.01, .5]), precision=3), np.array2string(np.quantile(np.linalg.eigvalsh(H3)[~closed, 0]/(mu_m*V[~closed]), [.01, .5]), precision=3)), flush=True)
        blk = np.zeros((n, 9, 9))
        for a in range(3):
            blk[:, 3*a:3*a+3, 3*a:3*a+3] = Mo
        KoF = (Bg.T@sp.block_diag([blk[s_] for s_ in range(n)], format='csr')@Bg).tocsr()
        Kf = (Ks+KoF-Ctot.T@sp.block_diag([Hf_[s_] for s_ in range(n)], format='csr')@Ctot).tocsr()
        print('  rigid spin in the whole cell\'s K: %.1e' % np.abs(Kf@ur).max(), flush=True)
        if (args.foam/'fem_modes_h0.0015.npz').exists():
            print('modes at rest on %s:' % args.foam.name, flush=True)
            report('read from F + torque, lam, variance', (Kc1+Kl+Kv).tocsr())
            report('read from F + torque + outer layer', (Kc1+Kl+Kv+Ko).tocsr())
            report('whole cell\'s rotation, lam, variance', (Kf+Kl+Kv).tocsr())
        free = np.where(depth >= 1)[0]; fd = (3*free[:, None]+np.arange(3)).ravel()
        print('uniform strain, %d of %d sites free: energy after relaxing / before; the free sites\' move / (|strain| h) median, 99 %%; the ring\'s strain read, |e - strain| / |strain| median, 99 %%' % (len(free), n), flush=True)
        h = float(np.median(V))**(1/3)
        Sh1, Sh2, St = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0.]]), np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0.]]), np.diag([1, -.5, -.5])
        for nm_, H0 in (('shear xy', Sh1), ('shear xz + yz', Sh2), ('stretch', St)):
            ue = (Xc@H0.T).ravel(); Hn = np.linalg.norm(H0)
            for form, K_ in (('relaxed rotation', Kr), ('F + torque', Kc1), ('whole cell', Kf)):
                Kt = (K_+Kl+Kv).tocsr()
                rf = (Kt@ue)[fd]
                Kff = Kt[fd][:, fd].tocsr()
                if len(fd) > 8000:                                                     # (a direct factor of this stencil would not fit)
                    du, inf_ = cg(Kff, -rf, M=sp.diags(1/Kff.diagonal()), rtol=1e-7, maxiter=20000)
                else:
                    du = -splu(Kff.tocsc()).solve(rf)
                E0 = .5*ue@(Kt@ue); Er = E0+.5*rf@du
                u_ = ue.copy(); u_[fd] += du
                x_ = (X.ravel()+u_).reshape(n, 3)
                d_ = x_[j]-x_[i]
                Am = np.zeros((n, 3, 3)); q_ = wm[:, None, None]*d_[:, :, None]*D[:, None, :]
                np.add.at(Am, i, q_); np.add.at(Am, j, q_)
                Fr = Am@Mri-np.eye(3)
                Sr = .5*(Fr+Fr.transpose(0, 2, 1))
                er = np.linalg.norm((Sr-H0)[depth >= 1], axis=(1, 2))/Hn
                mv = np.linalg.norm(du.reshape(-1, 3), axis=1)/(Hn*h)
                deep_ = depth[depth >= 1] >= 2
                print('  %-14s %-18s energy %.4f; moved median %.3f, 99%% %.3f; strain read off by median %.3f, 99%% %.3f; depth 2 and more (%d cells) median %.3f, 99%% %.3f' % (nm_, form, Er/E0, np.median(mv), np.quantile(mv, .99), np.median(er), np.quantile(er, .99), int(deep_.sum()), np.median(er[deep_]) if deep_.any() else 0., np.quantile(er[deep_], .99) if deep_.any() else 0.), flush=True)
    if 10 in args.test or 12 in args.test:
        # free flight from a deformed, spinning start, nothing else, the midpoint. Midpoint conserves angular momentum exactly for a force with no net moment, energy to second order for a gradient. Three forms of R: A polar(A), B polar(F) plain, C polar(F) with its torque's force
        def polar_(F):
            Us, s, Vt = np.linalg.svd(F)
            sg = np.sign(np.linalg.det(Us@Vt)); s = s.copy(); s[:, 2] *= sg; Us = Us.copy(); Us[:, :, 2] *= sg[:, None]
            return Us@Vt, s, Vt

        def cross_m(v):
            Z = np.zeros(v.shape[:-1]+(3, 3)); Z[..., 0, 1], Z[..., 0, 2], Z[..., 1, 0], Z[..., 1, 2], Z[..., 2, 0], Z[..., 2, 1] = -v[..., 2], v[..., 1], v[..., 2], -v[..., 0], -v[..., 1], v[..., 0]
            return Z

        Mcs = Mc
        Mo = 2*mu_m*Mc
        Xv = X.ravel()

        def eg(x, form):
            d = x[j]-x[i]
            Am = np.zeros((n, 3, 3)); q_ = wm[:, None, None]*d[:, :, None]*D[:, None, :]
            np.add.at(Am, i, q_); np.add.at(Am, j, q_)
            F = Am@Mri
            Fg = (Bg@(x.ravel()-Xv)).reshape(n, 3, 3)+np.eye(3)
            if form == 'A':
                R, _, _ = polar_(Am)
            elif form == 'D':
                R, _, _ = polar_(Am+Fg@Mo)                                             # (the whole cell's own rotation: the outer layer's virtual springs in the sum)
            else:
                R, s, Vt = polar_(F)
            qi, qj = np.einsum('fab,fb->fa', R[i], D), np.einsum('fab,fb->fa', R[j], D)
            mi, mj = d-qi, d-qj
            g_ = np.zeros((n, 3))
            fw = wm[:, None]*(mi+mj)
            np.add.at(g_, j, fw); np.add.at(g_, i, -fw)
            Es = .5*float((wm*((mi*mi).sum(1)+(mj*mj).sum(1))).sum())
            tq = np.zeros((n, 3)); np.add.at(tq, i, wm[:, None]*np.cross(qi, mi)); np.add.at(tq, j, wm[:, None]*np.cross(qj, mj))
            if form == 'C':
                tm = np.einsum('nba,nb->na', R, tq)
                U = (Vt.transpose(0, 2, 1)*s[:, None, :])@Vt
                Y = np.linalg.solve(s.sum(1)[:, None, None]*np.eye(3)-U, tm[:, :, None])[:, :, 0]
                RY = R@cross_m(Y)
                for s_ in (i, j):
                    e_x = -wm[:, None]*np.einsum('fab,fb->fa', RY[s_], np.einsum('fab,fb->fa', Mri[s_], D))
                    np.add.at(g_, j, e_x); np.add.at(g_, i, -e_x)
            Rg, sg_, Vg = polar_(Fg)
            Ug = Rg@Vg.transpose(0, 2, 1)
            J = np.prod(sg_, axis=1); J0 = .05
            Ls = np.where(J > J0, np.log(np.maximum(J, 1e-300)), np.log(J0)+(J-J0)/J0)
            dL = np.where(J > J0, 1/np.where(J > J0, J, 1.), 1/J0)
            cof = (Ug*np.stack([sg_[:, 1]*sg_[:, 2], sg_[:, 0]*sg_[:, 2], sg_[:, 0]*sg_[:, 1]], 1)[:, None, :])@Vg
            dLF = dL[:, None, None]*cof
            Ev = .5*lam_m*float((V*Ls*Ls).sum())
            Pt = (V*lam_m*Ls)[:, None, None]*dLF
            dl = Ls[i]-Ls[j]; cv = lam_m*A*L/24
            Evr = float((cv*dl*dl).sum())
            gL = np.zeros(n); np.add.at(gL, i, 2*cv*dl); np.add.at(gL, j, -2*cv*dl)
            Pt = Pt+gL[:, None, None]*dLF
            if form == 'D':
                FmR = Fg-R
                Po = 2*mu_m*FmR@Mcs
                Eo = .5*float((Po*FmR).sum())
                Pt = Pt+Po
            else:
                FmR = Fg-Rg
                Po = 2*mu_m*FmR@Mcs
                Eo = .5*float((Po*FmR).sum())
                Q = Po@Rg.transpose(0, 2, 1)
                tau_o = -np.stack([Q[:, 1, 2]-Q[:, 2, 1], Q[:, 2, 0]-Q[:, 0, 2], Q[:, 0, 1]-Q[:, 1, 0]], 1)
                Ug_ = (Vg.transpose(0, 2, 1)*sg_[:, None, :])@Vg
                Yo = np.linalg.solve(sg_.sum(1)[:, None, None]*np.eye(3)-Ug_, np.einsum('nba,nb->na', Rg, tau_o)[:, :, None])[:, :, 0]
                Pt = Pt+Po-Rg@cross_m(Yo)
            g_ = g_+(Bg.T@Pt.ravel()).reshape(n, 3)
            return Es+Ev+Evr+Eo, g_, float(np.linalg.norm(np.cross(x-x.mean(0), g_).sum(0)))

        h = float(np.median(V))**(1/3)
        T1 = 2*np.pi/np.sqrt(w1)
        rng = np.random.default_rng(3)
        # gradient check at a deformed state, all three
        phi = Ux[:, :, 0]+Ux[:, :, 2]
        phi = phi/np.abs(phi).max()
        # amplitude from the strain it reads: 99 % at about 10 %
        d_ = phi[j]-phi[i]; Am = np.zeros((n, 3, 3)); q_ = wm[:, None, None]*d_[:, :, None]*D[:, None, :]
        np.add.at(Am, i, q_); np.add.at(Am, j, q_)
        G_ = Am@Mri; S_ = .5*(G_+G_.transpose(0, 2, 1)); e_ = np.linalg.norm(S_-(S_.trace(axis1=1, axis2=2)/3)[:, None, None]*np.eye(3), axis=(1, 2))
        amp = .1/np.quantile(e_, .99)
        x_d = X+amp*phi                                                                # (a deformed state, for the gradient check and the force's scale)
        x0 = X.copy()
        mm = np.maximum(V, .1*np.median(V))                                            # (inertia only: a very small cell's period is no matter here, and Newton's way is easier)
        Om = 2*np.pi/(5*T1)*np.array([.3, .5, .8])/np.linalg.norm([.3, .5, .8])
        cm0 = (mm[:, None]*x0).sum(0)/mm.sum()
        # the start: at rest shape, moving: the spin and a mode's velocity, so the strain comes up smoothly to about 5 %
        phi0 = phi-(mm[:, None]*phi).sum(0)/mm.sum()
        v0 = np.cross(Om, x0-cm0)+.5*amp*np.sqrt(w1)*phi0
        print('free flight on %s: T1 %.4f s; start at the rest shape with a mode\'s velocity (amplitude %.3e m, %.2f h, strain 99 %% about 5 %%), spin one turn per 5 T1; mass floor 0.1 median' % (args.foam.name, T1, .5*amp, .5*amp/h), flush=True)
        u_ = rng.normal(size=x0.shape); e_fd = 1e-6*h
        x0, x_d = x_d, x0                                                              # (the gradient check at the deformed state)
        for form in 'ABCD':
            E_, g_, tq0 = eg(x0, form)
            fd_ = (eg(x0+e_fd*u_, form)[0]-eg(x0-e_fd*u_, form)[0])/(2*e_fd)
            print('  form %s: energy %.5e, gradient along a random direction %.8e, by differences %.8e, relative gap %.2e; net moment of the whole force %.2e, net force %.1e' % (form, E_, (g_*u_).sum(), fd_, abs((g_*u_).sum()-fd_)/abs(fd_), tq0, np.linalg.norm(g_.sum(0))), flush=True)
        x0, x_d = x_d, x0
        dt = T1/60; N = int(args.periods*60)
        Mm = np.repeat(mm*4/dt**2, 3)
        Kp = (Ks+Kl+Kv+Ko+sp.diags(Mm)).tocsc()
        lu = splu(Kp)
        Pre = LinearOperator(Kp.shape, matvec=lu.solve, dtype=np.float64)
        if 12 in args.test:
            # the same flight with a yield limit, form D. State: the half springs' q, each cell's P for its outer layer (virtual springs: q_k = P D_k, d_k = F_g D_k). Read at x: F = A M^-1 of the whole cell; the return from the step's start state; R = polar(A) of the returned state; the force w (d - R q_new), (F_g - R P_new) Mo. Force at the midpoint from a return at the midpoint, the state at the end from a return at the end
            kappa = float(args.rule[1:]) if args.rule else 1.5

            def ret_w(x, qi, qj, P):
                d = x[j]-x[i]
                Fg = (Bg@(x.ravel()-Xv)).reshape(n, 3, 3)+np.eye(3)
                Am = np.zeros((n, 3, 3)); Mt = np.zeros((n, 3, 3))
                np.add.at(Am, i, wm[:, None, None]*d[:, :, None]*qi[:, None, :]); np.add.at(Am, j, wm[:, None, None]*d[:, :, None]*qj[:, None, :])
                np.add.at(Mt, i, wm[:, None, None]*qi[:, :, None]*qi[:, None, :]); np.add.at(Mt, j, wm[:, None, None]*qj[:, :, None]*qj[:, None, :])
                MoP = Mo@P.transpose(0, 2, 1)
                F = (Am+Fg@MoP)@np.linalg.inv(Mt+P@MoP)
                Rf, s, Vt = polar_(F)
                ls = np.log(np.maximum(s, 1e-12)); lm = ls.mean(1, keepdims=True); dv = ls-lm; nr = np.linalg.norm(dv, axis=1); a = np.exp(lm[:, 0])
                ri, rj = d-np.einsum('fab,fb->fa', F[i], qi), d-np.einsum('fab,fb->fa', F[j], qj)
                ro = Fg-F@P
                Wr = np.einsum('nab,nbc,nac->n', ro, Mo, ro)
                np.add.at(Wr, i, wm*(ri*ri).sum(1)); np.add.at(Wr, j, wm*(rj*rj).sum(1))
                be = np.minimum(1., Ey/np.maximum(np.sqrt((a*nr)**2+kappa*Wr/(2*mu_m*V)), 1e-300))
                Ps = (Vt.transpose(0, 2, 1)*np.exp((1-be)[:, None]*dv)[:, None, :])@Vt
                qi2 = np.einsum('fab,fb->fa', Ps[i], qi)+(1-be[i])[:, None]*np.einsum('fba,fb->fa', Rf[i], ri)
                qj2 = np.einsum('fab,fb->fa', Ps[j], qj)+(1-be[j])[:, None]*np.einsum('fba,fb->fa', Rf[j], rj)
                P2 = Ps@P+(1-be)[:, None, None]*(Rf.transpose(0, 2, 1)@ro)
                return qi2, qj2, P2, be, Fg, np.sqrt(Wr/(2*mu_m*V)), a*nr

            def eg_state(x, qi, qj, P, Fg):
                """energy and gradient at x with this state, R the whole cell's minimum"""
                d = x[j]-x[i]
                Am = np.zeros((n, 3, 3))
                np.add.at(Am, i, wm[:, None, None]*d[:, :, None]*qi[:, None, :]); np.add.at(Am, j, wm[:, None, None]*d[:, :, None]*qj[:, None, :])
                R, _, _ = polar_(Am+Fg@Mo@P.transpose(0, 2, 1))
                mi, mj = d-np.einsum('fab,fb->fa', R[i], qi), d-np.einsum('fab,fb->fa', R[j], qj)
                g_ = np.zeros((n, 3))
                fw = wm[:, None]*(mi+mj)
                np.add.at(g_, j, fw); np.add.at(g_, i, -fw)
                Es = .5*float((wm*((mi*mi).sum(1)+(mj*mj).sum(1))).sum())
                Rg, sg_, Vg = polar_(Fg)
                Ug = Rg@Vg.transpose(0, 2, 1)
                J = np.prod(sg_, axis=1); J0 = .05
                Ls = np.where(J > J0, np.log(np.maximum(J, 1e-300)), np.log(J0)+(J-J0)/J0)
                dL = np.where(J > J0, 1/np.where(J > J0, J, 1.), 1/J0)
                cof = (Ug*np.stack([sg_[:, 1]*sg_[:, 2], sg_[:, 0]*sg_[:, 2], sg_[:, 0]*sg_[:, 1]], 1)[:, None, :])@Vg
                dLF = dL[:, None, None]*cof
                Ev = .5*lam_m*float((V*Ls*Ls).sum())
                Pt = (V*lam_m*Ls)[:, None, None]*dLF
                dl = Ls[i]-Ls[j]; cv = lam_m*A*L/24
                Evr = float((cv*dl*dl).sum())
                gL = np.zeros(n); np.add.at(gL, i, 2*cv*dl); np.add.at(gL, j, -2*cv*dl)
                Pt = Pt+gL[:, None, None]*dLF
                FmR = Fg-R@P
                Po = FmR@Mo
                Eo = .5*float((Po*FmR).sum())
                Pt = Pt+Po
                g_ = g_+(Bg.T@Pt.ravel()).reshape(n, 3)
                return Es+Ev+Evr+Eo, g_

            def g_pl(y_, qi, qj, P):
                qi2, qj2, P2, _, Fg, _, _ = ret_w(y_, qi, qj, P)
                return eg_state(y_, qi2, qj2, P2, Fg)[1]

            x, v = x0.copy(), v0.copy()
            qi, qj, Pc = D.copy(), D.copy(), np.tile(np.eye(3), (n, 1, 1))
            t0 = time.time()
            Fg0 = (Bg@(x.ravel()-Xv)).reshape(n, 3, 3)+np.eye(3)
            Ep0 = eg_state(x, qi, qj, Pc, Fg0)[0]; Ke0 = .5*float((mm*(v*v).sum(1)).sum()); E0 = Ep0+Ke0
            cm = (mm[:, None]*x).sum(0)/mm.sum(); vm = (mm[:, None]*v).sum(0)/mm.sum()
            L0 = (mm[:, None]*np.cross(x-cm, v-vm)).sum(0)
            gs = np.abs(eg(x_d, 'D')[1]).max(); nit = 0; nls = 0; nbad = 0; Wp = 0.; Wc = 0.
            print('with a yield limit %.2f %%, kappa %.2f, the part foam at this E; start energy %.3e, of which the spin\'s about %.3e' % (100*Ey, kappa, E0, .5*float((mm*(np.cross(Om, x-cm)**2).sum(1)).sum())), flush=True)
            for k in range(1, N+1):
                y = x+.5*dt*v

                def resid(y_):
                    return (mm[:, None]*4/dt**2)*(y_-x-.5*dt*v)+g_pl(y_, qi, qj, Pc)

                r_ = resid(y); rn = np.linalg.norm(r_)
                for it in range(40):
                    if np.abs(r_).max() < 1e-8*gs:
                        break
                    e_j = 1e-7*h
                    g0_ = g_pl(y, qi, qj, Pc)

                    def Jv(z, y_=y, g_=g0_):
                        nz = np.linalg.norm(z)
                        if nz == 0:
                            return z
                        s_ = e_j/nz*np.sqrt(z.size)
                        return ((g_pl(y_+s_*z.reshape(n, 3), qi, qj, Pc)-g_)/s_).ravel()+Mm*z
                    Jop = LinearOperator(Kp.shape, matvec=Jv, dtype=np.float64)
                    dy, _ = gmres(Jop, -r_.ravel(), M=Pre, rtol=1e-2, atol=0, restart=25, maxiter=1)
                    dy = dy.reshape(n, 3); al = 1.
                    for ls_ in range(12):
                        r1 = resid(y+al*dy); r1n = np.linalg.norm(r1)
                        if r1n < (1-1e-3*al)*rn:
                            break
                        al *= .5; nls += 1
                    y = y+al*dy; r_ = r1; rn = r1n; nit += 1
                else:
                    nbad += 1
                x1 = 2*y-x; v = 4*(y-x)/dt-v; x = x1
                qi2, qj2, P2, be, Fg, rr_, ee_ = ret_w(x, qi, qj, Pc)
                Etr = eg_state(x, qi, qj, Pc, Fg)[0]; Eaf = eg_state(x, qi2, qj2, P2, Fg)[0]
                # the energy the return lets go, E_tr - E_af = (1 - be^2) W_tr, holds the trial's overshoot: in 1D (1/2 k) (e_tr^2 - e_y^2) = s_y de + 1/2 k de^2. The plastic work is s_y de = 2 be (1 - be) W_tr, so per cell by 2 be / (1 + be)
                d_ = x[j]-x[i]
                Am_ = np.zeros((n, 3, 3))
                np.add.at(Am_, i, wm[:, None, None]*d_[:, :, None]*qi[:, None, :]); np.add.at(Am_, j, wm[:, None, None]*d_[:, :, None]*qj[:, None, :])
                R_, _, _ = polar_(Am_+Fg@Mo@Pc.transpose(0, 2, 1))
                m0i, m0j = d_-np.einsum('fab,fb->fa', R_[i], qi), d_-np.einsum('fab,fb->fa', R_[j], qj)
                m1i, m1j = d_-np.einsum('fab,fb->fa', R_[i], qi2), d_-np.einsum('fab,fb->fa', R_[j], qj2)
                dE = np.zeros(n)
                np.add.at(dE, i, .5*wm*((m0i*m0i).sum(1)-(m1i*m1i).sum(1))); np.add.at(dE, j, .5*wm*((m0j*m0j).sum(1)-(m1j*m1j).sum(1)))
                F0_, F1_ = Fg-R_@Pc, Fg-R_@P2
                dE += .5*(np.einsum('nab,nbc,nac->n', F0_, Mo, F0_)-np.einsum('nab,nbc,nac->n', F1_, Mo, F1_))
                Wc += float((dE*2*be/(1+be)).sum())
                Wp += Etr-Eaf
                qi, qj, Pc = qi2, qj2, P2
                if k % (N//24) == 0 or k == N:
                    Ke = .5*float((mm*(v*v).sum(1)).sum())
                    cm = (mm[:, None]*x).sum(0)/mm.sum(); vm = (mm[:, None]*v).sum(0)/mm.sum()
                    L_ = (mm[:, None]*np.cross(x-cm, v-vm)).sum(0)
                    print('  plastic t / T1 %.2f: kinetic %.4e, stored %.3e, let go %.4e, plastic work %.4e; balance by the work / start %.5f, by the energy let go %.5f; flowing %d of %d cells, strain read median %.3f %%, remainder read median %.3f %%; angular momentum off by %.2e, along it %.5f; Newton per step %.1f, halvings %d, unconverged %d; %.0f s' % (k*dt/T1, Ke, Eaf, Wp, Wc, (Ke+Eaf+Wc)/E0, (Ke+Eaf+Wp)/E0, int((be < 1).sum()), n, 100*np.median(ee_), 100*np.median(rr_), np.linalg.norm(L_-L0)/np.linalg.norm(L0), L_@L0/(L0@L0), nit/k, nls, nbad, time.time()-t0), flush=True)
        for form in (args.forms if 10 in args.test else ''):
            x, v = x0.copy(), v0.copy()
            t0 = time.time()
            Ep0 = eg(x, form)[0]; Ke0 = .5*float((mm*(v*v).sum(1)).sum()); E0 = Ep0+Ke0
            cm = (mm[:, None]*x).sum(0)/mm.sum(); vm = (mm[:, None]*v).sum(0)/mm.sum()
            L0 = (mm[:, None]*np.cross(x-cm, v-vm)).sum(0)
            gs = np.abs(eg(x_d, form)[1]).max(); nit = 0; nls = 0; nbad = 0

            def resid(y_, x_, v_):
                return (mm[:, None]*4/dt**2)*(y_-x_-.5*dt*v_)+eg(y_, form)[1]

            for k in range(1, N+1):
                y = x+.5*dt*v
                r_ = resid(y, x, v); rn = np.linalg.norm(r_)
                for it in range(40):
                    if np.abs(r_).max() < 1e-8*gs:
                        break
                    e_j = 1e-7*h
                    g0_ = eg(y, form)[1]

                    def Jv(z, y_=y, g_=g0_):
                        nz = np.linalg.norm(z)
                        if nz == 0:
                            return z
                        s_ = e_j/nz*np.sqrt(z.size)
                        return ((eg(y_+s_*z.reshape(n, 3), form)[1]-g_)/s_).ravel()+Mm*z
                    Jop = LinearOperator(Kp.shape, matvec=Jv, dtype=np.float64)
                    dy, _ = gmres(Jop, -r_.ravel(), M=Pre, rtol=1e-2, atol=0, restart=25, maxiter=1)
                    dy = dy.reshape(n, 3); al = 1.
                    for ls in range(12):
                        r1 = resid(y+al*dy, x, v); r1n = np.linalg.norm(r1)
                        if r1n < (1-1e-3*al)*rn:
                            break
                        al *= .5; nls += 1
                    y = y+al*dy; r_ = r1; rn = r1n; nit += 1
                else:
                    nbad += 1
                x1 = 2*y-x; v = 4*(y-x)/dt-v; x = x1
                if k % (N//24) == 0 or k == N:
                    Ep = eg(x, form)[0]; Ke = .5*float((mm*(v*v).sum(1)).sum())
                    cm = (mm[:, None]*x).sum(0)/mm.sum(); vm = (mm[:, None]*v).sum(0)/mm.sum()
                    L_ = (mm[:, None]*np.cross(x-cm, v-vm)).sum(0)
                    print('  form %s t / T1 %.2f: energy / start %.5f (potential %.3e, kinetic %.3e); angular momentum off by %.2e of its start, along it %.5f; momentum %.1e; Newton per step %.1f, halvings %d, steps unconverged %d; %.0f s' % (form, k*dt/T1, (Ep+Ke)/E0, Ep, Ke, np.linalg.norm(L_-L0)/np.linalg.norm(L0), L_@L0/(L0@L0), np.linalg.norm((mm[:, None]*v).sum(0)), nit/k, nls, nbad, time.time()-t0), flush=True)
    if 8 in args.test:
        # compression: a uniform state F = U, the surface sites pinned, so a body cannot buckle; for U >= 0.6 the continuum's law is stable (a transverse wave: mu k^2 (2 s - 1) / s, the volume term null), so any negative mode is the discretisation's. This is where the fitted F failed (10-29).
        # Springs, F read: L - C^T W T, W = 2 (tr U - U)^-1 from the polar's derivative. Volume on gg's F, Psi = lam / 2 ln^2 J: V lam [c (x) c - ln J T], c = vec U^-1, T the transpose in U^-1 dF; the variance with c. Relaxed rotation, U = s I: (K_0 - (1 - s) L) / s. Outer layer left out (its cells are pinned)
        free = np.where(depth >= 1)[0]
        fd = (3*free[:, None]+np.arange(3)).ravel()
        Md3 = np.repeat(V[free], 3)
        Tr9 = np.zeros((9, 9))
        print('compression, %d free sites of %d; columns: the state; springs read from F + volume: most negative / w1, count below 0, of the symmetric part, the 3 lowest real parts of the full / w1' % (len(free), n), flush=True)
        states = (('iso 0.9', (.9, .9, .9)), ('iso 0.8', (.8, .8, .8)), ('iso 0.7', (.7, .7, .7)), ('iso 0.6', (.6, .6, .6)), ('uniaxial 0.8', (1, 1, .8)), ('uniaxial 0.6', (1, 1, .6)), ('uniaxial 0.4', (1, 1, .4)))
        if len(fd) > 5000:
            states = (states[1], states[3], states[6])                                 # (dense: minutes each)
        for label, u_ in states:
            U_ = np.diag(u_).astype(float); J_ = np.prod(u_); lnJ = np.log(J_)
            Wu = 2*np.linalg.inv(np.trace(U_)*np.eye(3)-U_)
            Wb = sp.block_diag([Wu]*n, format='csr')
            # the springs' force at d = U D with q = D, R = I: rotation term with the lever q; second order of R is not a tangent of the force
            K_s = (Ks-Cr.T@Wb@TF).tocsr()
            c = np.array([1/u_[0], 0, 0, 0, 1/u_[1], 0, 0, 0, 1/u_[2]])
            for a in range(3):
                for b in range(3):
                    Tr9[3*a+b, 3*b+a] = 1/(u_[a]*u_[b])
            Bk = lam_m*(np.outer(c, c)-lnJ*Tr9)
            Dl_ = sp.block_diag([V[s_]*Bk for s_ in range(n)], format='csr')
            K_l = (Bg.T@Dl_@Bg).tocsr()
            ctr = sp.coo_matrix((np.tile(c[[0, 4, 8]], n), (np.repeat(np.arange(n), 3), (9*np.arange(n)[:, None]+np.array([0, 4, 8])).ravel())), shape=(n, 9*n)).tocsr()
            K_v = ((dT@ctr@Bg).T@sp.diags(2*lam_m*A*L/24)@(dT@ctr@Bg)).tocsr()
            K_ = (K_s+K_l+K_v).tocsr()[fd][:, fd]
            Sq = sp.diags(1/np.sqrt(Md3))
            Sy = (Sq@K_@Sq).toarray(); Sym_ = .5*(Sy+Sy.T)
            ev = np.linalg.eigvalsh(Sym_)
            nev = int((ev < 0).sum())
            lam_ = np.linalg.eigvals(Sy) if len(fd) < 5000 else None
            lows = 'skipped' if lam_ is None else np.array2string(np.sort(lam_.real)[:3]/w1, precision=2)
            line = '  %-13s J %.3f: symmetric part lowest %.3f, count below 0 %d; full, lowest real %s' % (label, J_, ev[0]/w1, nev, lows)
            if u_[0] == u_[2]:
                s_ = u_[0]
                # relaxed rotation at U = s I
                K_a = ((Kr-(1-s_)*Ks)/s_+K_l+K_v).tocsr()[fd][:, fd]
                eva = np.linalg.eigvalsh((Sq@K_a@Sq).toarray())
                line += '; relaxed rotation: lowest %.3f, count below 0 %d' % (eva[0]/w1, int((eva < 0).sum()))
                # springs alone, the pencil's lowest: the critical compression 1 - s of each form
            print(line, flush=True)
    if 7 in args.test:
        for name, K_b in (('springs alone', (Ks-Cr.T@TF).tocsr()), ('+ lam, variance, outer layer', (Ks-Cr.T@TF+Kl+Kv+Ko).tocsr())):
            t0 = time.time()
            Ad_ = K_b-K_b.T
            print('%s: |K - K^T| / |K| (Frobenius) %.3f' % (name, sp.linalg.norm(Ad_)/sp.linalg.norm(K_b)), flush=True)
            Am = (sp.diags(np.repeat(1/np.sqrt(V), 3))@K_b@sp.diags(np.repeat(1/np.sqrt(V), 3))).toarray()
            lam_ = np.linalg.eigvals(Am)
            o = np.argsort(lam_.real); lam_ = lam_[o]
            nz = int((np.abs(lam_) < 1e-6*w1).sum())
            a_, b_ = lam_.real, np.abs(lam_.imag)
            cx = b_ > 1e-7*w1
            fr = np.sqrt(np.clip(a_, 0, None))/(2*np.pi)
            print('  zeros %d, real part negative %d, complex %d of %d; largest real %.3e (w1 %.3e); first real / FEM %s' % (nz, int((a_ < -1e-6*w1).sum()), int(cx.sum()), len(lam_), a_.max(), w1, np.array2string(fr[nz:nz+6]/f_fem[:6], precision=3)), flush=True)
            pos = a_ > 1e-6*w1
            ra = b_[pos]/a_[pos]
            print('  |imag| / real: largest %.3f; by the modes in order of frequency: first 30 largest %.4f, first 100 %.4f, first 500 %.4f, all median %.4f, 99 %% %.4f' % (ra.max(), ra[:30].max(), ra[:100].max(), ra[:500].max(), np.median(ra), np.quantile(ra, .99)), flush=True)
            g_ = b_[pos]/(2*np.sqrt(a_[pos]))                                                  # (growth rate of the mode: lambda = a + i b, omega = sqrt(lambda))
            kk = np.argmax(g_)
            T1 = 2*np.pi/np.sqrt(w1)
            print('  growth: e-folds per the first FEM period: largest %.3f (at frequency %.2f x FEM first); first 30 modes largest %.4f; per its own cycle: first 30 largest %.4f, all largest %.4f; t %.0f s' % (g_[kk]*T1, np.sqrt(a_[pos][kk]/w1), (g_[:30]*T1).max(), (np.pi*ra[:30]).max(), np.pi*ra.max(), time.time()-t0), flush=True)
print('done', flush=True)
