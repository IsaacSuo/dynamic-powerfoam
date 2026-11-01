"""no foam, no geometry: a cell = N points with deviatoric strain mean + fluctuation, driven by a mean rate and a fluctuating rate at frequency om. Truth: each point on its own von Mises sphere. Models: mean + remainder, three laws.
A: limit on |mean|, remainder scaled by beta. B: the same, remainder by beta^2. C: limit on the total energy, sqrt(|mean|^2 + <|remainder|^2>), both scaled by the same beta.
Second order, g = |D| / e_y the relaxation rate: true excess of the dissipation over s_y |D| is mu g <|D'_perp|^2> / (g^2 + om^2); predicted model / truth, D' perpendicular to the mean rate: A 2, B 4 (g^2 + om^2) / (om^2 + 4 g^2), C 1. Radial part of D' included: all times 5/4 on the isotropic average"""
import numpy as np

ey = .01; g = .02; Dn = g*ey                                                          # (g per step, small: the first order return)
rng = np.random.default_rng(0)
N = 400
Dh = np.zeros(5); Dh[0] = 1.


def run(tau, oor, perp):
    om = oor*g; T = 2*np.pi/om
    dh = rng.normal(size=(N, 5))
    if perp:
        dh[:, 0] = 0
    dh /= np.linalg.norm(dh, axis=1, keepdims=True)
    # paired with minus, so the points' mean rate is exactly the mean rate
    dh = np.r_[dh, -dh]; ph = rng.uniform(0, 2*np.pi, N); ph = np.r_[ph, ph]
    amp = tau*Dn*np.sqrt(2)
    nb = int(np.ceil(T)) if T > 400 else int(np.ceil(400/T)*np.ceil(T))
    n_skip = int(8/g)
    n_all = n_skip+nb
    # truth
    e = np.zeros((2*N, 5)); WT = 0.
    # models: mean, remainders
    M = {k: [np.zeros(5), np.zeros((2*N, 5)), 0.] for k in 'ABC'}
    for k in range(n_all):
        Dp = (amp*np.sin(om*k+ph))[:, None]*dh
        et = e+Dn*Dh+Dp
        nr = np.linalg.norm(et, axis=1); be = np.minimum(1., ey/nr)
        e = be[:, None]*et
        if k >= n_skip:
            WT += float(((et*et).sum(1)-(e*e).sum(1)).mean())
        for name, s in M.items():
            mt = s[0]+Dn*Dh; rt = s[1]+Dp
            r2 = float((rt*rt).sum(1).mean())
            if name == 'C':
                be = min(1., ey/np.sqrt((mt*mt).sum()+r2)); br = be
            else:
                be = min(1., ey/np.linalg.norm(mt)); br = be if name == 'A' else be*be
            s[0] = be*mt; s[1] = br*rt
            if k >= n_skip:
                s[2] += float((mt*mt).sum()+r2-(s[0]*s[0]).sum()-(s[1]*s[1]).sum(1).mean())
    ref = 2*ey*Dn*nb                                                                   # (s_y |D| over the window, mu = 1)
    return WT/ref-1, [M[k][2]/ref-1 for k in 'ABC']


for perp in (True, False):
    print('fluctuating rate %s the mean rate; columns: om / g; true excess; model excess / true, A, B, C; predicted A, B, C' % ('perpendicular to' if perp else 'in all directions, against'), flush=True)
    for tau in (.2, .5, 1., 2.):
        for oor in (.1, 1/np.sqrt(2), np.sqrt(2), 5.):
            xt, xm = run(tau, oor, perp)
            f = 1. if perp else 5/4
            print('  tau %.1f, om / g %.2f: true %.4f; A %.2f, B %.2f, C %.2f; pred %.2f, %.2f, %.2f' % (tau, oor, xt, xm[0]/xt, xm[1]/xt, xm[2]/xt, 2*f, 4*(1+oor**2)/(oor**2+4)*f, f), flush=True)
