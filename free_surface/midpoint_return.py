"""one mass on one spring with a force limit: exact solution known. k = m = 1, limit c = 0.2, v0 = 1: elastic to x = 0.2, slides at force c to slip 2.4, then rings forever at amplitude 0.2 (energy 0.02).
th1: backward Euler, f = clip(m + k dx). end: midpoint rule, the return at the step's end, f = (m + clip(m + k dx)) / 2. mid: the midpoint with the return at the midpoint, extrapolated (m1 = 2 clip(m + k dx / 2) - m)"""
import numpy as np
c = .2


def clip(a):
    return np.clip(a, -c, c)


def bis(fun, lo=-50., hi=50.):
    for _ in range(90):
        q = .5*(lo+hi)
        if fun(q) > 0:
            hi = q
        else:
            lo = q
    return .5*(lo+hi)


def run(kind, dt, T):
    x, v, m, slip, Dmin, mmax = 0., 1., 0., 0., 0., 0.
    for _ in range(int(round(T/dt))):
        if kind == 'th1':
            f = lambda dx: clip(m+dx)
            dx = bis(lambda d: d/dt-(v-dt*f(d)))
            m1 = clip(m+dx); v1 = v-dt*m1; D = 0.
        elif kind == 'end':
            f = lambda dx: .5*(m+clip(m+dx))
            dx = bis(lambda d: d/dt-.5*(2*v-dt*f(d)))
            m1 = clip(m+dx); v1 = v-dt*f(dx); D = f(dx)*dx-.5*(m1*m1-m*m)
        else:
            f = lambda dx: clip(m+.5*dx)
            dx = bis(lambda d: d/dt-.5*(2*v-dt*f(d)))
            m1 = 2*clip(m+.5*dx)-m; v1 = v-dt*f(dx); D = f(dx)*dx-.5*(m1*m1-m*m)
        slip += dx-(m1-m); Dmin = min(Dmin, D); mmax = max(mmax, abs(m1))
        x, v, m = x+dx, v1, m1
    return slip, .5*v*v+.5*m*m, Dmin, mmax


print('exact: slip 2.4, energy after the slide 0.02')
for kind in ('th1', 'end', 'mid'):
    for dt in (.4, .2, .1, .05, .025):
        s, e, d, mm = run(kind, dt, 200.)
        print('%-4s dt %.3f: slip %.4f, energy at t 200 %.5f, smallest one-step dissipation %.2e, largest |memory| / limit %.3f' % (kind, dt, s, e, d, mm/c))
c = 1e9
for kind in ('th1', 'end'):
    print('no limit, %s dt 0.2, t 200: energy / start %.5f' % (kind, run(kind, .2, 200.)[1]/.5))
rng = np.random.default_rng(0)
worst, over = 1e9, 0.
for _ in range(200000):
    k = rng.uniform(.1, 2, 14)[:, None]; a = rng.normal(size=(14, 3))
    Lim = rng.uniform(.1, 3); U = .5*(k*a*a).sum()
    a = a*min(1, Lim/np.sqrt(U))*rng.uniform(.0, 1)**.25
    r = a+rng.normal(size=(14, 3))*rng.lognormal(-1, 1.5)
    Ur = np.sqrt(.5*(k*r*r).sum()); b = min(1., Lim/Ur); m1 = b*r
    D = .5*(k*(a+m1)*(r-m1)).sum()
    worst = min(worst, D/max(Lim*Lim, 1e-30)); over = max(over, np.sqrt(.5*(k*m1*m1).sum())/Lim)
print('cells of 14 vector halves, 200000 random steps: smallest dissipation / limit energy %.2e, largest store^(1/2) after the return / limit %.6f' % (worst, over))
