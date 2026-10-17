"""Float64 PyTorch reference of the Hertz-Mindlin contact + integration step (foam_dem_warp.hertz_integrate_kernel)
and a term-by-term comparison with the Warp kernel and its adjoint.

Reference (per body b, ground normal n pointing out of the ground, contact sphere radius R, offset o = theta[b, coc]):
  reach = R + o, d = reach - (x - p) . n; contact if d > 0
  lam = k_v e^th4, mu = mu_ref e^th5, nu = lam / (2 (lam + mu)), E = mu (3 lam + 2 mu) / (lam + mu)
  E* = E / (1 - nu^2), G* = mu / (2 - nu); F_n = s 4/3 E* sqrt(R) d^1.5, k_t = s 8 G* sqrt(R d)
  F_t = tangential part of the stored F_t; x sqrt(d / d_prev) if unloading (thornton); F_t -= k_t dt v_slip
  (v_slip = tangential part of v + w x (-R n)); capped at mu_f F_n, mu_f = exp(th7 + friction0)
  v' = v + dt ((f + F_n n + F_t) / m + g), w' = w + dt (tau + (-R n) x F_t) / I, x' = x + dt v',
  q' = normalize(normalize(1, dt w' / 2) * q), history' = (F_t, d) (zero without contact)
Tests: (1) one step on bodies drawn across all branches (no contact, loading, unloading, sticking, sliding, offsets
below -R): outputs and the gradients of a random linear functional of all outputs w.r.t. every input; (2) a short
trajectory of independent bodies hitting the ground with slip and spin: final state and gradients w.r.t. the theta
columns (4, 5, 7, offset) and the initial velocity / spin.
"""
import argparse

import numpy as np
import torch
import warp as wp

from experiments.foam_dem_warp import hertz_integrate_kernel

COC = 9


def ref_step(x, q, v, w, hc, f, tau, mass, inertia, fixed, R, g, p, n, th, k_v, mu_ref, friction0, kscale, coc, thornton, dt):
    reach = R+th[:, coc] if coc >= 0 else R
    d = reach-((x-p)*n).sum(-1)
    on = d > 0
    dp = torch.where(on, d, torch.ones_like(d))                     # safe value off contact (masked below)
    lam = k_v*torch.exp(th[:, 4]); mu = mu_ref*torch.exp(th[:, 5])
    nu = .5*lam/(lam+mu); E = mu*(3*lam+2*mu)/(lam+mu)
    Es = E/(1-nu*nu); Gs = mu/(2-nu)
    sR, sd = torch.sqrt(R), torch.sqrt(dp)
    Fn = kscale*4./3.*Es*sR*dp*sd
    kt = kscale*8.*Gs*sR*sd
    Ft = hc[:, :3]; Ft = Ft-(Ft*n).sum(-1, keepdim=True)*n
    d_prev = hc[:, 3]
    unl = (d_prev > 0) & (dp < d_prev) & bool(thornton)
    Ft = torch.where(unl[:, None], Ft*torch.sqrt(dp/torch.where(unl, d_prev, torch.ones_like(d_prev)))[:, None], Ft)
    rc = -R[:, None]*n
    vc = v+torch.cross(w, rc.expand_as(w), dim=-1)
    vs = vc-(vc*n).sum(-1, keepdim=True)*n
    Ft = Ft-(kt*dt)[:, None]*vs
    cap = torch.exp(th[:, 7]+friction0)*Fn
    L = Ft.norm(dim=-1)
    slide = L > cap
    Ft = torch.where(slide[:, None], Ft*(cap/torch.where(slide, L, torch.ones_like(L)))[:, None], Ft)
    m_ = on[:, None].to(x.dtype)
    Fc = m_*(Fn[:, None]*n+Ft)
    Tc = m_*torch.cross(rc.expand_as(Ft), Ft, dim=-1)
    free = (fixed == 0)[:, None].to(x.dtype)
    vn = v+free*dt*((f+Fc)/mass[:, None]+g)
    wn = w+free*dt*(tau+Tc)/inertia[:, None]
    xn = x+free*dt*vn
    hh = .5*dt*wn
    qs = torch.cat((torch.ones_like(hh[:, :1]), hh), -1); qs = qs/qs.norm(dim=-1, keepdim=True)
    a, b = qs, q
    qn = torch.stack((a[:, 0]*b[:, 0]-a[:, 1]*b[:, 1]-a[:, 2]*b[:, 2]-a[:, 3]*b[:, 3],
                      a[:, 0]*b[:, 1]+a[:, 1]*b[:, 0]+a[:, 2]*b[:, 3]-a[:, 3]*b[:, 2],
                      a[:, 0]*b[:, 2]-a[:, 1]*b[:, 3]+a[:, 2]*b[:, 0]+a[:, 3]*b[:, 1],
                      a[:, 0]*b[:, 3]+a[:, 1]*b[:, 2]-a[:, 2]*b[:, 1]+a[:, 3]*b[:, 0]), -1)
    qn = qn/qn.norm(dim=-1, keepdim=True)
    hn = torch.cat((m_*Ft, (m_[:, 0]*d)[:, None]), -1)
    return xn, qn, vn, wn, hn, dict(on=on, unl=unl & on, slide=slide & on)


def warp_step(inp, consts, coc, thornton, dt, tape=None):
    names = ('x', 'q', 'v', 'w', 'hc', 'f', 'tau')
    dt_ = dict(x=wp.vec3, q=wp.vec4, v=wp.vec3, w=wp.vec3, hc=wp.vec4, f=wp.vec3, tau=wp.vec3)
    arrs = {k: wp.from_torch(inp[k].float().contiguous().clone(), dtype=dt_[k], requires_grad=True) for k in names}
    th = wp.from_torch(inp['th'].float().contiguous().clone(), requires_grad=True)
    out = {k: wp.zeros(inp['x'].shape[0], dtype=dt_[k if k != 'h1' else 'hc'], requires_grad=True) for k in ('x', 'q', 'v', 'w', 'hc')}
    c = consts
    with (tape if tape is not None else wp.Tape()) as tp:
        wp.launch(hertz_integrate_kernel, inp['x'].shape[0], inputs=[arrs['x'], arrs['q'], arrs['v'], arrs['w'], arrs['hc'], arrs['f'], arrs['tau'],
                  c['mass'], c['inertia'], c['fixed'], c['R'], c['g'], c['p'], c['n'], th, c['k_v'], c['mu_ref'], c['friction0'], c['kscale'],
                  coc, thornton, dt, out['x'], out['q'], out['v'], out['w'], out['hc']])
    return tp, arrs, th, out


def rel(a, b):
    return float((a-b).abs().max()/(b.abs().max()+1e-30))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=4000)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    wp.init()
    torch.manual_seed(args.seed)
    dev, D = 'cuda', torch.float64
    N = args.n
    R = torch.full((N,), 1.6e-3, dtype=D, device=dev)*(1+.3*torch.rand(N, dtype=D, device=dev))
    n = torch.tensor([0., 0., -1.], dtype=D, device=dev); pg = torch.tensor([0., 0., .02], dtype=D, device=dev)
    g = torch.tensor([0., 0., 9.8], dtype=D, device=dev)
    k_v, mu_ref, friction0, kscale, dt = 4.e5, 2.e4, float(np.log(.5)), 1., 2.e-5
    mass = 4e-5*(1+torch.rand(N, dtype=D, device=dev)); inertia = .4*mass*R**2
    fixed = torch.zeros(N, dtype=torch.int32, device=dev)
    th = .3*torch.randn(N, 10, dtype=D, device=dev)
    th[:, COC] = 2.5e-3*(torch.rand(N, dtype=D, device=dev)*2-1)            # offsets -2.5 .. 2.5 mm (some below -R)
    reach = R+th[:, COC]
    d = (torch.rand(N, dtype=D, device=dev)*1.2-.2)*1e-4                     # penetration -20 .. 100 um
    x = torch.randn(N, 3, dtype=D, device=dev)*.01; x[:, 2] = pg[2]-(reach-d)    # (x - p).n = reach - d
    v = .05*torch.randn(N, 3, dtype=D, device=dev); w = 5*torch.randn(N, 3, dtype=D, device=dev)
    q = torch.randn(N, 4, dtype=D, device=dev); q = q/q.norm(dim=-1, keepdim=True)
    hc = torch.zeros(N, 4, dtype=D, device=dev)
    hc[:, :3] = 2e-3*torch.randn(N, 3, dtype=D, device=dev)                      # stored tangential force (N)
    hc[:, 3] = d*(.5+torch.rand(N, dtype=D, device=dev))                          # previous penetration: loading and unloading
    f = 1e-3*torch.randn(N, 3, dtype=D, device=dev); tau = 1e-7*torch.randn(N, 3, dtype=D, device=dev)
    inp = dict(x=x, q=q, v=v, w=w, hc=hc, f=f, tau=tau, th=th)
    consts = dict(mass=wp.from_torch(mass.float()), inertia=wp.from_torch(inertia.float()), fixed=wp.from_torch(fixed),
                  R=wp.from_torch(R.float()), g=wp.vec3(*g.tolist()), p=wp.vec3(*pg.tolist()), n=wp.vec3(*n.tolist()),
                  k_v=k_v, mu_ref=mu_ref, friction0=friction0, kscale=kscale)
    for thornton in (1, 0):
        print('=== one step, thornton %d' % thornton)
        # reference
        ti = {k: t_.clone().requires_grad_(True) for k, t_ in inp.items()}
        out_r = ref_step(ti['x'], ti['q'], ti['v'], ti['w'], ti['hc'], ti['f'], ti['tau'], mass, inertia, fixed, R, g, pg, n, ti['th'],
                         k_v, mu_ref, friction0, kscale, COC, thornton, dt)
        info = out_r[5]
        print('  branches: contact %d, unloading %d, sliding %d, offsets below -R %d of %d' % (
            int(info['on'].sum()), int(info['unl'].sum()), int(info['slide'].sum()), int((th[:, COC] < -R).sum()), N))
        seeds = [torch.randn_like(o) for o in out_r[:5]]
        # normalise the seeds by the output scale so every output counts
        seeds = [s_/(o.detach().abs().mean()+1e-30) for s_, o in zip(seeds, out_r[:5])]
        J = sum((s_*o).sum() for s_, o in zip(seeds, out_r[:5]))
        J.backward()
        # warp
        tp, arrs, thw, out_w = warp_step(inp, consts, COC, thornton, dt)
        for k, s_ in zip(('x', 'q', 'v', 'w', 'hc'), seeds):
            out_w[k].grad = wp.from_torch(s_.float().contiguous(), dtype=out_w[k].dtype)
        tp.backward()
        for k, o in zip(('x', 'q', 'v', 'w', 'hc'), out_r[:5]):
            ow = wp.to_torch(out_w[k]).double()
            dl = ow-o.detach()
            if k in ('x',):
                dl = dl-0.                                                    # absolute positions: compare increments
                print('  output %-3s max |warp - ref| / max |ref increment| %.2e' % (k, float(dl.abs().max()/((o.detach()-inp['x']).abs().max()))))
            else:
                print('  output %-3s rel. error %.2e' % (k, rel(ow, o.detach())))
        for k in ('x', 'q', 'v', 'w', 'hc', 'f', 'tau'):
            gw = wp.to_torch(arrs[k].grad).double(); gr = ti[k].grad
            print('  grad   %-3s rel. error %.2e' % (k, rel(gw, gr)))
        gw = wp.to_torch(thw.grad).double(); gr = ti['th'].grad
        for col in (4, 5, 7, COC):
            print('  grad   theta[:, %d] rel. error %.2e (|ref| max %.2e)' % (col, rel(gw[:, col], gr[:, col]), float(gr[:, col].abs().max())))
    # short trajectory: independent bodies dropped onto the ground with slip and spin
    print('=== trajectory: %d steps' % args.steps)
    M = 512
    Rt = R[:M].clone(); tht = th[:M].clone(); tht[:, COC] = 1.5e-3*(torch.rand(M, dtype=D, device=dev)*2-1)
    xt = torch.randn(M, 3, dtype=D, device=dev)*.01; xt[:, 2] = pg[2]-(Rt+tht[:, COC])-2e-4*torch.rand(M, dtype=D, device=dev)
    vt = torch.zeros(M, 3, dtype=D, device=dev); vt[:, 2] = .3+.3*torch.rand(M, dtype=D, device=dev); vt[:, :2] = .1*torch.randn(M, 2, dtype=D, device=dev)
    wt = 20*torch.randn(M, 3, dtype=D, device=dev)
    qt = q[:M].clone(); zero3 = torch.zeros(M, 3, dtype=D, device=dev)
    mt, It = mass[:M], inertia[:M]
    ti = dict(v=vt.clone().requires_grad_(True), w=wt.clone().requires_grad_(True), th=tht.clone().requires_grad_(True))
    s = (xt, qt, ti['v'], ti['w'], torch.zeros(M, 4, dtype=D, device=dev))
    for _ in range(args.steps):
        s = ref_step(*s, zero3, zero3, mt, It, fixed[:M], Rt, g, pg, n, ti['th'], k_v, mu_ref, friction0, kscale, COC, 1, dt)[:5]
    seeds = [torch.randn_like(o)/(o.detach().abs().mean()+1e-30) for o in s]
    sum((a*b).sum() for a, b in zip(seeds, s)).backward()
    cw = dict(consts, mass=wp.from_torch(mt.float()), inertia=wp.from_torch(It.float()), fixed=wp.from_torch(fixed[:M]), R=wp.from_torch(Rt.float()))
    dts = (wp.vec3, wp.vec4, wp.vec3, wp.vec3, wp.vec4)
    st = [wp.from_torch(t_.float().contiguous().clone(), dtype=dd, requires_grad=True) for t_, dd in zip((xt, qt, vt, wt, torch.zeros(M, 4, dtype=D, device=dev)), dts)]
    thw = wp.from_torch(tht.float().contiguous().clone(), requires_grad=True)
    zf = wp.zeros(M, dtype=wp.vec3); zt = wp.zeros(M, dtype=wp.vec3)
    states = [st]
    tape = wp.Tape()
    with tape:
        for _ in range(args.steps):
            nx = [wp.zeros(M, dtype=dd, requires_grad=True) for dd in dts]
            wp.launch(hertz_integrate_kernel, M, inputs=[*states[-1], zf, zt, cw['mass'], cw['inertia'], cw['fixed'], cw['R'], cw['g'], cw['p'], cw['n'],
                      thw, k_v, mu_ref, friction0, kscale, COC, 1, dt, *nx])
            states.append(nx)
    for a_, s_ in zip(states[-1], seeds):
        a_.grad = wp.from_torch(s_.float().contiguous(), dtype=a_.dtype)
    tape.backward()
    for k, a_, o in zip(('x', 'q', 'v', 'w', 'hc'), states[-1], s):
        ow = wp.to_torch(a_).double()
        if k == 'x':
            print('  final %-3s max |warp - ref| %.2e m (displacement scale %.2e m)' % (k, float((ow-o.detach()).abs().max()), float((o.detach()-xt).abs().max())))
        else:
            print('  final %-3s rel. error %.2e' % (k, rel(ow, o.detach())))
    print('  grad  v0 rel. error %.2e | w0 %.2e' % (rel(wp.to_torch(states[0][2].grad).double(), ti['v'].grad), rel(wp.to_torch(states[0][3].grad).double(), ti['w'].grad)))
    gw = wp.to_torch(thw.grad).double()
    for col in (4, 5, 7, COC):
        print('  grad  theta[:, %d] rel. error %.2e' % (col, rel(gw[:, col], ti['th'].grad[:, col])))


if __name__ == '__main__':
    main()
