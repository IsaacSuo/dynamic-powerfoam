"""Recut.cg's Jacobi-preconditioned CG in Warp kernels on the cut's site lists (11-06, stage 2 of the explicit step's
restructuring; the same iteration as recut.GraphedCG, to rounding):
  k_spmv    per site: (H q)_k = d_k q_k - sum over its pairs c q_other (geo_pipe's site entries; no atomics), the
            partial q . Hq into NP slots
  k_alpha   one thread: q . Hq, al = rz / q . Hq
  k_update  per site: x += al q, r -= al Hq, z = r / d, the partial r . z, the largest |r / scale|
  k_beta    one thread: rz1, beta = rz1 / rz, rz = rz1; the stop flag (the largest |r / scale| <= lim)
  k_dir     per site: q = z + beta q
`check` iterations per replay of one Warp graph on fixed buffers (pairs padded to a capacity, captured again when they
outgrow it); the flag read after each replay, as GraphedCG reads its stop."""
import torch
import warp as wp

f64, i32 = torch.float64, torch.int32
NP = 128


@wp.kernel(enable_backward=False)
def k_spmv(q: wp.array(dtype=wp.float64), d: wp.array(dtype=wp.float64), c: wp.array(dtype=wp.float64), pi: wp.array(dtype=wp.int32), pj: wp.array(dtype=wp.int32),
           soff: wp.array(dtype=wp.int32), ent: wp.array(dtype=wp.int32), npr: wp.array(dtype=wp.int32), Hq: wp.array(dtype=wp.float64), part: wp.array(dtype=wp.float64)):
    k = wp.tid()
    h = d[k]*q[k]
    np_ = npr[0]
    for e in range(soff[k], soff[k+1]):
        t = ent[e]
        f = t
        o = int(0)
        if t < np_:
            o = pj[f]
        else:
            f = t-np_
            o = pi[f]
        h = h-c[f]*q[o]
    Hq[k] = h
    wp.atomic_add(part, k % 128, q[k]*h)


@wp.kernel(enable_backward=False)
def k_alpha(part: wp.array(dtype=wp.float64), sc: wp.array(dtype=wp.float64)):
    """sc: 0 rz, 1 al, 2 beta, 3 lim, 4 the largest |r / scale|, 5 the stop flag"""
    s = wp.float64(0.0)
    for k in range(128):
        s = s+part[k]
        part[k] = wp.float64(0.0)
    sc[1] = sc[0]/s
    sc[4] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def k_update(q: wp.array(dtype=wp.float64), Hq: wp.array(dtype=wp.float64), d: wp.array(dtype=wp.float64), scale: wp.array(dtype=wp.float64), sc: wp.array(dtype=wp.float64),
             x: wp.array(dtype=wp.float64), rr: wp.array(dtype=wp.float64), z: wp.array(dtype=wp.float64), part: wp.array(dtype=wp.float64)):
    k = wp.tid()
    al = sc[1]
    x[k] = x[k]+al*q[k]
    r = rr[k]-al*Hq[k]
    rr[k] = r
    zk = r/d[k]
    z[k] = zk
    wp.atomic_add(part, k % 128, r*zk)
    wp.atomic_max(sc, 4, wp.abs(r/scale[k]))


@wp.kernel(enable_backward=False)
def k_beta(part: wp.array(dtype=wp.float64), sc: wp.array(dtype=wp.float64)):
    s = wp.float64(0.0)
    for k in range(128):
        s = s+part[k]
        part[k] = wp.float64(0.0)
    sc[2] = s/sc[0]
    sc[0] = s
    f = wp.float64(0.0)
    if sc[4] <= sc[3]:
        f = wp.float64(1.0)
    sc[5] = f


@wp.kernel(enable_backward=False)
def k_dir(z: wp.array(dtype=wp.float64), sc: wp.array(dtype=wp.float64), q: wp.array(dtype=wp.float64)):
    k = wp.tid()
    q[k] = z[k]+sc[2]*q[k]


class WarpCG:
    def __init__(self, n, check, dev):
        dev = str(dev)
        self.n, self.check, self.dev, self.cap, self.graph = n, check, dev, 0, None
        Z = lambda *s_, dt=f64: torch.zeros(*s_, dtype=dt, device=dev)
        self.q, self.x, self.rr, self.z, self.Hq, self.d, self.scale = Z(n), Z(n), Z(n), Z(n), Z(n), Z(n), Z(n)
        self.soff, self.npr = Z(n+1, dt=i32), Z(1, dt=i32)
        self.part, self.sc = Z(NP), Z(8)

    def _alloc(self, m):
        self.cap = int(1.25*m)+1024
        Z = lambda *s_, dt=f64: torch.zeros(*s_, dtype=dt, device=self.dev)
        self.c, self.pi, self.pj, self.ent = Z(self.cap), Z(self.cap, dt=i32), Z(self.cap, dt=i32), Z(2*self.cap, dt=i32)
        W_ = lambda t_, d_: wp.from_torch(t_, dtype=d_)
        F_, I_ = (lambda t_: W_(t_, wp.float64)), (lambda t_: W_(t_, wp.int32))
        n, dev = self.n, self.dev
        a = dict(q=F_(self.q), x=F_(self.x), rr=F_(self.rr), z=F_(self.z), Hq=F_(self.Hq), d=F_(self.d), scale=F_(self.scale), c=F_(self.c), pi=I_(self.pi),
                 pj=I_(self.pj), soff=I_(self.soff), ent=I_(self.ent), npr=I_(self.npr), part=F_(self.part), sc=F_(self.sc))

        def body():
            for _ in range(self.check):
                wp.launch(k_spmv, n, inputs=[a['q'], a['d'], a['c'], a['pi'], a['pj'], a['soff'], a['ent'], a['npr']], outputs=[a['Hq'], a['part']], device=dev)
                wp.launch(k_alpha, 1, inputs=[a['part'], a['sc']], device=dev)
                wp.launch(k_update, n, inputs=[a['q'], a['Hq'], a['d'], a['scale'], a['sc']], outputs=[a['x'], a['rr'], a['z'], a['part']], device=dev)
                wp.launch(k_beta, 1, inputs=[a['part'], a['sc']], device=dev)
                wp.launch(k_dir, n, inputs=[a['z'], a['sc']], outputs=[a['q']], device=dev)
        wp.capture_begin(device=dev)
        try:
            body()
        finally:
            self.graph = wp.capture_end(device=dev)

    def __call__(self, J, b, scale, rtol, maxit):
        i, j, c, d, soff, ent = J[:6]
        m = len(i)
        if self.graph is None or m > self.cap:
            self._alloc(m)
        self.c[:m] = c; self.pi[:m] = i; self.pj[:m] = j; self.ent[:2*m] = ent; self.soff.copy_(soff); self.npr.fill_(m)
        self.d.copy_(d); self.scale.copy_(scale); self.x.zero_(); self.rr.copy_(b)
        torch.div(b, d, out=self.z); self.q.copy_(self.z); self.part.zero_()
        self.sc[0] = b@self.z; self.sc[3] = rtol*(b/scale).abs().max(); self.sc[5] = 0.
        k = 0
        while k < maxit:
            wp.capture_launch(self.graph); k += self.check
            if self.sc[5].item() > 0:
                break
        return self.x.clone(), k
