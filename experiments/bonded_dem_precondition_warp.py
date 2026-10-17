"""Fused construction and CG vector operations for the SAME elastic preconditioner.

No changes to the Newton residual/Jacobian, material, mass, or CG stopping rule.
Reductions stay in Torch; fixed-size per-cell work runs in Warp.
"""
import torch
import warp as wp


@wp.kernel
def diagonal_blocks(edges: wp.array2d(dtype=wp.int64), n: wp.array(dtype=wp.vec3d),
                    ri: wp.array(dtype=wp.vec3d), rj: wp.array(dtype=wp.vec3d),
                    kn: wp.array(dtype=wp.float64), ks: wp.array(dtype=wp.float64),
                    kb: wp.array(dtype=wp.float64), kt: wp.array(dtype=wp.float64),
                    out: wp.array3d(dtype=wp.float64)):
    e = wp.tid()
    for endpoint in range(2):
        node = edges[e, endpoint]
        r = ri[e]
        if endpoint == 1:
            r = rj[e]
        for column in range(6):
            u, w = wp.vec3d(), wp.vec3d()
            if column < 3:
                u[column] = wp.float64(1.0)
            else:
                w[column-3] = wp.float64(1.0)
            du = u+wp.cross(w, r)
            f = ks[e]*du+(kn[e]-ks[e])*wp.dot(du, n[e])*n[e]
            m = wp.cross(r, f)+kb[e]*w+(kt[e]-kb[e])*wp.dot(w, n[e])*n[e]
            for row in range(3):
                wp.atomic_add(out, node, row, column, f[row])
                wp.atomic_add(out, node, row+3, column, m[row])


@wp.kernel
def cg_initialize(vector: wp.array2d(dtype=wp.float64), block: wp.array3d(dtype=wp.float64),
                  preconditioner: wp.array3d(dtype=wp.float64), units: wp.array2d(dtype=wp.float64),
                  fixed: wp.array(dtype=wp.bool), x: wp.array2d(dtype=wp.float64),
                  r: wp.array2d(dtype=wp.float64), direction: wp.array2d(dtype=wp.float64),
                  partial: wp.array(dtype=wp.float64)):
    i = wp.tid()
    for row in range(6):
        value = wp.float64(0.0)
        if not fixed[i]:
            for col in range(6):
                value = value+block[i, row, col]*units[i, col]*vector[i, col]
        r[i, row] = value
        x[i, row] = wp.float64(0.0)
    product = wp.float64(0.0)
    for row in range(6):
        value = wp.float64(0.0)
        for col in range(6):
            value = value+preconditioner[i, row, col]*r[i, col]
        direction[i, row] = value
        product = product+r[i, row]*value
    partial[i] = product


@wp.kernel
def cg_operator(mass: wp.array2d(dtype=wp.float64), fixed: wp.array(dtype=wp.bool),
                direction: wp.array2d(dtype=wp.float64), tangent: wp.array2d(dtype=wp.float64),
                alpha: wp.float64, ad: wp.array2d(dtype=wp.float64), partial: wp.array(dtype=wp.float64)):
    i = wp.tid()
    product = wp.float64(0.0)
    for row in range(6):
        value = mass[i, row]*direction[i, row]+alpha*tangent[i, row]
        if fixed[i]:
            value = direction[i, row]
        ad[i, row] = value
        product = product+direction[i, row]*value
    partial[i] = product


@wp.kernel
def cg_update(x: wp.array2d(dtype=wp.float64), r: wp.array2d(dtype=wp.float64),
              direction: wp.array2d(dtype=wp.float64), ad: wp.array2d(dtype=wp.float64),
              preconditioner: wp.array3d(dtype=wp.float64), rz: wp.array(dtype=wp.float64),
              denominator: wp.array(dtype=wp.float64), z: wp.array2d(dtype=wp.float64),
              partial: wp.array(dtype=wp.float64)):
    i = wp.tid()
    alpha = rz[0]/denominator[0]
    for row in range(6):
        x[i, row] = x[i, row]+alpha*direction[i, row]
        r[i, row] = r[i, row]-alpha*ad[i, row]
    product = wp.float64(0.0)
    for row in range(6):
        value = wp.float64(0.0)
        for col in range(6):
            value = value+preconditioner[i, row, col]*r[i, col]
        z[i, row] = value
        product = product+r[i, row]*value
    partial[i] = product


@wp.kernel
def cg_direction(direction: wp.array2d(dtype=wp.float64), z: wp.array2d(dtype=wp.float64),
                 following: wp.array(dtype=wp.float64), previous: wp.array(dtype=wp.float64)):
    i = wp.tid()
    beta = following[0]/previous[0]
    for row in range(6):
        direction[i, row] = z[i, row]+beta*direction[i, row]


class WarpDiagonal:
    def __init__(self, solver):
        wp.init()
        self.solver = solver
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(solver.device)) if solver.p.is_cuda else None

    def __call__(self):
        s = self.solver
        out = torch.zeros((len(s.p), 6, 6), dtype=s.p.dtype, device=s.p.device)
        vectors = [wp.from_torch(v.contiguous(), dtype=wp.vec3d) for v in (s.n, s.ri, s.rj)]
        wp.launch(diagonal_blocks, dim=len(s.edges), inputs=[wp.from_torch(s.edges), *vectors,
                  *(wp.from_torch(v) for v in (s.kn, s.ks, s.kb, s.kt)), wp.from_torch(out)],
                  device=str(s.p.device), stream=self.stream)
        return out


class WarpElastic:
    def __init__(self, solver):
        wp.init()
        self.solver = solver
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(solver.device)) if solver.p.is_cuda else None

    def __call__(self, vector, dt, tolerance=.03, max_iterations=80):
        s = self.solver
        x, r, direction, ad, z = [torch.empty_like(vector) for _ in range(5)]
        partial = torch.empty(len(vector), dtype=vector.dtype, device=vector.device)
        wrap = wp.from_torch
        wx, wr, wd, wa, wz = [wrap(v) for v in (x, r, direction, ad, z)]
        wpartial, wpcond = wrap(partial), wrap(s.preconditioner)
        def launch(kernel, inputs):
            wp.launch(kernel, dim=len(vector), inputs=inputs, device=str(vector.device), stream=self.stream)
        launch(cg_initialize, [wrap(vector), wrap(s.block), wpcond, wrap(s.units), wrap(s.fixed),
                               wx, wr, wd, wpartial])
        rz = partial.sum().reshape(1); initial = float(rz)
        if initial == 0: return x
        for iteration in range(max_iterations):
            tangent = s.stiffness(direction)
            launch(cg_operator, [wrap(s.mass), wrap(s.fixed), wd, wrap(tangent), .25*dt*dt, wa, wpartial])
            denominator = partial.sum().reshape(1)
            launch(cg_update, [wx, wr, wd, wa, wpcond, wrap(rz), wrap(denominator), wz, wpartial])
            following = partial.sum().reshape(1)
            if iteration % 5 == 0 and float(following) <= tolerance*tolerance*initial:
                break
            launch(cg_direction, [wd, wz, wrap(following), wrap(rz)])
            rz = following
        s.inner_iterations += iteration+1
        return x/s.units
