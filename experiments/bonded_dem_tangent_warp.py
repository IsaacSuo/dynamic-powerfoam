"""Fused CUDA evaluation of the existing frozen-geometry bond tangent."""
import torch
import warp as wp


@wp.kernel
def tangent(q: wp.array2d(dtype=wp.float64), edges: wp.array2d(dtype=wp.int64),
            n: wp.array(dtype=wp.vec3d), ri: wp.array(dtype=wp.vec3d), rj: wp.array(dtype=wp.vec3d),
            kn: wp.array(dtype=wp.float64), ks: wp.array(dtype=wp.float64),
            kb: wp.array(dtype=wp.float64), kt: wp.array(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    e = wp.tid()
    i, j = edges[e, 0], edges[e, 1]
    ui, uj = wp.vec3d(q[i, 0], q[i, 1], q[i, 2]), wp.vec3d(q[j, 0], q[j, 1], q[j, 2])
    wi, wj = wp.vec3d(q[i, 3], q[i, 4], q[i, 5]), wp.vec3d(q[j, 3], q[j, 4], q[j, 5])
    du, dw = uj-ui+wp.cross(wj, rj[e])-wp.cross(wi, ri[e]), wj-wi
    f = ks[e]*du+(kn[e]-ks[e])*wp.dot(du, n[e])*n[e]
    m = kb[e]*dw+(kt[e]-kb[e])*wp.dot(dw, n[e])*n[e]
    ti, tj = -wp.cross(ri[e], f)-m, wp.cross(rj[e], f)+m
    for axis in range(3):
        wp.atomic_add(out, i, axis, -f[axis])
        wp.atomic_add(out, j, axis, f[axis])
        wp.atomic_add(out, i, axis+3, ti[axis])
        wp.atomic_add(out, j, axis+3, tj[axis])


class WarpTangent:
    def __init__(self, solver):
        wp.init()
        self.solver = solver
        self.stream = wp.stream_from_torch(torch.cuda.current_stream())
        self.pointer = None

    def __call__(self, q):
        s = self.solver
        key = (s.n.data_ptr(), s.ri.data_ptr(), s.rj.data_ptr())
        if key != self.pointer:
            self.arrays = [wp.from_torch(s.edges), wp.from_torch(s.n, dtype=wp.vec3d),
                wp.from_torch(s.ri, dtype=wp.vec3d), wp.from_torch(s.rj, dtype=wp.vec3d),
                *(wp.from_torch(v) for v in (s.kn, s.ks, s.kb, s.kt))]
            self.pointer = key
        out = torch.zeros_like(q)
        wp.launch(tangent, dim=len(s.edges), inputs=[wp.from_torch(q), *self.arrays, wp.from_torch(out)],
                  device='cuda:0', stream=self.stream)
        return out


def check():
    import numpy as np
    from experiments.bonded_dem_linear_gpu import LinearBondsGPU
    rng = np.random.default_rng(81)
    p = rng.normal(size=(50, 3))
    pairs = np.array([(i, j) for i in range(50) for j in range(i+1, 50)])
    contacts = .5*(p[pairs[:, 0]]+p[pairs[:, 1]])+rng.normal(0, .1, (len(pairs), 3))
    s = LinearBondsGPU(p, np.ones(50), pairs, np.ones(len(pairs)), contacts, np.ones(50), np.zeros(50, bool))
    q = s.tensor(rng.normal(size=(50, 6)))
    fused = WarpTangent(s)
    error = float(torch.linalg.vector_norm(fused(q)-s.stiffness(q))/torch.linalg.vector_norm(s.stiffness(q)))
    assert error < 1e-12, error
    print('Warp/PyTorch tangent relative error:', error, flush=True)


if __name__ == '__main__':
    check()
