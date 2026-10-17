"""Fused analytic directional derivative of the UNCHANGED midpoint bond law.

Only replaces NewtonBonds.jacobian_vector. Primal trial, guards, line search,
convergence tolerances, history commit and the elastic preconditioner stay intact.
The caller supplies a valid primal iterate (already checked by trial).
"""
import torch
import warp as wp


@wp.func
def read3(a: wp.array2d(dtype=wp.float64), i: wp.int64, offset: int):
    return wp.vec3d(a[i, offset], a[i, offset+1], a[i, offset+2])


@wp.func
def rotate_jvp(v: wp.vec3d, n0: wp.vec3d, n: wp.vec3d, dn: wp.vec3d,
               spin: wp.vec3d, dspin: wp.vec3d):
    one = wp.float64(1.0)
    raw = wp.dot(n0, n)
    cosine = wp.clamp(raw, -one, one)
    dc = wp.dot(n0, dn)
    if raw < -one or raw > one:
        dc = wp.float64(0.0)
    k, dk = wp.cross(n0, n), wp.cross(n0, dn)
    kv = wp.cross(k, v)
    kkv = wp.cross(k, kv)
    t = v+kv+kkv/(one+cosine)
    dt = wp.cross(dk, v)+(wp.cross(dk, kv)+wp.cross(k, wp.cross(dk, v)))/(one+cosine)
    dt = dt-kkv*dc/((one+cosine)*(one+cosine))
    length = wp.length(n0+n)
    b = (n0+n)/length
    db = (dn-b*wp.dot(b, dn))/length
    angle = wp.dot(spin, b)
    da = wp.dot(dspin, b)+wp.dot(spin, db)
    c, s = wp.cos(angle), wp.sin(angle)
    nt = wp.dot(n, t)
    value = c*t+s*wp.cross(n, t)+(one-c)*nt*n
    derivative = c*dt-s*da*t+c*da*wp.cross(n, t)
    derivative = derivative+s*(wp.cross(dn, t)+wp.cross(n, dt))+s*da*nt*n
    derivative = derivative+(one-c)*((wp.dot(dn, t)+wp.dot(n, dt))*n+nt*dn)
    return value, derivative


@wp.kernel
def midpoint_jvp(y: wp.array2d(dtype=wp.float64), direction: wp.array2d(dtype=wp.float64),
                 x: wp.array(dtype=wp.vec3d), radii: wp.array(dtype=wp.float64),
                 edges: wp.array2d(dtype=wp.int64), normal: wp.array(dtype=wp.vec3d),
                 offset: wp.array(dtype=wp.vec3d), fn: wp.array(dtype=wp.float64),
                 fs: wp.array(dtype=wp.vec3d), mt: wp.array(dtype=wp.float64),
                 mb: wp.array(dtype=wp.vec3d), kn: wp.array(dtype=wp.float64),
                 ks: wp.array(dtype=wp.float64), kt: wp.array(dtype=wp.float64),
                 kb: wp.array(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    e = wp.tid()
    i, j = edges[e, 0], edges[e, 1]
    half, quarter = wp.float64(0.5), wp.float64(0.25)
    ui, uj = read3(y, i, 0), read3(y, j, 0)
    wi, wj = read3(y, i, 3), read3(y, j, 3)
    dui, duj = read3(direction, i, 0), read3(direction, j, 0)
    dwi, dwj = read3(direction, i, 3), read3(direction, j, 3)
    mi, mj = x[i]+half*ui, x[j]+half*uj
    displacement = mj-mi
    length = wp.length(displacement)
    n = displacement/length
    dd = half*(duj-dui)
    dn = (dd-n*wp.dot(n, dd))/length
    spin, dspin = quarter*(wi+wj), quarter*(dwi+dwj)
    rotated, drotated = rotate_jvp(offset[e], normal[e], n, dn, spin, dspin)
    contact = half*(mi+mj)+half*(radii[i]-radii[j])*n+rotated
    dcontact = quarter*(dui+duj)+half*(radii[i]-radii[j])*dn+drotated
    ri, rj = contact-mi, contact-mj
    dri, drj = dcontact-half*dui, dcontact-half*duj
    fs0, dfs0 = rotate_jvp(fs[e], normal[e], n, dn, spin, dspin)
    mb0, dmb0 = rotate_jvp(mb[e], normal[e], n, dn, spin, dspin)
    du = uj-ui+wp.cross(wj, rj)-wp.cross(wi, ri)
    ddu = duj-dui+wp.cross(dwj, rj)+wp.cross(wj, drj)-wp.cross(dwi, ri)-wp.cross(wi, dri)
    dw, ddw = wj-wi, dwj-dwi
    normal_displacement, twist = wp.dot(du, n), wp.dot(dw, n)
    dnormal_displacement = wp.dot(ddu, n)+wp.dot(du, dn)
    dtwist = wp.dot(ddw, n)+wp.dot(dw, dn)
    fn_mid = fn[e]+half*kn[e]*normal_displacement
    dfn_mid = half*kn[e]*dnormal_displacement
    fs_mid = fs0+half*ks[e]*(du-normal_displacement*n)
    dfs_mid = dfs0+half*ks[e]*(ddu-dnormal_displacement*n-normal_displacement*dn)
    mt_mid = mt[e]+half*kt[e]*twist
    dmt_mid = half*kt[e]*dtwist
    dmb_mid = dmb0+half*kb[e]*(ddw-dtwist*n-twist*dn)
    f = fn_mid*n+fs_mid
    df = dfn_mid*n+fn_mid*dn+dfs_mid
    dcouple = dmt_mid*n+mt_mid*dn+dmb_mid
    dti = wp.cross(dri, f)+wp.cross(ri, df)+dcouple
    dtj = -wp.cross(drj, f)-wp.cross(rj, df)-dcouple
    for axis in range(3):
        wp.atomic_add(out, i, axis, df[axis])
        wp.atomic_add(out, j, axis, -df[axis])
        wp.atomic_add(out, i, axis+3, dti[axis])
        wp.atomic_add(out, j, axis+3, dtj[axis])


class WarpJacobian:
    def __init__(self, solver):
        wp.init()
        self.solver = solver
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(solver.device)) if solver.p.is_cuda else None

    def __call__(self, y, direction, dt, external):
        s = self.solver
        out = torch.zeros_like(y)
        vector = lambda v: wp.from_torch(v.contiguous(), dtype=wp.vec3d)
        inputs = [wp.from_torch(y), wp.from_torch(direction), vector(s.x), wp.from_torch(s.radii),
                  wp.from_torch(s.edges), vector(s.normal), vector(s.offset), wp.from_torch(s.fn),
                  vector(s.fs), wp.from_torch(s.mt), vector(s.mb),
                  *(wp.from_torch(a) for a in (s.kn, s.ks, s.kt, s.kb)), wp.from_torch(out)]
        wp.launch(midpoint_jvp, dim=len(s.edges), inputs=inputs, device=str(y.device), stream=self.stream)
        result = s.mass*direction-.5*dt*dt*out
        return torch.where(s.fixed[:, None], direction, result)
