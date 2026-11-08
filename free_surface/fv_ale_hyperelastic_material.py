"""Isotropic compressible neo-Hookean material, with log isochoric metric.

Reference-volume energy (Sierra/LAME neo-Hookean volumetric potential):
  W = mu/2 (tr(Bbar)-3) + K/4 (J^2-1-2 log J).
H=log(Bbar), symmetric traceless; J=rho0/rho. Thus B=J^(2/3) exp(H)
is positive definite with det(B)=J^2. Only a single volume state is used.
This is athermal elasticity: numerical heat is total specific energy minus
kinetic and elastic energy, and has no thermal pressure coupling.

The log transport source is the exact spectral derivative of
  D_t Bbar = L Bbar + Bbar L^T - (2/3) tr(L) Bbar.
The use of logarithms is a representation change, not plastic return.
"""
import torch


def symmetric_deviator(a):
    a = .5 * (a + a.transpose(-1, -2))
    I = torch.eye(3, dtype=a.dtype, device=a.device)
    return a - a.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None] * I / 3


def metric(H):
    from free_surface.fv_ale_symmetric_eigen import eigh3
    values, vectors = eigh3(symmetric_deviator(H))
    eigenvalues = values.exp()
    B = (vectors * eigenvalues[..., None, :]) @ vectors.transpose(-1, -2)
    return B, values, vectors


def response(H, rho, mu, bulk, rho0, acoustic_bound='conservative'):
    B, values, vectors = metric(H)
    J = rho0 / rho
    trace = B.diagonal(dim1=-2, dim2=-1).sum(-1)
    I = torch.eye(3, dtype=H.dtype, device=H.device)
    shear = (mu / J)[..., None, None] * (B - trace[..., None, None] * I / 3)
    pressure = .5 * bulk * (1 / J - J)
    sigma = shear - pressure[..., None, None] * I
    # expm1 suppresses cancellation near the stress-free reference state.
    logJ = J.log()
    Wv = .25 * bulk * (torch.expm1(2 * logJ) - 2 * logJ)
    Ws = .5 * mu * torch.expm1(values).sum(-1)
    elastic = (Wv + Ws) / rho0
    volume_tangent = .5 * bulk * (J + 1 / J)
    # A bound on all directions/polarizations of the current acoustic tensor.
    maximum = values[..., -1].exp()
    c2 = (volume_tangent + (mu / J) * ((7 / 3) * maximum + (5 / 9) * trace)) / rho
    if acoustic_bound=='tight':
        c2=maximum_acoustic_eigenvalue(values.exp(),J,mu,bulk)/rho
    elif acoustic_bound!='conservative':
        raise ValueError('unknown acoustic bound')
    return pressure, shear, sigma, elastic, c2, B, J


def log_metric_rate(H, L):
    _, values, R = metric(H)
    D = symmetric_deviator(L)
    spin = .5 * (L - L.transpose(-1, -2))
    d = R.transpose(-1, -2) @ D @ R
    w = R.transpose(-1, -2) @ spin @ R
    difference = values[..., :, None] - values[..., None, :]
    x = .5 * difference.abs()
    factor = torch.where(x < 1e-4, 1 + x*x/3 - x**4/45,
                         x / x.tanh().clamp(min=1e-300))
    source = 2 * factor * d - difference * w
    return symmetric_deviator(R @ source @ R.transpose(-1, -2))


def acoustic_tensor(B, J, n, mu, bulk):
    """d(sigma)n / d(velocity-gradient amplitude a), L=a tensor n.

    Used for verification and future matrix impedance; the current production
    face adaptation uses the conservative scalar speed bound from response().
    """
    I = torch.eye(3, dtype=B.dtype, device=B.device)
    b = (B @ n[..., None])[..., 0]
    beta = (n * b).sum(-1)
    trace = B.diagonal(dim1=-2, dim2=-1).sum(-1)
    nn = n[..., :, None] * n[..., None, :]
    cross = b[..., :, None] * n[..., None, :] + n[..., :, None] * b[..., None, :]
    Q = (mu / J)[..., None, None] * (beta[..., None, None] * I - (2 / 3) * cross +
                                      (5 / 9) * trace[..., None, None] * nn)
    return Q + (.5 * bulk * (J + 1 / J))[..., None, None] * nn


def maximum_acoustic_eigenvalue(eigenvalues, J, mu, bulk, return_beta=False):
    """Exact all-direction maximum of the acoustic tensor above (stress units).

    Let beta=n.Bbar.n, a<=beta<=b be the extreme metric eigenvalues, and
    A=5 tr(Bbar)/9 + K (J^2+1)/(2 mu). In the span of n and Bbar*n,
    Q/(mu/J) is [[A-beta/3, -2 |Bbar*n-beta*n|/3], [same,beta]].
    At fixed beta the largest possible squared off-diagonal direction length
    is (b-beta)(beta-a), attained by n in the extreme-eigenvector plane.
    The larger eigenvalue is
      f(beta)=(A+2 beta/3+sqrt(D+C beta))/2,
      D=A^2-16 a b/9, C=16(a+b)/9-8 A/3.
    It is concave. Thus the maximum lies at an endpoint or at
    beta=(9 C^2/16-D)/C for C<0. Evaluate the equivalent noncancelling 2x2
    formula at those candidates. A tiny upward rounding margin is included.
    This tight bound is not enabled in response() by default.
    """
    a,b=eigenvalues[...,0],eigenvalues[...,-1]
    trace=eigenvalues.sum(-1)
    A=5*trace/9+bulk*(J.square()+1)/(2*mu)
    C=16*(a+b)/9-8*A/3
    D=A.square()-16*a*b/9
    safeC=torch.where(C<0,C,-torch.ones_like(C))
    beta=((9*C.square()/16-D)/safeC).clamp(min=a,max=b)
    beta=torch.where(C<0,beta,b)
    candidates=torch.stack((a,b,beta),-1)
    off2=((b[...,None]-candidates)*(candidates-a[...,None])).clamp(min=0)
    difference=A[...,None]-4*candidates/3
    roots=.5*(A[...,None]+2*candidates/3+torch.sqrt(difference.square()+16*off2/9))
    largest,index=roots.max(-1)
    value=(mu/J)*largest*(1+128*torch.finfo(eigenvalues.dtype).eps)
    if return_beta:return value,candidates.gather(-1,index[...,None])[...,0]
    return value
