"""Fixed-reference nodal tetrahedron volume energy for rigid-cell centers.

This adds no cell deformation DOFs. Connectivity is never rebuilt by rendering.
The builder currently supports equal-radius Cartesian specimens only; it is not
a general weighted Delaunay or nonconvex material-domain mesher.
"""
import itertools

import numpy as np
import torch


def grid_tetrahedra(points, radii, reflected=False):
    """Six Freudenthal simplices per grid box; B reflects A in the x axis.

    For equal radii these are valid choices of the degenerate Delaunay dual.
    The construction covers each box exactly and matches neighboring faces.
    """
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if not np.allclose(radii, radii[0], rtol=1e-12, atol=0):
        raise ValueError('This grid builder requires equal radii.')
    axes = [np.unique(points[:, k]) for k in range(3)]
    shape = tuple(map(len, axes))
    if min(shape) < 2 or np.prod(shape) != len(points):
        raise ValueError('Expected a complete three-dimensional Cartesian grid.')
    lookup = {tuple(np.searchsorted(axes[k], p[k]) for k in range(3)): i
              for i, p in enumerate(points)}
    if len(lookup) != len(points):
        raise ValueError('Duplicate centers.')
    tetrahedra = []
    for base in itertools.product(*(range(n-1) for n in shape)):
        for order in itertools.permutations(range(3)):
            vertex = np.array(base)
            vertices = [vertex.copy()]
            for axis in order:
                vertex = vertex.copy()
                vertex[axis] += 1
                vertices.append(vertex)
            if reflected:
                for vertex in vertices:
                    vertex[0] = shape[0]-1-vertex[0]
            tetrahedra.append([lookup[tuple(v)] for v in vertices])
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    p = points[tetrahedra]
    determinant = np.linalg.det(p[:, 1:]-p[:, :1])
    negative = determinant < 0
    tetrahedra[negative, 1], tetrahedra[negative, 2] = (
        tetrahedra[negative, 2].copy(), tetrahedra[negative, 1].copy())
    return tetrahedra


def mesh_topology(points, tetrahedra):
    """Combinatorial checks and boundary, independent of volume derivatives.

    Face incidence alone cannot certify absence of arbitrary geometric overlap.
    Non-overlap here follows from the Cartesian subdivision in the builder.
    """
    points = np.asarray(points)
    tetrahedra = np.asarray(tetrahedra)
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or not len(tetrahedra):
        raise ValueError('Expected nonempty tetrahedra with four indices each.')
    if np.any(tetrahedra < 0) or np.any(tetrahedra >= len(points)):
        raise ValueError('Invalid vertex index.')
    if len({tuple(sorted(t)) for t in tetrahedra}) != len(tetrahedra):
        raise ValueError('Duplicate tetrahedra.')
    p = points[tetrahedra]
    volumes = np.linalg.det(p[:, 1:]-p[:, :1])/6
    if np.any(volumes <= 0) or not np.isfinite(volumes).all():
        raise ValueError('Reference tetrahedra must be nondegenerate and positively oriented.')
    faces = {}
    for tet in tetrahedra:
        for opposite in range(4):
            face = [int(tet[k]) for k in range(4) if k != opposite]
            if opposite % 2:
                face[0], face[1] = face[1], face[0]
            sign = (-1)**sum(face[a] > face[b] for a in range(3) for b in range(a+1, 3))
            faces.setdefault(tuple(sorted(face)), []).append(sign)
    if any(len(signs) > 2 for signs in faces.values()):
        raise ValueError('A face has more than two incident tetrahedra.')
    if any(len(signs) == 2 and sum(signs) != 0 for signs in faces.values()):
        raise ValueError('Shared face orientations do not cancel.')
    boundary_faces = np.array([face for face, signs in faces.items() if len(signs) == 1],
                              dtype=np.int64)
    boundary = np.zeros(len(points), dtype=bool)
    boundary[boundary_faces.ravel()] = True
    if len(np.unique(tetrahedra)) != len(points):
        raise ValueError('Some centers have no incident tetrahedron.')
    return boundary, dict(vertices=len(points), tetrahedra=len(tetrahedra),
        boundary_faces=len(boundary_faces), interior_faces=len(faces)-len(boundary_faces),
        boundary_vertices=int(boundary.sum()), interior_vertices=int((~boundary).sum()),
        reference_volume=float(volumes.sum()), minimum_reference_volume=float(volumes.min()),
        nonoverlap_basis='Complete conforming Cartesian box subdivision; not face incidence alone.')


class NodalTetraVolume(torch.nn.Module):
    """E = c/2 sum_i k_v exp(theta_i) W_i (J_i-1)^2.

    W_i = sum_{T contains i} V_T^0 / 4, J_i = sum V_T(x) / sum V_T^0.
    c = material_volume / sum V_T^0. Masses and inertia are not modified.
    Forces, positions and multipliers remain differentiable, including the
    analytic force expression; no detach or autograd call inside force().
    """
    def __init__(self, reference, tetrahedra, material_volume, k_v=1.):
        super().__init__()
        reference = torch.as_tensor(reference, dtype=torch.float64)
        boundary, topology = mesh_topology(reference.cpu().numpy(), tetrahedra)
        if material_volume <= 0 or k_v <= 0:
            raise ValueError('Positive material volume and base volume modulus required.')
        self.register_buffer('reference', reference)
        self.register_buffer('tetrahedra', torch.as_tensor(tetrahedra, dtype=torch.long,
                                                         device=reference.device))
        self.register_buffer('boundary', torch.as_tensor(boundary, device=reference.device))
        volumes, _ = self.volumes_and_gradients(reference)
        self.register_buffer('reference_volumes', volumes)
        self.register_buffer('nodal_reference_sum', self.nodal_sum(volumes))
        self.register_buffer('weights', self.nodal_reference_sum/4)
        self.register_buffer('normalization', material_volume/volumes.sum())
        self.k_v = float(k_v)
        self.topology = topology

    def volumes_and_gradients(self, x):
        p = x[self.tetrahedra]
        b, c, d = (p[:, k]-p[:, 0] for k in (1, 2, 3))
        gb = torch.linalg.cross(c, d)/6
        gc = torch.linalg.cross(d, b)/6
        gd = torch.linalg.cross(b, c)/6
        volumes = (b*gb).sum(dim=-1)
        gradients = torch.stack((-gb-gc-gd, gb, gc, gd), dim=1)
        return volumes, gradients

    def nodal_sum(self, volumes):
        return volumes.new_zeros(len(self.reference)).index_add(
            0, self.tetrahedra.flatten(), volumes[:, None].expand(-1, 4).reshape(-1))

    def validate_configuration(self, x):
        volumes, _ = self.volumes_and_gradients(x)
        if not bool(torch.isfinite(volumes).all()) or bool((volumes <= 0).any()):
            raise ValueError('Nonpositive/nonfinite signed tetrahedron volume; reject this state.')

    def evaluate(self, x, log_multipliers):
        """Returns energy, center forces and J; call validation at solver acceptance.

        Rotation DOFs do not appear. Objectivity also implies zero total moment
        of center forces, not zero moment of each individual pair contribution.
        """
        if x.shape != self.reference.shape or log_multipliers.shape != (len(x),):
            raise ValueError('Expected positions (N,3) and per-cell log multipliers (N,).')
        volumes, gradients = self.volumes_and_gradients(x)
        J = self.nodal_sum(volumes)/self.nodal_reference_sum
        pressure = self.k_v*torch.exp(log_multipliers)*(J-1)
        energy = .5*self.normalization*(self.weights*pressure*(J-1)).sum()
        tet_pressure = pressure[self.tetrahedra].mean(dim=1)
        contributions = -self.normalization*tet_pressure[:, None, None]*gradients
        force = torch.zeros_like(x).index_add(0, self.tetrahedra.flatten(),
                                              contributions.reshape(-1, 3))
        return energy, force, J

    def reference_tangent(self, log_multipliers):
        """Analytic stress-free position Hessian; no geometric stiffness at J=1."""
        if log_multipliers.shape != (len(self.reference),):
            raise ValueError('Expected per-cell log multipliers (N,).')
        _, gradients = self.volumes_and_gradients(self.reference)
        n = len(self.reference)
        B = self.reference.new_zeros((n, 3*n))
        for local_owner in range(4):
            row = self.tetrahedra[:, local_owner]
            for local_vertex in range(4):
                for component in range(3):
                    column = 3*self.tetrahedra[:, local_vertex]+component
                    B.index_put_((row, column), gradients[:, local_vertex, component],
                                 accumulate=True)
        B = B/self.nodal_reference_sum[:, None]
        weight = self.normalization*self.k_v*torch.exp(log_multipliers)*self.weights
        return B.T@(weight[:, None]*B)
