"""Actual power-face section properties for an equal-radius bonded foam patch.

The face of bond (i, j) is the part of their bisector plane inside cell i's support
sphere and not closer to any other center (equal radii: Voronoi). Integrated on a
fine in-plane grid; area and centroid are checked against the stored CAD values.
Returns, per bond, the in-plane second-moment tensor about the face centroid
embedded in 3D (zero along the bond normal) and the polar moment.
"""
import numpy as np


def face_sections(geometry, resolution=801):
    points = np.asarray(geometry['points'], float)
    radii = np.asarray(geometry['radii'], float)
    if not np.allclose(radii, radii[0]):
        raise ValueError('Only equal-radius patches are supported.')
    r = radii[0]
    rows = []
    for (i, j) in np.asarray(geometry['pairs']):
        normal = points[j]-points[i]
        normal /= np.linalg.norm(normal)
        helper = np.eye(3)[np.argmin(np.abs(normal))]
        e1 = np.cross(normal, helper); e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        mid = .5*(points[i]+points[j])
        u = np.linspace(-r, r, resolution)
        du = u[1]-u[0]
        a, b = np.meshgrid(u, u, indexing='ij')
        x = mid+a[..., None]*e1+b[..., None]*e2
        inside = np.linalg.norm(x-points[i], axis=-1) <= r
        own = np.linalg.norm(x-points[i], axis=-1)
        for k in range(len(points)):
            if k not in (i, j):
                inside &= own <= np.linalg.norm(x-points[k], axis=-1)
        area = inside.sum()*du*du
        ca, cb = a[inside].mean(), b[inside].mean()
        da, db = a[inside]-ca, b[inside]-cb
        I2 = np.array([[np.sum(db*db), -np.sum(da*db)], [-np.sum(da*db), np.sum(da*da)]])*du*du
        # I2 is the second-moment tensor about in-plane axes (e1, e2): about e1 it is
        # the integral of b^2, about e2 of a^2. Embed in 3D, zero along the normal.
        basis = np.stack((e1, e2))
        rows.append(dict(area=area, centroid=mid+ca*e1+cb*e2, bending=basis.T@I2@basis,
                         polar=float(np.trace(I2))))
    return rows


if __name__ == '__main__':
    geometry = dict(np.load('output/foam_fem/reference_edge007/patch.npz'))
    rows = face_sections(geometry)
    area = np.array([row['area'] for row in rows])
    centroid = np.array([row['centroid'] for row in rows])
    print('area max rel error vs CAD', np.max(np.abs(area/geometry['areas']-1)))
    print('centroid max error vs CAD', np.max(np.abs(centroid-geometry['contacts'])))
    equivalent = geometry['areas']**2/(4*np.pi)
    for b in (0, 8, 10):
        eig = np.linalg.eigvalsh(rows[b]['bending'])[1:]
        print('bond', b, 'area %.4f, bending principal / equivalent circle:' % area[b], np.round(eig/equivalent[b], 3),
              ' polar / (2 I_eq): %.3f' % (rows[b]['polar']/(2*equivalent[b])))
