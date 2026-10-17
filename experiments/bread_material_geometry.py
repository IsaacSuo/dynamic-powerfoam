"""Bounded-power reference face clipping and support-volume quadrature.

The circle bounding each face is approximated by an inscribed polygon. All
regular neighbors clip it; faces are NOT the complete two-ball intersection
disks. Volumes use the divergence theorem about a point inside each cell,
including sampled spherical boundary patches. Optical density is not mass.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit, prange
from scipy import sparse
from scipy.stats import qmc

from experiments.powersim_bread_demo import CHECKPOINT


def neighbors(n, pairs):
    graph = sparse.coo_matrix((np.ones(2*len(pairs)),
                              (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                             shape=(n, n)).tocsr()
    return graph.indptr.astype(np.int64), graph.indices.astype(np.int64)


@njit(parallel=True, cache=True)
def clipped_faces(p, r, pairs, offsets, ids, segments=64):
    count = len(pairs)
    areas, centers = np.zeros(count), np.zeros((count, 3))
    for edge in prange(count):
        i, j = pairs[edge]
        d = p[j]-p[i]; length = np.sqrt(np.sum(d*d)); normal = d/length
        a = (length*length+r[i]*r[i]-r[j]*r[j])/(2*length)
        disk2 = r[i]*r[i]-a*a
        if disk2 <= 0:
            continue
        axis = np.array([1., 0., 0.]) if abs(normal[0]) < .8 else np.array([0., 1., 0.])
        u = np.cross(normal, axis); u /= np.sqrt(np.sum(u*u)); v = np.cross(normal, u)
        origin = p[i]+a*normal
        capacity = segments+offsets[i+1]-offsets[i]+4
        polygon, temporary = np.empty((capacity, 2)), np.empty((capacity, 2))
        size = segments
        for t in range(segments):
            theta = 2*np.pi*t/segments
            polygon[t, 0], polygon[t, 1] = np.sqrt(disk2)*np.cos(theta), np.sqrt(disk2)*np.sin(theta)
        for neighbor in range(offsets[i], offsets[i+1]):
            k = ids[neighbor]
            if k == j:
                continue
            delta = p[k]-p[i]
            alpha, beta = np.sum(delta*u), np.sum(delta*v)
            bound = .5*(np.sum(delta*delta)+r[i]*r[i]-r[k]*r[k])-a*np.sum(delta*normal)
            new_size = 0
            if size == 0:
                break
            last = size-1
            for current in range(size):
                va = alpha*polygon[last, 0]+beta*polygon[last, 1]-bound
                vb = alpha*polygon[current, 0]+beta*polygon[current, 1]-bound
                if (va <= 0) != (vb <= 0):
                    fraction = va/(va-vb)
                    temporary[new_size] = polygon[last]+fraction*(polygon[current]-polygon[last])
                    new_size += 1
                if vb <= 0:
                    temporary[new_size] = polygon[current]; new_size += 1
                last = current
            size = new_size
            polygon, temporary = temporary, polygon
        if size < 3:
            continue
        twice, cx, cy = 0., 0., 0.
        for t in range(size):
            nxt = (t+1)%size
            cross = polygon[t, 0]*polygon[nxt, 1]-polygon[nxt, 0]*polygon[t, 1]
            twice += cross
            cx += (polygon[t, 0]+polygon[nxt, 0])*cross
            cy += (polygon[t, 1]+polygon[nxt, 1])*cross
        if twice > 1e-14*disk2:
            areas[edge] = .5*twice
            centers[edge] = origin+(cx/(3*twice))*u+(cy/(3*twice))*v
    return areas, centers


@njit(parallel=True, cache=True)
def sphere_volume_terms(p, r, c, offsets, ids, rays, active):
    volume = np.zeros(len(p))
    hits = np.zeros(len(p), dtype=np.int64)
    for i in prange(len(p)):
        if not active[i]:
            continue
        # The bounded cell has no sphere patch if every ray fails. Sampling
        # only affects exposed support surfaces, not enclosed polyhedral cells.
        total = 0.
        for ray in rays:
            inside = True
            for neighbor in range(offsets[i], offsets[i+1]):
                j = ids[neighbor]; delta = p[j]-p[i]
                bound = .5*(np.sum(delta*delta)+r[i]*r[i]-r[j]*r[j])
                if r[i]*np.sum(delta*ray) > bound:
                    inside = False; break
            if inside:
                value = r[i]+np.sum((p[i]-c[i])*ray)
                total += max(value, 0.)
                hits[i] += 1
        volume[i] = total*4*np.pi*r[i]*r[i]/(3*len(rays))
    return volume, hits


def volume_from_faces(p, r, pairs, area, centers, offsets, ids, active, exponent):
    weight, centroid = np.zeros(len(p)), np.zeros_like(p)
    for endpoint in (0, 1):
        np.add.at(weight, pairs[:, endpoint], area)
        np.add.at(centroid, pairs[:, endpoint], area[:, None]*centers)
    have = weight > 0
    centroid[have] /= weight[have, None]
    centroid[~have] = p[~have]
    n = p[pairs[:, 1]]-p[pairs[:, 0]]; n /= np.linalg.norm(n, axis=1)[:, None]
    volume = np.zeros(len(p))
    for endpoint, sign in ((0, 1), (1, -1)):
        distance = sign*np.sum((centers-centroid[pairs[:, endpoint]])*n, axis=1)
        if np.min(distance[area > 0], initial=0.) < -1e-9:
            raise ValueError('Reference volume point is outside its clipped faces')
        np.add.at(volume, pairs[:, endpoint], area*np.maximum(distance, 0.)/3)
    samples = qmc.Sobol(2, scramble=True, seed=812).random_base2(exponent)
    z, phi = 1-2*samples[:, 0], 2*np.pi*samples[:, 1]
    rays = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
    surface, hits = sphere_volume_terms(p, r, centroid, offsets, ids, rays, active)
    return volume+surface, surface, hits


def geometry_checks():
    p = np.array([[-.2, 0., 0.], [.2, 0., 0.]])
    r = np.full(2, .4); pairs = np.array([[0, 1]], dtype=np.int64)
    offsets, ids = neighbors(2, pairs)
    area, c = clipped_faces(p, r, pairs, offsets, ids, 128)
    expected = np.pi*(.4**2-.2**2)
    assert abs(area[0]/expected-1) < .0005
    volume, _, _ = volume_from_faces(p, r, pairs, area, c, offsets, ids, np.ones(2, dtype=bool), 16)
    expected_volume = 4*np.pi*.4**3/3-np.pi*.2**2*(.4-.2/3)
    assert np.max(abs(volume/expected_volume-1)) < .001
    # Third sphere splits a two-ball disk in half (centers/radii chosen so its
    # power plane restricts y<=0 on x=0). This catches un-clipped lens areas.
    pp = np.vstack((p, [0., .2, 0.])); rr = np.r_[r, .4]
    pairs3 = np.array([[0, 1], [0, 2], [1, 2]])
    oo, ii = neighbors(3, pairs3)
    a3, _ = clipped_faces(pp, rr, pairs3, oo, ii, 128)
    assert abs(a3[0]/area[0]-.5) < 1e-12
    return dict(two_ball_face_relative_error=float(abs(area[0]/expected-1)),
                two_ball_volume_relative_error=float(np.max(abs(volume/expected_volume-1))),
                third_ball_clips_face_to_half=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('output/powersim_bread'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    import torch
    checks = geometry_checks(); print('Geometry analytic checks:', checks, flush=True)
    state = torch.load(CHECKPOINT/'model.pt', map_location='cpu', weights_only=True)
    p = state['points'].numpy().astype(float)
    r = torch.nn.functional.softplus(state['radii'], beta=100).numpy().astype(float)
    with np.load(args.out/'regular_candidates.npz') as archive:
        all_pairs = archive['pairs']; candidates = all_pairs[archive['possible']]
        active = np.zeros(len(p), dtype=bool); active[archive['active_unbounded']] = True
    offsets, ids = neighbors(len(p), all_pairs)
    print('Clipping', len(candidates), 'candidate faces', flush=True)
    area, centers = clipped_faces(p, r, candidates, offsets, ids, 64)
    keep = area > 0; pairs, area, centers = candidates[keep], area[keep], centers[keep]
    print('Positive polygon faces:', len(pairs), flush=True)
    # Refine a deterministic representative subset, including all tiny areas.
    probe = np.unique(np.r_[np.linspace(0, len(pairs)-1, 4096).astype(int), np.argsort(area)[:1024]])
    refined, _ = clipped_faces(p, r, pairs[probe], offsets, ids, 128)
    relative_area = float(np.sum(abs(refined-area[probe]))/np.sum(refined))
    print('Face refinement relative L1:', relative_area, flush=True)
    vol8, surf8, _ = volume_from_faces(p, r, pairs, area, centers, offsets, ids, active, 10)
    volume, surface, hits = volume_from_faces(p, r, pairs, area, centers, offsets, ids, active, 12)
    if np.any(~np.isfinite(volume)) or np.any(volume < 0):
        raise ValueError('Invalid support volume')
    positive = volume > 1e-18
    edge_keep = positive[pairs].all(axis=1)
    report = dict(checks=checks, positive_faces=int(keep.sum()), retained_edges=int(edge_keep.sum()),
                  positive_volume_sites=int(positive.sum()), removed_zero_volume_sites=int((~positive).sum()),
                  volume=float(volume.sum()), exposed_sphere_volume_term=float(surface.sum()),
                  sphere_quadrature_1024_vs_4096_relative_L1=float(np.sum(abs(volume-vol8))/volume.sum()),
                  face_64_vs_128_sample_relative_L1=relative_area,
                  mass_volume_quantiles=np.quantile(volume[positive], [0, .01, .5, .99, 1]).tolist(),
                  note='Material domain is bounded POWER SUPPORT, not optically inferred solid occupancy. Circle polygon and sphere quadrature approximations recorded.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(args.out/'material_geometry.npz', points=p, radii=r, volumes=volume,
                        pairs=pairs[edge_keep], areas=area[edge_keep], contacts=centers[edge_keep], keep=positive)
    (args.out/'material_geometry.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
