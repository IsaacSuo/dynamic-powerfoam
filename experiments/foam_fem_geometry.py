"""Shared bounded-power solid for independent FEM and bonded DEM.

Equal radii on a 4 x m x m lattice (m=2 bar, m=4 thick block) make power regions
axis-aligned boxes. Intersect
each support sphere with its exact region, then fragment all volumes to obtain
a conforming mesh. The extra x=3 cut resolves the loaded surface exactly.
"""
import argparse
import itertools
import json
from pathlib import Path

import gmsh
import numpy as np


def build(out, mesh_size, edge_size=None, cross=2, cells='sphere'):
    """cells='sphere': support sphere ∩ power region (original specimens).
    cells='cube': closed power cells as in a full Power Foam, where the material is
    bounded by air cells; for this lattice each cell is a unit cube and the solid is
    a box. The cube radius is set so the DEM inertia proxy 0.4 m r^2 equals the exact
    cube inertia m/6."""
    assert cells in ('sphere', 'cube')
    out.mkdir(parents=True, exist_ok=False)
    across = np.arange(cross)-(cross-1)/2
    points = np.array(list(itertools.product(range(4), across, across)), dtype=float)
    cell_count = len(points)
    # Interior faces at the Voronoi mid-planes; outer cells extend to their sphere.
    reach = .9 if cells == 'sphere' else .5
    def span(value, values):
        return (value-reach if value == values.min() else value-.5, value+reach if value == values.max() else value+.5)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('bounded_power_bar')
        occ = gmsh.model.occ
        pieces = []
        for x, y, z in points:
            left, right = (x-.5 if x else -reach), (x+.5 if x < 3 else 3+reach)
            yr, zr = span(y, across), span(z, across)
            intervals = [(left, right)] if x < 3 else [(left, 3.), (3., right)]
            for lo, hi in intervals:
                box = occ.addBox(lo, yr[0], zr[0], hi-lo, yr[1]-yr[0], zr[1]-zr[0])
                if cells == 'cube':
                    pieces.append((3, box))
                    continue
                sphere = occ.addSphere(x, y, z, .9)
                intersection, _ = occ.intersect([(3, sphere)], [(3, box)])
                pieces.extend(intersection)
        occ.fragment(pieces[:1], pieces[1:])
        occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        cell_masses = np.zeros(cell_count)
        face_owners, face_uses, volume_owner = {}, {}, {}
        for dim, tag in volumes:
            center = np.array(occ.getCenterOfMass(dim, tag))
            owner = int(np.argmin(np.linalg.norm(points-center, axis=1)))
            volume_owner[tag] = owner
            cell_masses[owner] += occ.getMass(dim, tag)
            for _, face in gmsh.model.getBoundary([(dim, tag)], oriented=False):
                face_owners.setdefault(face, set()).add(owner)
                face_uses[face] = face_uses.get(face, 0)+1
        faces = {}
        for face, owners in face_owners.items():
            if len(owners) != 2:
                continue
            pair = tuple(sorted(owners))
            area = occ.getMass(2, face)
            center = np.array(occ.getCenterOfMass(2, face))
            if pair not in faces:
                faces[pair] = [0., np.zeros(3)]
            faces[pair][0] += area
            faces[pair][1] += area*center
        pairs = np.array(sorted(faces), dtype=np.int64)
        areas = np.array([faces[tuple(pair)][0] for pair in pairs])
        contacts = np.array([faces[tuple(pair)][1]/faces[tuple(pair)][0] for pair in pairs])
        assert len(pairs) == 3*cross*cross+2*4*cross*(cross-1), len(pairs)  # axis neighbours
        assert np.all(cell_masses > 0)
        assert np.max(np.abs(np.sum((contacts-.5*(points[pairs[:, 0]]+points[pairs[:, 1]]))*
            (points[pairs[:, 1]]-points[pairs[:, 0]]), axis=1))) < 1e-8
        gmsh.option.setNumber('Mesh.MeshSizeMin', mesh_size if edge_size is None else edge_size)
        gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        refined_curves = []
        if edge_size is not None:
            # Refine physical sphere-intersection grooves and clamp rim. These
            # are edges shared by exterior surfaces of distinct power cells,
            # not arbitrary internal FEM partitions or smooth CAD seams.
            edge_owners = {}
            for face, count in face_uses.items():
                if count != 1:
                    continue
                for _, curve in gmsh.model.getBoundary([(2, face)], oriented=False):
                    edge_owners.setdefault(curve, set()).update(face_owners[face])
            refined_curves = [curve for curve, owners in edge_owners.items() if len(owners) > 1]
            assert refined_curves
            distance = gmsh.model.mesh.field.add('Distance')
            gmsh.model.mesh.field.setNumbers(distance, 'CurvesList', refined_curves)
            gmsh.model.mesh.field.setNumber(distance, 'Sampling', 150)
            threshold = gmsh.model.mesh.field.add('Threshold')
            for key, value in [('InField', distance), ('SizeMin', edge_size), ('SizeMax', mesh_size),
                               ('DistMin', .05), ('DistMax', .3)]:
                gmsh.model.mesh.field.setNumber(threshold, key, value)
            gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
            gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_tags)
        node_tags = node_tags[order]
        vertices = coordinates.reshape(-1, 3)[order]
        tetrahedra, owners = [], []
        for _, tag in volumes:
            kinds, _, nodes = gmsh.model.mesh.getElements(3, tag)
            for kind, indices in zip(kinds, nodes):
                assert kind == 11, kind
                # Gmsh -> meshio/scikit-fem tetra10 edge ordering.
                tet = np.searchsorted(node_tags, indices.reshape(-1, 10))[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
                tetrahedra.append(tet)
                owners.extend([volume_owner[tag]]*len(tet))
        tetrahedra = np.vstack(tetrahedra)
        owners = np.array(owners)
        gmsh.write(str(out/'mesh.msh'))
        from skfem import MeshTet2, Basis, ElementTetP2
        mesh = MeshTet2(vertices.T, tetrahedra.T)
        basis = Basis(mesh, ElementTetP2(), intorder=5)
        mesh_volumes = np.bincount(owners, weights=basis.dx.sum(axis=1), minlength=cell_count)
        np.savez_compressed(out/'geometry.npz', points=points, radii=np.full(cell_count, .9 if cells == 'sphere' else np.sqrt(1/2.4)),
            pairs=pairs, areas=areas, contacts=contacts, masses=cell_masses, fixed=points[:, 0] == 0,
            vertices=vertices, tetrahedra=tetrahedra, owners=owners, mesh_volumes=mesh_volumes)
        report = dict(cells=cell_count, cell_shape=cells, bonds=len(pairs), vertices=len(vertices), tetrahedra=len(tetrahedra),
            mesh_size=mesh_size, edge_size=edge_size, refined_boundary_curves=len(refined_curves),
            cad_volume=float(cell_masses.sum()), mesh_volume=float(mesh_volumes.sum()),
            volume_relative_error=float(abs(mesh_volumes.sum()/cell_masses.sum()-1)),
            per_cell_volume_max_relative_error=float(np.max(abs(mesh_volumes/cell_masses-1))),
            geometry='Exact bounded power cells: equal-radius support spheres intersected with Voronoi boxes.',
            mesh_geometry='Curved quadratic tetrahedra and P2 FEM fields; surface approximation checked under refinement.',
            inertia='DEM retains 0.4*m*r^2 proxy; matching mass does not imply matching continuum inertia.',
            gmsh_version=gmsh.__version__)
        (out/'geometry.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Geometry', json.dumps(report), flush=True)
        return report
    finally:
        gmsh.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mesh-size', type=float, default=.5)
    parser.add_argument('--edge-size', type=float)
    parser.add_argument('--cross', type=int, default=2, help='Cells across y and z (2: bar, 4: thick block).')
    parser.add_argument('--cells', choices=('sphere', 'cube'), default='sphere')
    args = parser.parse_args()
    build(args.out, args.mesh_size, args.edge_size, args.cross, args.cells)
