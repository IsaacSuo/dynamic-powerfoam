"""Serial static diagnostics of the existing four-mode bonded DEM.

Use the existing stress-free tangent, checked against FiniteBondsGPU.trial.
No replacement constitutive law, time integration, fitting, or FEM refinement.
Report finite-specimen anisotropic stiffness, not a presumed isotropic material.
The proposed volume term is audited separately at unit modulus, never installed
in the running model. Waiting for a completed training pipeline avoids parallel
simulation jobs and ensures that learned parameters are final checkpoints.
"""
import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

from experiments.bonded_dem_linear_gpu import LinearBondsGPU
from experiments.bonded_dem_finite_gpu import FiniteBondsGPU

KEYS = ('points', 'radii', 'pairs', 'areas', 'contacts', 'masses', 'fixed')
MODES = ('kn', 'ks', 'kb', 'kt')
STRAINS = ('xx', 'yy', 'zz', 'yz_engineering', 'xz_engineering', 'xy_engineering')


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def regular_patch(n=4):
    """Same sphere/box OCC geometry as the 16-cell bar, without any mesh."""
    import gmsh
    points = np.array(list(itertools.product(range(n), repeat=3)), dtype=float)
    radius = .9
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('bounded_power_static_patch')
        occ = gmsh.model.occ
        pieces = []
        for p in points:
            low = p-np.where(p == 0, radius, .5)
            high = p+np.where(p == n-1, radius, .5)
            ball = occ.addSphere(*p, radius)
            box = occ.addBox(*low, *(high-low))
            intersection, _ = occ.intersect([(3, ball)], [(3, box)])
            pieces.extend(intersection)
        occ.fragment(pieces[:1], pieces[1:])
        occ.synchronize()
        masses = np.zeros(len(points))
        owners = {}
        for dim, tag in gmsh.model.getEntities(3):
            centroid = np.asarray(occ.getCenterOfMass(dim, tag))
            owner = int(np.argmin(np.linalg.norm(points-centroid, axis=1)))
            masses[owner] += occ.getMass(dim, tag)
            for _, face in gmsh.model.getBoundary([(dim, tag)], oriented=False):
                owners.setdefault(face, set()).add(owner)
        faces = {}
        for face, cells in owners.items():
            if len(cells) != 2:
                continue
            pair = tuple(sorted(cells))
            area = occ.getMass(2, face)
            center = np.asarray(occ.getCenterOfMass(2, face))
            entry = faces.setdefault(pair, [0., np.zeros(3)])
            entry[0] += area
            entry[1] += area*center
        pairs = np.asarray(sorted(faces), dtype=np.int64)
        assert len(pairs) == 3*(n-1)*n*n
        assert np.all(masses > 0)
        return dict(points=points, radii=np.full(len(points), radius), pairs=pairs,
                    areas=np.array([faces[tuple(e)][0] for e in pairs]),
                    contacts=np.array([faces[tuple(e)][1]/faces[tuple(e)][0] for e in pairs]),
                    masses=masses, fixed=np.zeros(len(points), bool))
    finally:
        gmsh.finalize()


def model_for(geometry, theta):
    args = {key: geometry[key] for key in KEYS}
    args['fixed'] = np.zeros(len(geometry['points']), bool)
    model = LinearBondsGPU(**args, device='cpu')
    scales = np.exp(.5*(theta[geometry['pairs'][:, 0]]+theta[geometry['pairs'][:, 1]]))
    for c, key in enumerate(MODES):
        setattr(model, key, getattr(model, key)*model.tensor(scales[:, c]))
    return model


def apply(model, vectors):
    if vectors.ndim == 1:
        return model.stiffness(model.tensor(vectors.reshape(-1, 6))).numpy().ravel()
    return np.column_stack([apply(model, col) for col in vectors.T])


def strain_fields(points):
    n = len(points)
    tensors = np.zeros((6, 3, 3))
    for k in range(3):
        tensors[k, k, k] = 1.
    for k, (a, b) in enumerate(((1, 2), (0, 2), (0, 1)), 3):
        tensors[k, a, b] = tensors[k, b, a] = .5
    relative = points-points.mean(axis=0)
    fields = np.zeros((6*n, 6))
    for k, strain in enumerate(tensors):
        fields[:, k].reshape(n, 6)[:, :3] = relative@strain.T
    return fields


def boundary_cells(points):
    return np.any(np.isclose(points, points.min(axis=0)) |
                  np.isclose(points, points.max(axis=0)), axis=1)


def measures(C):
    S = np.linalg.inv(C)
    t = np.array([1., 1., 1., 0., 0., 0.])
    return dict(K_uniform_strain=float(t@C@t/9),
                K_uniform_stress=float(1/(t@S@t)),
                E_xyz=(1/np.diag(S)[:3]).tolist(),
                G_yz_xz_xy=(1/np.diag(S)[3:]).tolist(),
                nu_loading_row_transverse_column=[
                    [None if i == j else float(-S[j, i]/S[i, i]) for j in range(3)]
                    for i in range(3)],
                stiffness_eigenvalues=np.linalg.eigvalsh(C).tolist())


def scalar_vector(C):
    m = measures(C)
    nu = m['nu_loading_row_transverse_column']
    return np.array([m['K_uniform_strain'], m['K_uniform_stress'],
                     *m['E_xyz'], *m['G_yz_xz_xy'],
                     *[nu[i][j] for i in range(3) for j in range(3) if i != j]])


def dense_checks(model, geometry, K):
    n = len(geometry['points'])
    p = geometry['points']-geometry['points'].mean(axis=0)
    rigid = []
    for direction in np.eye(3):
        q = np.zeros((n, 6)); q[:, :3] = direction
        rigid.append(q.ravel())
        q = np.zeros((n, 6)); q[:, :3] = np.cross(direction, p); q[:, 3:] = direction
        rigid.append(q.ravel())
    rigid = np.array(rigid).T
    symmetry = np.linalg.norm(K-K.T)/np.linalg.norm(K)
    rigid_error = np.linalg.norm(K@rigid)/(np.linalg.norm(K)*np.linalg.norm(rigid))
    eigen = np.linalg.eigvalsh((K+K.T)/2)
    zero_count = int(np.sum(abs(eigen) < 1e-9*eigen[-1]))
    assert symmetry < 1e-12 and rigid_error < 1e-12
    assert zero_count == 6 and eigen[0] > -1e-10*eigen[-1]
    finite = FiniteBondsGPU(**{key: geometry[key] for key in KEYS}, device='cpu')
    for key in MODES:
        setattr(finite, key, getattr(model, key).clone())
    q = np.random.default_rng(861).normal(size=(n, 6))
    tangent = apply(model, q.ravel())
    errors = []
    for eps in (1e-5, 1e-6):
        trial = finite.trial(finite.tensor(eps*q))
        # From a stress-free state the midpoint force is half the endpoint
        # linear force. This checks the actual finite law, not a new spring.
        derivative = -2*trial['force'].numpy().ravel()/eps
        errors.append(float(np.linalg.norm(derivative-tangent)/np.linalg.norm(tangent)))
    assert errors[-1] < 1e-4, errors
    return dict(symmetry_error=float(symmetry), rigid_null_error=float(rigid_error),
                rigid_zero_modes=zero_count, finite_law_tangent_errors=errors)


def volume_audit(geometry):
    """Geometry/force audit of the proposed term, per unit k_v*m_v=1."""
    p, edges, area, V = [geometry[k] for k in ('points', 'pairs', 'areas', 'masses')]
    n = len(p)
    M = np.zeros((n, 3, 3)); G = np.zeros((n, n, 3))
    for (i, j), a in zip(edges, area):
        d = p[j]-p[i]; w = a/np.linalg.norm(d)
        M[i] += w*np.outer(d, d); M[j] += w*np.outer(d, d)
    inverse = np.linalg.inv(M)  # No ridge; reject rank-deficient neighborhoods.
    for (i, j), a in zip(edges, area):
        d = p[j]-p[i]; w = a/np.linalg.norm(d)
        G[i, j] += w*inverse[i]@d
        G[j, i] -= w*inverse[j]@d
    for i in range(n):
        G[i, i] = -G[i].sum(axis=0)
    F0 = np.einsum('ja,ijb->iab', p, G)
    assert np.max(abs(F0-np.eye(3))) < 1e-12
    B = np.zeros((n, 6*n))
    for j in range(n):
        B[:, 6*j:6*j+3] = G[:, j]
    Kvol = B.T@(V[:, None]*B)
    Q = strain_fields(p)
    hydro = Q[:, :3].sum(axis=1)
    affine_bulk = float(hydro@Kvol@hydro/(9*V.sum()))
    assert abs(affine_bulk-1) < 1e-12
    interior = ~boundary_cells(p)
    records = []
    for strain in (1e-4, 1e-3):
        s = 1+strain
        force = (-(s**3-1)*s*s*(B.T@V)).reshape(n, 6)[:, :3]
        energy = .5*V.sum()*(s**3-1)**2
        net = np.linalg.norm(force.sum(axis=0))
        torque = np.linalg.norm(np.cross(s*p, force).sum(axis=0))
        assert net < 1e-10 and torque < 1e-10
        records.append(dict(strain=strain, energy_per_unit_modulus=float(energy),
            interior_force_norm_per_unit_modulus=float(np.linalg.norm(force[interior])) if interior.any() else None,
            interior_force_norm_per_unit_first_piola_pressure=float(np.linalg.norm(force[interior])/((s**3-1)*s*s)) if interior.any() else None,
            interpretation='An area-valued geometry residual; not a measured material modulus.',
            net_force=float(net), net_torque=float(torque)))
    return dict(condition_numbers=np.linalg.cond(M).tolist(),
                affine_bulk_contribution_per_unit_modulus=affine_bulk,
                interior_cells=int(interior.sum()), dilation=records,
                scope='Only the proposed volume force evaluated on an imposed affine state; no augmented simulation or training.'), Kvol


def diagnose(geometry, theta, folder, sensitivities=True):
    folder.mkdir()
    model = model_for(geometry, theta)
    p = geometry['points']; V = float(geometry['masses'].sum()); n = len(p)
    K = apply(model, np.eye(6*n))
    checks = dense_checks(model, geometry, K)
    boundary = boundary_cells(p)
    prescribed = np.zeros((n, 6), bool); prescribed[boundary, :3] = True
    free = np.flatnonzero(~prescribed.ravel())
    Q0 = strain_fields(p)
    Q = Q0.copy()
    factor = cho_factor(K[np.ix_(free, free)], lower=True)
    Q[free] -= cho_solve(factor, (K@Q)[free])
    defect = float(np.linalg.norm((K@Q)[free])/max(np.linalg.norm(K@Q), 1e-30))
    assert defect < 1e-10
    C0, C = Q0.T@K@Q0/V, Q.T@K@Q/V
    assert np.linalg.eigvalsh((C0-C+C0.T-C.T)/2).min() > -1e-8*np.linalg.norm(C0)
    rows = []
    for strain in (1e-4, 1e-3):
        qa = strain*Q0[:, :3].sum(axis=1)
        qr = strain*Q[:, :3].sum(axis=1)
        fa = -(K@qa).reshape(n, 6)
        dq = (qr-qa).reshape(n, 6)
        interior = ~boundary
        ea, er = .5*qa@K@qa, .5*qr@K@qr
        rows.append(dict(strain=strain, affine_energy=float(ea), relaxed_energy=float(er),
            relative_energy_release=float(1-er/ea),
            interior_force_norm=float(np.linalg.norm(fa[interior, :3])) if interior.any() else None,
            interior_force_over_boundary_force=float(np.linalg.norm(fa[interior, :3])/max(np.linalg.norm(fa[boundary, :3]),1e-30)) if interior.any() else None,
            interior_torque_norm=float(np.linalg.norm(fa[interior, 3:])) if interior.any() else None,
            interior_displacement_relaxation_rms_over_affine_rms=float(np.linalg.norm(dq[interior,:3])/max(np.linalg.norm(qa.reshape(n,6)[interior,:3]),1e-30)) if interior.any() else None,
            max_rotation_relaxation_over_strain=float(np.linalg.norm(dq[:,3:],axis=1).max()/strain),
            relaxed_free_force_norm=float(np.linalg.norm((K@qr)[free]))))
    sensitivity_summary = None
    if sensitivities:
        derivatives = np.zeros((n, 4, 6, 6))
        for cell in range(n):
            incidence = .5*((geometry['pairs'][:,0] == cell).astype(float)+
                            (geometry['pairs'][:,1] == cell).astype(float))
            for mode, key in enumerate(MODES):
                partial = copy.copy(model)
                for other in MODES:
                    setattr(partial, other, torch.zeros_like(getattr(model, other)))
                setattr(partial, key, getattr(model, key)*model.tensor(incidence))
                # Envelope theorem: relaxed DOFs are stationary; fixed imposed
                # macro strain does not depend on parameters.
                derivatives[cell,mode] = Q.T@apply(partial,Q)/V
        assert np.linalg.norm(derivatives.sum(axis=(0,1))-C)/np.linalg.norm(C) < 1e-10
        global_derivatives = derivatives.sum(axis=0)
        step = 1e-5
        scalar_derivatives = np.stack([(scalar_vector(C+step*d)-scalar_vector(C-step*d))/(2*step)
                                       for d in global_derivatives])
        sensitivity_summary = dict(parameter='log positive multiplier theta, not bounded raw optimizer coordinate',
            columns=['K_strain','K_stress','Ex','Ey','Ez','Gyz','Gxz','Gxy','nu_xy','nu_xz','nu_yx','nu_yz','nu_zx','nu_zy'],
            global_family_rows=list(MODES), derivatives=scalar_derivatives.tolist())
        np.savez_compressed(folder/'sensitivities.npz', dC_dtheta=derivatives,
                            theta=theta, global_dC=global_derivatives)
        # Independent centered perturbation of a heterogeneous per-cell
        # parameter direction, including a new equilibrium solve.
        direction = np.random.default_rng(923).normal(size=theta.shape)
        changed = []
        for sign in (-1,1):
            other = model_for(geometry,theta+sign*step*direction)
            Ko = apply(other,np.eye(6*n)); Qo=Q0.copy()
            Qo[free] -= np.linalg.solve(Ko[np.ix_(free,free)],(Ko@Qo)[free])
            changed.append(Qo.T@Ko@Qo/V)
        fd=(changed[1]-changed[0])/(2*step)
        analytic=np.einsum('nm,nmab->ab',direction,derivatives)
        checks['modulus_sensitivity_relative_error']=float(np.linalg.norm(fd-analytic)/np.linalg.norm(fd))
        assert checks['modulus_sensitivity_relative_error'] < 1e-6
    vol, Kvol = volume_audit(geometry)
    result = dict(cells=n, interior_cells=int((~boundary).sum()), volume=V,
        strain_order=list(STRAINS), C_affine=C0.tolist(), C_relaxed=C.tolist(),
        affine=measures(C0), relaxed=measures(C), dilation=rows, checks=checks,
        equilibrium_relative_residual=defect, sensitivities=sensitivity_summary,
        volume_term_audit=vol,
        boundary_condition='All surface-cell center translations prescribed affinely; all microrotations and interior translations relaxed. The training clamp is not used.',
        interpretation='Finite-specimen apparent tangent at the stress-free reference. Keep full anisotropic C and directional moduli; not a unique isotropic K,G,nu.',
        limitations='Regular patch only. No finite-strain homogenization, rigid-cell rendering symmetric-difference measurement, or fifth-parameter fit in this diagnostic.')
    write(folder/'report.json',result)
    np.savez_compressed(folder/'fields.npz',K=K,K_volume_unit=Kvol,affine_fields=Q0,
                        relaxed_fields=Q,points=p,boundary=boundary,theta=theta)
    print(folder.name, 'K_stress=',result['relaxed']['K_uniform_stress'],
          'G=',result['relaxed']['G_yz_xz_xy'],'nu=',result['relaxed']['nu_loading_row_transverse_column'],flush=True)
    return result


def fem_affine(geometry_path):
    """Six numerical FEM energy probes on an already generated coarse mesh.

    No remeshing, time integration or sparse factorization. Homogeneous linear
    elasticity admits the imposed affine field exactly for its affine Dirichlet
    boundary data. Interior residual is checked rather than assumed.
    """
    from skfem import MeshTet2, Basis, ElementVector, ElementTetP2, asm
    from skfem.models.elasticity import linear_elasticity,lame_parameters
    with np.load(geometry_path) as a:
        mesh=MeshTet2(a['vertices'].T,a['tetrahedra'].T)
    b=Basis(mesh,ElementVector(ElementTetP2()),intorder=5)
    K=asm(linear_elasticity(*lame_parameters(1200.,.25)),b).tocsr()
    nodes=b.doflocs[:,::3].T
    Q=strain_fields(nodes).reshape(len(nodes),6,6)[:,:3,:].reshape(b.N,6)
    boundary=b.get_dofs().all(); free=np.setdiff1d(np.arange(b.N),boundary)
    R=K@Q
    residual=float(np.linalg.norm(R[free])/np.linalg.norm(R))
    C=Q.T@R/b.dx.sum()
    exact=np.zeros((6,6)); exact[:3,:3]=480.
    exact[np.arange(3),np.arange(3)]+=960.
    exact[np.arange(3,6),np.arange(3,6)]=480.
    error=float(np.linalg.norm(C-exact)/np.linalg.norm(exact))
    assert error<1e-10 and residual<1e-9,(error,residual)
    return dict(source_geometry=str(geometry_path), elements=int(mesh.nelements),
        C=C.tolist(),moduli=measures(C),constitutive_matrix_relative_error=error,
        interior_equilibrium_relative_residual=residual,
        scope='Numerical homogeneous FEM affine-field energy measurement; FEM boundary fields and DEM boundary-center constraints are different discretizations, so compare as apparent responses, not matched pointwise Dirichlet solutions.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inverse',type=Path,default=Path('output/foam_fem/inverse_exploratory_v1'))
    parser.add_argument('--reference',type=Path,default=Path('output/foam_fem/reference_edge007'))
    parser.add_argument('--wait-for-training',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    status=args.out/'status.json'
    def stage(name,**details):
        write(status,dict(stage=name,updated_unix=time.time(),**details))
        print(name,details,flush=True)
    try:
        if args.wait_for_training:
            stage('waiting_for_training',report=str(args.inverse/'report.json'))
            deadline=time.time()+12*3600
            while not (args.inverse/'report.json').exists():
                pipeline=json.loads(Path('output/foam_fem/pipeline_status.json').read_text())
                if pipeline['stage']=='failed':
                    raise RuntimeError('Training failed; no automatic simultaneous diagnostic.')
                if time.time()>deadline:
                    raise RuntimeError('Training wait exceeded 12 hours.')
                time.sleep(10)
        assert (args.inverse/'report.json').exists(),'Complete serial training first.'
        torch.set_num_threads(1)
        with np.load(args.reference/'patch.npz') as a:
            geometry={key:a[key] for key in KEYS}
        results={}
        for name in ('initial','shared','per_cell'):
            stage('static_moduli_16',model=name)
            theta=np.zeros((16,4)) if name=='initial' else np.load(args.inverse/name/'parameters.npz')['theta']
            results[name]=diagnose(geometry,theta,args.out/name)
        stage('fem_affine_energy')
        fem=fem_affine(Path('output/foam_fem/mesh_q050/geometry.npz'))
        write(args.out/'fem.json',fem)
        stage('geometry_64_no_mesh')
        geometry64=regular_patch()
        np.savez_compressed(args.out/'patch64.npz',**geometry64)
        stage('static_internal_cell_audit_64')
        results['regular64_initial']=diagnose(geometry64,np.zeros((64,4)),args.out/'regular64_initial',False)
        inverse=json.loads((args.inverse/'report.json').read_text())
        report=dict(complete=True,models=results,fem=fem,
            unchanged_pair_heldout={key:value['metrics'] for key,value in inverse['models'].items()},
            volume_term_adoption='Undecided: no fifth-parameter training or held-out comparison yet.',
            material_semantics='Reference cell rigid images carry material identity. J measures neighborhood volume, not the volume of a rigid cell. Rebuilt power geometry does not enter mechanics.',
            affine_bound='The affine trial field bounds the relaxed energy of the same constrained discrete problem; it does not establish an upper bound on the increment of the relaxed modulus after adding a term.')
        write(args.out/'report.json',report)
        stage('complete',report=str(args.out/'report.json'))
    except BaseException as error:
        stage('failed',error=repr(error))
        raise


if __name__=='__main__':
    main()
