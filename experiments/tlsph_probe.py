"""TLSPH numerical validation and controlled 3D beam release (our fixtures).

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 in conda om:
python -m experiments.tlsph_probe
No image fitting; numerical mechanics checks precede Power Foam visualization.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from experiments.tlsph_core import TLSPH, beam


def checks():
    X, V, _, _, dx = beam(3)
    rng = np.random.default_rng(42)
    X += rng.uniform(-.1, .1, X.shape)*dx
    V *= rng.uniform(.7, 1.3, len(V))
    model = TLSPH(X, V, 1.2*dx)
    result = {}
    R = Rotation.from_euler('xyz', [31, -17, 23], degrees=True).as_matrix()
    A = np.array([[1.04, .06, 0], [0, .98, .03], [0, 0, 1.02]])
    for name, matrix in [('rest', np.eye(3)), ('translation', np.eye(3)),
                          ('rotation', R), ('affine', A)]:
        x = X@matrix.T + (np.array([.4, -.3, .2]) if name != 'rest' else 0)
        f, data = model.evaluate(x)
        result[name] = dict(F_max_error=float(np.max(np.abs(data['F']-matrix))),
                            hourglass_force_max=float(np.abs(data['hourglass']).max()),
                            force_max=float(np.abs(f).max()))
        assert result[name]['F_max_error'] < 1e-12
        assert result[name]['hourglass_force_max'] < 1e-10
        if name != 'affine':
            assert result[name]['force_max'] < 1e-9
    x = X@A.T+rng.normal(0, .002, X.shape)
    f, data = model.evaluate(x)
    fr, rotated = model.evaluate(x@R.T + [1, 2, 3])
    result['objectivity'] = dict(force_relative_error=float(np.linalg.norm(fr-f@R.T)/np.linalg.norm(f)),
                                energy_difference=abs(rotated['energy']-data['energy']))
    assert result['objectivity']['force_relative_error'] < 1e-10
    for name in ('elastic', 'hourglass'):
        force = data[name]
        scale = np.linalg.norm(force)
        result[name+'_balance'] = dict(net_force=float(np.linalg.norm(force.sum(0))/scale),
            net_torque=float(np.linalg.norm(np.cross(x, force).sum(0))/scale))
        assert max(result[name+'_balance'].values()) < 1e-10
    directional = []
    for _ in range(4):
        d = rng.normal(size=x.shape); d /= np.linalg.norm(d)
        eps = 1e-6
        derivative = (model.evaluate(x+eps*d, alpha=0)[1]['energy']-
                      model.evaluate(x-eps*d, alpha=0)[1]['energy'])/(2*eps)
        analytic = -np.sum(data['elastic']*d)
        directional.append(abs(derivative-analytic)/max(1e-7, abs(analytic)))
    result['elastic_energy_gradient_relative_errors'] = directional
    assert max(directional) < 1e-5
    result['max_neighborhood_condition'] = float(model.condition.max())
    return result


def tangent(model, fixed, at=None):
    free = np.flatnonzero(np.repeat(~fixed, 3))
    K = np.empty((len(free), len(free)))
    # Central finite differences of the actual force, including paper hourglass
    # force. Do not silently symmetrize a non-variational stabilization tangent.
    eps = 1e-6*model.h
    at = model.X if at is None else at
    for column, dof in enumerate(free):
        plus = at.copy(); minus = at.copy()
        plus.flat[dof] += eps; minus.flat[dof] -= eps
        K[:, column] = -(model.evaluate(plus)[0]-model.evaluate(minus)[0]).ravel()[free]/(2*eps)
    return K, free


def static_case(resolution, alpha):
    X, V, fixed, tip, dx = beam(resolution)
    model = TLSPH(X, V, 1.2*dx, hourglass=alpha)
    K, free = tangent(model, fixed)
    factors = lu_factor(K)
    load = np.zeros_like(X); load[tip, 1] = -.01/tip.sum()
    displacement = np.zeros_like(X)
    displacement.ravel()[free] = lu_solve(factors, load.ravel()[free])
    L = 2-X[fixed, 0].max(); area=.5**2; inertia=.5**4/12
    theory = .01*(L**3/(3*model.young*inertia)+L/((5/6)*model.mu*area))
    deflection = -displacement[tip, 1].mean()
    residual = model.evaluate(X+displacement)[0]+load
    metrics = dict(particles=len(X), resolution=resolution, alpha=alpha,
        tip_deflection=float(deflection), timoshenko_estimate=float(theory),
        relative_error_to_beam_estimate=float(abs(deflection/theory-1)),
        note='3D discretized clamp/end traction; Timoshenko is a slender-beam approximation, not exact 3D truth.',
        nonlinear_force_residual=float(np.linalg.norm(residual[~fixed])),
        stiffness_asymmetry=float(np.linalg.norm(K-K.T)/np.linalg.norm(K)),
        max_condition=float(model.condition.max()),
        neighbor_range=[int(model.degree.min()), int(model.degree.max())])
    return model, fixed, tip, factors, free, displacement, metrics


def equilibrate(model, fixed, tip, factors, free, initial, target=.2):
    x = model.X+initial
    residuals = []
    for force in [.01, .04, .08, .12, target]:
        external = np.zeros_like(x); external[tip, 1] = -force/tip.sum()
        for iteration in range(20):
            residual=(model.evaluate(x)[0]+external).ravel()[free]
            norm=np.linalg.norm(residual)
            if norm < force*1e-7: break
            K,_=tangent(model,fixed,x)
            step=lu_solve(lu_factor(K),residual)
            for fraction in [1.,.5,.25,.125,.0625,.03125]:
                candidate=x.copy(); candidate.ravel()[free]+=fraction*step
                try: trial=(model.evaluate(candidate)[0]+external).ravel()[free]
                except ValueError: continue
                if np.linalg.norm(trial)<norm:
                    x=candidate; break
            else: raise RuntimeError(f'Newton stalled, load {force}, residual {norm}')
        if norm >= force*1e-6: raise RuntimeError(f'Equilibrium did not converge: {norm}')
        residuals.append(dict(load=force, newton_iterations=iteration, residual=float(norm)))
        print('Loaded equilibrium',residuals[-1],flush=True)
    return x, residuals


def release(model, fixed, tip, start, dt, duration=1.6, frames=65):
    count = int(np.ceil(duration/dt)); dt = duration/count
    x=start.copy(); velocity=np.zeros_like(x)
    force, data = model.evaluate(x); force[fixed]=0
    acceleration = force/model.mass[:, None]
    samples=[]; positions=[]; Fs=[]; next_sample=0
    work=0.; initial_energy=data['energy']; max_balance=0.
    for step in range(count+1):
        t=step*dt
        if t+1e-12 >= next_sample*duration/(frames-1):
            energy=data['energy']+.5*np.sum(model.mass[:, None]*velocity**2)
            samples.append([t, np.mean(x[tip, 1]-model.X[tip, 1]), energy,
                            work, data['J'].min(), data['J'].max(), data['hg_error']])
            positions.append(x.copy()); Fs.append(data['F'].copy()); next_sample+=1
        max_balance=max(max_balance, abs(data['energy']+.5*np.sum(model.mass[:, None]*velocity**2)-initial_energy-work))
        if step==count: break
        previous=x.copy(); old_hg=data['hourglass'].copy()
        velocity += .5*dt*acceleration
        x += dt*velocity; x[fixed]=model.X[fixed]
        force, data=model.evaluate(x); force[fixed]=0
        acceleration=force/model.mass[:, None]
        velocity += .5*dt*acceleration; velocity[fixed]=0
        work += .5*np.sum((old_hg+data['hourglass'])*(x-previous))
    samples=np.asarray(samples)
    peaks=find_peaks(-samples[:, 1])[0]
    metrics=dict(dt=dt, steps=count, duration=duration,
        initial_tip_displacement=float(samples[0, 1]), final_tip_displacement=float(samples[-1, 1]),
        max_absolute_tip_displacement=float(np.abs(samples[:, 1]).max()),
        min_detF=float(samples[:, 4].min()), max_detF=float(samples[:, 5].max()),
        elastic_plus_kinetic_relative_change=float((samples[-1, 2]-initial_energy)/initial_energy),
        hourglass_work_over_initial_energy=float(work/initial_energy),
        work_balance_max_relative_error=float(max_balance/initial_energy),
        observed_period=float(np.diff(samples[peaks, 0]).mean()) if len(peaks)>1 else None,
        energy_note='Hourglass force is accounted by measured work; elastic+kinetic energy alone is not claimed conserved.')
    return samples, np.asarray(positions), np.asarray(Fs), metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('output/tlsph'))
    parser.add_argument('--resume-static',action='store_true',help='Reuse completed static comparison metrics after an interrupted dynamic run')
    args=parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    started=time.time()
    results=dict(scope='Independent corrected TLSPH + Ganzenmueller radial hourglass implementation. Synthetic 3D solid, not a learned reconstruction or paper reproduction.',
                 material=dict(young=1000., poisson=.3, density=1.), checks=checks(), static=[])
    print('Algebra, finite-difference and invariance checks passed.', flush=True)
    selected=None
    if args.resume_static:
        results['static']=json.loads((args.out/'results.json').read_text())['static']
    for resolution, alpha in ([(4,10.)] if args.resume_static else [(4, 0.), (4, 10.), (4, 50.), (6, 10.)]):
        case=static_case(resolution, alpha)
        if not args.resume_static: results['static'].append(case[-1])
        print(json.dumps(case[-1]), flush=True)
        if resolution==4 and alpha==10: selected=case
    model,fixed,tip,factors,free,initial,_=selected
    (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    start, results['loading']=equilibrate(model,fixed,tip,factors,free,initial)
    print('Nonlinear loaded equilibrium:', results['loading'][-1], flush=True)
    dt=.15*model.h/model.wave_speed/np.sqrt(1+model.alpha)
    coarse,xs,Fs,results['release']=release(model,fixed,tip,start,dt)
    fine,_,_,results['release_half_dt']=release(model,fixed,tip,start,dt/2)
    results['time_convergence']=dict(tip_max_difference=float(np.max(np.abs(coarse[:, 1]-fine[:, 1]))),
        relative_to_initial_amplitude=float(np.max(np.abs(coarse[:, 1]-fine[:, 1]))/abs(coarse[0, 1])))
    np.savez_compressed(args.out/'trajectory.npz', reference=model.X, volumes=model.V,
        fixed=fixed, positions=xs, deformation=Fs, times=coarse[:, 0], samples=coarse)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    ax[0].plot(coarse[:,0],coarse[:,1],label='dt'); ax[0].plot(fine[:,0],fine[:,1],'--',label='dt/2')
    ax[0].set(xlabel='time',ylabel='mean tip displacement'); ax[0].legend()
    ax[1].plot(coarse[:,0],coarse[:,2]/coarse[0,2],label='elastic + kinetic')
    ax[1].plot(coarse[:,0],1+coarse[:,3]/coarse[0,2],'--',label='initial + HG work')
    ax[1].set(xlabel='time',ylabel='normalized energy'); ax[1].legend()
    for k in [0,16,32,48,64]:
        ax[2].scatter(xs[k,:,0],xs[k,:,1],s=3,label=f't={coarse[k,0]:.2f}')
    ax[2].set(xlabel='x',ylabel='y',aspect='equal'); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(args.out/'mechanics.png',dpi=150); plt.close(fig)
    results['elapsed_seconds']=time.time()-started
    (args.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:results[k] for k in ('release','release_half_dt','time_convergence','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__': main()
