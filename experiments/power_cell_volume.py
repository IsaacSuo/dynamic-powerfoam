"""Radial volume quadrature and weight Jacobian for bounded power cells.

Cell i = ball(p_i, sqrt(w_i)) intersect all power halfspaces. This evaluator
requires each generator to lie strictly inside its own cell, and rejects states
outside that domain; it is not a general empty/non-star-centered cell evaluator.
For direction u, rho=min(sqrt(w_i), b_ij/(u.n_ij)) over positive projections,
where b_ij=(d_ij^2+w_i-w_j)/(2*d_ij). V=int rho^3/3 dOmega.
The Jacobian differentiates this same hard-boundary quadrature, no soft labels.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import qmc


def directions(exponent=14, seed=19):
    q=qmc.Sobol(2,scramble=True,seed=seed).random_base2(exponent)
    z=1-2*q[:,0]; phi=2*np.pi*q[:,1]
    return np.column_stack([np.sqrt(1-z*z)*np.cos(phi),
                            np.sqrt(1-z*z)*np.sin(phi),z])


class RadialVolumes:
    def __init__(self, points, rays):
        self.points=np.asarray(points,dtype=float)
        self.rays=rays
        self.d=cdist(points,points)
        np.fill_diagonal(self.d,np.inf)
        if self.d.min()<1e-12: raise ValueError('Coincident generators')

    def evaluate(self, radii, jacobian=False):
        r=np.asarray(radii); w=r*r; n=len(r)
        if np.any(r<=0): raise ValueError('Nonpositive radius')
        numer=self.d**2+w[:,None]-w[None,:]
        if numer.min()<=1e-12:
            raise ValueError('A generator is outside its own bounded power cell')
        V=np.empty(n); A=np.zeros((n,n)) if jacobian else None
        factor=4*np.pi/len(self.rays)
        for i in range(n):
            ids=np.flatnonzero(self.d[i]<r[i]+r)
            if not len(ids):
                V[i]=4*np.pi*r[i]**3/3
                if jacobian: A[i,i]=2*np.pi*r[i]
                continue
            dist=self.d[i,ids]
            normals=(self.points[ids]-self.points[i])/dist[:,None]
            b=numer[i,ids]/(2*dist)
            # Chunks keep validation quadrature memory bounded.
            volume=0.
            for start in range(0,len(self.rays),16384):
                proj=self.rays[start:start+16384]@normals.T
                lengths=np.full_like(proj,np.inf)
                np.divide(b,proj,out=lengths,where=proj>0)
                active=np.argmin(lengths,axis=1)
                rows=np.arange(len(proj))
                plane_rho=lengths[rows,active]
                sphere=plane_rho>=r[i]
                rho=np.minimum(plane_rho,r[i])
                volume+=np.sum(rho**3)*factor/3
                if jacobian:
                    A[i,i]+=sphere.sum()*r[i]*factor/2
                    a=active[~sphere]
                    derivative=factor*rho[~sphere]**2/(2*dist[a]*proj[rows[~sphere],a])
                    A[i,i]+=derivative.sum()
                    A[i,ids]-=np.bincount(a,weights=derivative,minlength=len(ids))
            V[i]=volume
        return (V,A) if jacobian else V


def solve_radii(evaluator,target,initial,reference,tolerance=1e-9,max_iterations=15):
    r=initial.copy(); history=[]
    lower=(.5*reference)**2; upper=(2*reference)**2
    for iteration in range(max_iterations+1):
        V,A=evaluator.evaluate(r,True)
        residual=V-target
        error=float(np.sum(abs(residual))/sum(target))
        history.append(error)
        if error<tolerance:
            return r,dict(iterations=iteration,relative_l1=error,history=history)
        if iteration==max_iterations: break
        dw=np.linalg.solve(A,-residual)
        w=r*r
        for step in 2.**-np.arange(16):
            trial_w=w+step*dw
            if np.any(trial_w<=lower) or np.any(trial_w>=upper): continue
            trial=np.sqrt(trial_w)
            try: trial_V=evaluator.evaluate(trial)
            except ValueError: continue
            if np.sum(abs(trial_V-target))/sum(target)<error:
                r=trial; break
        else: raise RuntimeError(f'Radius line search stalled: {error}')
    raise RuntimeError(f'Radius solve failed to converge: {history}')


def checks():
    rays=directions(18,123)
    r=np.array([.4,.4]); x=np.array([[-.2,0,0],[.2,0,0]])
    ev=RadialVolumes(x,rays)
    V,A=ev.evaluate(r,True)
    sphere=4*np.pi*.4**3/3; height=.2
    analytic=sphere-np.pi*height**2*(.4-height/3)
    volume_error=float(np.max(abs(V/analytic-1)))
    assert volume_error<1e-5
    # Exact derivatives of equal-sphere clipped volumes w.r.t. squared radii.
    face_area=np.pi*(.4**2-.2**2)
    expected_offdiag=-face_area/(2*.4)
    expected_diag=2*np.pi*.4-np.pi*height+face_area/(2*.4)
    expected=np.array([[expected_diag,expected_offdiag],[expected_offdiag,expected_diag]])
    jac_error=float(np.max(abs(A-expected))/np.max(abs(expected)))
    assert jac_error<1e-4
    rng=np.random.default_rng(13)
    x=rng.normal(size=(12,3))*.3; r=np.full(12,.35)
    ev=RadialVolumes(x,directions(12,47)); V,A=ev.evaluate(r,True)
    direction=rng.normal(size=12); eps=1e-8
    fd=(ev.evaluate(np.sqrt(r*r+eps*direction))-ev.evaluate(np.sqrt(r*r-eps*direction)))/(2*eps)
    derivative_error=float(np.linalg.norm(fd-A@direction)/np.linalg.norm(fd))
    assert derivative_error<1e-6
    fitted,info=solve_radii(ev,V,r*1.005,r)
    recovery_error=float(np.max(abs(fitted/r-1)))
    assert recovery_error<1e-7
    scaled=RadialVolumes(x*1.2,directions(12,47)).evaluate(r*1.2)
    assert np.max(abs(scaled/V-1.2**3))<1e-12
    return dict(two_sphere_volume_relative_error=volume_error,
                two_sphere_analytic_jacobian_relative_error=jac_error,
                jacobian_finite_difference_relative_error=derivative_error,
                known_radius_recovery_relative_error=recovery_error)
