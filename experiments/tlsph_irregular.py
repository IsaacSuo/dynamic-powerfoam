"""Actual bounded power support volumes used as TLSPH rest integration weights.

This fixture deliberately declares the entire initial bounded cell to be solid
material. It does not infer physical occupancy from a learned dipole/density.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from experiments.tlsph_core import TLSPH,beam
from experiments.tlsph_probe import tangent,equilibrate,release
from experiments.tlsph_render import bounded_volumes


def main():
    out=Path('output/tlsph_irregular'); out.mkdir(parents=True,exist_ok=True)
    X,_,fixed,tip,dx=beam(4)
    rng=np.random.default_rng(91)
    X+=rng.uniform(-.08,.08,X.shape)*dx
    r0=1.1*dx*rng.uniform(.9,1.1,len(X))
    V=bounded_volumes(X,r0,16); Vlow=bounded_volumes(X,r0,14)
    uncertainty=float(np.sum(abs(Vlow-V))/sum(V))
    assert uncertainty<.01
    model=TLSPH(X,V,1.2*dx,hourglass=10.)
    K,free=tangent(model,fixed); factors=lu_factor(K)
    load=np.zeros_like(X); load[tip,1]=-.01/tip.sum()
    initial=np.zeros_like(X); initial.ravel()[free]=lu_solve(factors,load.ravel()[free])
    start,loading=equilibrate(model,fixed,tip,factors,free,initial,target=.16)
    dt=.15*model.h/model.wave_speed/np.sqrt(1+model.alpha)
    samples,xs,Fs,metrics=release(model,fixed,tip,start,dt,duration=1.6)
    np.savez_compressed(out/'trajectory.npz',reference=X,volumes=V,radii0=r0,fixed=fixed,
                       positions=xs,deformation=Fs,times=samples[:,0],samples=samples)
    result=dict(scope='Irregular full bounded power cells declared solid; their measured rest volumes set fixed particle masses and integration weights. Our synthetic fixture.',
        particles=len(X),radius_range=[float(r0.min()),float(r0.max())],
        reference_volume_range=[float(V.min()),float(V.max())],
        reference_volume_integration_relative_l1_14_vs_16=uncertainty,
        max_neighborhood_condition=float(model.condition.max()),loading=loading,release=metrics)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
