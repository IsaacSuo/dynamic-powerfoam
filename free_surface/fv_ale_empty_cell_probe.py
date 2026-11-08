"""CPU nearest-point certificate for a rejected actual power-ball cell.

Independent of spherical-area/volume formulas. This probe changes no state.
For the power polyhedron, minimize |(x-p_i)/r_i|. A minimum greater than one
certifies its ball intersection is empty. Enumerate the up-to-three active
planes of this three-dimensional convex quadratic problem.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch


def cpu_kernel_probe(p,r,k,active):
    """Run the actual area kernels on the independently certified local cell."""
    import math
    import warp as wp
    from free_surface.gpu_geometry import structure_overlaps,fibonacci
    from free_surface.geometry_warp import GeometryWarp,k_geo_site,BIG
    from free_surface.geometry_sphere_direct import k_geo_site_direct
    wp.init()
    torch.set_num_threads(1);ids=[k]+active;p=p[ids].contiguous();r=r[ids].contiguous()
    g=GeometryWarp(p,r,structure_overlaps(p,r));n=len(r);m=len(g.i)
    site=torch.cat((g.i,g.j));order=site.argsort(stable=True)
    off=torch.zeros(n+1,dtype=torch.int32);off[1:]=torch.cumsum(torch.bincount(site,minlength=n),0).to(torch.int32)
    ent=order.to(torch.int32)
    Vw=torch.zeros_like(r).index_add(0,g.i,g.wall_A*g.hi).index_add(0,g.j,g.wall_A*g.hj)
    W=lambda x,t:wp.from_torch(x.contiguous(),dtype=t)
    output={'unstable_sector_volume_m3':float(g.V[0])}
    for label,kernel in [('stable_sector',k_geo_site),('direct_integral',k_geo_site_direct)]:
        pn=torch.empty_like(p);area=torch.empty_like(r);volume=torch.empty_like(r);pole=torch.empty_like(p)
        wp.launch(kernel,n,inputs=[W(p,wp.vec3d),W(r,wp.float64),W(off,wp.int32),W(ent,wp.int32),m,
            W(g.nn,wp.vec3d),W(g.u,wp.vec3d),W(g.v,wp.vec3d),W(g.rho,wp.float64),W(g.hi,wp.float64),W(g.hj,wp.float64),
            W(g.arc_t0,wp.float64),W(g.arc_t1,wp.float64),W(g.narc,wp.int32),W(fibonacci(32,p.dtype,p.device),wp.vec3d),
            32,math.pi/6,BIG,4*math.pi,W(g.nbrs.to(torch.int32),wp.int32),W(g.hidden.to(torch.int32),wp.int32),
            1,0,0.,W(ent,wp.int32),0,W(Vw,wp.float64)],
            outputs=[W(pn,wp.vec3d),W(area,wp.float64),W(volume,wp.float64),W(pole,wp.vec3d)],device='cpu')
        output[label]={'volume_m3':float(volume[0]),'cap_area_m2':float(area[0]),'wall_volume_sum_m3':float(Vw[0])}
    return output


def two_plane_volume(p,r,k,active,dps=60,additional_halfspaces=()):
    """Positive slice integral, only when other planes provably do not cut it."""
    import mpmath as mp
    # A ball disjoint from the owner's entire ball cannot cut any point of
    # that power-ball cell: psi_owner<=0 while psi_neighbor>0. Use a deliberately
    # conservative floating-point separation bound; uncertain pairs remain in
    # the arbitrary-precision proof. This avoids constructing thousands of MP
    # coordinates for faraway cells on each exceptional tiny-wedge recut.
    points=np.asarray(p,dtype=np.float64);radii=np.asarray(r,dtype=np.float64)
    delta=points-points[k];distance2=(delta*delta).sum(1);sum_radius2=(radii+radii[k])**2
    arithmetic_scale=((np.abs(points)+np.abs(points[k]))**2).sum(1)+sum_radius2
    separation_error=512*np.finfo(np.float64).eps*arithmetic_scale+np.finfo(np.float64).tiny
    candidates=set(np.flatnonzero(distance2<=sum_radius2+separation_error).tolist())|{k}|set(active)
    with mp.workdps(dps):
        pp={j:mp.matrix([mp.mpf(float(x)) for x in points[j]]) for j in candidates}
        rr={j:mp.mpf(float(radii[j])) for j in candidates};norm=lambda x:mp.sqrt((x.T*x)[0])
        normals=[];bounds=[]
        for j in active:
            d=pp[j]-pp[k];distance=norm(d)
            normals.append(d/distance)
            bounds.append((distance**2+rr[k]**2-rr[j]**2)/(2*distance*rr[k]))
        N=mp.matrix([[normal[j] for j in range(3)] for normal in normals])
        coefficients=mp.lu_solve(N*N.T,mp.matrix(bounds))
        if any(c>0 for c in coefficients):
            return {'supported':False,'reason':'two-plane point is not the nearest feasible point'}
        q=N.T*coefficients;s=norm(q)
        if s>=1:return {'supported':True,'volume_m3':0.,'normalized_radius':float(s),'precision_digits':dps}
        enclosure=mp.sqrt(1-s*s);margin=mp.inf
        for j in sorted(candidates):
            if j==k or j in active:continue
            d=pp[j]-pp[k];distance=norm(d)
            h=(distance**2+rr[k]**2-rr[j]**2)/(2*distance*rr[k])
            # Every feasible point is within enclosure of this KKT point.
            margin=min(margin,h-(d.T*q)[0]/distance-enclosure)
        for direction,offset in additional_halfspaces:
            normal=mp.matrix([mp.mpf(float(x)) for x in direction]);length=norm(normal)
            h=(mp.mpf(float(offset))-(normal.T*pp[k])[0])/(length*rr[k])
            margin=min(margin,h-(normal.T*q)[0]/length-enclosure)
        if margin<=0:return {'supported':False,'other_plane_margin':float(margin)}
        x=q/s
        cross=lambda a,b:mp.matrix([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
        z=cross(normals[0],normals[1]);z/=norm(z);y=cross(z,x)
        lower=-mp.inf;upper=mp.inf
        for normal in normals:
            nx=(normal.T*x)[0];ny=(normal.T*y)[0]
            if ny>0:upper=min(upper,-nx/ny)
            else:lower=max(lower,-nx/ny)
        if not mp.isfinite(lower) or not mp.isfinite(upper) or lower>=upper:
            return {'supported':False,'reason':'non-wedge section'}
        height=1-s
        def section(fraction):
            t=height*fraction;R2=1-(s+t)**2
            if R2<=0:return mp.mpf(0)
            R=mp.sqrt(R2);lo=max(-R,lower*t);hi=min(R,upper*t)
            if hi<=lo:return mp.mpf(0)
            def F(a):return a*mp.sqrt(max(0,R2-a*a))+R2*mp.asin(max(-1,min(1,a/R)))
            return F(hi)-F(lo)
        splits=[mp.mpf(0),mp.mpf(1)]
        for slope in [lower,upper]:
            # Stable positive root of (1+slope^2)t^2+2 s t+s^2-1=0.
            t=(1-s*s)/(s+mp.sqrt(s*s+(1+slope*slope)*(1-s*s)))
            if 0<t<height:splits.append(t/height)
        integral=mp.quad(section,sorted(splits))*height*rr[k]**3
        return {'supported':True,'volume_m3':float(integral),'volume_high_precision':str(integral),
                'normalized_radius':float(s),'other_plane_margin':float(margin),
                'disjoint_ball_constraints_excluded':len(radii)-len(candidates),
                'arbitrary_precision_ball_constraints':len(candidates)-1,
                'precision_digits':dps,'section_slopes':[float(lower),float(upper)]}


def probe(p,r,k):
    d=p-p[k];distance=np.linalg.norm(d,axis=1)
    valid=distance>0
    h=np.full(len(r),np.inf)
    h[valid]=(distance[valid]**2+r[k]**2-r[valid]**2)/(2*distance[valid]*r[k])
    ids=np.flatnonzero(valid&(h<1.01));N=d[ids]/distance[ids,None];bounds=h[ids]
    best=None
    if np.all(bounds>=0):best={'minimum_normalized_radius':0.,'active':[],'point':[0.,0.,0.]}
    if best is None:
        for size in [1,2,3]:
            for active in itertools.combinations(range(len(ids)),size):
                A=N[list(active)];b=bounds[list(active)];gram=A@A.T
                if np.linalg.cond(gram)>1e14:continue
                lam=-np.linalg.solve(gram,b)
                if np.min(lam)<-1e-12:continue
                y=-A.T@lam
                violation=float(np.max(N@y-bounds))
                if violation>1e-10:continue
                radius=float(np.linalg.norm(y))
                if best is None or radius<best['minimum_normalized_radius']:
                    best={'minimum_normalized_radius':radius,'active':ids[list(active)].tolist(),
                          'point':y.tolist(),'max_constraint_violation':violation,'multipliers':lam.tolist(),
                          'dual_lower_bound_squared_radius':float(-np.dot(A.T@lam,A.T@lam)-2*np.dot(b,lam))}
    containing=r-distance-r[k];containing[k]=-np.inf
    return {'cell':k,'radius':float(r[k]),'candidate_planes':len(ids),
            'largest_single_ball_containment_margin_m':float(containing.max()),
            'best':best,'plane_ids':ids.tolist(),'normalized_plane_bounds':bounds.tolist()}


def independent_sampling_check(radius,precise,samples=1000000):
    """Independent targeted volume sampling, without sector/asin integration."""
    rng=np.random.default_rng(929);u=rng.random((samples,3))
    s=precise['normalized_radius'];h=1-s;zmax=np.sqrt(1-s*s)
    lower,upper=precise['section_slopes'];t=h*np.sqrt(u[:,0])
    y=t*(lower+(upper-lower)*u[:,1]);z=zmax*(2*u[:,2]-1)
    inside=2*s*t+t*t+y*y+z*z<=1-s*s
    fraction=inside.mean();proposal=(upper-lower)*h*h*zmax*radius**3
    estimate=proposal*fraction;se=proposal*np.sqrt(fraction*(1-fraction)/samples)
    difference=estimate-precise['volume_m3']
    assert abs(difference)<5*se,(estimate,precise,se)
    return {'samples':samples,'volume_m3':estimate,'standard_error_m3':se,
            'difference_in_standard_errors':difference/se}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot',type=Path);parser.add_argument('cell',type=int)
    parser.add_argument('--cpu-kernels',action='store_true')
    args=parser.parse_args();data=torch.load(args.snapshot,map_location='cpu',weights_only=False)
    result={}
    for key,state in [('accepted',data),('rejected',data.get('rejected_geometry'))]:
        if state is None or 'p' not in state:continue
        result[key]=probe(state['p'].numpy(),state['r'].numpy(),args.cell)
        best=result[key]['best']
        if best and len(best['active'])==2:
            result[key]['independent_volume']=two_plane_volume(state['p'].numpy(),state['r'].numpy(),args.cell,best['active'])
            precise=result[key]['independent_volume']
            if precise.get('supported') and precise['volume_m3']>0:
                result[key]['targeted_sampling']=independent_sampling_check(float(state['r'][args.cell]),precise)
            if args.cpu_kernels and result[key]['independent_volume']['supported']:
                result[key]['cpu_kernels']=cpu_kernel_probe(state['p'],state['r'],args.cell,best['active'])
    g=data['groups'];members=torch.nonzero(g==g[args.cell]).flatten()
    result['group_members']=members.tolist();result['group_mass']=float(data['q'][members,0].sum())
    result['time']=data['time'];result['rejected_reason']=data.get('rejected_geometry',{}).get('reason')
    path=args.snapshot.with_name(args.snapshot.stem+'_cell'+str(args.cell)+'_empty_probe.json')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
