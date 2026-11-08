"""Affine patch check of density as well as deviatoric shape on real cells.

Compare the currently used isotropic radius rate with the spherical-cap
integrated affine normal velocity, without advancing or changing the solver.
"""
import json
import argparse
from pathlib import Path
import torch
from free_surface.fv_ale_hyperelastic import HyperelasticALE
from free_surface.fv_ale_reconstruction import group_centroids,group_sum


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cap-motion',choices=['volume','balanced'],default='volume')
    args=parser.parse_args()
    torch.set_num_threads(1);root=Path('output/ale')
    data=torch.load(root/'fv_state_dt8us.pt',weights_only=False);m=data['material']
    sol=HyperelasticALE(data['p'],data['r'],m['E'],m['nu'],m['ys'],groups=data['groups'],
        ground=-10.,gravity=(0.,0.,0.),contact_model='plane',mechanical_reconstruction='compatible',
        affine_stencil='extended',interface_dissipation=0.,cap_dissipation=0.)
    s=sol.initial;mesh=s.mesh
    b={'q':s.q,'p':s.p,'r':s.r,'V':mesh.volume,'groups':sol.groups,'i':mesh.i,'j':mesh.j,
       'A':mesh.area,'n':mesh.normal,'c':mesh.centroid,'N':mesh.cap_normal_integral,'Q':mesh.cap_Q,
       'cap':mesh.cap_area,'surface':mesh.surface,'floor':mesh.ground_area,
       'floor_c':mesh.ground_centroid,'floor_n':mesh.ground_normal}
    b={k:v.cpu() for k,v in b.items()}
    centers=group_centroids(b);g=b['groups'];V=group_sum(b['V'],g);roots=g==torch.arange(len(g))
    tests={'rotation':[[0.,-1.,.2],[1.,0.,-.3],[-.2,.3,0.]],
           'shear':[[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]],
           'deviatoric_stretch':[[1.,0.,0.],[0.,-.5,0.],[0.,0.,-.5]],
           'dilation':[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}
    result={}
    for name,matrix in tests.items():
        L=torch.tensor(matrix,dtype=s.p.dtype);v=centers@L.T
        sigma=torch.zeros((len(v),3,3),dtype=v.dtype);Z=torch.ones(len(v),dtype=v.dtype)
        ops=sol.device_ops();w,dr,_,cg,cr,us,*_=ops.mechanical_kinematics(b,v,sigma,Z)
        i,j,A,n=b['i'],b['j'],b['A'],b['n']
        def incidence(x):return torch.zeros_like(b['V']).index_add(0,i,x).index_add(0,j,-x)
        face=incidence(A*(us*n).sum(1))
        old_cap=(b['N']*w).sum(1)+b['cap']*dr
        G=ops._compatible['gradient']
        target_dr=b['r']*(G*b['Q']).sum((1,2))/b['cap'].clamp(min=1e-300)
        target_dr=torch.where(b['surface'],target_dr,dr)
        new_cap=(b['N']*w).sum(1)+b['cap']*target_dr
        sol.affine_cap_motion=args.cap_motion
        wv,drv,_,cgv,_,usv,*_=ops.mechanical_kinematics(b,v,sigma,Z)
        sol.affine_cap_motion='isotropic'
        actual_cap=(b['N']*wv).sum(1)+b['cap']*drv
        assert torch.allclose(actual_cap,new_cap,atol=1e-14,rtol=1e-12)
        raw=torch.zeros_like(sigma).index_add(0,i,A[:,None,None]*(usv-v[i])[:,:,None]*n[:,None,:])
        raw.index_add_(0,j,-A[:,None,None]*(usv-v[j])[:,:,None]*n[:,None,:])
        actual_gradient=group_sum(raw+cgv,g)[roots]/V[roots,None,None]
        row={}
        for label,cap in [('isotropic_radius',old_cap),('affine_cap_volume',new_cap)]:
            divergence=group_sum(face+cap,g)[roots]/V[roots]
            error=divergence-L.trace()
            row[label]={'max_divergence_error_per_second':float(error.abs().max()),
                        'volume_weighted_rms_per_second':float((V[roots]*error.square()).sum().div(V[roots].sum()).sqrt()),
                        'global_volume_rate_error_m3_s':float((V[roots]*error).sum())}
        assert row['affine_cap_volume']['max_divergence_error_per_second']<1e-9,row
        row['production_full_gradient_max_error']=float((actual_gradient-L).abs().max())
        assert row['production_full_gradient_max_error']<1e-9,row
        result[name]=row
    suffix='' if args.cap_motion=='volume' else '_'+args.cap_motion
    (root/('hyperelastic_affine_volume_check'+suffix+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
