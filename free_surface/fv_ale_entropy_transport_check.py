"""Independent entropy-potential, covariance and density-bound checks."""
import json
from pathlib import Path
import torch
from free_surface.fv_ale_hyperelastic_material import response,symmetric_deviator
from free_surface.fv_ale_entropy_transport import entropy_state


def main():
    torch.manual_seed(1381);torch.set_num_threads(1);n=2000;dtype=torch.float64
    mu,bulk,rho0=6e4,1.4e5,1000.
    H=symmetric_deviator(torch.randn(2*n,3,3,dtype=dtype)*.8)
    v=torch.randn(2*n,3,dtype=dtype)*4;rho=rho0*torch.exp(torch.randn(2*n,dtype=dtype))
    # Include equal states, nearly equal states and constant-density pairs.
    H[n:n+20]=H[:20];v[n:n+20]=v[:20];rho[n:n+20]=rho[:20]
    H[n+20:n+40]=H[20:40]+symmetric_deviator(torch.randn(20,3,3,dtype=dtype)*1e-9)
    rho[n+20:n+100]=rho[20:100]
    for k,exponent in enumerate(range(-14,1)):
        sl=slice(100+20*k,120+20*k);sr=slice(n+sl.start,n+sl.stop)
        H[sr]=H[sl]+10.**exponent*symmetric_deviator(torch.randn(20,3,3,dtype=dtype))
        rho[sr]=rho[sl]
    i=torch.arange(n);j=i+n
    def calc(H,v):
        p,_,_,e,_,B,_=response(H,rho,mu,bulk,rho0)
        T=mu/(2*rho0)*symmetric_deviator(B)
        shape=mu/(2*rho0)*(B.diagonal(dim1=1,dim2=2).sum(1)-3)
        face=entropy_state(rho,v,H,T,shape,rho0,i,j)
        chi=torch.zeros((2*n,16),dtype=dtype)
        chi[:,0]=e+p/rho-(T*H).sum((1,2))-.5*v.square().sum(1)
        chi[:,1:4]=v;chi[:,5:14]=T.reshape(-1,9)
        return face,chi,p
    face,chi,p=calc(H,v)
    defect=((chi[i]-chi[j])*face).sum(1)-(p[i]-p[j])
    scale=(p[i].abs()+p[j].abs()+mu+1)
    entropy_error=float((defect.abs()/scale).max())
    density_error=float(torch.maximum(face[:,0]-torch.maximum(rho[i],rho[j]),torch.minimum(rho[i],rho[j])-face[:,0]).clamp(min=0).max())
    hf=face[:,5:14].reshape(-1,3,3)/face[:,0,None,None]
    mean_bound=float(((hf-.5*(H[i]+H[j])).norm(dim=(1,2))-.5*(H[i]-H[j]).norm(dim=(1,2))).clamp(min=0).max())
    trace_error=float(hf.diagonal(dim1=1,dim2=2).sum(1).abs().max())
    R=torch.matrix_exp(torch.tensor([[0.,-.2,.7],[.2,0.,-.3],[-.7,.3,0.]],dtype=dtype))
    boost=torch.tensor([1.3,-.7,.2],dtype=dtype)
    other,_,_=calc(R@H@R.T,v@R.T+boost)
    expected_H=R@face[:,5:14].reshape(-1,3,3)@R.T
    rotation_error=float((other[:,5:14].reshape(-1,3,3)-expected_H).norm()/expected_H.norm())
    velocity_error=float((other[:,1:4]-face[:,1:4]@R.T-face[:,:1]*boost).norm()/face[:,1:4].norm())
    swap=entropy_state(rho,v,H,mu/(2*rho0)*symmetric_deviator(response(H,rho,mu,bulk,rho0)[5]),
        mu/(2*rho0)*(response(H,rho,mu,bulk,rho0)[5].diagonal(dim1=1,dim2=2).sum(1)-3),rho0,j,i)
    symmetry_error=float((swap-face).norm()/face.norm())
    assert entropy_error<1e-11 and density_error<1e-10,(entropy_error,density_error)
    assert mean_bound<1e-10 and trace_error<1e-10,(mean_bound,trace_error)
    assert max(rotation_error,velocity_error,symmetry_error)<1e-10,(rotation_error,velocity_error,symmetry_error)
    result=dict(entropy_identity_scaled_error=entropy_error,density_bound_error=density_error,
        rotation_relative_error=rotation_error,boost_relative_error=velocity_error,swap_relative_error=symmetry_error,
        shape_mean_ball_violation=mean_bound,shape_trace_error=trace_error)
    Path('output/ale/hyperelastic_entropy_transport_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
