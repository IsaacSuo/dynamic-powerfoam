"""Large-strain/near-incompressible CPU checks of the production material."""
import json,math
from pathlib import Path
import torch
from free_surface.fv_ale_hyperelastic_check import from_F
from free_surface.fv_ale_hyperelastic_material import response,log_metric_rate,acoustic_tensor


def main():
    torch.set_num_threads(1);torch.manual_seed(820);dtype=torch.float64
    F=torch.matrix_exp(torch.randn((128,3,3),dtype=dtype)*.65)
    F=torch.cat((F,torch.diag_embed(torch.tensor([[5.,.5,.4],[.15,2.,3.],[.5,.5,.5],[2.,2.,2.]],dtype=dtype))))
    H,rho=from_F(F);rho0=1000.;E=2e5;I=torch.eye(3,dtype=dtype)
    L=torch.randn_like(F)*.2;h=1e-6;results={}
    for nu in [.0,.3,.45,.49,.499]:
        mu=E/(2*(1+nu));bulk=E/(3*(1-2*nu))
        out=response(H,rho,mu,bulk,rho0);sigma,energy,c2,B,J=out[2:]
        Hp,rp=from_F(torch.matrix_exp(h*L)@F);Hm,rm=from_F(torch.matrix_exp(-h*L)@F)
        derivative=(response(Hp,rp,mu,bulk,rho0)[3]-response(Hm,rm,mu,bulk,rho0)[3])/(2*h)
        work=(sigma*L).sum((1,2))/rho
        work_error=float((derivative-work).norm()/work.norm())
        rate=log_metric_rate(H,L);kin_error=float(((Hp-Hm)/(2*h)-rate).norm()/rate.norm())
        assert work_error<1e-7 and kin_error<1e-7,(nu,work_error,kin_error)
        lower=math.inf;upper=0.
        for _ in range(32):
            n=torch.randn((len(F),3),dtype=dtype);n/=n.norm(dim=1)[:,None]
            eig=torch.linalg.eigvalsh(acoustic_tensor(B,J,n,mu,bulk))/rho[:,None]
            lower=min(lower,float(eig[:,0].min()));upper=max(upper,float((eig[:,-1]/c2).max()))
        assert lower>0 and upper<=1+1e-12,(nu,lower,upper)
        results[str(nu)]={'work_relative_error':work_error,'log_rate_relative_error':kin_error,
                         'minimum_acoustic_eigenvalue':lower,'maximum_eigenvalue_over_speed_bound':upper}
    # A finite rigid rotation must rotate an already stretched material without
    # changing its stored energy; test time discretization, not just covariance.
    initial=H[-4:-3].clone();omega=torch.tensor([[0.,-1.,.4],[1.,0.,-.2],[-.4,.2,0.]],dtype=dtype)[None]
    mu=E/2.6;bulk=E/(3*.4);rotation=[];duration=1.
    for steps in [32,64,128,256]:
        dt=duration/steps;state=initial.clone()
        for _ in range(steps):
            rate=log_metric_rate(state,omega);trial=state+dt*rate
            state+=.5*dt*(rate+log_metric_rate(trial,omega))
        R=torch.matrix_exp(duration*omega);exact=R@initial@R.transpose(1,2)
        e0=response(initial,rho[-4:-3],mu,bulk,rho0)[3]
        e1=response(state,rho[-4:-3],mu,bulk,rho0)[3]
        rotation.append({'steps':steps,'shape_error':float((state-exact).norm()),
                         'relative_energy_error':float((e1-e0)/e0)})
    assert rotation[-1]['shape_error']<rotation[0]['shape_error']/40,rotation
    report={'positive_determinant_range':[float(torch.linalg.det(F).min()),float(torch.linalg.det(F).max())],
            'largest_stretch':float(torch.linalg.svdvals(F).max()),'materials':results,'finite_rotation':rotation}
    Path('output/ale/hyperelastic_material_extended.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
