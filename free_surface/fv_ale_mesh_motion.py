"""Optional ALE interior mesh-motion least squares, on the actual geometry.

Only the mesh velocity changes. The acoustic material face state, forces,
constitutive rate and all current free/contact boundary motion stay fixed.
Minimise area-weighted face-normal slip plus a positive diagonal penalty on
the correction. This is an experimental choice within ALE mesh freedom, not
a new material law or a claim of reproducing a particular published scheme.
"""
import torch


def fit_motion(b,w,dr,us,iterations=24,regularization=.1,allow_radius=False):
    p,r=b['p'],b['r'];i,j,A,n,c=b['i'],b['j'],b['A'],b['n'],b['c']
    g=b['groups'];distance=(p[j]-p[i]).norm(dim=1)
    distance=torch.where(A>0,distance,torch.ones_like(distance))
    ai=torch.cat(((c-p[i])/distance[:,None],(r[i]/distance)[:,None]),1)
    aj=torch.cat((-(c-p[j])/distance[:,None],(-r[j]/distance)[:,None]),1)
    free=(~b['surface'])&(b['floor']==0)&(b['V']>0)
    mask=free[:,None].expand(-1,4).clone()
    if not allow_radius:mask[:,3]=False
    base=torch.cat((w,dr[:,None]),1)
    slip=(us*n).sum(1)-(ai*base[i]+aj*base[j]).sum(1)
    weight=torch.where(g[i]!=g[j],A,0.)
    ai=ai*mask[i];aj=aj*mask[j]
    def forward(x):return (ai*x[i]+aj*x[j]).sum(1)
    def transpose(y):
        return torch.zeros_like(base).index_add(0,i,ai*y[:,None]).index_add(0,j,aj*y[:,None])
    diag=torch.zeros_like(base).index_add(0,i,weight[:,None]*ai.square())
    diag.index_add_(0,j,weight[:,None]*aj.square())
    diagonal=diag.clamp(min=1e-300)
    rhs=transpose(weight*slip)
    def product(x):return transpose(weight*forward(x))+regularization*diagonal*x
    correction=torch.zeros_like(base);residual=rhs.clone()
    z=residual/((1+regularization)*diagonal);direction=z.clone()
    rz=(residual*z).sum();initial=rz
    for _ in range(iterations):
        applied=product(direction);pap=(direction*applied).sum()
        active=(rz>1e-16*initial)&(pap>0)
        step=torch.where(active,rz/pap.clamp(min=1e-300),0.)
        correction=correction+step*direction;residual=residual-step*applied
        z=residual/((1+regularization)*diagonal);newrz=(residual*z).sum()
        beta=torch.where(active,newrz/rz.clamp(min=1e-300),0.)
        direction=z+beta*direction;rz=newrz
    # The finite-iteration solve should decrease this strictly convex objective.
    # Retain the original mesh motion if roundoff ever defeats that property.
    original=(weight*slip.square()).sum()
    after=(weight*(slip-forward(correction)).square()).sum()
    penalty=regularization*(diagonal*correction.square()).sum()
    accepted=torch.isfinite(after+penalty)&(after+penalty<=original*(1+1e-12)+1e-30)
    correction=torch.where(accepted,correction,torch.zeros_like(correction))
    result=base+correction
    diagnostics=torch.stack((original,(weight*(slip-forward(correction)).square()).sum(),
                             correction[:,:3].norm(dim=1).max(),correction[:,3].abs().max(),
                             (residual.square().sum()/rhs.square().sum().clamp(min=1e-300)).sqrt(),
                             accepted.to(w.dtype)))
    return result[:,:3],result[:,3],diagnostics
