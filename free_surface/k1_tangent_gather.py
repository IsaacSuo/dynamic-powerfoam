"""Production tangent with a gathered fit and fused pointwise operations.

The original wall/spring scatter remains: replacing it with expanded stencil
tables increased memory traffic in measured production states.
"""
import torch
import warp as wp

@wp.kernel(enable_backward=False)
def fit_rows(v:wp.array2d(dtype=wp.float64),offsets:wp.array(dtype=int),edges:wp.array(dtype=int),
             O:wp.array(dtype=int),R:wp.array(dtype=wp.vec3d),G:wp.array2d(dtype=wp.float64)):
    b,r=wp.tid();a=wp.vec3d()
    for q in range(offsets[b],offsets[b+1]):
        e=edges[q];a=a+(v[O[e],r]-v[b,r])*R[e]
    G[b,3*r]=a[0];G[b,3*r+1]=a[1];G[b,3*r+2]=a[2]

@wp.kernel(enable_backward=False)
def geometry_cell(v:wp.array(dtype=wp.vec3d),Sg:wp.array(dtype=wp.mat33d),Ge:wp.array(dtype=wp.mat33d),
                  a:wp.array(dtype=wp.vec3d),Mc:wp.array(dtype=wp.mat33d),Vi:wp.array(dtype=wp.float64),
                  F:wp.array(dtype=wp.mat33d),R:wp.array(dtype=wp.mat33d),er:wp.array(dtype=wp.mat33d),
                  be:wp.array(dtype=wp.float64),V:wp.array(dtype=wp.float64),C:wp.array(dtype=wp.mat33d),
                  shear:wp.float64,kap:wp.float64,D:wp.array(dtype=wp.mat33d),
                  Gb:wp.array(dtype=wp.mat33d),L:wp.array(dtype=wp.float64),Lb:wp.array(dtype=wp.float64)):
    i=wp.tid();I=wp.identity(n=3,dtype=wp.float64)
    d=(Sg[i]+wp.outer(v[i],a[i])+Ge[i]@Mc[i])*Vi[i];D[i]=d
    X=wp.transpose(R[i])@d@F[i]
    de=wp.float64(.5)*(X+wp.transpose(X));de=de-(wp.trace(de)/wp.float64(3.))*I
    r=er[i]
    if be[i]<wp.float64(1.):
        de=be[i]*(de-r*(wp.ddot(r,de)/wp.ddot(r,r)))
    Gb[i]=(wp.float64(2.)*shear*V[i])*(R[i]@de@wp.transpose(F[i]))
    dl=wp.ddot(C[i],d);L[i]=dl;Lb[i]=V[i]*kap*dl

@wp.kernel(enable_backward=False)
def variance(wi:wp.array(dtype=int),wj:wp.array(dtype=int),wv:wp.array(dtype=wp.float64),
             L:wp.array(dtype=wp.float64),Lb:wp.array(dtype=wp.float64)):
    e=wp.tid();i=wi[e];j=wj[e]
    if i!=j:
        f=wp.float64(2.)*wv[e]*(L[i]-L[j])
        wp.atomic_add(Lb,i,f);wp.atomic_add(Lb,j,-f)

@wp.kernel(enable_backward=False)
def pressure_adjoint(Gb:wp.array(dtype=wp.mat33d),Lb:wp.array(dtype=wp.float64),C:wp.array(dtype=wp.mat33d),
                     a:wp.array(dtype=wp.vec3d),Mc:wp.array(dtype=wp.mat33d),Vi:wp.array(dtype=wp.float64),
                     g:wp.array(dtype=wp.vec3d),Gs:wp.array(dtype=wp.mat33d),Gbe:wp.array(dtype=wp.mat33d)):
    i=wp.tid();p=Gb[i]+Lb[i]*C[i];Gb[i]=p;h=p*Vi[i]
    Gs[i]=h;g[i]=g[i]+h@a[i];Gbe[i]=h@wp.transpose(Mc[i])

@wp.kernel(enable_backward=False)
def fit_adjoint_init(Gbe:wp.array(dtype=wp.mat33d),used:wp.array(dtype=int),Gb0:wp.array(dtype=wp.mat33d)):
    i=wp.tid();a=wp.mat33d()
    if used[i]==0:a=Gbe[i]
    Gb0[i]=a

class Gather:
    def __init__(self,W):
        self.W=W;self.token=None;self.views={}
        self.nodes=torch.arange(W.n+1,dtype=torch.long,device='cuda')
        self.off=torch.empty(W.n+1,dtype=torch.int32,device='cuda')
        self.edges=torch.empty(2*W.ul['Kw'],dtype=torch.int32,device='cuda')
        self.offsets=wp.from_torch(self.off);self.order=wp.from_torch(self.edges)

    def refresh(self,force=False):
        W=self.W;token=(getattr(W,'_ul_gen',None),id(getattr(W,'_ulG',None)))
        if not force and token==self.token:return False
        s,o=(wp.to_torch(W.ul[k]).long() for k in ('S','O'))
        keys=torch.where(s!=o,s,W.n);order=torch.argsort(keys,stable=True)
        self.edges.copy_(order);self.off.copy_(torch.searchsorted(keys[order],self.nodes))
        self.token=token
        return False  # Fixed allocations remain valid in captured CUDA graphs.

    def view(self,a,components):
        key=(a.ptr,components)
        if key not in self.views:self.views[key]=wp.from_torch(wp.to_torch(a).reshape(-1,components))
        return self.views[key]

    def run(self,T):
        from free_surface import merged_ul as mu,closure_warp as cw
        from free_surface.k1_tangent import spring,ball,inertia
        W,CG,B=T.W,T.CG,T.B;U,S=W.ul,W.sp;n,K=W.n,W.ul['Kw'];v=CG.a['d'];g=CG.a['Hd']
        B['Ga'].zero_();B['Sg'].zero_();g.zero_()
        wp.launch(fit_rows,(n,3),inputs=[self.view(v,3),self.offsets,self.order,U['O'],U['R']],outputs=[self.view(B['G0'],9)])
        wp.launch(cw.k_Aul,K,inputs=[B['G0'],U['As'],U['Ao'],U['Aw']],outputs=[B['Ga']])
        wp.launch(mu.k_Ge,n,inputs=[B['G0'],B['Ga'],U['used']],outputs=[B['Ge']])
        wallargs=[U[k] for k in ('wi','wj','wa','wn','wt','wtau')]
        wp.launch(mu.k_gg_wall,K,inputs=[v,T.zero,*wallargs,B['Ge']],outputs=[B['Sg']])
        wp.launch(geometry_cell,n,inputs=[v,B['Sg'],B['Ge'],U['a'],U['Mc'],U['Vi'],S['F0'],S['Rh'],B['er'],B['be'],U['V0'],U['Q'],W.mu,W.kap],outputs=[B['G'],B['Gb'],B['L'],B['Lb']])
        wp.launch(variance,K,inputs=[U['wi'],U['wj'],U['wv'],B['L']],outputs=[B['Lb']])
        wp.launch(spring,S['K'],inputs=[v,B['G'],S['si'],S['sj'],S['sP'],S['sk'],S['sb']],outputs=[g,B['Gb']])
        wp.launch(pressure_adjoint,n,inputs=[B['Gb'],B['Lb'],U['Q'],U['a'],U['Mc'],U['Vi']],outputs=[g,B['Gs'],B['Gbe']])
        wp.launch(ball,W.ball['k'],inputs=[v,W.ball['body'],T.Hball],outputs=[g])
        wp.launch(mu.k_gg_wall_adj,K,inputs=[*wallargs,B['Gs']],outputs=[g,B['Gbe']])
        wp.launch(fit_adjoint_init,n,inputs=[B['Gbe'],U['used']],outputs=[B['Gb0']])
        wp.launch(mu.k_Aul_adj2,K,inputs=[B['Gbe'],U['As'],U['Ao'],U['Aw']],outputs=[B['Gb0']])
        wp.launch(cw.k_Gul_adj,2*K,inputs=[B['Gb0'],U['S'],U['O'],U['R']],outputs=[g])
        wp.launch(inertia,n,inputs=[v,CG.a['cm']],outputs=[g])
