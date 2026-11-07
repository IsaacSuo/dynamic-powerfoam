"""Derivative of the actual MergedUL force, with its production G and adjoint.
Analytic tangent used by the optional persistent trust-region solver.
"""
import torch
import warp as wp
from free_surface import merged_ul as mu
from free_surface import closure_warp as cw

@wp.kernel(enable_backward=False)
def cell(D:wp.array(dtype=wp.mat33d),F:wp.array(dtype=wp.mat33d),R:wp.array(dtype=wp.mat33d),
         er:wp.array(dtype=wp.mat33d),be:wp.array(dtype=wp.float64),V:wp.array(dtype=wp.float64),
         C:wp.array(dtype=wp.mat33d),shear:wp.float64,kap:wp.float64,
         Gb:wp.array(dtype=wp.mat33d),L:wp.array(dtype=wp.float64),Lb:wp.array(dtype=wp.float64)):
    i=wp.tid();I=wp.identity(n=3,dtype=wp.float64)
    X=wp.transpose(R[i])@D[i]@F[i]
    de=wp.float64(.5)*(X+wp.transpose(X));de=de-(wp.trace(de)/wp.float64(3.))*I
    r=er[i]
    if be[i]<wp.float64(1.):
        de=be[i]*(de-r*(wp.ddot(r,de)/wp.ddot(r,r)))
    Gb[i]=(wp.float64(2.)*shear*V[i])*(R[i]@de@wp.transpose(F[i]))
    dl=wp.ddot(C[i],D[i]);L[i]=dl;Lb[i]=V[i]*kap*dl

@wp.kernel(enable_backward=False)
def spring(v:wp.array(dtype=wp.vec3d),D:wp.array(dtype=wp.mat33d),si:wp.array(dtype=int),sj:wp.array(dtype=int),
           sp:wp.array(dtype=wp.vec3d),sk:wp.array(dtype=wp.float64),sb:wp.array(dtype=wp.float64),
           g:wp.array(dtype=wp.vec3d),Gb:wp.array(dtype=wp.mat33d)):
    e=wp.tid();i=si[e];j=sj[e]
    if i==j:return
    half=wp.float64(.5);p=sp[e]
    f=(sk[e]*sb[e])*(v[j]-v[i]-half*((D[i]+D[j])@p))
    wp.atomic_add(g,j,f);wp.atomic_add(g,i,-f)
    q=-half*wp.outer(f,p);wp.atomic_add(Gb,i,q);wp.atomic_add(Gb,j,q)

@wp.kernel(enable_backward=False)
def ball(v:wp.array(dtype=wp.vec3d),body:wp.array(dtype=int),H:wp.array(dtype=wp.mat33d),g:wp.array(dtype=wp.vec3d)):
    i=wp.tid();b=body[i];wp.atomic_add(g,b,H[i]@v[b])

@wp.kernel(enable_backward=False)
def inertia(v:wp.array(dtype=wp.vec3d),cm:wp.array(dtype=wp.float64),g:wp.array(dtype=wp.vec3d)):
    i=wp.tid();g[i]=g[i]+cm[i]*v[i]

@wp.kernel(enable_backward=False)
def dhd(v:wp.array(dtype=wp.vec3d),H:wp.array(dtype=wp.vec3d),tr:wp.array(dtype=wp.float64)):
    i=wp.tid();wp.atomic_add(tr,cw.TR_DHD,wp.dot(v[i],H[i]))

class Tangent:
    def __init__(self,CG,gather=False):
        self.CG=CG;self.W=CG.W;W=self.W;n=W.n
        self.B={k:wp.zeros(n,dtype=wp.mat33d) for k in ('G0','Ga','Ge','Sg','G','Gb','Gs','Gbe','Gb0','er')}
        self.B.update({k:wp.zeros(n,dtype=wp.float64) for k in ('L','Lb','be')})
        self.zero=wp.zeros(n,dtype=wp.vec3d)
        self.terms=wp.zeros_like(W.B['terms'])
        self.Hball=wp.zeros(W.ball['k'],dtype=wp.mat33d)
        self.gather=None
        if gather:
            from free_surface.k1_tangent_gather import Gather
            self.gather=Gather(W)
        self.setbase()
        self.run();wp.capture_begin();self.run();self.graph=wp.capture_end()

    def setbase(self):
        self.refresh_geometry()
        W,CG=self.W,self.CG
        W.tx.copy_(CG.t['z']);W._run()
        wp.copy(self.B['er'],W.B['er']);wp.copy(self.B['be'],W.B['be'])
        bl=W.ball;tt=lambda a:wp.to_torch(a)
        ids=tt(bl['body']).long();z=CG.t['z'][ids]
        cs,wt,c0,Es,fric=tt(bl['par']).tolist();ng=torch.tensor(list(bl['ng']),device='cuda',dtype=torch.float64)
        xn=tt(bl['xn']);rad=tt(bl['R']);kt=tt(bl['kt']);bound=tt(bl['bound'])
        xe=cs*z-(cs-1)*xn;dn=rad-(xe@ng-c0)
        nn=ng[:,None]*ng[None,:];T=torch.eye(3,device='cuda',dtype=torch.float64)-nn
        H=(2*Es*torch.sqrt((rad*dn.clamp(min=0)).clamp(min=0)))[:,None,None]*nn
        dx=xe-xn;s=dx-(dx@ng)[:,None]*ng
        q=tt(bl['lamt'])-kt[:,None]*s;qn=q.norm(dim=1)
        good=(bound>0)&(fric>0);slip=good&(qn>bound)
        H=H+(good*kt)[:,None,None]*T
        if bool(slip.any()):
            u=q[slip]/qn[slip,None]
            J=bound[slip,None,None]/qn[slip,None,None]*(T-u[:,:,None]*u[:,None,:])
            H[slip]+=kt[slip,None,None]*(J-T)
        tt(self.Hball).copy_(cs*cs*wt*H)

    def refresh_geometry(self,force=False):
        if self.gather is not None:self.gather.refresh(force)

    def run(self):
        if self.gather is not None:
            self.gather.run(self)
            return
        W,CG,B=self.W,self.CG,self.B;U,S=W.ul,W.sp;n,K=W.n,W.ul['Kw'];v=CG.a['d'];g=CG.a['Hd']
        for k in ('G0','Ga','Sg','Gbe','Gb0'):B[k].zero_()
        g.zero_();self.terms.zero_()
        wp.launch(cw.k_Gul,2*K,inputs=[v,self.zero,U['S'],U['O'],U['R']],outputs=[B['G0']])
        wp.launch(cw.k_Aul,K,inputs=[B['G0'],U['As'],U['Ao'],U['Aw']],outputs=[B['Ga']])
        wp.launch(mu.k_Ge,n,inputs=[B['G0'],B['Ga'],U['used']],outputs=[B['Ge']])
        wall=[U[k] for k in ('wi','wj','wa','wn','wt','wtau')]
        wp.launch(mu.k_gg_wall,K,inputs=[v,self.zero]+wall+[B['Ge']],outputs=[B['Sg']])
        wp.launch(mu.k_gg_cell,n,inputs=[v,self.zero,B['Sg'],B['Ge'],U['a'],U['Mc'],U['Vi']],outputs=[B['G']])
        wp.launch(cell,n,inputs=[B['G'],S['F0'],S['Rh'],B['er'],B['be'],U['V0'],U['Q'],W.mu,W.kap],outputs=[B['Gb'],B['L'],B['Lb']])
        wp.launch(mu.k_var,K,inputs=[U['wi'],U['wj'],U['wv'],B['L']],outputs=[B['Lb'],self.terms])
        wp.launch(spring,S['K'],inputs=[v,B['G'],S['si'],S['sj'],S['sP'],S['sk'],S['sb']],outputs=[g,B['Gb']])
        wp.launch(mu.k_Lb,n,inputs=[B['Lb'],U['Q']],outputs=[B['Gb']])
        wp.launch(ball,W.ball['k'],inputs=[v,W.ball['body'],self.Hball],outputs=[g])
        wp.launch(mu.k_gg_cell_adj,n,inputs=[B['Gb'],U['a'],U['Mc'],U['Vi']],outputs=[g,B['Gs'],B['Gbe']])
        wp.launch(mu.k_gg_wall_adj,K,inputs=wall+[B['Gs']],outputs=[g,B['Gbe']])
        wp.launch(mu.k_Ge_adj,n,inputs=[B['Gbe'],U['used']],outputs=[B['Gb0']])
        wp.launch(mu.k_Aul_adj2,K,inputs=[B['Gbe'],U['As'],U['Ao'],U['Aw']],outputs=[B['Gb0']])
        wp.launch(cw.k_Gul_adj,2*K,inputs=[B['Gb0'],U['S'],U['O'],U['R']],outputs=[g])
        wp.launch(inertia,n,inputs=[v,CG.a['cm']],outputs=[g])
