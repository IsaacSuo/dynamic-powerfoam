"""Device-controlled implementation of k1_krylov's existing search rules.

The CUDA graph contains the direction loop, radius solve, KKT verification,
and the existing single joint-block retry. Python reads only final results.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
import weakref

import torch
import warp as wp
from free_surface.k1_tangent import Tangent

CAP=300
K,RUN,ADV,CHECK,VERIFY,SPEC,DENSE,GROUP,JOINT,CERT,ALLOW,PRODUCTS,CHECKS,INITIAL,RECORDS,ERROR,ICAP,REDO,NS=range(19)
GN,BN,RADIUS,STOP,LAMBDA,RN,PROJECTION,PMP,GP,PHP,RR,ND=range(12)
NAMES='g L Q AQ v co diag off rhs yc y ay p hp pp res smat ev work Lj jc je jwork part dval inputd inputh cm pinv'.split()

def library():
    source=Path(__file__).with_suffix('.cu')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache=Path(tempfile.gettempdir())/'powerfoam_krylov_gpu'
    cache.mkdir(exist_ok=True)
    so=cache/('device_'+digest+'.so')
    cuda=Path(os.environ.get('K1_CUDA_HOME','/usr/local/cuda-12.9'))
    if not so.exists():
        subprocess.run([str(cuda/'bin/nvcc'),'-shared','-Xcompiler','-fPIC','-O3',
                        '-std=c++17','-arch=sm_120','--fmad=false',str(source),
                        '-o',str(so),'-lcudart'],check=True)
    lib=ctypes.CDLL(str(so))
    lib.create_engine.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*5+[ctypes.c_int]
    lib.create_engine.restype=ctypes.c_void_p
    lib.phase.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int]
    lib.phase.restype=ctypes.c_int
    lib.destroy_engine.argtypes=[ctypes.c_void_p]
    return lib

class DeviceKrylovTR:
    def __init__(self,ST):
        self.ST,self.CG=ST,ST.CG
        self.T=Tangent(self.CG,gather=True)
        self.ng=torch.tensor(tuple(ST.W.ball['ng']),device=self.CG.t['d'].device,dtype=torch.float64)
        self.eye=torch.eye(3,device=self.CG.t['d'].device,dtype=torch.float64)
        self.lib=library()
        self.valid=False
        self.records=[]
        self.graph_build_seconds=0.
        self.n=self.CG.t['d'].numel()
        self.buffers={}
        def empty(*shape):return torch.empty(shape,device=self.CG.t['d'].device,dtype=torch.float64)
        for name in NAMES:self.buffers[name]=empty(1)
        for name in 'g v y ay p hp pp res'.split():self.buffers[name]=empty(self.n)
        self.buffers['L']=empty(self.n//3,3,3)
        self.buffers['Q']=empty(CAP+1,self.n)
        self.buffers['AQ']=empty(CAP,self.n)
        for name in 'co diag off rhs yc ev'.split():self.buffers[name]=empty(CAP)
        self.buffers['smat']=empty(CAP,CAP)
        self.buffers['Lj']=empty(18,18);self.buffers['jc']=empty(18,18);self.buffers['je']=empty(18)
        self.buffers['dval']=empty(ND)
        self.buffers['inputd']=self.CG.t['d'];self.buffers['inputh']=self.CG.t['Hd']
        self.buffers['cm']=self.CG.t['cm'];self.buffers['pinv']=self.CG.t['Pinv']
        self.state=torch.zeros(NS,dtype=torch.int32,device='cuda')
        self.ids=torch.zeros(6,dtype=torch.int32,device='cuda')
        self.info=torch.zeros(1,dtype=torch.int32,device='cuda')
        self.jinfo=torch.zeros_like(self.info)
        self.record_buffer=empty(2,9)
        ptrs=(ctypes.c_void_p*len(NAMES))(*(self.buffers[n].data_ptr() for n in NAMES))
        self.handle=self.lib.create_engine(ptrs,self.state.data_ptr(),self.ids.data_ptr(),self.info.data_ptr(),self.jinfo.data_ptr(),self.record_buffer.data_ptr(),self.n)
        self._cleanup=weakref.finalize(self,self.lib.destroy_engine,self.handle)
        self._cleanup.atexit=False  # CUDA tears down the process context itself.
        self.flags={i:wp.from_torch(self.state[i:i+1]) for i in (RUN,ADV,CHECK,VERIFY,SPEC,JOINT)}
        self.stream=wp.Stream(device='cuda')
        self.tstream=wp.stream_to_torch(self.stream)
        self._capture()
        self.signature=self._signature()

    def _signature(self):
        W=self.ST.W
        items=[(W.n,W.mu,W.kap,tuple(W.ball['ng']))]
        for name,table in [('ul',W.ul),('sp',W.sp),('ball',W.ball),('base',W.B),('cg',self.CG.a)]:
            for key,value in sorted(table.items()):
                if isinstance(value,wp.array):items.append((name,key,value.ptr,value.shape))
                elif isinstance(value,(float,int)):items.append((name,key,value))
        return tuple(items)

    def _phase(self,op,extra=0,radius=0.,stop=0.,cap=CAP):
        code=self.lib.phase(self.handle,op,wp.get_stream().cuda_stream,radius,stop,cap,extra)
        if code:raise RuntimeError('GPU Krylov phase %d launch error %d'%(op,code))

    def _capture(self):
        def advance():
            self._phase(2);self.T.run();self._phase(3)
        def candidate():
            self._phase(4)
            wp.capture_if(self.flags[SPEC],lambda:self._phase(5))
            self._phase(6)
        def verify():
            self._phase(7);self.T.run();self._phase(8)
        def joint():
            self._phase(10)
            for j in range(18):
                self._phase(11,extra=j);self.T.run();self._phase(12,extra=j)
            self._phase(13)
        def body():
            wp.capture_if(self.flags[ADV],advance)
            wp.capture_if(self.flags[CHECK],candidate)
            wp.capture_if(self.flags[VERIFY],verify)
            self._phase(9)
            wp.capture_if(self.flags[JOINT],joint)
        start=time.perf_counter()
        with torch.cuda.stream(self.tstream),wp.ScopedStream(self.stream):
            with wp.ScopedCapture() as capture:
                wp.capture_while(self.flags[RUN],body)
                self._phase(14)
        self.graph=capture.graph
        self.graph_build_seconds+=time.perf_counter()-start

    def invalidate(self):self.valid=False

    def _setbase(self):
        self.T.refresh_geometry()
        # Same tangent as Tangent.setbase, with parameters and slip decisions
        # kept on the device instead of .tolist() / bool(slip.any()).
        W,CG=self.ST.W,self.CG
        W.tx.copy_(CG.t['z']);W._run()
        wp.copy(self.T.B['er'],W.B['er']);wp.copy(self.T.B['be'],W.B['be'])
        bl=W.ball;tt=wp.to_torch
        ids=tt(bl['body']).long();z=CG.t['z'][ids]
        par=tt(bl['par']);cs,wt,c0,Es,fric=par.unbind()
        ng=self.ng
        xn,rad,kt,bound=(tt(bl[key]) for key in ('xn','R','kt','bound'))
        xe=cs*z-(cs-1)*xn;dn=rad-(xe@ng-c0)
        nn=ng[:,None]*ng[None,:];tangent=self.eye-nn
        H=(2*Es*torch.sqrt((rad*dn.clamp(min=0)).clamp(min=0)))[:,None,None]*nn
        dx=xe-xn;s=dx-(dx@ng)[:,None]*ng
        q=tt(bl['lamt'])-kt[:,None]*s;qn=q.norm(dim=1)
        good=(bound>0)&(fric>0);slip=good&(qn>bound)
        H=H+(good*kt)[:,None,None]*tangent
        safe=torch.where(slip,qn,torch.ones_like(qn))
        u=torch.where(slip[:,None],q/safe[:,None],0.)
        J=bound[:,None,None]/safe[:,None,None]*(tangent-u[:,:,None]*u[:,None,:])
        H=H+kt[:,None,None]*torch.where(slip[:,None,None],J-tangent,0.)
        tt(self.T.Hball).copy_(cs*cs*wt*H)

    def solve(self,g,delta,stop,cap,allow_joint=True):
        if cap>CAP:raise ValueError('GPU Krylov capacity is 300')
        if self._signature()!=self.signature:
            # A recut may replace arrays captured by the tangent; do not reuse
            # a graph whose pointers or launch sizes belong to the old cut.
            records=self.records
            builds=self.graph_build_seconds
            wp.synchronize_stream(self.stream)
            self.graph=None
            self._cleanup()
            self.handle=None
            self.__init__(self.ST)
            self.records=records
            self.graph_build_seconds+=builds
        fresh=not self.valid
        if fresh:
            self._setbase()
            self.buffers['g'].copy_(wp.to_torch(g).flatten())
        previous=torch.cuda.current_stream()
        self.tstream.wait_stream(previous)
        with torch.cuda.stream(self.tstream),wp.ScopedStream(self.stream):
            self._phase(0,extra=int(fresh),radius=float(delta),stop=float(stop),cap=cap)
            if not allow_joint:self.state[ALLOW]=0
            if fresh:self._phase(1)
            wp.capture_launch(self.graph)
            # One packed result transfer after the entire device-controlled search.
            packed=torch.cat((self.state.double(),self.buffers['dval'],self.record_buffer.flatten(),self.ids.double())).cpu().numpy()
        previous.wait_stream(self.tstream)
        s=packed[:NS].astype(int);d=packed[NS:NS+ND]
        if s[ERROR]:raise RuntimeError('GPU Krylov numerical failure %d'%s[ERROR])
        self.CG.t['p'].copy_(self.buffers['p'].reshape(-1,3))
        self.CG.t['Hp'].copy_(self.buffers['hp'].reshape(-1,3))
        self.CG.t['r'].copy_((-self.buffers['g']-self.buffers['hp']).reshape(-1,3))
        ids=packed[-6:].astype(int).tolist()
        rec=packed[NS+ND:NS+ND+18].reshape(2,9)
        for j in range(s[RECORDS]):
            r=rec[j]
            self.records.append(dict(basis=int(r[0]),added=int(r[0])-s[INITIAL] if j==0 else int(r[0]),
                reused=not fresh and j==0,products=int(r[7]-(rec[j-1,7] if j else 0)),joint=ids if r[1] else None,
                dense=bool(r[2]),certified=bool(r[3]),kkt_relative=float(r[4]),
                radius=float(d[RADIUS]),multiplier=float(r[5]),norm=float(r[8]),projection_error=float(r[6])))
        self.valid=True
        return dict(K=int(s[PRODUCTS]),EDGE=d[LAMBDA]>0,NEG=False,PMP=float(d[PMP]),DELTA=float(d[RADIUS]),GP=float(d[GP]),PHP=float(d[PHP]),RR=float(d[RR]))
