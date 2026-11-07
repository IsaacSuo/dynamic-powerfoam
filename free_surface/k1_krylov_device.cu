#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <algorithm>

// Fixed storage capacity, variable active prefix. All decisions are device-side.
constexpr int C=300, B=256;
enum { K,RUN,ADV,CHECK,VERIFY,SPEC,DENSE,GROUP,JOINT,CERT,ALLOW,PRODUCTS,
       CHECKS,INITIAL,RECORDS,ERROR,CAP,REDO,NS };
enum { GN,BN,RADIUS,STOP,LAMBDA,RN,PROJECTION,PMP,GP,PHP,RR,ND };
enum { G,L,Q,AQ,V,CO,DIAG,OFF,RHS,YC,Y,AY,P,HP,PP,RES,SMAT,EV,WORK,
       LJ,JC,JE,JWORK,PART,DVAL,INPUTD,INPUTH,CM,PINV,NB };
struct Engine {
  double* a[NB]; int *s,*ids,*info,*jinfo;
  int n; double* records; int jacobi_blocks;
};
__device__ double sumblock(double x) {
  __shared__ double sh[B]; int t=threadIdx.x; sh[t]=x; __syncthreads();
  for(int d=B/2;d;d/=2) {if(t<d)sh[t]+=sh[t+d];__syncthreads();} double out=sh[0];__syncthreads();return out;
}
__device__ double maxblock(double x) {
  __shared__ double sh[B]; int t=threadIdx.x; sh[t]=x;__syncthreads();
  for(int d=B/2;d;d/=2) {if(t<d)sh[t]=fmax(sh[t],sh[t+d]);__syncthreads();}double out=sh[0];__syncthreads();return out;
}
__device__ double mapval(Engine e,const double* x,int i,bool trans,bool inverse=false) {
  int cell=i/3,axis=i%3,gi=-1;
  if(e.s[GROUP])for(int j=0;j<6;j++)if(e.ids[j]==cell)gi=3*j+axis;
  if(gi>=0) {
    // Inverse transpose is handled by a separate, tiny triangular solve.
    double z=0;for(int j=0;j<18;j++)z+=(trans?e.a[LJ][j*18+gi]:e.a[LJ][gi*18+j])*x[3*e.ids[j/3]+j%3];return z;
  }
  double z=0;for(int j=0;j<3;j++)z+=(trans?e.a[L][cell*9+j*3+axis]:e.a[L][cell*9+axis*3+j])*x[cell*3+j];return z;
}
__global__ void init_context(Engine e) {
  int cell=blockIdx.x*B+threadIdx.x;if(cell>=e.n/3)return;
  const double* A=e.a[PINV]+9*cell;double* l=e.a[L]+9*cell;
  for(int j=0;j<9;j++)l[j]=0;
  l[0]=sqrt(A[0]);l[3]=A[3]/l[0];l[6]=A[6]/l[0];
  l[4]=sqrt(A[4]-l[3]*l[3]);l[7]=(A[7]-l[6]*l[3])/l[4];l[8]=sqrt(A[8]-l[6]*l[6]-l[7]*l[7]);
  if(!isfinite(l[8]))atomicExch(e.s+ERROR,1);
}
__global__ void reset(Engine e,double radius,double stop,int cap,int fresh) {
  if(threadIdx.x)return;
  if(fresh){for(int j=0;j<NS;j++)e.s[j]=0;e.s[ALLOW]=1;e.a[DVAL][LAMBDA]=0;}
  e.s[INITIAL]=e.s[K];e.s[PRODUCTS]=0;e.s[CHECKS]=0;e.s[RECORDS]=0;
  e.s[CAP]=cap;e.s[RUN]=1;e.s[ADV]=(e.s[K]==0);e.s[CHECK]=!e.s[ADV];
  e.s[VERIFY]=e.s[JOINT]=e.s[CERT]=e.s[REDO]=0;
  e.a[DVAL][RADIUS]=radius;e.a[DVAL][STOP]=stop;
}
__global__ void make_b(Engine e) {int i=blockIdx.x*B+threadIdx.x;if(i<e.n)e.a[V][i]=mapval(e,e.a[G],i,true);}
__global__ void map_residual(Engine e) {int i=blockIdx.x*B+threadIdx.x;if(i<e.n)e.a[V][i]=mapval(e,e.a[RES],i,true);}
__global__ void norm_b(Engine e) {
  double b=0,g=0;for(int i=threadIdx.x;i<e.n;i+=B){b+=e.a[V][i]*e.a[V][i];g+=e.a[G][i]*e.a[G][i];}
  b=sumblock(b);g=sumblock(g);if(!threadIdx.x){e.a[DVAL][BN]=sqrt(b);e.a[DVAL][GN]=sqrt(g);if(e.a[DVAL][RADIUS]<=0)e.a[DVAL][RADIUS]=sqrt(b);}
}
__global__ void q0(Engine e) {int i=blockIdx.x*B+threadIdx.x;if(i<e.n)e.a[Q][i]=-e.a[V][i]/e.a[DVAL][BN];}
__global__ void make_direction(Engine e,int kind,int col) {
  int i=blockIdx.x*B+threadIdx.x;if(i>=e.n)return;double v=0;
  if(kind==0)v=mapval(e,e.a[Q]+e.s[K]*e.n,i,false);
  else if(kind==1)v=e.a[P][i];
  else v=(i==3*e.ids[col/3]+col%3)?1.:0.;
  e.a[INPUTD][i]=v;
}
__global__ void map_product(Engine e) {
  int i=blockIdx.x*B+threadIdx.x;if(i<e.n){double v=mapval(e,e.a[INPUTH],i,true);e.a[V][i]=v;e.a[AQ][e.s[K]*e.n+i]=v;}
}
// Split long dot products across four blocks; both complete projection passes
// remain unchanged. PART is reused by norm reduction and the later spectral solve.
__global__ void dot_parts(Engine e,int kind) {
  constexpr int S=4;
  int j=blockIdx.x,k=e.s[K],part=blockIdx.y;
  if(!(kind==0?j<=k:j<k))return;
  double x=0;
  for(int i=part*B+threadIdx.x;i<e.n;i+=S*B)x+=e.a[Q][j*e.n+i]*e.a[V][i];
  x=sumblock(x);if(!threadIdx.x)e.a[PART][j*S+part]=x;
}
__global__ void dots(Engine e,int kind) {
  constexpr int S=4;
  int j=blockIdx.x*B+threadIdx.x,k=e.s[K];
  if(!(kind==0?j<=k:j<k))return;
  double x=0;for(int p=0;p<S;p++)x+=e.a[PART][j*S+p];
  if(kind==1)e.a[RHS][j]=x;else e.a[CO][j]=x;
}
// Accumulate the squared norm while the second subtraction is in registers.
__global__ void subtract(Engine e,int first) {
  int i=blockIdx.x*B+threadIdx.x,k=e.s[K];double v=0;
  if(i<e.n){double x=0;for(int j=0;j<=k;j++)x+=e.a[CO][j]*e.a[Q][j*e.n+i];
    v=e.a[V][i]-x;e.a[V][i]=v;}
  if(first){if(i==0)e.a[DIAG][k]=e.a[CO][k];}
  else{double n=sumblock(v*v);if(!threadIdx.x)e.a[PART][blockIdx.x]=n;}
}
__global__ void norm_v(Engine e) {
  int blocks=(e.n+B-1)/B;double x=0;
  for(int j=threadIdx.x;j<blocks;j+=B)x+=e.a[PART][j];
  x=sumblock(x);if(!threadIdx.x)e.a[OFF][e.s[K]]=sqrt(x);
}
__global__ void next_q(Engine e) {int i=blockIdx.x*B+threadIdx.x,k=e.s[K];if(i<e.n)e.a[Q][(k+1)*e.n+i]=e.a[V][i]/fmax(e.a[OFF][k],1e-300);}
__global__ void advance_done(Engine e) {if(threadIdx.x)return;int k=++e.s[K];e.s[PRODUCTS]++;e.s[CHECK]=(k==1||k==2||k%5==0||k==e.s[CAP]);e.s[VERIFY]=0;e.s[REDO]=0;}

// Parallel cyclic reduction solves the same shifted tridiagonal system.
// Its result is tested against the unchanged radius and KKT tolerances.
__device__ void tri_solve(Engine e,double lam,const double* rhs,double* result,double* a,double* b,double* c,double* d,int k) {
  int t=threadIdx.x;
  for(int i=t;i<512;i+=B){a[i]=(i>0&&i<k)?e.a[OFF][i-1]:0;c[i]=(i+1<k)?e.a[OFF][i]:0;b[i]=i<k?e.a[DIAG][i]+lam:1;d[i]=i<k?rhs[i]:0;}
  __syncthreads();
  for(int stride=1;stride<k;stride*=2){
    double aa[2],bb[2],cc[2],dd[2];
    for(int z=0;z<2;z++){int i=t+z*B;double l=i>=stride?-a[i]/b[i-stride]:0,r=i+stride<512?-c[i]/b[i+stride]:0;
      aa[z]=i>=stride?l*a[i-stride]:0;cc[z]=i+stride<512?r*c[i+stride]:0;
      bb[z]=b[i]+(i>=stride?l*c[i-stride]:0)+(i+stride<512?r*a[i+stride]:0);
      dd[z]=d[i]+(i>=stride?l*d[i-stride]:0)+(i+stride<512?r*d[i+stride]:0);
    }__syncthreads();for(int z=0;z<2;z++){int i=t+z*B;a[i]=aa[z];b[i]=bb[z];c[i]=cc[z];d[i]=dd[z];}__syncthreads();
  }
  for(int i=t;i<k;i+=B)result[i]=d[i]/b[i];__syncthreads();
}
__global__ void reduced_tri(Engine e) {
  __shared__ double a[512],b[512],c[512],d[512],rhs[512],y[512],invy[512],yhi[512],z[12];
  __shared__ int good,done;
  int t=threadIdx.x,k=e.s[K];if(!t){e.s[SPEC]=e.s[DENSE];good=1;double piv=0;for(int j=0;j<k;j++){piv=e.a[DIAG][j]-(j?e.a[OFF][j-1]*e.a[OFF][j-1]/piv:0);if(!(piv>0)){good=0;break;}}if(!good)e.s[SPEC]=1;}
  __syncthreads();if(e.s[SPEC])return;
  for(int i=t;i<k;i+=B)rhs[i]=-e.a[RHS][i];__syncthreads();
  tri_solve(e,0,rhs,y,a,b,c,d,k);double s=0;for(int i=t;i<k;i+=B)s+=y[i]*y[i];s=sqrt(sumblock(s));
  if(s<=e.a[DVAL][RADIUS]){for(int i=t;i<k;i+=B)e.a[YC][i]=y[i];if(!t)e.a[DVAL][LAMBDA]=0;return;}
  s=0;for(int i=t;i<k;i+=B)s+=rhs[i]*rhs[i];s=sqrt(sumblock(s));
  if(!t){z[0]=0;z[1]=fmax(fmax(e.a[DVAL][LAMBDA],s/e.a[DVAL][RADIUS]),1e-14);done=0;}__syncthreads();
  for(int it=0;it<100;it++){
    tri_solve(e,z[1],rhs,yhi,a,b,c,d,k);s=0;for(int i=t;i<k;i+=B)s+=yhi[i]*yhi[i];s=sqrt(sumblock(s));
    if(s<=e.a[DVAL][RADIUS])break;if(!t)z[1]*=2;__syncthreads();
  }
  if(!t){z[2]=fmin(fmax(e.a[DVAL][LAMBDA],z[0]),z[1]);if(!(z[2]>z[0]&&z[2]<z[1]))z[2]=.5*(z[0]+z[1]);}__syncthreads();
  for(int it=0;it<45;it++){
    tri_solve(e,z[2],rhs,y,a,b,c,d,k);s=0;for(int i=t;i<k;i+=B)s+=y[i]*y[i];s=sqrt(sumblock(s));
    if(s<=e.a[DVAL][RADIUS]){for(int i=t;i<k;i+=B)yhi[i]=y[i];if(!t){z[1]=z[2];done=e.a[DVAL][RADIUS]-s<=1e-9*e.a[DVAL][RADIUS];}}
    else if(!t)z[0]=z[2];__syncthreads();if(done)break;
    tri_solve(e,z[2],y,invy,a,b,c,d,k);double prod=0;for(int i=t;i<k;i+=B)prod+=y[i]*invy[i];prod=sumblock(prod);
    if(!t){double candidate=z[2]-(s-e.a[DVAL][RADIUS])/(-prod/s);z[2]=(candidate>z[0]&&candidate<z[1])?candidate:.5*(z[0]+z[1]);}__syncthreads();
  }
  for(int i=t;i<k;i+=B)e.a[YC][i]=yhi[i];if(!t)e.a[DVAL][LAMBDA]=z[1];
}
__global__ void dense_matrix(Engine e) {
  int i=blockIdx.x/C,j=blockIdx.x%C,k=e.s[K];if(i>=k||j>=k)return;double x=0;
  if(e.s[DENSE]){for(int q=threadIdx.x;q<e.n;q+=B)x+=.5*(e.a[Q][i*e.n+q]*e.a[AQ][j*e.n+q]+e.a[Q][j*e.n+q]*e.a[AQ][i*e.n+q]);x=sumblock(x);}
  else x=i==j?e.a[DIAG][i]:i+1==j?e.a[OFF][i]:j+1==i?e.a[OFF][j]:0;
  if(!threadIdx.x)e.a[SMAT][j*C+i]=x;
}
__global__ void pad_matrix(Engine e) {
  int k=e.s[K];double norm=0;for(int i=threadIdx.x;i<k;i+=B){double r=0;for(int j=0;j<k;j++)r+=fabs(e.a[SMAT][j*C+i]);norm=fmax(norm,r);}norm=maxblock(norm);
  for(int q=threadIdx.x;q<C*C;q+=B){int i=q%C,j=q/C;if(i>=k||j>=k)e.a[SMAT][q]=(i==j)?2*norm+1:0;}
}
// Cyclic parallel Jacobi: disjoint pairs rotate together; no host polling and
// no library call that can synchronize inside a conditional CUDA graph.
__global__ void jacobi(Engine e,int joint) {
  int t=threadIdx.x,n=joint?18:e.s[K],ld=joint?18:C,m=n+(n%2);
  double* u=e.a[joint?JC:SMAT];double* a=e.a[joint?JWORK:WORK];double* b=a+ld*ld;double* ev=e.a[joint?JE:EV];
  __shared__ double cs[C],sn[C];__shared__ int partner[C],perm[C],done;
  for(int q=t;q<n*n;q+=B){int r=q%n,c=q/n;a[c*ld+r]=u[c*ld+r];u[c*ld+r]=(r==c)?1.:0.;}__syncthreads();
  done=0;__syncthreads();
  for(int sweep=0;sweep<100;sweep++){
    double off=0,scale=0;for(int q=t;q<n*n;q+=B){int r=q%n,c=q/n;if(r!=c)off=fmax(off,fabs(a[c*ld+r]));else scale=fmax(scale,fabs(a[c*ld+r]));}
    off=maxblock(off);scale=maxblock(scale);if(off<=1e-15*fmax(scale,1e-300)){if(!t)done=1;__syncthreads();break;}
    for(int round=0;round<m-1;round++){
      for(int j=t;j<m/2;j+=B){int p=j?(round+j)%(m-1):m-1,q=j?(round-j+m-1)%(m-1):round;
        if(p>=n||q>=n){int i=p<n?p:q;if(i<n){cs[i]=1;sn[i]=0;partner[i]=i;}continue;}
        double z=a[q*ld+p],c=1,s=0;
        if(z!=0){double tau=(a[q*ld+q]-a[p*ld+p])/(2*z);double v=copysign(1.,tau)/(fabs(tau)+hypot(1.,tau));c=1/sqrt(1+v*v);s=v*c;}
        cs[p]=cs[q]=c;sn[p]=-s;sn[q]=s;partner[p]=q;partner[q]=p;
      }__syncthreads();
      for(int q=t;q<n*n;q+=B){int r=q%n,c=q/n;b[c*ld+r]=cs[c]*a[c*ld+r]+sn[c]*a[partner[c]*ld+r];}
      for(int q=t;q<(m/2)*n;q+=B){int j=q/n,r=q%n,p=j?(round+j)%(m-1):m-1,c=j?(round-j+m-1)%(m-1):round;if(p<n&&c<n){double up=u[p*ld+r],uq=u[c*ld+r];u[p*ld+r]=cs[p]*up+sn[p]*uq;u[c*ld+r]=cs[c]*uq+sn[c]*up;}}
      __syncthreads();for(int q=t;q<n*n;q+=B){int r=q%n,c=q/n;a[c*ld+r]=cs[r]*b[c*ld+r]+sn[r]*b[c*ld+partner[r]];}__syncthreads();
    }
  }
  if(!t){if(!done)e.s[ERROR]=joint?3:2;for(int j=0;j<n;j++){ev[j]=a[j*ld+j];perm[j]=j;}for(int i=0;i<n;i++){int p=i;for(int j=i+1;j<n;j++)if(ev[j]<ev[p])p=j;double v=ev[p];ev[p]=ev[i];ev[i]=v;int pi=perm[p];perm[p]=perm[i];perm[i]=pi;}}
  __syncthreads();for(int q=t;q<n*n;q+=B){int r=q%n,c=q/n;b[c*ld+r]=u[perm[c]*ld+r];}__syncthreads();for(int q=t;q<n*n;q+=B){int r=q%n,c=q/n;u[c*ld+r]=b[c*ld+r];}
  if(!t){if(joint)*e.jinfo=done?0:1;else *e.info=done?0:1;}
}

// Same cyclic Jacobi rotations and convergence criterion as jacobi(), spread
// over resident blocks. Grid barriers preserve the original round ordering.
__global__ void jacobi_grid(Engine e) {
  auto grid=cooperative_groups::this_grid();
  int t=threadIdx.x,id=blockIdx.x*B+t,stride=gridDim.x*B,n=e.s[K],m=n+(n%2),ld=C;
  double* u=e.a[SMAT];double* a=e.a[WORK];double* b=a+C*C;double* ev=e.a[EV];
  double* cs=e.a[PART];double* sn=cs+C;double* partner=sn+C;double* perm=partner+C;double* stat=perm+C;
  for(int q=id;q<n*n;q+=stride){int r=q%n,c=q/n;a[c*ld+r]=u[c*ld+r];u[c*ld+r]=(r==c)?1.:0.;}
  if(id==0)stat[2*gridDim.x]=0;grid.sync();
  for(int sweep=0;sweep<100;sweep++){
    double off=0,scale=0;for(int q=id;q<n*n;q+=stride){int r=q%n,c=q/n;if(r!=c)off=fmax(off,fabs(a[c*ld+r]));else scale=fmax(scale,fabs(a[c*ld+r]));}
    off=maxblock(off);scale=maxblock(scale);if(!t){stat[blockIdx.x]=off;stat[gridDim.x+blockIdx.x]=scale;}grid.sync();
    if(id==0){off=scale=0;for(int j=0;j<gridDim.x;j++){off=fmax(off,stat[j]);scale=fmax(scale,stat[gridDim.x+j]);}stat[2*gridDim.x]=(off<=1e-15*fmax(scale,1e-300));}grid.sync();
    if(stat[2*gridDim.x])break;
    for(int round=0;round<m-1;round++){
      for(int j=id;j<m/2;j+=stride){int p=j?(round+j)%(m-1):m-1,q=j?(round-j+m-1)%(m-1):round;
        if(p>=n||q>=n){int i=p<n?p:q;if(i<n){cs[i]=1;sn[i]=0;partner[i]=i;}continue;}
        double z=a[q*ld+p],c=1,v=0;
        if(z!=0){double tau=(a[q*ld+q]-a[p*ld+p])/(2*z);double w=copysign(1.,tau)/(fabs(tau)+hypot(1.,tau));c=1/sqrt(1+w*w);v=w*c;}
        cs[p]=cs[q]=c;sn[p]=-v;sn[q]=v;partner[p]=q;partner[q]=p;
      }grid.sync();
      for(int q=id;q<n*n;q+=stride){int r=q%n,c=q/n;b[c*ld+r]=cs[c]*a[c*ld+r]+sn[c]*a[int(partner[c])*ld+r];}
      for(int q=id;q<(m/2)*n;q+=stride){int j=q/n,r=q%n,p=j?(round+j)%(m-1):m-1,c=j?(round-j+m-1)%(m-1):round;if(p<n&&c<n){double up=u[p*ld+r],uq=u[c*ld+r];u[p*ld+r]=cs[p]*up+sn[p]*uq;u[c*ld+r]=cs[c]*uq+sn[c]*up;}}
      grid.sync();for(int q=id;q<n*n;q+=stride){int r=q%n,c=q/n;a[c*ld+r]=cs[r]*b[c*ld+r]+sn[r]*b[c*ld+int(partner[r])];}grid.sync();
    }
  }
  if(id==0){bool done=stat[2*gridDim.x];if(!done)e.s[ERROR]=2;*e.info=done?0:1;
    for(int j=0;j<n;j++){ev[j]=a[j*ld+j];perm[j]=j;}for(int i=0;i<n;i++){int p=i;for(int j=i+1;j<n;j++)if(ev[j]<ev[p])p=j;double v=ev[p];ev[p]=ev[i];ev[i]=v;double pi=perm[p];perm[p]=perm[i];perm[i]=pi;}}
  grid.sync();for(int q=id;q<n*n;q+=stride){int r=q%n,c=q/n;b[c*ld+r]=u[int(perm[c])*ld+r];}grid.sync();for(int q=id;q<n*n;q+=stride){int r=q%n,c=q/n;u[c*ld+r]=b[c*ld+r];}
}

__global__ void spectral_step(Engine e) {
  __shared__ double a[512],y[512],sh[8];__shared__ int hard,first;
  int t=threadIdx.x,k=e.s[K];double rhsn=0;
  for(int j=t;j<k;j+=B){double x=0;for(int i=0;i<k;i++)x+=e.a[SMAT][j*C+i]*e.a[RHS][i];a[j]=x;rhsn+=e.a[RHS][j]*e.a[RHS][j];}
  rhsn=sqrt(sumblock(rhsn));double mx=0;for(int j=t;j<k;j+=B)mx=fmax(mx,fabs(e.a[EV][j]));mx=maxblock(mx);
  if(!t){sh[0]=fmax(0.,-e.a[EV][0]);sh[1]=2.220446049250313e-16*fmax(1.,mx)*16;first=k;if(*e.info)e.s[ERROR]=2;}__syncthreads();
  double yn=0,sn=0;for(int j=t;j<k;j+=B){double den=e.a[EV][j]+sh[0];bool sing=den<=sh[1];y[j]=sing?0:-a[j]/den;yn+=y[j]*y[j];if(sing){sn+=a[j]*a[j];atomicMin(&first,j);}}
  yn=sumblock(yn);sn=sumblock(sn);if(!t)hard=(sqrt(sn)<=sh[1]*fmax(1.,rhsn)&&sqrt(yn)<=e.a[DVAL][RADIUS]);__syncthreads();
  if(hard){if(!t&&sh[0]>0)y[first]=sqrt(fmax(e.a[DVAL][RADIUS]*e.a[DVAL][RADIUS]-yn,0.));if(!t)e.a[DVAL][LAMBDA]=sh[0];}
  else{
    if(!t){sh[2]=fmax(sh[0]+1.,rhsn/e.a[DVAL][RADIUS]+sh[0]);sh[3]=sh[0];}__syncthreads();
    for(int it=0;it<100;it++){double s=0;for(int j=t;j<k;j+=B){double x=a[j]/(e.a[EV][j]+sh[2]);s+=x*x;}s=sqrt(sumblock(s));if(s<=e.a[DVAL][RADIUS])break;if(!t)sh[2]=sh[0]+2*(sh[2]-sh[0]);__syncthreads();}
    for(int it=0;it<60;it++){double mid=.5*(sh[2]+sh[3]),s=0;for(int j=t;j<k;j+=B){double x=a[j]/(e.a[EV][j]+mid);s+=x*x;}s=sqrt(sumblock(s));if(!t){if(s>e.a[DVAL][RADIUS])sh[3]=mid;else sh[2]=mid;}__syncthreads();}
    for(int j=t;j<k;j+=B)y[j]=-a[j]/(e.a[EV][j]+sh[2]);if(!t)e.a[DVAL][LAMBDA]=sh[2];
  }__syncthreads();for(int i=t;i<k;i+=B){double x=0;for(int j=0;j<k;j++)x+=e.a[SMAT][j*C+i]*y[j];e.a[YC][i]=x;}
}
__global__ void combine(Engine e) {int i=blockIdx.x*B+threadIdx.x;if(i>=e.n)return;double y=0,a=0;for(int j=0;j<e.s[K];j++){y+=e.a[YC][j]*e.a[Q][j*e.n+i];a+=e.a[YC][j]*e.a[AQ][j*e.n+i];}e.a[Y][i]=y;e.a[AY][i]=a;}
__global__ void unmap(Engine e) {
  int cell=blockIdx.x*B+threadIdx.x;if(cell>=e.n/3)return;int i=3*cell;const double* l=e.a[L]+9*cell;
  for(int which=0;which<2;which++){const double* x=e.a[which?Y:AY]+i;double* o=e.a[which?PP:HP]+i;double c=x[2]/l[8],b=(x[1]-l[7]*c)/l[4],a=(x[0]-l[3]*b-l[6]*c)/l[0];o[0]=a;o[1]=b;o[2]=c;}
  for(int a=0;a<3;a++)e.a[P][i+a]=mapval(e,e.a[Y],i+a,false);
}
__global__ void unmap_joint(Engine e) {
  if(threadIdx.x||!e.s[GROUP])return;
  for(int which=0;which<2;which++){double x[18];for(int j=17;j>=0;j--){double v=e.a[which?Y:AY][3*e.ids[j/3]+j%3];for(int q=j+1;q<18;q++)v-=e.a[LJ][q*18+j]*x[q];x[j]=v/e.a[LJ][j*18+j];}for(int j=0;j<18;j++)e.a[which?PP:HP][3*e.ids[j/3]+j%3]=x[j];}
}
__global__ void residual(Engine e,int fresh) {int i=blockIdx.x*B+threadIdx.x;if(i<e.n){if(fresh)e.a[HP][i]=e.a[INPUTH][i];e.a[RES][i]=e.a[G][i]+e.a[HP][i]+e.a[DVAL][LAMBDA]*e.a[PP][i];}}
__global__ void metrics(Engine e,int fresh) {
  double rn=0,proj=0,pmp=0;for(int i=threadIdx.x;i<e.n;i+=B){rn+=e.a[RES][i]*e.a[RES][i];pmp+=e.a[Y][i]*e.a[Y][i];}rn=sqrt(sumblock(rn));pmp=sumblock(pmp);
  if(!fresh){for(int j=threadIdx.x;j<e.s[K];j+=B)proj+=e.a[CO][j]*e.a[CO][j];proj=sqrt(sumblock(proj))/fmax(e.a[DVAL][BN],1e-300);}
  if(!threadIdx.x){e.a[DVAL][RN]=rn;e.a[DVAL][PMP]=pmp;if(!isfinite(rn))e.s[ERROR]=5;if(fresh){e.s[PRODUCTS]++;e.s[CHECKS]++;e.s[CERT]=(rn<=e.a[DVAL][STOP]);}
    else {e.a[DVAL][PROJECTION]=proj;e.s[REDO]=(proj>1e-7&&!e.s[DENSE]);e.s[VERIFY]=!e.s[REDO]&&(rn<=e.a[DVAL][STOP]||e.s[K]>=e.s[CAP]);}}
}
__device__ void record(Engine e) {
  int j=e.s[RECORDS]++;if(j>=2)return;double* r=e.records+j*9;r[0]=e.s[K];r[1]=e.s[GROUP];r[2]=e.s[DENSE];r[3]=e.s[CERT];r[4]=e.a[DVAL][RN]/fmax(e.a[DVAL][GN],1e-300);r[5]=e.a[DVAL][LAMBDA];r[6]=e.a[DVAL][PROJECTION];r[7]=e.s[PRODUCTS];r[8]=sqrt(e.a[DVAL][PMP]);
}
__global__ void decide(Engine e) {
  if(threadIdx.x)return;e.s[JOINT]=0;
  if(e.s[ERROR]){e.s[RUN]=0;return;}
  if(!e.s[CHECK]){e.s[ADV]=1;return;}
  if(e.s[REDO]){e.s[DENSE]=1;e.s[ADV]=0;return;}
  if(e.s[VERIFY]&&(e.s[CERT]||e.s[K]>=e.s[CAP])){
    record(e);
    if(!e.s[CERT]&&e.s[K]>=e.s[CAP]&&e.a[DVAL][LAMBDA]==0&&!e.s[GROUP]&&e.s[ALLOW]){e.s[JOINT]=1;e.s[ALLOW]=0;}
    else e.s[RUN]=0;return;
  }
  if(e.s[VERIFY]&&!e.s[DENSE]){e.s[DENSE]=1;e.s[ADV]=0;return;}
  e.s[ADV]=1;e.s[CHECK]=0;
}
__global__ void pick_joint(Engine e) {
  __shared__ double val[B];__shared__ int idx[B];int t=threadIdx.x;
  for(int j=0;j<6;j++){double best=-1;int bi=0;for(int i=t;i<e.n/3;i+=B){bool used=false;for(int z=0;z<j;z++)if(e.ids[z]==i)used=true;double x=0;for(int q=0;q<3;q++){double v=e.a[G][3*i+q]+e.a[HP][3*i+q];x+=v*v;}if(!used&&(x>best||(x==best&&i<bi))){best=x;bi=i;}}val[t]=best;idx[t]=bi;__syncthreads();for(int d=B/2;d;d/=2){if(t<d&&(val[t+d]>val[t]||(val[t+d]==val[t]&&idx[t+d]<idx[t]))){val[t]=val[t+d];idx[t]=idx[t+d];}__syncthreads();}if(!t)e.ids[j]=idx[0];__syncthreads();}
}
__global__ void joint_column(Engine e,int col) {int j=threadIdx.x;if(j<18)e.a[JC][col*18+j]=e.a[INPUTH][3*e.ids[j/3]+j%3];}
__global__ void joint_sym(Engine e) {int j=threadIdx.x;if(j<18*18){int r=j%18,c=j/18;if(r<c){double x=.5*(e.a[JC][j]+e.a[JC][r*18+c]);e.a[JC][j]=e.a[JC][r*18+c]=x;}}}
__global__ void joint_factor(Engine e) {
  __shared__ double a[324];int t=threadIdx.x;double floor=1e300;for(int j=0;j<6;j++)floor=fmin(floor,e.a[CM][e.ids[j]]);
  for(int i=t;i<324;i+=B){int r=i/18,c=i%18;double x=0;for(int j=0;j<18;j++)x+=e.a[JC][j*18+r]*e.a[JC][j*18+c]/fmax(e.a[JE][j],floor);a[i]=x;}__syncthreads();
  if(!t){if(*e.jinfo)e.s[ERROR]=3;for(int i=0;i<18;i++)for(int j=0;j<=i;j++){double x=a[i*18+j];for(int z=0;z<j;z++)x-=a[i*18+z]*a[j*18+z];a[i*18+j]=i==j?sqrt(x):x/a[j*18+j];if(!isfinite(a[i*18+j]))e.s[ERROR]=4;}}
  __syncthreads();for(int i=t;i<324;i+=B)e.a[LJ][i]=(i%18<=i/18)?a[i]:0;
  if(!t){e.s[K]=0;e.s[DENSE]=0;e.s[GROUP]=1;e.s[ADV]=1;e.s[CHECK]=0;e.s[VERIFY]=0;e.s[REDO]=0;e.s[CERT]=0;e.s[PRODUCTS]+=18;e.a[DVAL][LAMBDA]=0;}
}
__global__ void finish(Engine e) {
  double pmp=0,gp=0,php=0,rr=0;for(int i=threadIdx.x;i<e.n;i+=B){pmp+=e.a[Y][i]*e.a[Y][i];gp+=e.a[G][i]*e.a[P][i];php+=e.a[P][i]*e.a[HP][i];double v=e.a[G][i]+e.a[HP][i];rr+=v*v;}
  pmp=sumblock(pmp);gp=sumblock(gp);php=sumblock(php);rr=sumblock(rr);if(!threadIdx.x){e.a[DVAL][PMP]=pmp;e.a[DVAL][GP]=gp;e.a[DVAL][PHP]=php;e.a[DVAL][RR]=rr;}
}
extern "C" {
void* create_engine(void** ptrs,int* state,int* ids,int* info,int* jinfo,double* records,int n) {
  auto* e=new Engine{};for(int j=0;j<NB;j++)e->a[j]=(double*)ptrs[j];e->s=state;e->ids=ids;e->info=info;e->jinfo=jinfo;e->records=records;e->n=n;
  cudaMalloc(&e->a[WORK],2*C*C*sizeof(double));cudaMalloc(&e->a[JWORK],2*18*18*sizeof(double));
  int device=0,active=0;cudaDeviceProp prop{};cudaGetDevice(&device);cudaGetDeviceProperties(&prop,device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,jacobi_grid,B,0);
  e->jacobi_blocks=std::min(32,active*prop.multiProcessorCount);
  cudaMalloc(&e->a[PART],std::max(4*C+2*32+1,(n+B-1)/B)*sizeof(double));return e;
}
int phase(void* handle,int op,uintptr_t stream,double radius,double stop,int cap,int extra) {
  auto e=*(Engine*)handle;cudaStream_t st=(cudaStream_t)stream;int grid=(e.n+B-1)/B;
  switch(op){
    case 0:reset<<<1,1,0,st>>>(e,radius,stop,cap,extra);break;
    case 1:init_context<<<(e.n/3+B-1)/B,B,0,st>>>(e);make_b<<<grid,B,0,st>>>(e);norm_b<<<1,B,0,st>>>(e);q0<<<grid,B,0,st>>>(e);break;
    case 2:make_direction<<<grid,B,0,st>>>(e,0,0);break;
    case 3:map_product<<<grid,B,0,st>>>(e);dot_parts<<<dim3(C,4),B,0,st>>>(e,0);dots<<<(C+B-1)/B,B,0,st>>>(e,0);subtract<<<grid,B,0,st>>>(e,1);dot_parts<<<dim3(C,4),B,0,st>>>(e,0);dots<<<(C+B-1)/B,B,0,st>>>(e,0);subtract<<<grid,B,0,st>>>(e,0);norm_v<<<1,B,0,st>>>(e);next_q<<<grid,B,0,st>>>(e);advance_done<<<1,1,0,st>>>(e);break;
    case 4:make_b<<<grid,B,0,st>>>(e);dot_parts<<<dim3(C,4),B,0,st>>>(e,1);dots<<<(C+B-1)/B,B,0,st>>>(e,1);reduced_tri<<<1,B,0,st>>>(e);break;
    case 5:{dense_matrix<<<C*C,B,0,st>>>(e);void* args[]={&e};auto err=cudaLaunchCooperativeKernel((void*)jacobi_grid,dim3(e.jacobi_blocks),dim3(B),args,0,st);if(err)return int(err);spectral_step<<<1,B,0,st>>>(e);break;}
    case 6:combine<<<grid,B,0,st>>>(e);unmap<<<(e.n/3+B-1)/B,B,0,st>>>(e);unmap_joint<<<1,1,0,st>>>(e);residual<<<grid,B,0,st>>>(e,0);map_residual<<<grid,B,0,st>>>(e);dot_parts<<<dim3(C,4),B,0,st>>>(e,2);dots<<<(C+B-1)/B,B,0,st>>>(e,2);metrics<<<1,B,0,st>>>(e,0);break;
    case 7:make_direction<<<grid,B,0,st>>>(e,1,0);break;
    case 8:residual<<<grid,B,0,st>>>(e,1);metrics<<<1,B,0,st>>>(e,1);break;
    case 9:decide<<<1,1,0,st>>>(e);break;
    case 10:pick_joint<<<1,B,0,st>>>(e);break;
    case 11:make_direction<<<grid,B,0,st>>>(e,2,extra);break;
    case 12:joint_column<<<1,32,0,st>>>(e,extra);break;
    case 13:joint_sym<<<1,512,0,st>>>(e);jacobi<<<1,B,0,st>>>(e,1);joint_factor<<<1,B,0,st>>>(e);make_b<<<grid,B,0,st>>>(e);norm_b<<<1,B,0,st>>>(e);q0<<<grid,B,0,st>>>(e);break;
    case 14:finish<<<1,B,0,st>>>(e);break;
  }
  return (int)cudaPeekAtLastError();
}
void destroy_engine(void* handle){auto* e=(Engine*)handle;cudaFree(e->a[WORK]);cudaFree(e->a[JWORK]);cudaFree(e->a[PART]);delete e;}
}
