"""Graph-capturable double-precision symmetric 3x3 Jacobi eigensolver."""
import torch
import warp as wp
wp.init()


@wp.kernel
def jacobi3(A: wp.array(dtype=wp.mat33d), values: wp.array(dtype=wp.vec3d), vectors: wp.array(dtype=wp.mat33d)):
    tid = wp.tid()
    M = A[tid]
    R = wp.identity(n=3, dtype=wp.float64)
    for sweep in range(10):
        for pair in range(3):
            p = int(0); q = int(1)
            if pair == 1:
                p = 0; q = 2
            elif pair == 2:
                p = 1; q = 2
            off = M[p,q]
            threshold = wp.float64(1e-16) * (wp.abs(M[p,p]) + wp.abs(M[q,q])) + wp.float64(1e-30)
            if wp.abs(off) > threshold:
                tau = (M[q,q] - M[p,p]) / (wp.float64(2.) * off)
                sign = wp.float64(1.)
                if tau < wp.float64(0.): sign = wp.float64(-1.)
                t = sign / (wp.abs(tau) + wp.sqrt(wp.float64(1.) + tau*tau))
                c = wp.float64(1.) / wp.sqrt(wp.float64(1.) + t*t)
                s = t*c
                pp = M[p,p]; qq = M[q,q]
                M[p,p] = pp - t*off; M[q,q] = qq + t*off
                M[p,q] = wp.float64(0.); M[q,p] = wp.float64(0.)
                for k in range(3):
                    if k != p and k != q:
                        kp = M[k,p]; kq = M[k,q]
                        M[k,p] = c*kp - s*kq; M[p,k] = M[k,p]
                        M[k,q] = s*kp + c*kq; M[q,k] = M[k,q]
                    rp = R[k,p]; rq = R[k,q]
                    R[k,p] = c*rp - s*rq; R[k,q] = s*rp + c*rq
    d = wp.vec3d(M[0,0],M[1,1],M[2,2])
    for p in range(2):
        for q in range(p+1,3):
            if d[q] < d[p]:
                old = d[p]; d[p] = d[q]; d[q] = old
                for k in range(3):
                    old_r = R[k,p]; R[k,p] = R[k,q]; R[k,q] = old_r
    values[tid] = d; vectors[tid] = R


def eigh3(A):
    if not A.is_cuda or A.dtype != torch.float64:
        return torch.linalg.eigh(A)
    shape = A.shape[:-2]
    flat = A.contiguous().reshape(-1,3,3)
    values = torch.empty((len(flat),3), device=A.device, dtype=A.dtype)
    vectors = torch.empty_like(flat)
    wp.launch(jacobi3,dim=len(flat),inputs=[wp.from_torch(flat,dtype=wp.mat33d)],
              outputs=[wp.from_torch(values,dtype=wp.vec3d),wp.from_torch(vectors,dtype=wp.mat33d)],
              device=str(A.device),stream=wp.stream_from_torch(torch.cuda.current_stream(A.device)))
    return values.reshape(shape+(3,)), vectors.reshape(shape+(3,3))
