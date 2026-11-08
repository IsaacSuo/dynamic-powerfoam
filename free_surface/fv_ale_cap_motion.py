"""Integrated regularized cap map shared by sphere and moving-plane controls."""
import torch


def regularized_map(b,epsilon):
    cap,N,Q,r=b['cap'],b['N'],b['Q'],b['r'];i,j,A=b['i'],b['j'],b['A']
    area=cap.clamp(min=0)+b['floor']+torch.zeros_like(cap).index_add(0,i,A).index_add(0,j,A)
    if 'plane_i' in b:area.index_add_(0,b['plane_i'],b['plane_A'])
    norm2=N.square().sum(1)+cap.square()
    denominator=(norm2+(epsilon*area).square()).clamp(min=1e-300)
    Q0=cap[:,None,None]*torch.eye(3,dtype=r.dtype,device=r.device)/3
    B=(r/denominator)[:,None,None]*(Q-Q0)
    effective=Q0+(norm2/denominator)[:,None,None]*(Q-Q0)
    return B,effective
