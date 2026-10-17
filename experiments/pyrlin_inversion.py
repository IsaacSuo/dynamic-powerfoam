"""Which pyramids invert in pyrlin (element NH + linear pyramid penalty), and why?

pyrlin beta=8, GT E/nu, noslip ground. At frames 4.5-8 (quarter frames): inverted pyramids
(J_s < 0) vs all pyramids by size (|s| / |E|), height above ground, whether they touch a vertex
on the ground, and the penalty energy they hold vs the total strain energy.
"""
import json
import numpy as np
import torch

from experiments.vem_dynamics import VEMSolid

gt = json.load(open('data/pacnerf/gt/elastic.json'))['0']
m = VEMSolid('output/pacnerf_vem/elastic_0_f0/vem_cap1_bodies_pyr.npz', 'cuda', torch.float32, stab='pyrlin')
m.ground_mode = 'noslip'
p = tuple(torch.tensor(t, device='cuda') for t in (gt['E'], gt['nu'], 8.))
sub = int(np.ceil(m.max_frequency(p)/24)); dt = torch.tensor(1/24/sub, device='cuda')
mu = gt['E']/(2*(1+gt['nu']))
rel = (m.s_vol/m.volume[m.s_elem]).cpu().numpy()
edges = [0, 1e-4, 1e-3, 1e-2, .05, 1]
all_hist = np.histogram(rel, edges)[0]
print(f'{sub} substeps/frame; pyramids {m.ns}; size share |s|/|E| bins {edges}: all {all_hist.tolist()}', flush=True)
x, v = m.X.clone(), torch.tensor(gt['v'], device='cuda').expand(m.nv, 3).clone()
step = torch.compile(m.step, dynamic=False)
with torch.no_grad():
    for n in range(1, 8*sub+1):
        x, v = step(x, v, p, dt)
        f = n/sub
        if n % (sub//4) == 0 and f >= 4.49:
            xbar, F, _ = m.kinematics(x)
            Fs = m.sub_F(x, xbar)
            J = torch.linalg.det(Fs).cpu().numpy()
            D = Fs-F[m.s_elem]
            pen = (.5*8*mu*m.s_vol*(D*D).sum((1, 2))).cpu().numpy()
            E_total = float(m.energy(x, p))
            pts = m._points(x)
            hp = (torch.zeros(m.ns, device='cuda').index_add_(0, m.p_id, pts[m.p_pt, 1])/torch.bincount(m.p_id, minlength=m.ns)).cpu().numpy()
            on_ground = (x[:, 1] <= 1e-6)
            touch = torch.zeros(m.ns, dtype=torch.bool, device='cuda')
            vrow = m.p_pt < m.nv
            touch.index_fill_(0, m.p_id[vrow][on_ground[m.p_pt[vrow]]], True)
            touch = touch.cpu().numpy()
            inv = J < 0
            if inv.sum() == 0:
                print(f'f{f:.2f}: no inverted pyramids; penalty energy {pen.sum():.2f} J of strain {E_total:.2f} J', flush=True); continue
            h_inv = np.histogram(rel[inv], edges)[0]
            print(f'f{f:.2f}: inverted {inv.sum()} ({inv.mean()*100:.2f}%) | by size bin {h_inv.tolist()} -> share of bin inverted '
                  f'{np.round(h_inv/np.maximum(all_hist, 1)*100, 2).tolist()}% | height median inv {np.median(hp[inv])*1e3:.0f} mm (all {np.median(hp)*1e3:.0f}), '
                  f'below 2 cm {np.mean(hp[inv] < .02)*100:.0f}% | touching ground {np.mean(touch[inv])*100:.0f}% (all {np.mean(touch)*100:.2f}%) | '
                  f'penalty energy: inverted {pen[inv].sum():.2f} J, all pyramids {pen.sum():.2f} J, total strain {E_total:.2f} J | J min {J.min():.1f}', flush=True)
