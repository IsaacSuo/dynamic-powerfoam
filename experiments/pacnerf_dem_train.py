"""Train the shared EnergyFoam parameters on a PAC-NeRF ground-truth trajectory (time split).

Seven shared log multipliers (kn, ks, kb, kt bonds; Neo-Hookean lam, mu; dashpot strength),
starting from the GT (E, nu) split of pacnerf_dem_forward.build (theta = 0). Loss: normalized
MSE of body centres vs the particle-mean targets over frames 1..train_frames; frames after that
are held out (prediction of unseen motion). Adam (contact makes the loss piecewise smooth).
The stable substep count is recomputed from the current parameters every few iterations.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pacnerf_dem_forward import build, cell_targets


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene', default='elastic/0')
    a.add_argument('--geometry', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/geometry.npz'))
    a.add_argument('--frames', type=int, default=15)
    a.add_argument('--train-frames', type=int, default=10)
    a.add_argument('--alpha', type=float, default=.25)
    a.add_argument('--damping', type=float, default=.3)
    a.add_argument('--iterations', type=int, default=40)
    a.add_argument('--lr', type=float, default=.05)
    a.add_argument('--out', type=Path, default=Path('output/pacnerf_dem/elastic_0_f0/train_v1'))
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    gt = json.loads(Path('data/pacnerf/gt/elastic.json').read_text())[args.scene.split('/')[1]]
    geometry = dict(np.load(args.geometry))
    cells = dict(np.load(args.geometry.parent/'cells.npz'))
    targets, has, _ = cell_targets(cells, geometry['body_of_cell'], geometry['points'],
                                   Path('data/pacnerf/simulation_data')/args.scene, args.frames)
    model = build(geometry, gt['E'], gt['nu'], 'cuda', 'noslip', args.alpha, args.damping, torch.float32)
    dtype = model.dtype
    tg = torch.tensor(targets, dtype=dtype, device='cuda')
    hs = torch.tensor(has, device='cuda')
    x0, q0, _, w0 = model.initial_state()
    v0 = torch.tensor(gt['v'], dtype=dtype, device='cuda').expand(model.n, 3).clone()
    shared = torch.zeros(7, dtype=dtype, device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([shared], lr=args.lr)

    def rollout(frames, substeps, grad=True):
        theta = shared.expand(model.n, 7) if grad else shared.detach().expand(model.n, 7)
        pos, _ = model.rollout(theta, (x0, q0, v0, w0), 1/24/substeps, substeps, frames, compile=True)
        return pos-pos[:1]+tg[:1]  # compare motion (frame-0 centre vs particle-mean gap removed)

    def nrmse(pos, lo, hi):
        p, t = pos[lo:hi+1, hs], tg[lo:hi+1, hs]
        return float(torch.sqrt(((p-t)**2).mean()/((t-tg[:1, hs])**2).mean()))

    def evaluate(tag, substeps):
        with torch.no_grad():
            pos = rollout(args.frames, substeps, grad=False)
        res = dict(train=nrmse(pos, 1, args.train_frames), test=nrmse(pos, args.train_frames+1, args.frames),
                   all=nrmse(pos, 1, args.frames), centroid_y=pos[:, hs, 1].mean(1).tolist(),
                   gt_centroid_y=tg[:, hs, 1].mean(1).tolist(), parameters=shared.detach().tolist(), substeps=substeps)
        np.savez_compressed(args.out/f'positions_{tag}.npz', positions=pos.cpu().numpy(), targets=targets, has=has)
        return res

    substeps = int(np.ceil(1/24/model.stable_dt(shared.detach().expand(model.n, 7))))
    report = dict(config={k: str(v) for k, v in vars(args).items()}, gt=gt, initial=evaluate('initial', substeps))
    print('initial', json.dumps({k: report['initial'][k] for k in ('train', 'test', 'all', 'substeps')}), flush=True)
    history, best = [], (np.inf, None)
    start = time.perf_counter()
    for it in range(args.iterations):
        if it and it % 5 == 0:
            substeps = int(np.ceil(1/24/model.stable_dt(shared.detach().expand(model.n, 7))))
        optimizer.zero_grad()
        pos = rollout(args.train_frames, substeps)
        p, t = pos[1:, hs], tg[1:args.train_frames+1, hs]
        loss = ((p-t)**2).mean()/((t-tg[:1, hs])**2).mean()
        loss.backward()
        value = float(loss.detach())
        if not np.isfinite(value) or not torch.isfinite(shared.grad).all():
            print('non-finite at iteration', it, flush=True); break
        if value < best[0]:
            best = (value, shared.detach().clone())
        optimizer.step()
        with torch.no_grad():
            shared.clamp_(-3, 3)
        history.append(dict(iteration=it, train_nrmse=float(np.sqrt(value)), parameters=shared.detach().tolist(),
                            substeps=substeps, seconds=time.perf_counter()-start))
        print(json.dumps(history[-1]), flush=True)
        (args.out/'history.json').write_text(json.dumps(history, indent=1))
    with torch.no_grad():
        shared.copy_(best[1])
    substeps = int(np.ceil(1/24/model.stable_dt(shared.detach().expand(model.n, 7))))
    report['best'] = evaluate('best', substeps)
    report['seconds'] = time.perf_counter()-start
    names = ('kn', 'ks', 'kb', 'kt', 'lam', 'mu', 'damping')
    report['multipliers'] = dict(zip(names, np.exp(best[1].cpu().numpy()).tolist()))
    (args.out/'report.json').write_text(json.dumps(report, indent=1))
    print('best', json.dumps({k: report['best'][k] for k in ('train', 'test', 'all', 'substeps')}),
          json.dumps(report['multipliers']), flush=True)


if __name__ == '__main__':
    main()
