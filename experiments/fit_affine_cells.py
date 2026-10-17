"""Fit original bounded-power parameters to known affine material regions.

Run: python -m experiments.fit_affine_cells
Uses the existing om environment. No renderer or MPM code is modified.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.affine_probe import GPURenderer, membership, save_image, scene, texture, transforms


def power(query, centers, radii):
    return (query.square().sum(1)[:, None] + centers.square().sum(1)[None]
            - 2 * query @ centers.T - radii.square()[None])


def geometry_loss(qin, labels, qout, centers, radii, length2):
    inside = power(qin, centers, radii) / length2
    own = inside.gather(1, labels[:, None]).squeeze(1)
    rivals = inside.scatter(1, labels[:, None], float("inf")).amin(1)
    # A material sample must beat both every other cell and background power 0.
    material = torch.relu(own - torch.minimum(rivals, torch.zeros_like(rivals))).square().mean()
    outside = torch.relu(-power(qout, centers, radii).amin(1) / length2).square().mean()
    return material, outside


def sample_sets(centers, radii, seed, train_count, test_count):
    rng = np.random.default_rng(seed)
    low = (centers - radii[:, None]).min(0) - 0.05
    high = (centers + radii[:, None]).max(0) + 0.05
    uniform = rng.uniform(low, high, (train_count // 2, 3))
    candidates = rng.uniform(low, high, (train_count * 3, 3))
    margins = []
    for start in range(0, len(candidates), 4096):
        q = candidates[start:start+4096]
        phi = ((q[:, None] - centers)**2).sum(-1) - radii**2
        # Background competes as one additional site with constant power zero.
        ordered = np.partition(np.column_stack((phi, np.zeros(len(q)))), 1, axis=1)
        margins.append(ordered[:, 1] - ordered[:, 0])
    margins = np.concatenate(margins)
    boundary_pool = np.argsort(margins)[:len(candidates)//4]
    selected = rng.choice(boundary_pool, train_count - len(uniform), replace=False)
    train = np.concatenate((uniform, candidates[selected]))
    test = rng.uniform(low, high, (test_count, 3))
    return train, membership(train, centers, radii), test, membership(test, centers, radii), low, high


def evaluate(query, labels, centers, radii):
    actual = membership(query, centers, radii)
    inside, outside = labels >= 0, labels < 0
    covered = inside & (actual >= 0)
    switched = covered & (actual != labels)
    return dict(identity_loss_pct=100*float((actual[inside] != labels[inside]).mean()),
                support_miss_pct=100*float((actual[inside] < 0).mean()),
                owner_switch_pct_covered=100*float(switched.sum()/covered.sum()),
                exterior_false_occupied_pct=100*float((actual[outside] >= 0).mean()) if outside.any() else 0.)


def fit(query, labels, p_phys, r_prior, h, limit, args, device):
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    p, r, spacing = tensor(p_phys), tensor(r_prior), tensor(h)
    qin, qout = tensor(query[labels >= 0]), tensor(query[labels < 0])
    owners = torch.as_tensor(labels[labels >= 0], dtype=torch.long, device=device)
    shift = torch.nn.Parameter(torch.zeros((len(p), 2), device=device))
    log_radius = torch.nn.Parameter(torch.zeros(len(p), device=device))
    optimizer = torch.optim.Adam([log_radius] + ([shift] if limit else []), lr=args.lr)
    length2 = float(np.median(h)**2)
    history, best = [], None
    start = time.perf_counter()
    for step in range(args.steps + 1):
        offset = torch.cat((shift * spacing[:, None], torch.zeros_like(spacing[:, None])), 1)
        centers, radii = p + offset, r * log_radius.exp()
        material, outside = geometry_loss(qin, owners, qout, centers, radii, length2)
        objective = (material + outside + args.position_weight * shift.square().sum(1).mean()
                     + args.radius_weight * log_radius.square().mean())
        value = objective.item()
        if not np.isfinite(value):
            raise RuntimeError("Non-finite optimization objective")
        if best is None or value < best[0]:
            best = (value, centers.detach().clone(), radii.detach().clone(), step)
        if step % 100 == 0 or step == args.steps:
            history.append(dict(step=step, objective=value, material=material.item(), exterior=outside.item()))
        if step == args.steps:
            break
        optimizer.zero_grad()
        objective.backward()
        optimizer.param_groups[0]["lr"] = args.lr * (0.03 + 0.97 * (1 + np.cos(np.pi*step/args.steps))/2)
        optimizer.step()
        with torch.no_grad():
            if limit:
                shift.mul_(torch.clamp(limit / shift.norm(dim=1, keepdim=True).clamp_min(1e-12), max=1.))
            log_radius.clamp_(-np.log(2.), np.log(2.))
    torch.cuda.synchronize()
    centers, radii = best[1].cpu().numpy().astype(float), best[2].cpu().numpy().astype(float)
    relative_shift = np.linalg.norm(centers-p_phys, axis=1)/h
    assert relative_shift.max() <= limit + 2e-6
    assert np.all((radii/r_prior >= 0.5-1e-6) & (radii/r_prior <= 2+1e-6))
    return centers, radii, history, dict(best_training_objective=best[0], best_step=best[3],
                                       optimization_seconds=time.perf_counter()-start)


def check_gradients():
    # Away from ties, check the actual loss derivatives used by the optimizer.
    dtype = torch.float64
    p = torch.tensor([[0.,0.,0.], [0.7,0.3,0.]], dtype=dtype, requires_grad=True)
    r = torch.tensor([0.4,0.35], dtype=dtype, requires_grad=True)
    qi = torch.tensor([[0.52,0.04,0.1], [1.2,0.4,0.]], dtype=dtype)
    qo = torch.tensor([[0.1,0.1,0.05], [0.65,0.25,0.]], dtype=dtype)
    label = torch.tensor([0,1])
    assert torch.autograd.gradcheck(lambda a,b: sum(geometry_loss(qi,label,qo,a,b,0.04)), (p,r))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/affine_fit"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-count", type=int, default=16384)
    parser.add_argument("--test-count", type=int, default=65536)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--position-weight", type=float, default=0.01)
    parser.add_argument("--radius-weight", type=float, default=0.01)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.train_count < 128 or args.test_count < 128 or args.resolution < 32:
        parser.error("Invalid steps, sample count, or resolution")
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Use the CUDA-enabled om environment")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    check_gradients()
    print("Analytic loss gradient check passed", flush=True)
    p0, r0, sites0, colors = scene(args.seed, False)
    distance = np.linalg.norm(p0[:,None]-p0[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    h = np.sort(distance, axis=1)[:, :4].mean(1)
    train, ytrain, test, ytest, low, high = sample_sets(p0,r0,args.seed+100,args.train_count,args.test_count)
    # Float64 global affine oracle must retain every training/test label exactly.
    cases = {k:v for k,v in transforms().items() if k in ("stretch_J1", "shear_J1")}
    x,y = np.meshgrid(np.linspace(-.95,.95,args.resolution),np.linspace(.95,-.95,args.resolution))
    surface = np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    ysurface = membership(surface,p0,r0)
    reference = texture(surface,ysurface,sites0,colors,r0)
    methods = [("baseline",None),("radius_only",0.),("shift_05",.05),("shift_10",.10),("shift_20",.20)]
    gpu = None if args.no_render else GPURenderer(args.resolution)
    ownership_fig, ownership_axes = plt.subplots(2,5,figsize=(17,7),squeeze=False)
    gpu_fig, gpu_axes = plt.subplots(2,6,figsize=(19,7),squeeze=False) if gpu else (None,None)
    metrics, histories = [], {}
    for row,(name,(f,t)) in enumerate(cases.items()):
        p_phys, r_prior = p0@f.T+t, r0*np.cbrt(np.linalg.det(f))
        qtrain,qtest,qsurface = train@f.T+t,test@f.T+t,surface@f.T+t
        np.testing.assert_array_equal(membership((qtest-t)@np.linalg.inv(f).T,p0,r0),ytest)
        # Check a transform representable by the original parameterization.
        np.testing.assert_array_equal(membership(test*1.6,p0*1.6,r0*1.6),ytest)
        if gpu:
            wx,wy=np.meshgrid(np.linspace(-3,3,args.resolution),np.linspace(3,-3,args.resolution))
            canonical=(np.column_stack((wx.ravel(),wy.ravel(),np.zeros(wx.size)))-t)@np.linalg.inv(f).T
            expected=texture(canonical,membership(canonical,p0,r0),sites0,colors,r0).reshape(args.resolution,args.resolution,3)
            gpu_axes[row,0].imshow(expected); gpu_axes[row,0].set_title(f"{name}\nAdvected surface reference")
        for col,(method,limit) in enumerate(methods):
            print(f"Running {name}/{method}",flush=True)
            if limit is None:
                p,r,history,info=p_phys.copy(),r_prior.copy(),[],{}
            else:
                p,r,history,info=fit(qtrain,ytrain,p_phys,r_prior,h,limit,args,"cuda:0")
            row_metrics=dict(case=name,method=method,shift_limit=limit or 0.,**info)
            for prefix,q,labels in [("test3d",qtest,ytest),("train3d",qtrain,ytrain),("surface",qsurface,ysurface)]:
                row_metrics.update({f"{prefix}_{k}":v for k,v in evaluate(q,labels,p,r).items()})
            delta=p-p_phys
            row_metrics.update(mean_shift_over_h=float((np.linalg.norm(delta,axis=1)/h).mean()),
                max_shift_over_h=float((np.linalg.norm(delta,axis=1)/h).max()),
                min_radius_ratio=float((r/r_prior).min()),max_radius_ratio=float((r/r_prior).max()))
            dist=np.linalg.norm(p[:,None]-p[None],axis=-1)
            edges=dist < r[:,None]+r[None]; np.fill_diagonal(edges,False)
            row_metrics["mean_degree"]=float(edges.sum(1).mean())
            # Explicit geometric error of material centers vs renderer proxies.
            owner=membership(qsurface,p,r)
            shifted_sites=sites0@f.T+t+delta[:,None,:]
            image=texture(qsurface,owner,shifted_sites,colors,r)
            row_metrics["surface_texture_mae"]=float(np.abs(image-reference)[ysurface>=0].mean())
            # Geometry-only diagnostic: original material chart without center correction.
            canonical_texture=texture(surface,owner,sites0,colors,r0)
            row_metrics["surface_ownership_texture_mae"]=float(np.abs(canonical_texture-reference)[ysurface>=0].mean())
            map_rgb=np.full((len(surface),3),.92)
            map_rgb[(owner>=0)&(owner!=ysurface)]=[1.,.38,.08]
            map_rgb[owner<0]=[.12,.22,.75]
            ownership_axes[row,col].imshow(map_rgb.reshape(args.resolution,args.resolution,3))
            ownership_axes[row,col].set_title(f"{name}/{method}\nidentity loss {row_metrics['surface_identity_loss_pct']:.1f}%")
            if gpu:
                rgb,alpha=gpu.render(p,r,shifted_sites,colors,f)
                save_image(args.out/f"{name}_{method}_gpu.png",rgb)
                gpu_axes[row,col+1].imshow(rgb); gpu_axes[row,col+1].set_title(f"{method}\ntexture MAE {row_metrics['surface_texture_mae']:.3f}")
            np.savez_compressed(args.out/f"{name}_{method}.npz",centers=p,radii=r,material_centers=p_phys,spacing=h)
            metrics.append(row_metrics);histories[f"{name}/{method}"]=history
            print(f"  3D identity loss={row_metrics['test3d_identity_loss_pct']:.2f}%, "
                  f"surface={row_metrics['surface_identity_loss_pct']:.2f}%, "
                  f"outside={row_metrics['test3d_exterior_false_occupied_pct']:.2f}%, "
                  f"shift max={row_metrics['max_shift_over_h']:.3f}h",flush=True)
            # Save intermediate records so a long run remains inspectable.
            (args.out/"metrics_partial.json").write_text(json.dumps(metrics,indent=2)+"\n")
    for fig,axes,title,filename in [(ownership_fig,ownership_axes,"Held-out surface material ownership: orange switched; blue unsupported","ownership_comparison.png"),
                                   (gpu_fig,gpu_axes,"Frozen appearance: original Power Foam GPU rasterizer","gpu_comparison.png")]:
        if fig is None: continue
        for ax in axes.ravel(): ax.axis("off")
        fig.suptitle(title);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(args.out/filename,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,name in zip(axes,cases):
        for method,_ in methods[1:]:
            history=histories[f"{name}/{method}"]
            ax.plot([v['step'] for v in history],[v['objective'] for v in history],label=method)
        ax.set_title(name);ax.set_xlabel("step");ax.set_ylabel("training objective");ax.set_yscale("log");ax.legend()
    fig.tight_layout();fig.savefig(args.out/"convergence.png",dpi=150);plt.close(fig)
    metadata=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),train_inside=int((ytrain>=0).sum()),
        test_inside=int((ytest>=0).sum()),reference_domain=[low.tolist(),high.tolist()],
        selection="best training objective only; test points never used for checkpoint selection",
        spacing="mean distance to four nearest reference sites",position_dofs="XY only; z fixed for planar test",
        checks=["float64 loss gradcheck","affine pullback labels","uniform-scale labels","hard parameter bounds","GPU adjacency equality if rendering"])
    (args.out/"results.json").write_text(json.dumps(dict(metadata=metadata,metrics=metrics,histories=histories),indent=2)+"\n")
    fields=sorted(set().union(*(v.keys() for v in metrics)))
    with (args.out/"metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(metrics)
    print(f"Saved {args.out.resolve()}",flush=True)


if __name__=="__main__":
    main()
