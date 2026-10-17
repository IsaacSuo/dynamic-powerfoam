"""Plot measured macro amplitudes and compare original-renderer animations."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads((args.root/'job.json').read_text())
    baseline = Path('output/powersim_bread/finite_dem/newton_fast_after_resume')
    paths = [baseline, *(Path(r['path']) for r in job['accepted'])]
    labels = ['Baseline', *(f"Target {100*r['target']:g}% (measured {100*r['achieved_fraction']:.2f}%)" for r in job['accepted'])]
    width = job['reference_width']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    summaries = []
    for path, label in zip(paths, labels):
        m = json.loads((path/'mechanics.json').read_text())
        with np.load(path/'trajectory.npz') as a: samples=a['samples']; period=float(a['period'])
        rows = [json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines()]
        if rows and 'downward_fraction_width' in rows[0]:
            t = np.r_[0., [r['time'] for r in rows]]
            displacement = np.r_[0., [r['downward_fraction_width']*100 for r in rows]]
        else:
            t=samples[:, 0]; displacement=-samples[:, 1]/width*100
        ax[0].plot(t/period, displacement, label=label)
        ax[1].plot(t/period, m['total_peak_force']*np.where(t<period, np.sin(np.pi*t/period)**2, 0.), label=label)
        ax[2].plot(samples[:, 0]/period, samples[:, 2], label=label+' energy')
        ax[2].plot(samples[:, 0]/period, samples[:, 3], '--', label=label+' work')
        summaries.append(dict(path=str(path), label=label, peak_downward_percent=float(displacement.max()),
                              total_peak_force=m['total_peak_force'], checks_pass=m['checks_pass']))
    for target in job['targets']: ax[0].axhline(100*target, color='gray', linewidth=.7, linestyle=':')
    ax[0].set(ylabel='Mean downward displacement / initial width (%)')
    ax[1].set(ylabel='Applied total force (model units)')
    ax[2].set(ylabel='Energy / external work (model units)')
    for axis in ax:
        axis.set_xlabel('Time / loading duration T'); axis.axvline(1., color='gray', linewidth=.7)
        axis.legend(fontsize=7)
    fig.savefig(args.root/'macro_comparison.png', dpi=160); plt.close(fig)
    (args.root/'macro_comparison.json').write_text(json.dumps(dict(runs=summaries,
        display_displacement_scale=1, scope='Force-calibrated macro amplitude; no independent temporal refinement or material calibration claimed'), indent=2)+'\n')
    if not all((p/'bread_dem.gif').exists() for p in paths): return
    sources = [Image.open(p/'bread_dem.gif') for p in paths]
    try:
        assert all(s.n_frames == 49 for s in sources)
        frames = []
        for index in range(49):
            frame = Image.new('RGB', (512*len(sources), 512), 'black')
            draw = ImageDraw.Draw(frame)
            for column, (source, label) in enumerate(zip(sources, labels)):
                source.seek(index)
                frame.paste(source.convert('RGB').crop((512, 0, 1024, 512)), (512*column, 0))
                draw.rectangle((512*column, 0, 512*(column+1)-1, 58), fill='black')
                draw.text((512*column+10, 10), label, fill='white')
                draw.text((512*column+10, 30), 'Actual simulated motion | display scale 1x', fill='white')
            frames.append(frame)
        frames[0].save(args.root/'macro_comparison.gif', save_all=True, append_images=frames[1:], duration=85, loop=0)
    finally:
        for source in sources: source.close()
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
