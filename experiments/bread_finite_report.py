"""Check temporal refinement and compare finite-motion vs linear bread."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse', type=Path, default=Path('output/powersim_bread/finite_dem/adaptive_coarse'))
    parser.add_argument('--fine', type=Path, default=Path('output/powersim_bread/finite_dem/fine'))
    args = parser.parse_args()
    root = Path('output/powersim_bread/finite_dem')
    fine = args.fine
    def read(path):
        with np.load(path/'trajectory.npz') as a:
            return a['q'], a['samples']
    qc, sc = read(args.coarse); qf, sf = read(fine)
    ql, sl = read(Path('output/powersim_bread/linear_dem_10x'))
    assert np.allclose(sc[:, 0], sf[:, 0], atol=1e-12) and np.allclose(sl[:, 0], sf[:, 0], atol=1e-12)
    amplitude = np.linalg.norm(qf[:, :, :3], axis=2).max()
    positional = float(np.linalg.norm(qf[:, :, :3]-qc[:, :, :3], axis=2).max()/amplitude)
    angular = 0.
    for a, b in zip(qc, qf):
        error = (Rotation.from_rotvec(a[:, 3:]).inv()*Rotation.from_rotvec(b[:, 3:])).magnitude()
        angular = max(angular, float(error.max()))
    report = dict(coarse_vs_fine_max_position_error_over_amplitude=positional,
        coarse_vs_fine_max_orientation_error_rad=angular,
        finite_vs_linear_max_position_difference_over_finite_amplitude=float(
            np.linalg.norm(qf[:, :, :3]-ql[:, :, :3], axis=2).max()/amplitude),
        finite_vs_linear_loaded_region_max_difference=float(abs(sf[:, 1]-sl[:, 1]).max()),
        passes_position_threshold_3_percent=positional < .03,
        passes_orientation_threshold_0p01_rad=angular < .01)
    (root/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    ax[0].plot(sf[:, 0], sf[:, 1], label='Finite-motion, fine dt')
    ax[0].plot(sc[:, 0], sc[:, 1], '.', label='Finite-motion, coarse dt')
    ax[0].plot(sl[:, 0], sl[:, 1], '--', label='Previous linear solver, same force')
    ax[0].set(xlabel='time (uncalibrated)', ylabel='Loaded region mean z displacement')
    ax[0].legend()
    ax[1].plot(sf[:, 0], sf[:, 2], label='Total energy')
    ax[1].plot(sf[:, 0], sf[:, 3], '--', label='External work')
    ax[1].legend(); fig.savefig(root/'comparison.png', dpi=160); plt.close(fig)
    assert report['passes_position_threshold_3_percent'] and report['passes_orientation_threshold_0p01_rad'], report
    if not (fine/'bread_dem.gif').exists(): return
    left = Image.open('output/powersim_bread/linear_dem_10x/bread_dem.gif')
    right = Image.open(fine/'bread_dem.gif')
    assert left.n_frames == right.n_frames == 49
    frames = []
    for index in range(right.n_frames):
        frame = Image.new('RGB', (1024, 512), 'black')
        for column, (source, title) in enumerate(((left, 'Previous: linear mechanics'),
                                                (right, 'New: finite-motion bond histories'))):
            source.seek(index)
            frame.paste(source.convert('RGB').crop((512, 0, 1024, 512)), (512*column, 0))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((512*column, 0, 512*(column+1)-1, 55), fill='black')
            draw.text((512*column+10, 10), title, fill='white')
            draw.text((512*column+10, 28), 'Same applied load | original Power Foam renderer', fill='white')
        frames.append(frame)
    frames[0].save(root/'linear_vs_finite.gif', save_all=True, append_images=frames[1:], duration=85, loop=0)


if __name__ == '__main__':
    main()
