"""Put existing 1x/10x force renders side by side, without warping images."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('output/powersim_bread/linear_dem'))
    parser.add_argument('--stronger', type=Path, default=Path('output/powersim_bread/linear_dem_10x'))
    args = parser.parse_args()
    import json
    report = json.loads((args.stronger/'mechanics.json').read_text())
    multiplier = report['force_multiplier']
    a, b = Image.open(args.baseline/'bread_dem.gif'), Image.open(args.stronger/'bread_dem.gif')
    if a.n_frames != b.n_frames or a.size != b.size or a.size != (1024, 512):
        raise ValueError('Expected matching dual-view animations')
    frames = []
    for index in range(a.n_frames):
        a.seek(index); b.seek(index)
        # Same original camera 99 from each run, with no spatial scaling.
        out = Image.new('RGB', (1024, 512), 'black')
        for column, (source, label) in enumerate(((a, 'Previous: 1x applied force'),
                         (b, f'New solve: {multiplier:g}x applied force'))):
            out.paste(source.convert('RGB').crop((512, 0, 1024, 512)), (512*column, 0))
            draw = ImageDraw.Draw(out)
            draw.rectangle((512*column, 0, 512*(column+1)-1, 55), fill='black')
            draw.text((512*column+10, 10), label, fill='white')
            draw.text((512*column+10, 28), 'Linear mechanics | no display displacement scaling', fill='white')
        frames.append(out)
    frames[0].save(args.stronger/'force_comparison.gif', save_all=True, append_images=frames[1:],
                   duration=85, loop=0)
    print(args.stronger/'force_comparison.gif')


if __name__ == '__main__':
    main()
