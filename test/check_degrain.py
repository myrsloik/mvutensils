#!/usr/bin/env python3
"""mvgpu.Degrain against mvu.Degrain, byte for byte.

Both denoise the same 8-bit clip with the same vectors. By default they are mvgpu.AnalyseMany's:
mvu.Degrain gets them on a carrier clip that holds mvgpu's analysis description under mvu's property
names, so mvu never searches. With --vectors mvu they are mvu.Analyse's, and mvgpu.Degrain gets them
on the frames of its super, as mvgpu.Analyse would attach them: for grids and formats mvgpu.Analyse
doesn't search yet. Each side builds its super with the same settings (mvgpu.Super's planes are
mvu.Super's). Every pixel of every frame and plane is compared.

    check_degrain.py --src noisy.nv12 --size 1920x1080 --frames 52 --pel 4 --blksize 16 --overlap 8
                     [--format YUV444P8] [--vectors mvu] [--radius 2] [--thsad 400 300] [--thsad2 150]
                     [--planes 0 2] [--limit 3 2] [--thscd1 400] [--thscd2 51] [--weights 1 2 3 2 1]
                     [--centersuper] [--crop 1914x1074]

--format converts the 8-bit 4:2:0 source first (resize.Bicubic). --centersuper gives both a separate
centre super, of the clip blurred, as scripts that denoise with the super of another clip do.
"""
import argparse
import os
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # an embedded Python leaves the script's directory out
from mvtest import gpu_vectors, mvu_vectors, nv12_clip  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--format', default='YUV420P8', help='a VapourSynth preset name: YUV420P8 or YUV444P8')
    ap.add_argument('--vectors', choices=['mvgpu', 'mvu'], default='mvgpu', help="whose vectors both sides use")
    ap.add_argument('--blksize', type=int, default=16)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--thsad', type=int, nargs='+')
    ap.add_argument('--thsad2', type=int, nargs='+')
    ap.add_argument('--planes', type=int, nargs='+')
    ap.add_argument('--limit', type=float, nargs='+')
    ap.add_argument('--thscd1', type=int)
    ap.add_argument('--thscd2', type=float)
    ap.add_argument('--weights', type=int, nargs='+')
    ap.add_argument('--centersuper', action='store_true', help='a centre super of the clip blurred, on both sides')
    ap.add_argument('--crop', help='WxH: crop the frames to this size first, for grids that end inside a block')
    ap.add_argument('--render', type=int, nargs=2, metavar=('BLKSIZE', 'OVERLAP'),
                    help="Degrain's super with this block size and overlap instead of the analysis super (the same padded frame), on both sides")
    ap.add_argument('--plugin', help='the MVGPUtensils library to load, unless it autoloads')
    ap.add_argument('--mvu', default=r'C:\Libraries\mvutensils\msvc\x64\Release\MVUtensils.dll', help='the MVUtensils library, unless it autoloads')
    args = ap.parse_args()

    core = vs.core
    if args.plugin:
        core.std.LoadPlugin(args.plugin)
    if not hasattr(core, 'mvu'):
        core.std.LoadPlugin(args.mvu)
    w, h = (int(v) for v in args.size.split('x'))
    clip = nv12_clip(core, args.src, w, h, args.frames)
    if args.format != 'YUV420P8':
        clip = core.resize.Bicubic(clip, format=getattr(vs, args.format))
    if args.crop:
        w, h = (int(v) for v in args.crop.split('x'))
        clip = core.std.CropAbs(clip, w, h)
    gclip = core.std.GPUUpload(clip)

    sk = dict(blksize=args.blksize, overlap=args.overlap, pel=args.pel)
    dk = {k: getattr(args, k) for k in ('thsad', 'thsad2', 'planes', 'limit', 'thscd1', 'thscd2', 'weights') if getattr(args, k) is not None}

    gsup = core.mvgpu.Super(gclip, **sk)
    csup = core.mvu.Super(clip, **sk)
    if args.vectors == 'mvgpu':
        fields = core.mvgpu.AnalyseMany(gsup, radius=args.radius)
        cvec = [mvu_vectors(core, an, clip, args.frames) for an in fields]
    else:
        deltas = [d for r in range(1, args.radius + 1) for d in (r, -r)]
        cvec = [core.mvu.Analyse(csup, delta=d, blksize=args.blksize, overlap=args.overlap) for d in deltas]
        fields = [gpu_vectors(core, an, gsup) for an in cvec]
    # Degrain's supers: the analysis one, or one with the render block size and overlap
    rk = dict(sk, blksize=args.render[0], overlap=args.render[1]) if args.render else sk
    if args.render:
        gsup, csup = core.mvgpu.Super(gclip, onelevel=True, **rk), core.mvu.Super(clip, onelevel=True, **rk)
    if args.centersuper:
        soft = core.std.BoxBlur(clip, hradius=2, vradius=2)
        dk_gpu = dict(dk, centersuper=core.mvgpu.Super(core.std.GPUUpload(soft), onelevel=True, **rk))
        dk_cpu = dict(dk, centersuper=core.mvu.Super(soft, onelevel=True, **rk))
    else:
        dk_gpu = dk_cpu = dk
    gout = core.std.GPUDownload(core.mvgpu.Degrain(gclip, gsup, fields, **dk_gpu))
    cout = core.mvu.Degrain(clip, csup, cvec, **dk_cpu)

    differ = [0, 0, 0]
    worst = [0, 0, 0]
    changed = 0
    first = None
    for n in range(args.frames):
        a, b, s = gout.get_frame(n), cout.get_frame(n), clip.get_frame(n)
        for p in range(3):
            pa, pb = np.asarray(a[p]).astype(np.int16), np.asarray(b[p]).astype(np.int16)
            changed += np.count_nonzero(pa != np.asarray(s[p]))
            d = np.count_nonzero(pa != pb)
            if d:
                differ[p] += d
                worst[p] = max(worst[p], int(np.abs(pa - pb).max()))
                if first is None:
                    ys, xs = np.nonzero(pa != pb)
                    first = (n, p, int(xs[0]), int(ys[0]), int(pa[ys[0], xs[0]]), int(pb[ys[0], xs[0]]))
    total = args.frames * w * h
    total_c = args.frames * (w >> clip.format.subsampling_w) * (h >> clip.format.subsampling_h)
    print(f'{args.format} {args.frames} frames, {100 * changed / (total + 2 * total_c):.1f}% of the pixels denoised: differing pixels Y {differ[0]} of {total}, '
          f'U {differ[1]}, V {differ[2]} of {total_c} each'
          + (f'; largest difference {worst}; first at frame {first[0]} plane {first[1]} ({first[2]}, {first[3]}): '
             f'mvgpu {first[4]}, mvu {first[5]}' if first else ''))
    sys.exit(0 if sum(differ) == 0 else 1)


if __name__ == '__main__':
    main()
