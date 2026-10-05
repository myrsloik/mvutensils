#!/usr/bin/env python3
"""mvgpu.VectorLengthMask, SADMask, OcclusionMask and SCDetection against mvu's.

Both sides make the masks, and SCDetection's properties, from the same vectors:

  mvgpu   mvgpu.Analyse's, mvu's side through mvgpu.ToMVU.
  mvu     mvu.Analyse's, mvgpu's side through mvgpu.FromMVU.
  random  mvu.Analyse's description with every block given a vector and a SAD of its own, and on
          some frames just enough blocks or one block too many above thscd1 for a scene change
          (mvtest.random_vectors).

Every pixel of every mask frame is compared (floats bit for bit), and the masks' _Range;
SCDetection's properties on every frame, with the clip GPU resident and not. A gamma other than 1
(other than 1 or 2 for VectorLengthMask) goes through the GPU's pow, which differs from the C
library's in the last bits: --tolerance is the largest difference allowed.

    check_masks.py --src clip.nv12 --size 1920x1080 --frames 20 [--filter all|length|sad|occlusion|scd]
                   [--vectors mvgpu] [--format YUV420P8] [--analyse8] [--crop 1914x1074] [--pel 2]
                   [--blksize 16] [--overlap 8] [--pad 16] [--delta 1] [--ml 100] [--gamma 1]
                   [--time 100] [--scval 0] [--thscd1 400] [--thscd2 51] [--tolerance 0]
--extreme makes every other frame dark and the others bright first (mvtest.extreme), for SADs past 2^31.
"""
import argparse
import os
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # an embedded Python leaves the script's directory out
from mvtest import eight_bit, extreme, format_clip, grid_label, nv12_clip, plane_pair, random_vectors, scene_limits  # noqa: E402

MASKS = {'length': 'VectorLengthMask', 'sad': 'SADMask', 'occlusion': 'OcclusionMask'}
SCD_PROPS = ('_SceneChangePrev', '_SceneChangeNext')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--format', default='YUV420P8', help='a VapourSynth preset name: YUV420P8, YUV444P16, GRAYS, ...')
    ap.add_argument('--extreme', action='store_true', help='every other frame dark, the others bright (mvtest.extreme)')
    ap.add_argument('--analyse8', action='store_true', help='analyse an 8-bit copy of the clip')
    ap.add_argument('--crop', help='WxH: crop the frames to this size first, for grids that end inside a block')
    ap.add_argument('--filter', choices=['all', 'length', 'sad', 'occlusion', 'scd'], default='all')
    ap.add_argument('--vectors', choices=['mvgpu', 'mvu', 'random'], default='mvgpu', help='whose vectors both sides use')
    ap.add_argument('--blksize', type=int, nargs='+', default=[16], help='the block width, and its height when it differs')
    ap.add_argument('--overlap', type=int, nargs='+', default=[8], help='the overlap, and the vertical one when it differs')
    ap.add_argument('--analyse-blksize', type=int, nargs='+', help="the grid both sides analyse, where it isn't the super's")
    ap.add_argument('--analyse-overlap', type=int, nargs='+')
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--pad', type=int, nargs='+', default=[16], help='the padding, and the vertical one when it differs')
    ap.add_argument('--delta', type=int, default=1)
    ap.add_argument('--ml', type=float)
    ap.add_argument('--gamma', type=float)
    ap.add_argument('--time', type=float)
    ap.add_argument('--scval', type=float)
    ap.add_argument('--thscd1', type=int)
    ap.add_argument('--thscd2', type=float)
    ap.add_argument('--tolerance', type=float, default=0, help='the largest difference allowed (for a gamma the GPU takes the pow of)')
    ap.add_argument('--seed', type=int, default=1, help='random: the vectors drawn')
    ap.add_argument('--plugin', help='the MVGPUtensils library to load, unless it autoloads')
    ap.add_argument('--mvu', default=r'C:\Libraries\mvutensils\msvc\x64\Release\MVUtensils.dll', help='the MVUtensils library, unless it autoloads')
    args = ap.parse_args()

    core = vs.core
    if args.plugin:
        core.std.LoadPlugin(args.plugin)
    if not hasattr(core, 'mvu'):
        core.std.LoadPlugin(args.mvu)
    w, h = (int(v) for v in args.size.split('x'))
    clip = format_clip(core, nv12_clip(core, args.src, w, h, args.frames), args.format)
    if args.extreme:
        clip = extreme(core, clip)
    if args.crop:
        w, h = (int(v) for v in args.crop.split('x'))
        clip = core.std.CropAbs(clip, w, h)
    gclip = core.std.GPUUpload(clip)

    sk = dict(blksize=args.blksize, overlap=args.overlap, pel=args.pel, pad=args.pad)
    aclip = eight_bit(core, clip) if args.analyse8 else clip
    gasup = core.mvgpu.Super(core.std.GPUUpload(aclip), **sk)
    casup = core.mvu.Super(aclip, **sk)
    thscd1 = 400 if args.thscd1 is None else args.thscd1
    thscd2 = 51.0 if args.thscd2 is None else args.thscd2
    ag = dict(blksize=args.analyse_blksize or args.blksize, overlap=args.analyse_overlap if args.analyse_overlap is not None else args.overlap)
    if args.vectors == 'mvgpu':
        gvec = core.mvgpu.Analyse(gasup, delta=args.delta, **ag)
        cvec = core.mvgpu.ToMVU(gvec)
    else:
        cvec = core.mvu.Analyse(casup, delta=args.delta, **ag)
        if args.vectors == 'random':
            th1, scd = scene_limits(cvec.get_frame(0).props, thscd1, thscd2)
            cvec = random_vectors(core, cvec, (min(args.pad) - 1) * args.pel, th1, scd, args.seed)
        gvec = core.mvgpu.FromMVU(cvec, gasup)

    sk2 = {k: getattr(args, k) for k in ('thscd1', 'thscd2') if getattr(args, k) is not None}
    mk = dict(sk2, **{k: getattr(args, k) for k in ('ml', 'gamma', 'time', 'scval') if getattr(args, k) is not None})
    filters = list(MASKS) + ['scd'] if args.filter == 'all' else [args.filter]
    results = []
    ok = True
    for name in filters:
        if name == 'scd':
            cout = core.mvu.SCDetection(clip, cvec, **sk2)
            outs = [('gpu', core.std.GPUDownload(core.mvgpu.SCDetection(gclip, gvec, **sk2))), ('cpu', core.mvgpu.SCDetection(clip, gvec, **sk2))]
            differ = 0
            changes = 0
            for n in range(cout.num_frames):
                b = cout.get_frame(n).props
                changes += sum(int(b.get(k, 0)) for k in SCD_PROPS)
                for _, out in outs:
                    a = out.get_frame(n).props
                    if any(a.get(k) != b.get(k) for k in SCD_PROPS):
                        differ += 1
                        if differ == 1:
                            print(f'SCDetection frame {n}: mvgpu {[a.get(k) for k in SCD_PROPS]}, mvu {[b.get(k) for k in SCD_PROPS]}')
            ok &= differ == 0
            results.append(f'SCDetection {differ} of {2 * cout.num_frames} frames differ ({changes} scene changes)')
            continue
        gout = core.std.GPUDownload(getattr(core.mvgpu, MASKS[name])(gvec, **mk))
        cout = getattr(core.mvu, MASKS[name])(cvec, **mk)
        if (gout.format.id, gout.width, gout.height, gout.num_frames) != (cout.format.id, cout.width, cout.height, cout.num_frames):
            print(f'{MASKS[name]}: mvgpu makes {gout.format.name} {gout.width}x{gout.height} x {gout.num_frames}, mvu {cout.format.name} {cout.width}x{cout.height} x {cout.num_frames}')
            sys.exit(1)
        differ = 0
        worst = 0.0
        props = 0
        first = None
        nonzero = 0
        for n in range(cout.num_frames):
            a, b = gout.get_frame(n), cout.get_frame(n)
            if a.props.get('_Range') != b.props.get('_Range'):
                props += 1
            pa, pb, dv = plane_pair(a, b, 0)
            nonzero += np.count_nonzero(np.asarray(b[0]))
            bad = pa != pb
            d = np.count_nonzero(bad)
            if d:
                differ += d
                worst = max(worst, dv.max().item())
                if first is None:
                    ys, xs = np.nonzero(bad)
                    first = (n, int(xs[0]), int(ys[0]), np.asarray(a[0])[ys[0], xs[0]].item(), np.asarray(b[0])[ys[0], xs[0]].item())
        total = cout.num_frames * cout.width * cout.height
        ok &= props == 0 and (differ == 0 or worst <= args.tolerance)
        results.append(f'{name} {differ} of {total} differ ({100 * nonzero / total:.0f}% nonzero)' + (f', largest {worst:g}' if differ else '')
                       + (f', first at frame {first[0]} ({first[1]}, {first[2]}): mvgpu {first[3]}, mvu {first[4]}' if first else '')
                       + (f', {props} _Range differ' if props else ''))
    print(f'{args.format}{" analysed on 8 bits" if args.analyse8 else ""} pel {args.pel} {grid_label(args.blksize, args.overlap)} {args.vectors} vectors delta {args.delta}, '
          f'{args.frames} frames: ' + '; '.join(results))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
