#!/usr/bin/env python3
"""mvgpu.FlowInter, FlowFPS, Flow, FlowBlur and Compensate against mvu's, byte for byte.

Both run the filter on the same clip with the same vectors, each side with its own super made with
the same settings (mvgpu.Super's planes are mvu.Super's). The vectors:

  mvgpu  mvgpu.Analyse's, mvu's side through mvgpu.ToMVU.
  mvu    mvu.Analyse's, mvgpu's side through mvgpu.FromMVU: vectors mvgpu's search wouldn't find.
  const  mvu.Analyse's description with every block of a frame given one vector, a different one for
         each frame and direction, and the SADs of some frames' blocks at and just past the scene
         change limits, so that the scene change fallback runs on some frames and not on frames
         beside them.

Every pixel of every output frame and plane is compared, and FlowFPS's frame durations.

    check_flow.py --src clip.nv12 --size 1920x1080 --frames 30 [--filter inter|fps|flow|blur|compensate]
                  [--vectors mvgpu] [--pel 2] [--blksize 16 [8]] [--overlap 8 [4]] [--format YUV444P8]
                  [--crop 1914x1074] [--delta 1] [--time 50] [--num 60 --den 1] [--extramask 0]
                  [--ml 100] [--blend 0] [--blur 50] [--prec 1] [--thsad 10000] [--thscd1 400] [--thscd2 51]
                  [--fps 30000/1001] [--other-super]

--format converts the 8-bit 4:2:0 source first (resize.Bicubic). --fps sets the clip's frame rate
(24 by default), which with --num and --den decides FlowFPS's times. Flow and Compensate take the
vectors of --delta (either sign), the others those of --delta and -delta.
--other-super gives Flow and Compensate the super of another clip (the clip inverted) than the one
the vectors come from: Compensate's blocks not under thsad come from it too.
"""
import argparse
import math
import os
import sys
from fractions import Fraction

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # an embedded Python leaves the script's directory out
from mvtest import const_vectors, eight_bit, format_clip, grid_label, nv12_clip, plane_pair, scene_limits  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--format', default='YUV420P8', help='a VapourSynth preset name: YUV420P8, YUV444P16, YUV420P10, ...')
    ap.add_argument('--analyse8', action='store_true', help='analyse an 8-bit copy of the clip')
    ap.add_argument('--crop', help='WxH: crop the frames to this size first, for grids that end inside a block')
    ap.add_argument('--filter', choices=['inter', 'fps', 'flow', 'blur', 'compensate'], default='inter')
    ap.add_argument('--vectors', choices=['mvgpu', 'mvu', 'const'], default='mvgpu', help='whose vectors both sides use')
    ap.add_argument('--blksize', type=int, nargs='+', default=[16], help='the block width, and its height when it differs')
    ap.add_argument('--overlap', type=int, nargs='+', default=[8], help='the overlap, and the vertical one when it differs')
    ap.add_argument('--analyse-blksize', type=int, nargs='+', help="the grid both sides analyse, where it isn't the super's")
    ap.add_argument('--analyse-overlap', type=int, nargs='+')
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--pad', type=int, nargs='+', default=[16], help='the padding, and the vertical one when it differs')
    ap.add_argument('--delta', type=int, default=1, help='the frames the vectors span')
    ap.add_argument('--time', type=float, help='FlowInter, Flow, Compensate')
    ap.add_argument('--num', type=int, help='FlowFPS')
    ap.add_argument('--den', type=int, help='FlowFPS')
    ap.add_argument('--extramask', type=int, help='FlowFPS')
    ap.add_argument('--ml', type=float)
    ap.add_argument('--blend', type=int)
    ap.add_argument('--blur', type=float, help='FlowBlur')
    ap.add_argument('--prec', type=int, help='FlowBlur')
    ap.add_argument('--thsad', type=int, help='Compensate')
    ap.add_argument('--other-super', action='store_true', help="Flow and Compensate: the super of the clip inverted")
    ap.add_argument('--thscd1', type=int)
    ap.add_argument('--thscd2', type=float)
    ap.add_argument('--fps', help="N/D: the clip's frame rate")
    ap.add_argument('--seed', type=int, default=1, help='const: the vectors drawn')
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
    if args.crop:
        w, h = (int(v) for v in args.crop.split('x'))
        clip = core.std.CropAbs(clip, w, h)
    if args.fps:
        num, den = (int(v) for v in args.fps.split('/'))
        clip = core.std.AssumeFPS(clip, fpsnum=num, fpsden=den)
    gclip = core.std.GPUUpload(clip)

    sk = dict(blksize=args.blksize, overlap=args.overlap, pel=args.pel, pad=args.pad)
    gsup = core.mvgpu.Super(gclip, **sk)
    csup = core.mvu.Super(clip, **sk)
    # The supers the vectors come from: the clip's, or an 8-bit copy's
    aclip = eight_bit(core, clip) if args.analyse8 else clip
    gasup = core.mvgpu.Super(core.std.GPUUpload(aclip), **sk) if args.analyse8 else gsup
    casup = core.mvu.Super(aclip, **sk) if args.analyse8 else csup
    if args.other_super:
        oclip = core.std.Invert(clip)
        gsup, csup = core.mvgpu.Super(core.std.GPUUpload(oclip), **sk), core.mvu.Super(oclip, **sk)
    thscd1 = 400 if args.thscd1 is None else args.thscd1
    thscd2 = 51.0 if args.thscd2 is None else args.thscd2
    deltas = (args.delta,) if args.filter in ('flow', 'compensate') else (args.delta, -args.delta)  # mvbw, mvfw
    ag = dict(blksize=args.analyse_blksize or args.blksize, overlap=args.analyse_overlap if args.analyse_overlap is not None else args.overlap)
    if args.vectors == 'mvgpu':
        gvec = [core.mvgpu.Analyse(gasup, delta=d, **ag) for d in deltas]
        cvec = [core.mvgpu.ToMVU(an) for an in gvec]
    else:
        cvec = [core.mvu.Analyse(casup, delta=d, **ag) for d in deltas]
        if args.vectors == 'const':
            th1, scd = scene_limits(cvec[0].get_frame(0).props, thscd1, thscd2)
            cvec = [const_vectors(core, an, (min(args.pad) - 1) * args.pel, th1, scd, args.seed * 2 + i) for i, an in enumerate(cvec)]
        gvec = [core.mvgpu.FromMVU(an, gasup) for an in cvec]

    fk = {k: getattr(args, k) for k in ('thscd1', 'thscd2') if getattr(args, k) is not None}
    if args.filter in ('inter', 'fps'):
        fk.update({k: getattr(args, k) for k in ('ml', 'blend') if getattr(args, k) is not None})
    if args.filter == 'flow':
        fk.update({k: getattr(args, k) for k in ('time',) if getattr(args, k) is not None})
        gout = core.mvgpu.Flow(gclip, gsup, gvec[0], **fk)
        cout = core.mvu.Flow(clip, csup, cvec[0], **fk)
    elif args.filter == 'compensate':
        fk.update({k: getattr(args, k) for k in ('time', 'thsad') if getattr(args, k) is not None})
        gout = core.mvgpu.Compensate(gclip, gsup, gvec[0], **fk)
        cout = core.mvu.Compensate(clip, csup, cvec[0], **fk)
    elif args.filter == 'blur':
        fk.update({k: getattr(args, k) for k in ('blur', 'prec') if getattr(args, k) is not None})
        gout = core.mvgpu.FlowBlur(gclip, gsup, gvec, **fk)
        cout = core.mvu.FlowBlur(clip, csup, cvec, **fk)
    elif args.filter == 'inter':
        fk.update({k: getattr(args, k) for k in ('time',) if getattr(args, k) is not None})
        gout = core.mvgpu.FlowInter(gclip, gsup, gvec, **fk)
        cout = core.mvu.FlowInter(clip, csup, cvec, **fk)
    else:
        fk.update({k: getattr(args, k) for k in ('num', 'den', 'extramask') if getattr(args, k) is not None})
        gout = core.mvgpu.FlowFPS(gclip, gsup, gvec, **fk)
        cout = core.mvu.FlowFPS(clip, csup, cvec, **fk)
    gout = core.std.GPUDownload(gout)
    if (gout.num_frames, gout.fps) != (cout.num_frames, cout.fps):
        print(f'mvgpu makes {gout.num_frames} frames at {gout.fps}, mvu {cout.num_frames} at {cout.fps}')
        sys.exit(1)

    frames = cout.num_frames
    # The output frame's frame before, to tell how much of it the filters changed
    step = Fraction(clip.fps.numerator, clip.fps.denominator) / Fraction(cout.fps.numerator, cout.fps.denominator) if args.filter == 'fps' else 1
    changed = 0
    differ = [0, 0, 0]
    worst = [0, 0, 0]
    sq = [0.0, 0.0, 0.0]
    props_differ = 0
    first = None
    for n in range(frames):
        a, b = gout.get_frame(n), cout.get_frame(n)
        before = clip.get_frame(min(int(n * step), clip.num_frames - 1))
        for k in ('_DurationNum', '_DurationDen'):
            if a.props.get(k) != b.props.get(k):
                props_differ += 1
        for p in range(clip.format.num_planes):
            pa, pb, dv = plane_pair(a, b, p)
            changed += np.count_nonzero(np.asarray(b[p]) != np.asarray(before[p]))
            d = np.count_nonzero(pa != pb)
            if d:
                differ[p] += d
                worst[p] = max(worst[p], dv.max().item())
                sq[p] += float(np.square(dv).sum())
                if first is None:
                    ys, xs = np.nonzero(pa != pb)
                    first = (n, p, int(xs[0]), int(ys[0]), np.asarray(a[p])[ys[0], xs[0]].item(), np.asarray(b[p])[ys[0], xs[0]].item())
    total = frames * w * h
    total_c = frames * (w >> clip.format.subsampling_w) * (h >> clip.format.subsampling_h) if clip.format.num_planes == 3 else 0
    sizes = (total, total_c, total_c)
    peak = 1.0 if clip.format.sample_type == vs.FLOAT else (1 << clip.format.bits_per_sample) - 1
    psnr = ['inf' if not sq[p] else f'{10 * math.log10(peak * peak * sizes[p] / sq[p]):.1f}' for p in range(clip.format.num_planes)]
    print(f'{args.filter} {args.format}{" analysed on 8 bits" if args.analyse8 else ""} pel {args.pel} {grid_label(args.blksize, args.overlap)} {args.vectors} vectors, {frames} frames, '
          f'{100 * changed / (total + 2 * total_c):.1f}% of the pixels changed from the frame before: differing pixels Y {differ[0]} of {total}'
          + (f', U {differ[1]}, V {differ[2]} of {total_c} each' if total_c else '')
          + (f'; largest difference {worst}, PSNR {psnr}; first at frame {first[0]} plane {first[1]} ({first[2]}, {first[3]}): '
             f'mvgpu {first[4]}, mvu {first[5]}' if first else '')
          + (f'; {props_differ} duration properties differ' if props_differ else ''))
    sys.exit(0 if props_differ == 0 and sum(differ) == 0 else 1)


if __name__ == '__main__':
    main()
