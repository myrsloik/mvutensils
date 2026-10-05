#!/usr/bin/env python3
"""mvgpu.Recalculate against mvu.Recalculate, vector for vector.

Both recalculate the same old vectors on supers of the same clip made with the same settings. The
old vectors:

  mvgpu   mvgpu.Analyse's, of a super with the --old-blksize grid (the super's own by default), mvu's
          side through mvgpu.ToMVU.
  mvu     mvu.Analyse's, mvgpu's side through mvgpu.FromMVU with mvgpu's super of that grid.
  random  mvu.Analyse's description with every block given a vector of its own (mvtest.random_vectors).

Every block's vector and SAD is compared, and the analysis description. Float SADs are those of mvu's
AVX-512 build, which round apart from its AVX2 and SSE2 builds' (mvu sums float differences in the order
each CPU's kernels take): run on an AVX-512 CPU. mvu's SAD is 64 bits, mvgpu's records' 32: a float SAD
past 2^32 - 1 (--overrange) compares saturated.

    check_recalculate.py --src clip.nv12 --size 1920x1080 --frames 10 [--vectors mvgpu] [--format YUV420P8]
                         [--crop 1914x1074] [--pel 2] [--blksize 16] [--overlap 8] [--pad 16]
                         [--old-blksize 32 --old-overlap 16] [--old-pel 4] [--delta 1] [--thsad 200] [--smooth 0]
                         [--search 2] [--searchparam 2] [--mvlambda 1000] [--chroma 0] [--pnew 25] [--satd 1]
--extreme makes every other frame dark and the others bright first (mvtest.extreme), for SADs past 2^31;
--overrange multiplies a float clip's frames by 32 and -32 by turns (mvtest.overrange), for float SADs past
what integer samples make, past 2^32 summed and saturated.
"""
import argparse
import os
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # an embedded Python leaves the script's directory out
from mvtest import extreme, format_clip, grid_label, nv12_clip, overrange, random_vectors, scene_limits  # noqa: E402


def first_clip(c):
    return c[0] if isinstance(c, list) else c


# The arguments both Recalculates take as they come
RECALC_ARGS = ('thsad', 'smooth', 'search', 'searchparam', 'mvlambda', 'chroma', 'pnew', 'satd')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--format', default='YUV420P8', help='a VapourSynth preset name: YUV420P8, YUV444P16, GRAYS, ...')
    ap.add_argument('--extreme', action='store_true', help='every other frame dark, the others bright (mvtest.extreme)')
    ap.add_argument('--overrange', action='store_true', help="a float clip's frames multiplied by 32 and -32 by turns (mvtest.overrange)")
    ap.add_argument('--crop', help='WxH: crop the frames to this size first, for grids that end inside a block')
    ap.add_argument('--vectors', choices=['mvgpu', 'mvu', 'random'], default='mvgpu', help='whose old vectors both sides recalculate')
    ap.add_argument('--blksize', type=int, nargs='+', default=[16], help='the block width, and its height when it differs')
    ap.add_argument('--overlap', type=int, nargs='+', default=[8], help='the overlap, and the vertical one when it differs')
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--pad', type=int, nargs='+', default=[16], help='the padding, and the vertical one when it differs')
    ap.add_argument('--old-blksize', type=int, nargs='+', help="the old vectors' grid (the super's by default)")
    ap.add_argument('--old-overlap', type=int, nargs='+')
    ap.add_argument('--old-pel', type=int, help="the old vectors' pel (the super's by default), rescaled to the super's")
    ap.add_argument('--new-blksize', type=int, nargs='+', help="the grid both Recalculates make, where it isn't the super's")
    ap.add_argument('--new-overlap', type=int, nargs='+')
    ap.add_argument('--delta', type=int, default=1)
    for k in RECALC_ARGS:
        ap.add_argument('--' + k, type=int)
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
    if args.overrange:
        clip = overrange(core, clip)
    if args.crop:
        w, h = (int(v) for v in args.crop.split('x'))
        clip = core.std.CropAbs(clip, w, h)
    gclip = core.std.GPUUpload(clip)

    sk = dict(blksize=args.blksize, overlap=args.overlap, pel=args.pel, pad=args.pad)
    ok_ = dict(sk, blksize=args.old_blksize or args.blksize, overlap=args.old_overlap if args.old_overlap is not None else args.overlap,
               pel=args.old_pel or args.pel)
    gsup, csup = core.mvgpu.Super(gclip, **sk), core.mvu.Super(clip, **sk)
    gsup_old = core.mvgpu.Super(gclip, **ok_) if ok_ != sk else gsup
    csup_old = core.mvu.Super(clip, **ok_) if ok_ != sk else csup
    if args.vectors == 'mvgpu':
        gold = core.mvgpu.Analyse(gsup_old, delta=args.delta)
        cold = core.mvgpu.ToMVU(gold)
    else:
        cold = core.mvu.Analyse(csup_old, delta=args.delta, blksize=ok_['blksize'], overlap=ok_['overlap'])
        if args.vectors == 'random':
            th1, scd = scene_limits(cold.get_frame(0).props, 400, 51.0)
            # (vectors past the padding are refused by mvu)
            cold = random_vectors(core, cold, (min(args.pad) - 1) * ok_['pel'], th1, scd, args.seed)
        gold = core.mvgpu.FromMVU(cold, gsup_old)

    rk = {k: getattr(args, k) for k in RECALC_ARGS if getattr(args, k) is not None}
    if args.new_blksize:
        rk.update(blksize=args.new_blksize, overlap=args.new_overlap if args.new_overlap is not None else args.overlap)
    gout = first_clip(core.mvgpu.Recalculate(gsup, gold, **rk))
    cout = first_clip(core.mvu.Recalculate(csup, cold, **rk))

    # A frame whose reference frame is outside the clip has no old vectors and gets no new ones, only
    # the new grid's description
    frames = args.frames
    props = cout.get_frame(0).props
    nbx, nby = props['MVUtensilsAnalysisNBlkX'], props['MVUtensilsAnalysisNBlkY']
    records = core.std.GPUDownload(gout)
    differ = 0
    desc = 0
    worst = None
    for n in range(frames):
        a, b = gout.get_frame(n).props, cout.get_frame(n).props
        for k in b.keys():
            if k.startswith('MVUtensilsAnalysis') and k not in ('MVUtensilsAnalysisVectors', 'MVUtensilsAnalysisSAD'):
                if a.get(k.replace('MVUtensils', 'MVGPUtensils', 1)) != b[k]:
                    desc += 1
                    if desc == 1:
                        print(f'frame {n}: {k} mvgpu {a.get(k.replace("MVUtensils", "MVGPUtensils", 1))}, mvu {b[k]}')
        has_g, has_c = a.get('MVGPUtensilsAnalysisHasVectors', 0) == 1, 'MVUtensilsAnalysisVectors' in b
        if has_g != has_c:
            print(f'frame {n}: mvgpu {"has" if has_g else "lacks"} vectors, mvu {"has" if has_c else "lacks"} them')
            sys.exit(1)
        if not has_c:
            continue
        rec = np.asarray(records.get_frame(n)[0]).view(np.int32)[:, :4 * nbx].reshape(nby * nbx, 4)
        packed = np.array(b['MVUtensilsAnalysisVectors'], np.int64)
        cx, cy = (packed & 0xFFFFFFFF).astype(np.uint32).view(np.int32), (packed >> 32).astype(np.int32)
        csad = np.minimum(np.array(b['MVUtensilsAnalysisSAD'], np.int64), 0xFFFFFFFF)  # (mvgpu's records' 32 bits)
        bad = (rec[:, 0] != cx) | (rec[:, 1] != cy) | (rec[:, 2].astype(np.uint32) != csad)
        differ += int(np.count_nonzero(bad))
        if worst is None and bad.any():
            i = int(np.nonzero(bad)[0][0])
            worst = (n, i % nbx, i // nbx, tuple(int(v) for v in rec[i, :3]), (int(cx[i]), int(cy[i]), int(csad[i])))
    total = frames * nbx * nby
    print(f'{args.format} pel {args.pel} {grid_label(args.blksize, args.overlap)} from {grid_label(ok_["blksize"], ok_["overlap"])}'
          + (f' pel {ok_["pel"]}' if ok_['pel'] != args.pel else '') + f' {args.vectors} vectors delta {args.delta}'
          + (f' {rk}' if rk else '') + f', {frames} frames: {differ} of {total} blocks differ'
          + (f'; first at frame {worst[0]} block ({worst[1]}, {worst[2]}): mvgpu {worst[3]}, mvu {worst[4]}' if worst else '')
          + (f'; {desc} description properties differ' if desc else ''))
    sys.exit(0 if differ == 0 and desc == 0 else 1)


if __name__ == '__main__':
    main()
