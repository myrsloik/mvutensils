#!/usr/bin/env python3
"""mvgpu.Recalculate against mvu.Recalculate, vector for vector.

Both recalculate the same old vectors on supers of the same clip made with the same settings. The
old vectors:

  mvgpu   mvgpu.Analyse's, of a super with the --old-blksize grid (the super's own by default), on a
          carrier clip for mvu (mvtest.mvu_vectors).
  mvu     mvu.Analyse's, on the frames of mvgpu's super of that grid (mvtest.gpu_vectors).
  random  mvu.Analyse's description with every block given a vector of its own (mvtest.random_vectors).

Every block's vector and SAD is compared, and the analysis description. mvgpu searches float supers
as the 16-bit samples they stand for, as mvgpu.Analyse does (mvu sums float differences in an order its
CPU kernels decide): with a float --format, mvu recalculates the clip quantized to 16 bits the same
way, which matches mvgpu's float super exactly at pel 1, where its samples are the clip's (at pel 2 and
4 the sub-pel samples come from different arithmetic).

    check_recalculate.py --src clip.nv12 --size 1920x1080 --frames 10 [--vectors mvgpu] [--format YUV420P8]
                         [--crop 1914x1074] [--pel 2] [--blksize 16] [--overlap 8] [--pad 16]
                         [--old-blksize 32 --old-overlap 16] [--delta 1] [--thsad 200] [--smooth 0]
                         [--search 2] [--searchparam 2] [--mvlambda 1000] [--chroma 0] [--pnew 25]
"""
import argparse
import os
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # an embedded Python leaves the script's directory out
from mvtest import format_clip, gpu_vectors, mvu_vectors, nv12_clip, random_vectors, scene_limits  # noqa: E402


def quantized16(core, clip):
    """A float clip as the 16-bit samples mvgpu's search takes it for: luma's 0 .. 1 and chroma's
    -0.5 .. 0.5 scaled to 0 .. 65535, rounded and clamped, in float32 arithmetic (refine_common.glsl's
    Quantize)"""
    f = clip.format
    q = core.query_video_format(f.color_family, vs.INTEGER, 16, f.subsampling_w, f.subsampling_h)
    blank = core.std.BlankClip(clip, format=q.id)

    def fill(n, f):
        out = f[0].copy()
        for p in range(clip.format.num_planes):
            v = np.asarray(f[1][p]).astype(np.float32)
            if p:
                v = v + np.float32(0.5)
            v = np.clip(v, np.float32(0), np.float32(1))
            s = v * np.float32(65535) + np.float32(0.5)
            np.asarray(out[p])[:, :] = s.astype(np.uint16)
        return out

    return core.std.ModifyFrame(blank, [blank, clip], fill)


def first_clip(c):
    return c[0] if isinstance(c, list) else c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--format', default='YUV420P8', help='a VapourSynth preset name: YUV420P8, YUV444P16, GRAYS, ...')
    ap.add_argument('--crop', help='WxH: crop the frames to this size first, for grids that end inside a block')
    ap.add_argument('--vectors', choices=['mvgpu', 'mvu', 'random'], default='mvgpu', help='whose old vectors both sides recalculate')
    ap.add_argument('--blksize', type=int, default=16)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--pad', type=int, default=16)
    ap.add_argument('--old-blksize', type=int, help="the old vectors' grid (the super's by default)")
    ap.add_argument('--old-overlap', type=int)
    ap.add_argument('--delta', type=int, default=1)
    for k in ('thsad', 'smooth', 'search', 'searchparam', 'mvlambda', 'chroma', 'pnew'):
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
    if args.crop:
        w, h = (int(v) for v in args.crop.split('x'))
        clip = core.std.CropAbs(clip, w, h)
    isfloat = clip.format.sample_type == vs.FLOAT
    if isfloat and args.pel != 1:
        ap.error('a float clip compares with mvu only at pel 1')
    cclip = quantized16(core, clip) if isfloat else clip  # what mvu searches
    gclip = core.std.GPUUpload(clip)

    sk = dict(blksize=args.blksize, overlap=args.overlap, pel=args.pel, pad=args.pad)
    ok_ = dict(sk, blksize=args.old_blksize or args.blksize, overlap=args.old_overlap if args.old_overlap is not None else args.overlap)
    gsup, csup = core.mvgpu.Super(gclip, **sk), core.mvu.Super(cclip, **sk)
    gsup_old = core.mvgpu.Super(gclip, **ok_) if ok_ != sk else gsup
    csup_old = core.mvu.Super(cclip, **ok_) if ok_ != sk else csup
    bits_g, bits_c = clip.format.bits_per_sample, cclip.format.bits_per_sample
    if args.vectors == 'mvgpu':
        gold = core.mvgpu.Analyse(gsup_old, delta=args.delta)
        cold = core.std.SetFrameProps(mvu_vectors(core, gold, cclip, args.frames), MVUtensilsAnalysisBitsPerSample=bits_c)
    else:
        cold = core.mvu.Analyse(csup_old, delta=args.delta, blksize=ok_['blksize'], overlap=ok_['overlap'])
        if args.vectors == 'random':
            th1, scd = scene_limits(cold.get_frame(0).props, 400, 51.0)
            # (vectors past the padding are refused by mvu)
            cold = random_vectors(core, cold, (args.pad - 1) * args.pel, th1, scd, args.seed)
        gold = core.std.SetFrameProps(gpu_vectors(core, cold, gsup_old), MVGPUtensilsAnalysisBitsPerSample=bits_g)

    rk = {k: getattr(args, k) for k in ('thsad', 'smooth', 'search', 'searchparam', 'mvlambda', 'chroma', 'pnew') if getattr(args, k) is not None}
    gout = first_clip(core.mvgpu.Recalculate(gsup, gold, **rk))
    cout = first_clip(core.mvu.Recalculate(csup, cold, **rk))

    # Every frame has vectors: one whose reference frame is outside the clip is recalculated from zero
    # vectors against the reference frame clamped to the clip (as mvu does it)
    frames = args.frames
    props = cout.get_frame(0).props
    nbx, nby = props['MVUtensilsAnalysisNBlkX'], props['MVUtensilsAnalysisNBlkY']
    records = core.std.GPUDownload(core.std.PropToClip(gout, prop='MVGPUtensilsAnalysisVectors'))
    differ = 0
    desc = 0
    worst = None
    for n in range(frames):
        a, b = gout.get_frame(n).props, cout.get_frame(n).props
        for k in b.keys():
            # (float: mvu's description is the quantized 16-bit clip's)
            if k.startswith('MVUtensilsAnalysis') and k not in ('MVUtensilsAnalysisVectors', 'MVUtensilsAnalysisSAD') and not (isfloat and k == 'MVUtensilsAnalysisBitsPerSample'):
                if a.get(k.replace('MVUtensils', 'MVGPUtensils', 1)) != b[k]:
                    desc += 1
                    if desc == 1:
                        print(f'frame {n}: {k} mvgpu {a.get(k.replace("MVUtensils", "MVGPUtensils", 1))}, mvu {b[k]}')
        has_g, has_c = 'MVGPUtensilsAnalysisVectors' in a, 'MVUtensilsAnalysisVectors' in b
        if has_g != has_c:
            print(f'frame {n}: mvgpu {"has" if has_g else "lacks"} vectors, mvu {"has" if has_c else "lacks"} them')
            sys.exit(1)
        if not has_c:
            continue
        rec = np.asarray(records.get_frame(n)[0]).view(np.int32)[:, :4 * nbx].reshape(nby * nbx, 4)
        packed = np.array(b['MVUtensilsAnalysisVectors'], np.int64)
        cx, cy = (packed & 0xFFFFFFFF).astype(np.uint32).view(np.int32), (packed >> 32).astype(np.int32)
        csad = np.array(b['MVUtensilsAnalysisSAD'], np.int64)
        bad = (rec[:, 0] != cx) | (rec[:, 1] != cy) | (rec[:, 2] != csad)
        differ += int(np.count_nonzero(bad))
        if worst is None and bad.any():
            i = int(np.nonzero(bad)[0][0])
            worst = (n, i % nbx, i // nbx, tuple(int(v) for v in rec[i, :3]), (int(cx[i]), int(cy[i]), int(csad[i])))
    total = frames * nbx * nby
    print(f'{args.format} pel {args.pel} {args.blksize}/{args.overlap} from {ok_["blksize"]}/{ok_["overlap"]} {args.vectors} vectors delta {args.delta}'
          + (f' {rk}' if rk else '') + f', {frames} frames: {differ} of {total} blocks differ'
          + (f'; first at frame {worst[0]} block ({worst[1]}, {worst[2]}): mvgpu {worst[3]}, mvu {worst[4]}' if worst else '')
          + (f'; {desc} description properties differ' if desc else ''))
    sys.exit(0 if differ == 0 and desc == 0 else 1)


if __name__ == '__main__':
    main()
