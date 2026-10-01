#!/usr/bin/env python3
"""mvgpu.Super + mvgpu.AnalyseMany (or Analyse per delta) against the CPU reference, byte for byte.

The reference is a vector file from the prototype's CPU implementation (probe/seedrefine.cpp,
--out), run on the same raw 8-bit NV12 frames with matching settings, for example

    seedrefine --src noisy.nv12 --size 1920x1080 --frames 52 --grid mvu_full_16_8.bin
               --blksize 16 --overlap 8 --seeds pyramid --fallback 40 --fallback-step 2
               --badsad 4000 --pairs 1 --out ref.bin

which corresponds to

    check_reference.py --src noisy.nv12 --size 1920x1080 --frames 52 --ref ref.bin
                       --blksize 16 --overlap 8 --badsad 1000 --badrange 40 --badstep 2

(seedrefine's --badsad is the threshold for the block itself, mvgpu's per 8x8 block, so it is
scaled by blksize^2 / 64; seedrefine --no-chain matches --standalone). Every field the reference
holds is compared: x, y and SAD of every block.
"""
import argparse
import sys

import numpy as np
import vapoursynth as vs


def load_fields(path):
    """The reference's fields: (n, d) -> (nbx, nby, int32 array of x, y and SAD rows)"""
    out = {}
    data = open(path, 'rb').read()
    pos = 0
    while pos < len(data):
        n, d, nbx, nby, _pel, _scale = np.frombuffer(data, np.int32, 6, pos)
        pos += 24
        count = int(nbx) * int(nby)
        out[(int(n), int(d))] = (int(nbx), int(nby), np.frombuffer(data, np.int32, 3 * count, pos).reshape(3, count))
        pos += 12 * count
    return out


def nv12_clip(core, path, w, h, frames):
    """Raw NV12 frames as a YUV420P8 clip"""
    size = w * h * 3 // 2
    raw = np.memmap(path, np.uint8, 'r', shape=(frames, size))
    blank = core.std.BlankClip(width=w, height=h, format=vs.YUV420P8, length=frames)

    def fill(n, f):
        out = f.copy()
        np.asarray(out[0])[:, :] = raw[n, :w * h].reshape(h, w)
        uv = raw[n, w * h:].reshape(h // 2, w // 2, 2)
        np.asarray(out[1])[:, :] = uv[:, :, 0]
        np.asarray(out[2])[:, :] = uv[:, :, 1]
        return out

    return core.std.ModifyFrame(blank, blank, fill)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--ref', required=True, help='the CPU reference vector file')
    ap.add_argument('--blksize', type=int, default=16)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--badsad', type=int, default=1000)
    ap.add_argument('--badrange', type=int, default=40)
    ap.add_argument('--badstep', type=int, default=2)
    ap.add_argument('--standalone', action='store_true', help='an Analyse per delta: no chained or inverted seeds')
    ap.add_argument('--plugin', help='the MVGPUtensils library to load, unless it autoloads')
    args = ap.parse_args()

    core = vs.core
    if args.plugin:
        core.std.LoadPlugin(args.plugin)
    w, h = (int(v) for v in args.size.split('x'))
    ref = load_fields(args.ref)
    clip = core.std.GPUUpload(nv12_clip(core, args.src, w, h, args.frames))
    sup = core.mvgpu.Super(clip, blksize=args.blksize, overlap=args.overlap)
    kw = dict(badsad=args.badsad, badrange=args.badrange, badstep=args.badstep)
    if args.standalone:
        fields = [core.mvgpu.Analyse(sup, delta=s * r, **kw) for r in range(1, args.radius + 1) for s in (1, -1)]
    else:
        fields = core.mvgpu.AnalyseMany(sup, radius=args.radius, **kw)

    compared = differ = blocks = missing = 0
    for an in fields:
        delta = an.get_frame(0).props['MVGPUtensilsAnalysisDeltaFrame']
        # Frames whose reference frame is outside the clip carry no vectors, and PropToClip takes
        # its format from its first frame
        start, end = max(0, -delta), min(args.frames, args.frames - delta)
        vectors = core.std.GPUDownload(core.std.PropToClip(an[start:end], prop='MVGPUtensilsAnalysisVectors'))
        for n in range(start, end):
            if (n, delta) not in ref:
                missing += 1
                continue
            nbx, nby, expected = ref[(n, delta)]
            rec = np.asarray(vectors.get_frame(n - start)[0]).view(np.int32)[:, :4 * nbx].reshape(nby * nbx, 4)
            differ += int(np.count_nonzero((rec[:, :3].T != expected).any(axis=0)))
            blocks += nbx * nby
            compared += 1
    print(f'{compared} fields, {differ} of {blocks} blocks differ' + (f', {missing} fields missing from the reference' if missing else ''))
    sys.exit(0 if differ == 0 and compared > 0 else 1)


if __name__ == '__main__':
    main()
