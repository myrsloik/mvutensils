#!/usr/bin/env python3
"""mvgpu.Super + mvgpu.AnalyseMany (or Analyse per delta) against the CPU reference, byte for byte.

The reference is test/reference/reference.cpp, the search in plain C++. With --reference the script
runs it on the frames it gives mvgpu, dumped as raw planar frames, with the same arguments; with
--ref it reads a vector file the reference wrote before. Every field the reference holds is
compared: x, y and SAD of every block.

    check_reference.py --src noisy.nv12 --size 1920x1080 --frames 52 --reference reference.exe
                       [--format YUV444P8] [--blksize 16] [--overlap 8] [--pel 2] [--pad 16]
                       [--radius 2] [--delta 1] [--standalone] [--chroma 0] [--plevel 2]
                       [--mvlambda 1000] [--lsad 400] [--badsad 1000] [--badrange 40] [--badstep 2]
                       [--crop WxH] [--work DIR]

--format converts the 8-bit 4:2:0 source first (mvtest.format_clip: resize.Bicubic, shifted a quarter
pixel to more than 8 bits). --standalone makes an Analyse per delta, without chained or inverted
seeds, instead of AnalyseMany. --crop crops the frames first, for grids that end inside a block.
--work keeps the dumped frames and the reference's vectors in that directory instead of a
temporary one.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # an embedded Python leaves the script's directory out
from mvtest import format_clip, nv12_clip  # noqa: E402


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


def dump(clip, path):
    """The clip's frames as raw planar Y, U, V"""
    with open(path, 'wb') as f:
        for frame in clip.frames():
            for p in range(frame.format.num_planes):
                f.write(np.asarray(frame[p]).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--format', default='YUV420P8', help='a VapourSynth preset name, 4:2:0 or 4:4:4 at 8 to 16 bits: YUV420P8, YUV444P16, ...')
    ref = ap.add_mutually_exclusive_group(required=True)
    ref.add_argument('--reference', help='the reference executable, run with matching arguments')
    ref.add_argument('--ref', help='a vector file the reference wrote')
    ap.add_argument('--blksize', type=int, default=16)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--pad', type=int, default=16)
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--delta', type=int, default=1, help="AnalyseMany's delta, the frames between its fields")
    ap.add_argument('--standalone', action='store_true', help='an Analyse per delta: no chained or inverted seeds')
    ap.add_argument('--chroma', type=int, default=1)
    ap.add_argument('--plevel', type=int, default=1)
    ap.add_argument('--mvlambda', type=int, default=1000)
    ap.add_argument('--lsad', type=int, default=400)
    ap.add_argument('--badsad', type=int, default=1000)
    ap.add_argument('--badrange', type=int, default=40)
    ap.add_argument('--badstep', type=int, default=2)
    ap.add_argument('--crop', help='WxH: crop the frames to this size first')
    ap.add_argument('--work', help='a directory to keep the dumped frames and the reference vectors in')
    ap.add_argument('--verbose', action='store_true', help='list every field with differing blocks')
    ap.add_argument('--dump', help="write mvgpu's fields to this file, in the reference's format")
    ap.add_argument('--plugin', help='the MVGPUtensils library to load, unless it autoloads')
    args = ap.parse_args()

    core = vs.core
    if args.plugin:
        core.std.LoadPlugin(args.plugin)
    w, h = (int(v) for v in args.size.split('x'))
    clip = format_clip(core, nv12_clip(core, args.src, w, h, args.frames), args.format)
    if args.crop:
        w, h = (int(v) for v in args.crop.split('x'))
        clip = core.std.CropAbs(clip, w, h)

    search = dict(chroma=args.chroma, plevel=args.plevel, mvlambda=args.mvlambda, lsad=args.lsad, badsad=args.badsad, badrange=args.badrange,
                  badstep=args.badstep)
    if args.reference:
        work = args.work or tempfile.mkdtemp(prefix='mvgpu_reference_')
        os.makedirs(work, exist_ok=True)
        frames, vectors = os.path.join(work, 'frames.yuv'), os.path.join(work, 'reference.bin')
        cmd = [args.reference, '--src', frames, '--size', f'{w}x{h}', '--frames', str(args.frames), '--out', vectors,
               '--format', 'gray' if clip.format.color_family == vs.GRAY else '420' if clip.format.subsampling_w else '444',
               '--bits', str(clip.format.bits_per_sample),
               '--blksize', str(args.blksize), '--overlap', str(args.overlap), '--pel', str(args.pel), '--pad', str(args.pad),
               '--radius', str(args.radius), '--delta', str(args.delta)] + (['--standalone'] if args.standalone else [])
        for k, v in search.items():
            cmd += [f'--{k}', str(v)]
        try:
            dump(clip, frames)
            subprocess.run(cmd, check=True)
            expected = load_fields(vectors)
        finally:
            if not args.work:
                for f in (frames, vectors):
                    if os.path.exists(f):
                        os.remove(f)
                os.rmdir(work)
    else:
        expected = load_fields(args.ref)

    sup = core.mvgpu.Super(core.std.GPUUpload(clip), blksize=args.blksize, overlap=args.overlap, pel=args.pel, pad=args.pad)
    if args.standalone:
        fields = [core.mvgpu.Analyse(sup, delta=s * r * args.delta, **search) for r in range(1, args.radius + 1) for s in (1, -1)]
    else:
        fields = core.mvgpu.AnalyseMany(sup, radius=args.radius, delta=args.delta, **search)

    compared = differ = blocks = missing = 0
    first = None
    dumped = open(args.dump, 'wb') if args.dump else None
    for an in fields:
        delta = an.get_frame(0).props['MVGPUtensilsAnalysisDeltaFrame']
        # The field's frames are its vectors' records; those whose reference frame is outside the
        # clip have none
        start, end = max(0, -delta), min(args.frames, args.frames - delta)
        vectors = core.std.GPUDownload(an)
        for n in range(start, end):
            if (n, delta) not in expected:
                missing += 1
                continue
            nbx, nby, exp = expected[(n, delta)]
            rec = np.asarray(vectors.get_frame(n)[0]).view(np.int32)[:, :4 * nbx].reshape(nby * nbx, 4)
            if dumped:
                dumped.write(np.array([n, delta, nbx, nby, args.pel, 1], np.int32).tobytes())
                dumped.write(np.ascontiguousarray(rec[:, :3].T).tobytes())
            bad = (rec[:, :3].T != exp).any(axis=0)
            if bad.any() and first is None:
                b = int(np.argmax(bad))
                first = (n, delta, b % nbx, b // nbx, tuple(int(v) for v in rec[b, :3]), tuple(int(v) for v in exp[:, b]))
            differ += int(np.count_nonzero(bad))
            if args.verbose and bad.any():
                b = int(np.argmax(bad))
                print(f'field ({n}, {delta:+d}): {int(np.count_nonzero(bad))} blocks differ, first ({b % nbx}, {b // nbx}): mvgpu {tuple(int(v) for v in rec[b, :3])}, '
                      f'reference {tuple(int(v) for v in exp[:, b])}')
            blocks += nbx * nby
            compared += 1
    if dumped:
        dumped.close()
    print(f'{compared} fields, {differ} of {blocks} blocks differ' + (f', {missing} fields missing from the reference' if missing else '')
          + (f'; first in field ({first[0]}, {first[1]:+d}) at block ({first[2]}, {first[3]}): mvgpu {first[4]}, reference {first[5]}' if first else ''))
    sys.exit(0 if differ == 0 and missing == 0 and compared > 0 else 1)


if __name__ == '__main__':
    main()
