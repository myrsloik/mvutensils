#!/usr/bin/env python3
"""Wall-clock throughput of mvgpu on a GPU source: the source alone, then with Super, then with
AnalyseMany, then Degrain on top, frames requested in parallel as a script would.

    bench.py SOURCE [--start 0] [--frames 300] [--radius 2] [--blksize 16 --overlap 8] [--pel 2] [--plugin PATH] [--vram MB]

SOURCE is anything bs.VideoSource opens as 8-bit 4:2:0; it decodes straight to GPU frames (gpu=True).
"""
import argparse
import os
import time

import vapoursynth as vs


def run(clip, label, unit):
    for _ in clip[:8].frames():  # warm up: indexing, pipelines, caches
        pass
    start = time.perf_counter()
    count = sum(1 for _ in clip.frames())
    secs = time.perf_counter() - start
    print(f'  {label:30} {count / secs:8.1f} {unit}/s  ({secs * 1000 / count:.2f} ms each)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--blksize', type=int, default=16)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--badsad', type=int, default=1000)
    ap.add_argument('--badrange', type=int, default=40)
    ap.add_argument('--plugin', help='the MVGPUtensils library to load, unless it autoloads')
    ap.add_argument('--vram', type=int, help="the core's GPU frame cache limit, MB: at 4K its default can leave the decoder too little VRAM")
    args = ap.parse_args()

    core = vs.core
    if args.plugin:
        core.std.LoadPlugin(args.plugin)
    if args.vram:
        core.max_vram_cache_size = args.vram
    src = core.bs.VideoSource(args.source, gpu=True)[args.start:args.start + args.frames]
    print(f'{os.path.basename(args.source)}: {src.width}x{src.height} {src.format.name}, {src.num_frames} frames, {core.num_threads} threads')
    run(src, 'source', 'frames')
    sup = core.mvgpu.Super(src, blksize=args.blksize, overlap=args.overlap, pel=args.pel)
    run(sup, 'source + Super', 'frames')
    fields = core.mvgpu.AnalyseMany(sup, radius=args.radius, badsad=args.badsad, badrange=args.badrange)
    run(core.std.Interleave(fields), f'+ AnalyseMany(radius={args.radius})', 'fields')
    run(core.mvgpu.Degrain(src, sup, fields), '+ Degrain', 'frames')


if __name__ == '__main__':
    main()
