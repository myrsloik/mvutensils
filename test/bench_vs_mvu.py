#!/usr/bin/env python3
"""mvgpu's speed against mvu's: the same pipelines on both plugins, frames per second and the CPU threads each
keeps busy (bench.py measures mvgpu alone, stage by stage, on a source decoded to GPU frames).

    bench_vs_mvu.py --src clip.nv12 --size 1920x1080 --frames 52 [--format YUV420P16] [--length 300]
                    [--blksize 16] [--overlap 8] [--pel 2] [--radius 2] [--runs 2]
                    [--pipelines source transfer search degrain hybrid flowfps] [--threads 16] [--cache 8192]
                    [--vram MB] [--plugin MVGPUtensils.dll] [--mvu MVUtensils.dll]

The clip (raw 8-bit NV12 frames, converted to --format by mvtest.format_clip) is rendered into memory once and
served from there, played forth and back to --length frames, so that neither the source nor the conversion
counts. Each pipeline runs --runs times, each time in a process of its own, so that no cache or GPU memory
carries over, and the median is printed; its frames are requested core.num_threads at a time, as vspipe requests
them, and the first tenth of them (the filters' setup and the kernels' compiles among them) aren't timed. mvgpu's
pipelines include uploading the clip and downloading the result. The pipelines:

  source    the frames from memory alone
  transfer  GPUUpload and GPUDownload
  search    Super and AnalyseMany, the vectors on the CPU (mvgpu's through ToMVU)
  degrain   Super, AnalyseMany and Degrain
  hybrid    mvgpu's Super and AnalyseMany, then mvgpu.ToMVU and mvu.Degrain on the CPU with an onelevel=True mvu
            super: the same frames as mvgpu's Degrain
  flowfps   Super, AnalyseMany of radius 1 and FlowFPS doubling the frame rate, in output frames per second
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # an embedded Python leaves the script's directory out
from mvtest import format_clip, nv12_clip  # noqa: E402

# Each pipeline's measurements: mvu's and mvgpu's, or the one there is
SIDES = {'source': ['memory'], 'transfer': ['gpu'], 'search': ['mvu', 'mvgpu'], 'degrain': ['mvu', 'mvgpu'], 'hybrid': ['hybrid'],
         'flowfps': ['mvu', 'mvgpu']}


def measure(args, side, pipeline):
    """The pipeline in this process: frames per second (source frames; output frames for flowfps), the CPU threads
    busy on average and the CPU milliseconds per source frame"""
    core = vs.core
    core.max_cache_size = args.cache
    if args.vram:
        core.max_vram_cache_size = args.vram
    if args.threads:
        core.num_threads = args.threads
    if args.plugin:
        core.std.LoadPlugin(args.plugin)
    if not hasattr(core, 'mvu'):
        core.std.LoadPlugin(args.mvu)
    w, h = (int(v) for v in args.size.split('x'))
    src = format_clip(core, nv12_clip(core, args.src, w, h, args.frames), args.format)
    frames = [src.get_frame(n) for n in range(args.frames)]
    period = max(1, 2 * args.frames - 2)

    def from_memory(n, f):
        i = n % period
        return frames[i if i < args.frames else period - i]

    blank = core.std.BlankClip(src, length=args.length)
    clip = core.std.ModifyFrame(blank, blank, from_memory)
    sk = dict(blksize=args.blksize, overlap=args.overlap, pel=args.pel)
    doubled = dict(num=2 * clip.fps.numerator, den=clip.fps.denominator)
    if pipeline == 'source':
        out = clip
    elif pipeline == 'transfer':
        out = core.std.GPUDownload(core.std.GPUUpload(clip))
    elif side == 'mvu':
        sup = core.mvu.Super(clip, **sk)
        if pipeline == 'search':
            out = core.std.Interleave(core.mvu.AnalyseMany(sup, radius=args.radius))
        elif pipeline == 'degrain':
            out = core.mvu.Degrain(clip, sup, core.mvu.AnalyseMany(sup, radius=args.radius))
        else:
            out = core.mvu.FlowFPS(clip, sup, core.mvu.AnalyseMany(sup, radius=1)[:2], **doubled)
    else:
        g = core.std.GPUUpload(clip)
        sup = core.mvgpu.Super(g, **sk)
        if pipeline == 'search':
            out = core.std.Interleave(core.mvgpu.ToMVU(core.mvgpu.AnalyseMany(sup, radius=args.radius)))
        elif pipeline == 'degrain':
            out = core.std.GPUDownload(core.mvgpu.Degrain(g, sup, core.mvgpu.AnalyseMany(sup, radius=args.radius)))
        elif pipeline == 'hybrid':
            vectors = core.mvgpu.ToMVU(core.mvgpu.AnalyseMany(sup, radius=args.radius))
            out = core.mvu.Degrain(clip, core.mvu.Super(clip, onelevel=True, **sk), vectors)
        else:
            out = core.std.GPUDownload(core.mvgpu.FlowFPS(g, sup, core.mvgpu.AnalyseMany(sup, radius=1)[:2], **doubled))
    per = len(out) // args.length  # output frames per source frame: the interleaved fields, FlowFPS's doubling
    warm = max(8, args.length // 10) * per
    for i, _ in enumerate(out.frames()):
        if i == warm - 1:
            t0, c0 = time.perf_counter(), time.process_time()
    secs, cpu = time.perf_counter() - t0, time.process_time() - c0
    timed = len(out) - warm
    return timed / secs / (1 if pipeline == 'flowfps' else per), cpu / secs, 1000 * cpu / (timed / per)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True, help='the frames in --src')
    ap.add_argument('--format', default='YUV420P16', help='a VapourSynth preset name: YUV420P16, YUV420P8, YUV444PS, ...')
    ap.add_argument('--length', type=int, default=300, help='the frames each pipeline runs, the clip played forth and back')
    ap.add_argument('--blksize', type=int, default=16)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--runs', type=int, default=2, help='processes per measurement, of which the median is printed')
    ap.add_argument('--pipelines', nargs='+', choices=list(SIDES), default=list(SIDES))
    ap.add_argument('--threads', type=int, help="the core's threads (its default: the CPU's)")
    ap.add_argument('--cache', type=int, default=8192, help="the core's frame cache, MB")
    ap.add_argument('--vram', type=int, help="the core's GPU frame cache, MB")
    ap.add_argument('--plugin', help='the MVGPUtensils library to load, unless it autoloads')
    ap.add_argument('--mvu', default=r'C:\Libraries\mvutensils\msvc\x64\Release\MVUtensils.dll', help='the MVUtensils library, unless it autoloads')
    ap.add_argument('--run', nargs=2, metavar=('SIDE', 'PIPELINE'), help=argparse.SUPPRESS)  # one measurement, in a child process
    args = ap.parse_args()
    if args.length < 20:
        ap.error('--length must be at least 20')
    if args.run:
        print('RESULT', *measure(args, *args.run), flush=True)
        return

    print(f'{args.size} {args.format}, {args.blksize}/{args.overlap} pel {args.pel} radius {args.radius}, {args.length} frames, '
          f'{args.threads or os.cpu_count()} threads, median of {args.runs} runs', flush=True)
    child = [sys.executable, os.path.abspath(__file__)] + sys.argv[1:]
    fps = {}
    for pipeline in args.pipelines:
        for side in SIDES[pipeline]:
            runs = []
            for _ in range(args.runs):
                r = subprocess.run(child + ['--run', side, pipeline], capture_output=True, text=True)
                line = next((s for s in r.stdout.splitlines() if s.startswith('RESULT ')), None)
                if line is None:
                    sys.exit(f'{pipeline} on {side} failed:\n{r.stderr[-3000:]}')
                runs.append([float(v) for v in line.split()[1:]])
            f, busy, ms = (statistics.median(c) for c in zip(*runs))
            fps[pipeline, side] = f
            mvu = fps.get(('degrain' if pipeline == 'hybrid' else pipeline, 'mvu'))
            ratio = f', {f / mvu:.2f}x mvu' + ("'s degrain" if pipeline == 'hybrid' else '') if mvu and side != 'mvu' else ''
            unit = 'output frames' if pipeline == 'flowfps' else 'frames'
            print(f'  {pipeline:9} {side:7} {f:8.1f} {unit}/s, {busy:5.1f} threads busy, {ms:7.1f} CPU ms per source frame{ratio}', flush=True)


if __name__ == '__main__':
    main()
