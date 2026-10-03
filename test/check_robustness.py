#!/usr/bin/env python3
"""mvgpu's checks of the clips its filters are given, against mvu's: both plugins must refuse the same
mistakes with the same message, or do the same with them.

  pad       Super with a padding the chroma subsampling doesn't divide (refused), and with one it does
            (an odd vertical padding at 4:2:2, taken)
  length    super, centersuper and vector clips shorter than the clip they go with, for every filter
            that pairs them (Recalculate: vectors shorter than the super)
  super     a super clip spliced from two supers made with different arguments (but the same frame
            size), refused at the first frame of the second
  vectors   a vector clip spliced from two analyses with different arguments, refused at the first
            frame with vectors of the second
  pairs     vector clips analysed at different bit depths passed to one Degrain or FlowInter
  reference Recalculate with a super shorter than the vector clip: the frames whose reference frame
            is past the super's end get no vectors, though the vector clip has some for them

    check_robustness.py --src clip.nv12 --size 1920x1080 --frames 10
"""
import argparse
import os
import sys

import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # an embedded Python leaves the script's directory out
from mvtest import format_clip, nv12_clip  # noqa: E402

LENGTH = 'must have at least as many frames as'
SPLICED_SUPER = 'a super clip frame was made with different Super arguments than its first frame'
SPLICED_VECTORS = 'a vector clip frame was made with different Analyse/Recalculate arguments than its first frame'


class Results:
    def __init__(self):
        self.cases = 0
        self.problems = 0

    def fail(self, message):
        self.problems += 1
        print('  ' + message)


def outcome(make, frames):
    """None when make() and its frames succeed, else the error"""
    try:
        clip = make()
        clip = clip[0] if isinstance(clip, list) else clip
        for n in frames:
            clip.get_frame(n)
        return None
    except vs.Error as e:
        return str(e)


def both_refuse(results, label, make_mvu, make_gpu, text, frames=(0,)):
    """Both plugins must refuse, with an error containing text"""
    results.cases += 1
    for name, make in (('mvu', make_mvu), ('mvgpu', make_gpu)):
        error = outcome(make, frames)
        if error is None:
            results.fail(f'{label}: {name} accepts it')
        elif text not in error:
            results.fail(f'{label}: {name} refuses with "{error}", without "{text}"')


def both_accept(results, label, make_mvu, make_gpu, frames=(0,)):
    results.cases += 1
    for name, make in (('mvu', make_mvu), ('mvgpu', make_gpu)):
        error = outcome(make, frames)
        if error is not None:
            results.fail(f'{label}: {name} refuses it: {error}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--plugin', help='the MVGPUtensils library to load, unless it autoloads')
    ap.add_argument('--mvu', default=r'C:\Libraries\mvutensils\msvc\x64\Release\MVUtensils.dll', help='the MVUtensils library, unless it autoloads')
    args = ap.parse_args()

    core = vs.core
    if args.plugin:
        core.std.LoadPlugin(args.plugin)
    if not hasattr(core, 'mvu'):
        core.std.LoadPlugin(args.mvu)
    w, h = (int(v) for v in args.size.split('x'))
    frames = args.frames
    clip = nv12_clip(core, args.src, w, h, frames)
    gclip = core.std.GPUUpload(clip)
    last = frames - 1
    mid = frames // 2
    results = Results()

    sk = dict(blksize=16, overlap=8, pel=2)
    csup, gsup = core.mvu.Super(clip, **sk), core.mvgpu.Super(gclip, **sk)
    cvec = core.mvu.AnalyseMany(csup, radius=1)
    gvec = core.mvgpu.AnalyseMany(gsup, radius=1)
    mvu, gpu = core.mvu, core.mvgpu

    # pad
    both_refuse(results, 'Super pad 7 at 4:2:0', lambda: mvu.Super(clip, pad=7, **sk), lambda: gpu.Super(gclip, pad=7, **sk),
                'pad must be divisible by the chroma subsampling')
    c422 = format_clip(core, clip, 'YUV422P8')
    both_refuse(results, 'Super pad 7x8 at 4:2:2', lambda: mvu.Super(c422, pad=[7, 8], **sk), lambda: gpu.Super(core.std.GPUUpload(c422), pad=[7, 8], **sk),
                'pad must be divisible by the chroma subsampling')
    both_accept(results, 'Super pad 8x9 at 4:2:2', lambda: mvu.Super(c422, pad=[8, 9], **sk), lambda: gpu.Super(core.std.GPUUpload(c422), pad=[8, 9], **sk))

    # length
    short = frames - 1
    for label, make_mvu, make_gpu, text in [
        ('Degrain vectors', lambda: mvu.Degrain(clip, csup, [cvec[0][:short], cvec[1]]), lambda: gpu.Degrain(gclip, gsup, [gvec[0][:short], gvec[1]]),
         'vectors ' + LENGTH + ' clip'),
        ('Degrain super', lambda: mvu.Degrain(clip, csup[:short], cvec), lambda: gpu.Degrain(gclip, gsup[:short], gvec), 'super ' + LENGTH + ' clip'),
        ('Degrain centersuper', lambda: mvu.Degrain(clip, csup, cvec, centersuper=csup[:short]),
         lambda: gpu.Degrain(gclip, gsup, gvec, centersuper=gsup[:short]), 'centersuper ' + LENGTH + ' clip'),
        ('Compensate vectors', lambda: mvu.Compensate(clip, csup, cvec[0][:short]), lambda: gpu.Compensate(gclip, gsup, gvec[0][:short]),
         'vectors ' + LENGTH + ' clip'),
        ('Compensate super', lambda: mvu.Compensate(clip, csup[:short], cvec[0]), lambda: gpu.Compensate(gclip, gsup[:short], gvec[0]),
         'super ' + LENGTH + ' clip'),
        ('Flow vectors', lambda: mvu.Flow(clip, csup, cvec[0][:short]), lambda: gpu.Flow(gclip, gsup, gvec[0][:short]), 'vectors ' + LENGTH + ' clip'),
        ('FlowBlur super', lambda: mvu.FlowBlur(clip, csup[:short], cvec), lambda: gpu.FlowBlur(gclip, gsup[:short], gvec), 'super ' + LENGTH + ' clip'),
        ('FlowInter mvfw', lambda: mvu.FlowInter(clip, csup, [cvec[0], cvec[1][:short]]), lambda: gpu.FlowInter(gclip, gsup, [gvec[0], gvec[1][:short]]),
         'vectors ' + LENGTH + ' clip'),
        ('FlowFPS mvbw', lambda: mvu.FlowFPS(clip, csup, [cvec[0][:short], cvec[1]], num=60, den=1),
         lambda: gpu.FlowFPS(gclip, gsup, [gvec[0][:short], gvec[1]], num=60, den=1), 'vectors ' + LENGTH + ' clip'),
        ('SCDetection vectors', lambda: mvu.SCDetection(clip, cvec[0][:short]), lambda: gpu.SCDetection(gclip, gvec[0][:short]),
         'vectors ' + LENGTH + ' clip'),
        ('Recalculate vectors', lambda: mvu.Recalculate(csup, cvec[0][:short]), lambda: gpu.Recalculate(gsup, gvec[0][:short]),
         'vectors ' + LENGTH + ' super'),
    ]:
        both_refuse(results, 'length: ' + label, make_mvu, make_gpu, text)

    # super: spliced from two supers of other grids, whose frames are the same size on both sides
    csup_b, gsup_b = core.mvu.Super(clip, blksize=8, overlap=4, pel=2), core.mvgpu.Super(gclip, blksize=8, overlap=4, pel=2)
    csplice = core.std.Splice([csup[:mid], csup_b[mid:]], mismatch=True)
    gsplice = core.std.Splice([gsup[:mid], gsup_b[mid:]], mismatch=True)
    later = range(mid, last)
    both_refuse(results, 'spliced super: Degrain', lambda: mvu.Degrain(clip, csplice, cvec), lambda: gpu.Degrain(gclip, gsplice, gvec), SPLICED_SUPER, later)
    both_refuse(results, 'spliced super: Analyse', lambda: mvu.Analyse(csplice, delta=1), lambda: gpu.Analyse(gsplice, delta=1), SPLICED_SUPER, later)
    both_refuse(results, 'spliced super: FlowInter', lambda: mvu.FlowInter(clip, csplice, cvec), lambda: gpu.FlowInter(gclip, gsplice, gvec), SPLICED_SUPER, later)
    both_refuse(results, 'spliced super: Recalculate', lambda: mvu.Recalculate(csplice, cvec[0]), lambda: gpu.Recalculate(gsplice, gvec[0]), SPLICED_SUPER, later)

    # vectors: spliced from analyses with other deltas, whose frames are the same size on both sides
    cvec2 = core.mvu.Analyse(csup, delta=2)
    gvec2 = core.mvgpu.Analyse(gsup, delta=2)
    cvsplice = core.std.Splice([cvec[0][:mid], cvec2[mid:]], mismatch=True)
    gvsplice = core.std.Splice([gvec[0][:mid], gvec2[mid:]], mismatch=True)
    vlater = range(mid, frames - 2)
    for label, make_mvu, make_gpu in [
        ('Compensate', lambda: mvu.Compensate(clip, csup, cvsplice), lambda: gpu.Compensate(gclip, gsup, gvsplice)),
        ('Degrain', lambda: mvu.Degrain(clip, csup, [cvsplice, cvec[1]]), lambda: gpu.Degrain(gclip, gsup, [gvsplice, gvec[1]])),
        ('Flow', lambda: mvu.Flow(clip, csup, cvsplice), lambda: gpu.Flow(gclip, gsup, gvsplice)),
        ('FlowInter', lambda: mvu.FlowInter(clip, csup, [cvsplice, cvec[1]]), lambda: gpu.FlowInter(gclip, gsup, [gvsplice, gvec[1]])),
        ('SADMask', lambda: mvu.SADMask(cvsplice), lambda: gpu.SADMask(gvsplice)),
        ('SCDetection', lambda: mvu.SCDetection(clip, cvsplice), lambda: gpu.SCDetection(gclip, gvsplice)),
        ('Recalculate', lambda: mvu.Recalculate(csup, cvsplice), lambda: gpu.Recalculate(gsup, gvsplice)),
    ]:
        both_refuse(results, 'spliced vectors: ' + label, make_mvu, make_gpu, SPLICED_VECTORS, vlater)

    # pairs analysed at different bit depths
    c16 = format_clip(core, clip, 'YUV420P16')
    g16 = core.std.GPUUpload(c16)
    csup16, gsup16 = core.mvu.Super(c16, **sk), core.mvgpu.Super(g16, **sk)
    cvec16, gvec16 = core.mvu.AnalyseMany(csup16, radius=1), core.mvgpu.AnalyseMany(gsup16, radius=1)
    both_refuse(results, 'pairs: Degrain 8 and 16 bits', lambda: mvu.Degrain(c16, csup16, [cvec[0], cvec16[1]]),
                lambda: gpu.Degrain(g16, gsup16, [gvec[0], gvec16[1]]), 'not compatible with each other')
    both_refuse(results, 'pairs: FlowInter 8 and 16 bits', lambda: mvu.FlowInter(c16, csup16, [cvec[0], cvec16[1]]),
                lambda: gpu.FlowInter(g16, gsup16, [gvec[0], gvec16[1]]), 'mvfw and mvbw must be compatible with each other')

    # reference: the super two frames shorter than the vector clip of delta 1
    results.cases += 1
    keep = frames - 2
    cr = mvu.Recalculate(csup[:keep], cvec[0])
    gr = gpu.Recalculate(gsup[:keep], gvec[0])
    for n in range(keep):
        has_c = 'MVUtensilsAnalysisVectors' in cr.get_frame(n).props
        has_g = gr.get_frame(n).props.get('MVGPUtensilsAnalysisHasVectors') == 1
        if has_c != has_g:
            results.fail(f'reference: Recalculate frame {n}: mvu {"has" if has_c else "lacks"} vectors, mvgpu {"has" if has_g else "lacks"} them')
        if n == keep - 1 and (has_c or has_g):
            results.fail(f'reference: Recalculate frame {n}, whose reference is past the super, has vectors')

    print(f'{frames} frames: {results.cases} cases, {results.problems} problems')
    sys.exit(0 if results.problems == 0 else 1)


if __name__ == '__main__':
    main()
