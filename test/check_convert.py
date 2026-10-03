#!/usr/bin/env python3
"""mvgpu.ToMVU and mvgpu.FromMVU, which move vector clips between mvgpu's form and mvu's.

  mvgpu -> mvu  mvgpu.AnalyseMany's vectors (and with --recalculate mvgpu.Recalculate's of them)
                through ToMVU, against the records they come from: mvu's description and arrays on
                every frame that has vectors, no arrays on the others, mvgpu's properties kept; then
                back through FromMVU, which must give the records and every property again.
  mvu -> mvgpu  mvu.AnalyseMany's (and mvu.Recalculate's) through FromMVU, against mvu's arrays:
                the records, mvgpu's description and the super's, the source's own properties kept
                and mvu's gone (its super's planes among them); then back through ToMVU, which must
                give mvu's description and arrays again.
  errors        either plugin's vectors where the other's belong, a super of another grid, vectors
                that would read outside the padded frame, negative SADs, a vector clip without the
                description of its super.

Both filters only move the vectors and SADs, so the filters of both plugins make the same frames from
them (check_degrain, check_flow, check_masks and check_recalculate run mvu's filters on mvgpu's
vectors and mvgpu's on mvu's through them). With --analyse8 the vectors come from an 8-bit copy of a
high bit depth clip, and FromMVU gets the clip's own super: vectors analysed on another bit depth
serve it, as in mvu.

    check_convert.py --src clip.nv12 --size 1920x1080 --frames 10 [--format YUV420P8] [--crop 1914x1074]
                     [--pel 2] [--blksize 16] [--overlap 8] [--radius 2] [--recalculate 8 4]
                     [--analyse8] [--mvuprefix Other]
"""
import argparse
import os
import sys

import numpy as np
import vapoursynth as vs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # an embedded Python leaves the script's directory out
from mvtest import eight_bit, format_clip, nv12_clip  # noqa: E402

GPU = 'MVGPUtensils'
DESCRIPTION = ['Width', 'Height', 'RealWidth', 'RealHeight', 'HPad', 'VPad', 'Pel', 'Levels', 'Chroma', 'XRatioUV', 'YRatioUV',
               'BlkSizeX', 'BlkSizeY', 'OverlapX', 'OverlapY', 'NBlkX', 'NBlkY', 'DeltaFrame', 'BitsPerSample']
SOURCE_KEY = 'ConvertCheckSource'  # a property of the source's frames, which the vector frames carry on both sides


class Problems:
    def __init__(self):
        self.count = 0
        self.error_cases = 0

    def add(self, message):
        self.count += 1
        if self.count <= 10:
            print('  ' + message)


def records(frame, nbx, nby):
    """A frame of records as (blocks, 4) int32"""
    return np.asarray(frame[0]).view(np.int32)[:, :4 * nbx].reshape(nby * nbx, 4)


def unpack(props, prefix):
    """mvu's arrays as x, y and SAD per block"""
    packed = np.array(props[prefix + 'AnalysisVectors'], np.int64)
    return (packed & 0xFFFFFFFF).astype(np.uint32).view(np.int32), (packed >> 32).astype(np.int32), np.array(props[prefix + 'AnalysisSAD'], np.int64)


def first(c):
    return c[0] if isinstance(c, list) else c


def check_to_mvu(core, field, frames, mvuprefix, label, problems):
    """ToMVU's frames against the records they come from; returns ToMVU's clip"""
    conv = core.mvgpu.ToMVU(field, mvuprefix=mvuprefix)
    down = core.std.GPUDownload(field)
    for n in range(frames):
        g, c = down.get_frame(n), conv.get_frame(n)
        gp, cp = g.props, c.props
        for k in DESCRIPTION:
            if cp.get(mvuprefix + 'Analysis' + k) != gp[GPU + 'Analysis' + k]:
                problems.add(f'{label} ToMVU frame {n}: {mvuprefix}Analysis{k} {cp.get(mvuprefix + "Analysis" + k)}, the records have {gp[GPU + "Analysis" + k]}')
        has = gp[GPU + 'AnalysisHasVectors'] == 1
        if (mvuprefix + 'AnalysisVectors' in cp) != has or (mvuprefix + 'AnalysisSAD' in cp) != has:
            problems.add(f'{label} ToMVU frame {n}: arrays {"missing" if has else "present"}, HasVectors {int(has)}')
            continue
        for k in gp.keys():  # still mvgpu's vector frame
            if k.startswith(GPU) and cp.get(k) != gp[k]:
                problems.add(f'{label} ToMVU frame {n}: {k} changed')
        if has:
            nbx, nby = gp[GPU + 'AnalysisNBlkX'], gp[GPU + 'AnalysisNBlkY']
            rec = records(g, nbx, nby)
            x, y, sad = unpack(cp, mvuprefix)
            bad = int(np.count_nonzero((rec[:, 0] != x) | (rec[:, 1] != y) | (rec[:, 2] != sad)))
            if bad:
                problems.add(f'{label} ToMVU frame {n}: {bad} blocks differ from the records')
    return conv


def check_back_to_gpu(core, field, conv, gsup, frames, mvuprefix, label, problems):
    """FromMVU of ToMVU's clip against the vectors ToMVU got: the same records and properties"""
    back = core.std.GPUDownload(core.mvgpu.FromMVU(conv, gsup, mvuprefix=mvuprefix))
    down = core.std.GPUDownload(field)
    for n in range(frames):
        g, b = down.get_frame(n), back.get_frame(n)
        gp, bp = g.props, b.props
        keys = {k for k in gp.keys() if k.startswith(GPU)} | {k for k in bp.keys() if k.startswith(GPU)}
        for k in sorted(keys):
            if gp.get(k) != bp.get(k):
                problems.add(f'{label} FromMVU(ToMVU) frame {n}: {k} {bp.get(k)}, was {gp.get(k)}')
        if gp[GPU + 'AnalysisHasVectors'] == 1:
            nbx, nby = gp[GPU + 'AnalysisNBlkX'], gp[GPU + 'AnalysisNBlkY']
            bad = int(np.count_nonzero((records(g, nbx, nby) != records(b, nbx, nby)).any(axis=1)))
            if bad:
                problems.add(f'{label} FromMVU(ToMVU) frame {n}: {bad} records differ')


def check_from_mvu(core, cfield, gsup, frames, mvuprefix, label, problems):
    """FromMVU's frames against mvu's arrays; returns FromMVU's clip"""
    conv = core.mvgpu.FromMVU(cfield, gsup, mvuprefix=mvuprefix)
    down = core.std.GPUDownload(conv)
    super_props = {k: v for k, v in gsup.get_frame(0).props.items() if k.startswith(GPU + 'Super')}
    for n in range(frames):
        c, g = cfield.get_frame(n), down.get_frame(n)
        cp, gp = c.props, g.props
        for k in DESCRIPTION:
            if gp.get(GPU + 'Analysis' + k) != cp[mvuprefix + 'Analysis' + k]:
                problems.add(f'{label} FromMVU frame {n}: {GPU}Analysis{k} {gp.get(GPU + "Analysis" + k)}, mvu has {cp[mvuprefix + "Analysis" + k]}')
        for k, v in super_props.items():
            if gp.get(k) != v:
                problems.add(f'{label} FromMVU frame {n}: {k} {gp.get(k)}, the super has {v}')
        if gp.get(SOURCE_KEY) != cp[SOURCE_KEY]:
            problems.add(f'{label} FromMVU frame {n}: the source property {SOURCE_KEY} is lost')
        if mvuprefix != GPU:
            left = [k for k in gp.keys() if k.startswith(mvuprefix + 'Analysis') or k.startswith(mvuprefix + 'Super')]
            if left:
                problems.add(f'{label} FromMVU frame {n}: mvu\'s properties left: {left[:3]}')
        has = mvuprefix + 'AnalysisVectors' in cp
        if gp.get(GPU + 'AnalysisHasVectors') != int(has):
            problems.add(f'{label} FromMVU frame {n}: HasVectors {gp.get(GPU + "AnalysisHasVectors")}, mvu {"has" if has else "lacks"} vectors')
            continue
        if has:
            nbx, nby = cp[mvuprefix + 'AnalysisNBlkX'], cp[mvuprefix + 'AnalysisNBlkY']
            rec = records(g, nbx, nby)
            x, y, sad = unpack(cp, mvuprefix)
            bad = int(np.count_nonzero((rec[:, 0] != x) | (rec[:, 1] != y) | (rec[:, 2] != sad) | (rec[:, 3] != 0)))
            if bad:
                problems.add(f'{label} FromMVU frame {n}: {bad} records differ from mvu\'s arrays')
    return conv


def check_back_to_mvu(core, cfield, conv, frames, mvuprefix, label, problems):
    """ToMVU of FromMVU's clip against mvu's vectors: the same description and arrays"""
    back = core.mvgpu.ToMVU(conv, mvuprefix=mvuprefix)
    for n in range(frames):
        cp, bp = cfield.get_frame(n).props, back.get_frame(n).props
        for k in DESCRIPTION + ['Vectors', 'SAD']:
            key = mvuprefix + 'Analysis' + k
            a, b = cp.get(key), bp.get(key)
            if (list(a) if isinstance(a, (list, tuple)) else a) != (list(b) if isinstance(b, (list, tuple)) else b):
                problems.add(f'{label} ToMVU(FromMVU) frame {n}: {key} differs')


def expect_error(label, make, text, problems, frame=2):
    """make() must fail, when created or at frame, with an error containing text"""
    problems.error_cases += 1
    try:
        clip = first(make())
        clip.get_frame(frame)
        problems.add(f'error case {label}: no error')
    except vs.Error as e:
        if text not in str(e):
            problems.add(f'error case {label}: "{e}" lacks "{text}"')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, required=True)
    ap.add_argument('--format', default='YUV420P8', help='a VapourSynth preset name: YUV420P8, YUV444P16, GRAYS, ...')
    ap.add_argument('--crop', help='WxH: crop the frames to this size first, for grids that end inside a block')
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--blksize', type=int, default=16)
    ap.add_argument('--overlap', type=int, default=8)
    ap.add_argument('--radius', type=int, default=2)
    ap.add_argument('--recalculate', type=int, nargs=2, metavar=('BLKSIZE', 'OVERLAP'), help="also both Recalculates' vectors, for this grid")
    ap.add_argument('--analyse8', action='store_true', help='analyse an 8-bit copy of the clip')
    ap.add_argument('--mvuprefix', default='MVUtensils', help="mvu's property prefix on both sides")
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
    clip = core.std.SetFrameProps(clip, **{SOURCE_KEY: 7})
    gclip = core.std.GPUUpload(clip)
    frames = args.frames
    mp = args.mvuprefix

    sk = dict(blksize=args.blksize, overlap=args.overlap, pel=args.pel)
    gsup = core.mvgpu.Super(gclip, **sk)
    aclip = eight_bit(core, clip) if args.analyse8 else clip
    gasup = core.mvgpu.Super(core.std.GPUUpload(aclip), **sk) if args.analyse8 else gsup
    casup = core.mvu.Super(aclip, prefix=mp, **sk)
    gfields = core.mvgpu.AnalyseMany(gasup, radius=args.radius)
    cfields = core.mvu.AnalyseMany(casup, radius=args.radius, prefix=mp)
    # (field clip, its super, label): the vectors either plugin's filters would take
    gsets = [(f, gasup, f'mvgpu delta {f.get_frame(0).props[GPU + "AnalysisDeltaFrame"]}') for f in gfields]
    csets = [(f, gsup, f'mvu delta {f.get_frame(0).props[mp + "AnalysisDeltaFrame"]}') for f in cfields]
    if args.recalculate:
        rk = dict(sk, blksize=args.recalculate[0], overlap=args.recalculate[1])
        gsup_r = core.mvgpu.Super(core.std.GPUUpload(aclip), **rk)
        csup_r = core.mvu.Super(aclip, prefix=mp, **rk)
        gsets += [(first(core.mvgpu.Recalculate(gsup_r, f)), gsup_r, 'mvgpu recalculated ' + label) for f, _, label in gsets[:2]]
        csets += [(first(core.mvu.Recalculate(csup_r, f, prefix=mp)), core.mvgpu.Super(gclip, **rk), 'mvu recalculated ' + label) for f, _, label in csets[:2]]

    problems = Problems()
    for field, sup, label in gsets:
        conv = check_to_mvu(core, field, frames, mp, label, problems)
        check_back_to_gpu(core, field, conv, sup, frames, mp, label, problems)
    for field, sup, label in csets:
        conv = check_from_mvu(core, field, sup, frames, mp, label, problems)
        check_back_to_mvu(core, field, conv, frames, mp, label, problems)

    # The errors: each plugin's vectors where the other's belong, with or without a matching prefix
    gf, cf = gfields[0], cfields[0]
    hint = 'FromMVU' + (f" and mvuprefix='{mp}'" if mp != 'MVUtensils' else '')
    expect_error('mvgpu.Degrain on mvu vectors', lambda: core.mvgpu.Degrain(gclip, gsup, cfields), hint, problems)
    expect_error("mvgpu.Degrain on mvu vectors, mvu's prefix",
                 lambda: core.mvgpu.Degrain(gclip, core.mvgpu.Super(gclip, prefix=mp, **sk), cfields, prefix=mp), hint, problems)
    expect_error('ToMVU on mvu vectors', lambda: core.mvgpu.ToMVU(cf, mvuprefix=mp), hint, problems)
    expect_error('FromMVU on mvgpu vectors', lambda: core.mvgpu.FromMVU(gf, gsup, mvuprefix=mp), "mvgpu's already", problems)
    other = dict(sk, blksize=8 if args.blksize != 8 else 16, overlap=4 if args.blksize != 8 else 8)
    expect_error('FromMVU with a super of another grid', lambda: core.mvgpu.FromMVU(cf, core.mvgpu.Super(gclip, **other), mvuprefix=mp),
                 "blksize and overlap", problems)

    def corrupt(what):
        def modify(n, f):
            g = f.copy()
            if mp + 'AnalysisVectors' in g.props:
                if what == 'vector':
                    v = list(g.props[mp + 'AnalysisVectors'])
                    v[0] = (-(1 << 20)) & 0xFFFFFFFF  # x far left of the padding, y 0
                    g.props[mp + 'AnalysisVectors'] = v
                else:
                    s = list(g.props[mp + 'AnalysisSAD'])
                    s[5] = -1
                    g.props[mp + 'AnalysisSAD'] = s
            return g
        return core.std.ModifyFrame(cf, cf, modify)

    expect_error('FromMVU on a vector outside the padded frame', lambda: core.mvgpu.FromMVU(corrupt('vector'), gsup, mvuprefix=mp),
                 'outside the padded frame', problems)
    expect_error('FromMVU on a negative SAD', lambda: core.mvgpu.FromMVU(corrupt('sad'), gsup, mvuprefix=mp), 'negative', problems)
    bare = core.std.RemoveFrameProps(core.mvgpu.FromMVU(cf, gsup, mvuprefix=mp), props=[GPU + 'SuperGPULayout'])
    expect_error('mvgpu.Compensate on vectors without their super description', lambda: core.mvgpu.Compensate(gclip, gsup, bare),
                 'the description of the super it was analysed on', problems)

    print(f'{args.format}{" analysed on 8 bits" if args.analyse8 else ""} pel {args.pel} {args.blksize}/{args.overlap} radius {args.radius}'
          + (f' + recalculated {args.recalculate[0]}/{args.recalculate[1]}' if args.recalculate else '') + (f' mvuprefix {mp}' if mp != 'MVUtensils' else '')
          + f', {frames} frames: {len(gsets)} mvgpu and {len(csets)} mvu vector clips converted both ways, {problems.error_cases} error cases: {problems.count} problems')
    sys.exit(0 if problems.count == 0 else 1)


if __name__ == '__main__':
    main()
