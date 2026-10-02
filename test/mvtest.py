"""Helpers the test scripts share: raw clips, and moving vectors between mvu and mvgpu.

mvu attaches an analysis description (MVUtensilsAnalysis* integers) and the vectors as properties
of each analysis frame: x and y packed into one int64 per block (x in the low 32 bits) and the SADs
in another array. mvgpu attaches the same description under its prefix and the vectors as a GPU
frame of 32-bit records (x, y, SAD, 0) per block, a row of records per row of blocks, on frames of
the super clip, whose own properties describe the super (SuperLayout.h). Frames whose reference
frame lies outside the clip carry no vectors on either side.
"""
import numpy as np
import vapoursynth as vs


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


def format_clip(core, clip, fmt):
    """The 8-bit clip in the VapourSynth preset format fmt (resize.Bicubic). Converted to more than 8
    bits it is shifted a quarter pixel on the way, so that the low bits hold picture, not zeros."""
    f = core.get_video_format(getattr(vs, fmt))
    if f.bits_per_sample > 8:
        return core.resize.Bicubic(clip, format=f.id, src_left=0.25, src_top=0.25)
    return clip if f.id == clip.format.id else core.resize.Bicubic(clip, format=f.id)


def eight_bit(core, clip):
    """An 8-bit copy of a clip, rounded, for analysing a high bit depth clip on 8 bits"""
    f = clip.format
    if f.bits_per_sample == 8:
        return clip
    eight = core.query_video_format(f.color_family, vs.INTEGER, 8, f.subsampling_w, f.subsampling_h)
    return core.resize.Point(clip, format=eight.id, dither_type='none')


def mvu_vectors(core, analysis, carrier, frames):
    """mvgpu's vectors of one field clip on a CPU carrier clip, as mvu.Analyse would attach them"""
    props = analysis.get_frame(0).props
    desc = {k.replace('MVGPUtensils', 'MVUtensils', 1): int(props[k]) for k in props.keys()
            if k.startswith('MVGPUtensilsAnalysis') and k != 'MVGPUtensilsAnalysisVectors'}
    delta, nbx, nby = desc['MVUtensilsAnalysisDeltaFrame'], desc['MVUtensilsAnalysisNBlkX'], desc['MVUtensilsAnalysisNBlkY']
    # Frames whose reference frame is outside the clip carry no vectors, and PropToClip takes its
    # format from its first frame
    start, end = max(0, -delta), min(frames, frames - delta)
    records = core.std.GPUDownload(core.std.PropToClip(analysis[start:end], prop='MVGPUtensilsAnalysisVectors'))
    vectors = {}
    for n in range(start, end):
        rec = np.asarray(records.get_frame(n - start)[0]).view(np.int32)[:, :4 * nbx].reshape(nby * nbx, 4)
        packed = (rec[:, 0].astype(np.int64) & 0xFFFFFFFF) | (rec[:, 1].astype(np.int64) << 32)
        vectors[n] = (packed.tolist(), rec[:, 2].astype(np.int64).tolist())
    carrier = core.std.SetFrameProps(carrier, **desc)

    def attach(n, f):
        g = f.copy()
        if n in vectors:
            g.props['MVUtensilsAnalysisVectors'] = vectors[n][0]
            g.props['MVUtensilsAnalysisSAD'] = vectors[n][1]
        return g

    return core.std.ModifyFrame(carrier, carrier, attach)


def gpu_vectors(core, analysis, gsup):
    """mvu.Analyse's vectors of one field clip as mvgpu.Analyse would attach them, on the frames of
    gsup, the mvgpu super of the same clip with the same arguments: for testing mvgpu's consumers on
    vectors mvgpu's search wouldn't find, and on frames narrower than mvgpu.Analyse takes"""
    props = analysis.get_frame(0).props
    desc = {k.replace('MVUtensils', 'MVGPUtensils', 1): int(props[k]) for k in props.keys()
            if k.startswith('MVUtensilsAnalysis') and k not in ('MVUtensilsAnalysisVectors', 'MVUtensilsAnalysisSAD')}
    nbx, nby = desc['MVGPUtensilsAnalysisNBlkX'], desc['MVGPUtensilsAnalysisNBlkY']
    gray32 = core.query_video_format(vs.GRAY, vs.INTEGER, 32, 0, 0)
    blank = core.std.BlankClip(format=gray32.id, width=4 * nbx, height=nby, length=analysis.num_frames, keep=True)

    def fill(n, f):
        out = f[0].copy()
        r = np.zeros((nby * nbx, 4), np.int32)
        p = f[1].props
        if 'MVUtensilsAnalysisVectors' in p:
            packed = np.array(p['MVUtensilsAnalysisVectors'], np.int64)
            r[:, 0] = (packed & 0xFFFFFFFF).astype(np.uint32).view(np.int32)
            r[:, 1] = (packed >> 32).astype(np.int32)
            r[:, 2] = np.array(p['MVUtensilsAnalysisSAD'], np.int64).astype(np.int32)
        # whole rows: the plane's rows may be strided, so a reshaped slice of it would be a copy
        np.asarray(out[0]).view(np.int32)[:, :4 * nbx] = r.reshape(nby, 4 * nbx)
        return out

    records = core.std.GPUUpload(core.std.ModifyFrame(blank, [blank, analysis], fill))

    def attach(n, f):
        g = f[0].copy()
        for k, v in desc.items():
            g.props[k] = v
        if 'MVUtensilsAnalysisVectors' in f[2].props:
            g.props['MVGPUtensilsAnalysisVectors'] = f[1]
        elif 'MVGPUtensilsAnalysisVectors' in g.props:
            del g.props['MVGPUtensilsAnalysisVectors']
        return g

    return core.std.ModifyFrame(gsup, [gsup, records, analysis], attach)
