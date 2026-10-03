"""Helpers the test scripts share: raw clips, formats, and vectors made up for the filters.

mvu attaches an analysis description (MVUtensilsAnalysis* integers) and the vectors as properties
of each analysis frame: x and y packed into one int64 per block (x in the low 32 bits) and the SADs
in another array. mvgpu's analysis frames are the vectors themselves, GPU frames of 32-bit records
(x, y, SAD, 0) per block, a row of records per row of blocks, carrying the same description under
its prefix and the properties of the super frame they were analysed on (SuperLayout.h). Frames
whose reference frame lies outside the clip carry no vectors on either side (mvgpu's records then
don't count: MVGPUtensilsAnalysisHasVectors 0). mvgpu.ToMVU and mvgpu.FromMVU convert vector clips
between the two (check_convert.py checks them), which is how the checks give each plugin's filters
the other's vectors.
"""
import math

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


def plane_pair(fa, fb, p):
    """Plane p of two frames as arrays to compare exactly, integers widened and floats as their bits
    (bit for bit, signed zeros and NaNs too), and the absolute differences of their values"""
    a, b = np.asarray(fa[p]), np.asarray(fb[p])
    if a.dtype.kind == 'f':
        return a.view(np.uint32), b.view(np.uint32), np.abs(a.astype(np.float64) - b.astype(np.float64))
    a, b = a.astype(np.int64), b.astype(np.int64)
    return a, b, np.abs(a - b)


def eight_bit(core, clip):
    """An 8-bit copy of a clip, rounded, for analysing a high bit depth clip on 8 bits"""
    f = clip.format
    if f.bits_per_sample == 8:
        return clip
    eight = core.query_video_format(f.color_family, vs.INTEGER, 8, f.subsampling_w, f.subsampling_h)
    return core.resize.Point(clip, format=eight.id, dither_type='none')


def scene_limits(props, thscd1, thscd2):
    """mvu's scaled thscd1 and the largest count of blocks above it that isn't a scene change
    (ScaleThSCD, IsSceneChange), for the vectors' bit depth"""
    blk, nbx, nby = props['MVUtensilsAnalysisBlkSizeX'], props['MVUtensilsAnalysisNBlkX'], props['MVUtensilsAnalysisNBlkY']
    chroma = props['MVUtensilsAnalysisChroma']
    ratio = props['MVUtensilsAnalysisXRatioUV'] * props['MVUtensilsAnalysisYRatioUV']
    depth = ((1 << min(16, props['MVUtensilsAnalysisBitsPerSample'])) - 1) / 255.0
    scale = blk * props['MVUtensilsAnalysisBlkSizeY'] / 64.0 * ((1.0 + 2.0 / ratio) if chroma else 1.0) * depth
    th1 = int(thscd1 * scale + 0.5)
    blocks = np.float32(float(np.float32(thscd2)) * nbx * nby / 100.0)
    return th1, int(math.floor(blocks))


def const_vectors(core, analysis, limit, th1, scd, seed):
    """analysis (mvu.Analyse's) with every block of a frame given one vector of at most limit in
    each direction, drawn per frame; by the frame, the SADs below thscd1, at it, or the first scd or
    scd + 1 blocks just above it (a scene change only then)"""
    props = analysis.get_frame(0).props
    nb = props['MVUtensilsAnalysisNBlkX'] * props['MVUtensilsAnalysisNBlkY']

    def modify(n, f):
        g = f.copy()
        if 'MVUtensilsAnalysisVectors' in g.props:
            rng = np.random.default_rng(seed * 1000003 + n)
            vx, vy = (int(v) for v in rng.integers(-limit, limit + 1, size=2))
            g.props['MVUtensilsAnalysisVectors'] = [(vx & 0xFFFFFFFF) | (vy << 32)] * nb
            kind = n % 5
            sad = np.full(nb, th1 if kind == 1 else th1 // 2, np.int64)
            if kind >= 3:
                sad[:scd + (kind - 3)] = th1 + 1
            g.props['MVUtensilsAnalysisSAD'] = sad.tolist()
        return g

    return core.std.ModifyFrame(analysis, analysis, modify)


def random_vectors(core, analysis, limit, th1, scd, seed):
    """analysis (mvu.Analyse's) with every block given a vector of its own, at most limit in each
    direction, drawn per frame, and a SAD up to thscd1; then, by the frame, a few blocks, scd of them
    or scd + 1 (a scene change only then) given SADs above thscd1, up to four times it"""
    props = analysis.get_frame(0).props
    nb = props['MVUtensilsAnalysisNBlkX'] * props['MVUtensilsAnalysisNBlkY']

    def modify(n, f):
        g = f.copy()
        if 'MVUtensilsAnalysisVectors' in g.props:
            rng = np.random.default_rng(seed * 1000003 + n)
            vx = rng.integers(-limit, limit + 1, size=nb, dtype=np.int64)
            vy = rng.integers(-limit, limit + 1, size=nb, dtype=np.int64)
            g.props['MVUtensilsAnalysisVectors'] = ((vx & 0xFFFFFFFF) | (vy << 32)).tolist()
            sad = rng.integers(0, th1 + 1, size=nb, dtype=np.int64)
            kind = n % 5
            above = min(nb, [3, 0, scd // 2, scd, scd + 1][kind])
            sad[rng.permutation(nb)[:above]] = rng.integers(th1 + 1, 4 * th1 + 2, size=above, dtype=np.int64)
            g.props['MVUtensilsAnalysisSAD'] = sad.tolist()
        return g

    return core.std.ModifyFrame(analysis, analysis, modify)
