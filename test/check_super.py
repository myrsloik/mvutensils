#!/usr/bin/env python3
"""mvgpu.Super against mvu.Super, byte for byte, and its coarse levels against a model of them.

Both make the super of the same clip with the same arguments. Every sub-pel plane of level 0, padding
included, is compared sample for sample (float ones bit for bit): mvu.Super's level-0 frames
(MVUtensilsSuperLevel0, the sub-pel planes stacked) against mvgpu.Super's storage frames (SuperLayout.h).
At pel 4, where mvgpu keeps only luma's half-pel planes (and 4:4:4 chroma's), its quarter planes are
computed from them the way the kernels read them and compared with mvu's, and subsampled chroma's
sixteen are read out of its quarter-pel image (every fourth sample of every fourth row from the phase
on); the quarter samples mvu leaves unset
(the last column of x + 3/4, the last row of y + 3/4) are skipped. mvgpu's coarse levels, which mvu
doesn't have, are compared with a numpy model of their reduction (rfilter's filter on clamped reads,
SuperLayout.h's level table).

    check_super.py --src noisy.nv12 --size 1920x1080 --frames 3 [--format YUV420P10] [--pel 4] [--sharp 1]
                   [--rfilter 0] [--blksize 16 8] [--overlap 4 2] [--pad 16 8] [--pelclip] [--onelevel]
                   [--crop 1914x1074]

--format converts the 8-bit 4:2:0 source first (resize.Bicubic); --pelclip gives both a pelclip, the clip
upscaled pel times by Spline36. mvgpu.Super refuses a pelclip at pel 4 (its luma keeps only the half-pel
planes), which --pelclip --pel 4 checks instead.
"""
import argparse
import sys

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


def aligned(size, blk, overlap):
    """mvu.Super's BlockAlignedDimension"""
    step = blk - overlap
    b = step * ((size - overlap) // step) + overlap if size >= overlap else overlap
    return b + step if b < size else b


def bits_of(a):
    return a.view(np.uint32) if a.dtype == np.float32 else a


def avg(a, b):
    if a.dtype == np.float32:
        return (a + b) * np.float32(0.5)
    return ((a.astype(np.int32) + b + 1) >> 1).astype(a.dtype)


def quarters(p0, p2, p8, p10):
    """GeneratePelQuarters from the four half-pel planes, the last column and row clamped"""
    def nx(a):
        return np.concatenate([a[:, 1:], a[:, -1:]], axis=1)

    def ny(a):
        return np.concatenate([a[1:], a[-1:]], axis=0)

    q = {0: p0, 2: p2, 8: p8, 10: p10}
    d4, d6, d4x = avg(p0, p8), avg(p2, p10), avg(nx(p0), nx(p8))
    d12, d14, d12x = avg(ny(p0), p8), avg(ny(p2), p10), avg(nx(ny(p0)), nx(p8))
    q[1], q[9], q[4], q[6], q[5] = avg(p0, p2), avg(p8, p10), d4, d6, avg(d4, d6)
    q[3], q[11], q[7] = avg(nx(p0), p2), avg(nx(p8), p10), avg(d4x, d6)
    q[12], q[14], q[13], q[15] = d12, d14, avg(d12, d14), avg(d12x, d14)
    return q


def unset_mask(k, h, w):
    """The samples of quarter plane k mvu leaves unset"""
    m = np.ones((h, w), bool)
    if k in (3, 7, 11, 15):
        m[:, -1] = False
    if k in (12, 13, 14, 15):
        m[-1, :] = False
    return m


def reduce_plane(s, rfilter):
    """One coarse level from the one below, pyr_reduce.comp's arithmetic on clamped reads"""
    h, w = s.shape
    dh, dw = (h + 1) // 2, (w + 1) // 2
    flt = s.dtype == np.float32
    S = s.astype(np.float32) if flt else s.astype(np.int64)

    def rows(r):
        return S[np.clip(r, 0, h - 1)]

    def cols(a, c):
        return a[:, np.clip(c, 0, w - 1)]

    y = np.arange(dh)
    x = np.arange(dw)
    if rfilter == 0:
        top, bottom = rows(2 * y), rows(2 * y + 1)
        tl, tr, br, bl = cols(top, 2 * x), cols(top, 2 * x + 1), cols(bottom, 2 * x + 1), cols(bottom, 2 * x)
        out = ((tl + tr) + br + bl) * np.float32(0.25) if flt else (tl + tr + br + bl + 2) >> 2
        return out.astype(s.dtype)
    # The vertical pass, at every source column
    r = 2 * y[:, None]
    edge = (y == 0) | (y == dh - 1)
    if flt:
        v_edge = (rows(r[:, 0]) + rows(r[:, 0] + 1)) * np.float32(0.5)
        if rfilter == 1:
            v_mid = ((rows(r[:, 0] - 1) + (rows(r[:, 0]) + rows(r[:, 0] + 1)) * np.float32(3)) + rows(r[:, 0] + 2)) * np.float32(0.125)
        else:
            m2 = (rows(r[:, 0]) + rows(r[:, 0] + 1)) * np.float32(10)
            m1 = (rows(r[:, 0] - 1) + rows(r[:, 0] + 2)) * np.float32(5)
            v_mid = (rows(r[:, 0] - 2) + ((rows(r[:, 0] + 3) + m2) + m1)) * np.float32(1 / 32)
    else:
        v_edge = (rows(r[:, 0]) + rows(r[:, 0] + 1) + 1) >> 1
        if rfilter == 1:
            v_mid = (rows(r[:, 0] - 1) + (rows(r[:, 0]) + rows(r[:, 0] + 1)) * 3 + rows(r[:, 0] + 2) + 4) >> 3
        else:
            m2 = (rows(r[:, 0]) + rows(r[:, 0] + 1)) * 10
            m1 = (rows(r[:, 0] - 1) + rows(r[:, 0] + 2)) * 5
            v_mid = (rows(r[:, 0] - 2) + rows(r[:, 0] + 3) + m2 + m1 + 16) >> 5
    V = np.where(edge[:, None], v_edge, v_mid)
    if not flt:
        V = V.astype(s.dtype).astype(np.int64)  # the vertical pass's results are samples
    c = 2 * x
    edge = (x == 0) | (x == dw - 1)
    if flt:
        h_edge = (cols(V, c) + cols(V, c + 1)) * np.float32(0.5)
        if rfilter == 1:
            h_mid = ((cols(V, c - 1) + (cols(V, c) + cols(V, c + 1)) * np.float32(3)) + cols(V, c + 2)) * np.float32(0.125)
        else:
            m2 = (cols(V, c) + cols(V, c + 1)) * np.float32(10)
            m1 = (cols(V, c - 1) + cols(V, c + 2)) * np.float32(5)
            h_mid = (cols(V, c - 2) + ((cols(V, c + 3) + m2) + m1)) * np.float32(1 / 32)
    else:
        h_edge = (cols(V, c) + cols(V, c + 1) + 1) >> 1
        if rfilter == 1:
            h_mid = (cols(V, c - 1) + (cols(V, c) + cols(V, c + 1)) * 3 + cols(V, c + 2) + 4) >> 3
        else:
            m2 = (cols(V, c) + cols(V, c + 1)) * 10
            m1 = (cols(V, c - 1) + cols(V, c + 2)) * 5
            h_mid = (cols(V, c - 2) + cols(V, c + 3) + m2 + m1 + 16) >> 5
    return np.where(edge[None, :], h_edge, h_mid).astype(s.dtype)


def level_table(w, h, chroma, xr, yr, pad):
    """SuperLayout::Make's coarse levels: (w, h, wc, hc, offY, offU, offV, strideY, strideC, borderY, borderC) per level"""
    def up(v, a):
        return (v + a - 1) // a * a

    top, cw = 0, w
    while cw // 2 >= 96:
        top += 1
        cw = (cw + 1) // 2
    levels, offset = [], 0
    wc, hc = (w // xr, h // yr) if chroma else (0, 0)
    for L in range(1, top + 1):
        w, h, wc, hc = (w + 1) // 2, (h + 1) // 2, (wc + 1) // 2, (hc + 1) // 2
        lp = max(1, pad >> L)
        by, bc = lp + 3, max((lp + xr - 1) // xr, (lp + yr - 1) // yr) + 4
        sy = up(w + 2 * by, 4)
        offY = offset + by * sy + by
        offset = up(offset + sy * (h + 2 * by), 16)
        sc = offU = offV = 0
        if chroma:
            sc = up(wc + 2 * bc, 4)
            offU = offset + bc * sc + bc
            offset = up(offset + sc * (hc + 2 * bc), 16)
            offV = offset + bc * sc + bc
            offset = up(offset + sc * (hc + 2 * bc), 16)
        levels.append((w, h, wc, hc, offY, offU, offV, sy, sc, by, bc))
    return levels


def flat(frame, dtype):
    """A storage frame's samples as the flat buffer the kernels index"""
    a = np.asarray(frame[0])
    stride = frame.get_stride(0) // a.itemsize
    return np.lib.stride_tricks.as_strided(a, shape=(a.shape[0] * stride,), strides=(a.itemsize,)).view(dtype), stride


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='raw 8-bit NV12 frames')
    ap.add_argument('--size', required=True, help='WxH')
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--format', default='YUV420P8', help='a VapourSynth preset name, e.g. YUV444P16, GRAYS')
    ap.add_argument('--crop', help='WxH: crop the frames to this size first')
    ap.add_argument('--pel', type=int, default=2)
    ap.add_argument('--sharp', type=int, default=2)
    ap.add_argument('--rfilter', type=int, default=1)
    ap.add_argument('--blksize', type=int, nargs='+', default=[16])
    ap.add_argument('--overlap', type=int, nargs='+', default=[8])
    ap.add_argument('--pad', type=int, nargs='+', default=[16])
    ap.add_argument('--pelclip', action='store_true')
    ap.add_argument('--onelevel', action='store_true')
    ap.add_argument('--plugin', help='the MVGPUtensils library to load, unless it autoloads')
    ap.add_argument('--mvu', default=r'C:\Libraries\mvuplus\msvc\x64\Release\MVUtensils.dll', help='the MVUtensils library, unless it autoloads')
    args = ap.parse_args()

    core = vs.core
    if args.plugin:
        core.std.LoadPlugin(args.plugin)
    if not hasattr(core, 'mvu'):
        core.std.LoadPlugin(args.mvu)
    w, h = (int(v) for v in args.size.split('x'))
    clip = nv12_clip(core, args.src, w, h, args.frames)
    if args.crop:
        w, h = (int(v) for v in args.crop.split('x'))
        clip = core.std.CropAbs(clip, w, h)
    fmt = getattr(vs, args.format, None)
    if fmt is None:  # a 4:4:0 format beyond 8 bits has no preset: YUV440P10 etc.
        bits = int(args.format.removeprefix('YUV440P'))
        fmt = core.query_video_format(vs.YUV, vs.INTEGER, bits, 0, 1).id
    if clip.format.id != fmt:
        clip = core.resize.Bicubic(clip, format=fmt)
    f = clip.format
    chroma = f.color_family != vs.GRAY
    xr, yr = (1 << f.subsampling_w, 1 << f.subsampling_h) if chroma else (1, 1)
    flt = f.sample_type == vs.FLOAT
    dtype = np.float32 if flt else (np.uint8 if f.bits_per_sample == 8 else np.uint16)

    kw = dict(blksize=args.blksize, overlap=args.overlap, pad=args.pad, pel=args.pel, sharp=args.sharp, rfilter=args.rfilter, onelevel=args.onelevel)
    gclip = core.std.GPUUpload(clip)
    if args.pelclip and args.pel == 4:
        pc = core.std.GPUUpload(core.resize.Spline36(clip, w * 4, h * 4))
        try:
            core.mvgpu.Super(gclip, pelclip=pc, **kw)
        except vs.Error as e:
            refused = 'pelclip isn\'t supported at pel=4' in str(e)
            print(f'{args.format} pel 4 pelclip: refused{" as documented" if refused else " with an unexpected message: " + str(e)}')
            sys.exit(0 if refused else 1)
        print(f'{args.format} pel 4 pelclip: accepted, but mvgpu.Super must refuse it')
        sys.exit(1)
    if args.pelclip and args.pel > 1:
        pc = core.resize.Spline36(clip, w * args.pel, h * args.pel)
        csup = core.mvu.Super(clip, pelclip=pc, **kw)
        gsup = core.mvgpu.Super(gclip, pelclip=core.std.GPUUpload(pc), **kw)
    else:
        csup = core.mvu.Super(clip, **kw)
        gsup = core.mvgpu.Super(gclip, **kw)

    bx, by = args.blksize[0], args.blksize[-1]
    ox, oy = args.overlap[0], args.overlap[-1]
    padx, pady = args.pad[0], args.pad[-1]
    aw, ah = aligned(w, bx, ox), aligned(h, by, oy)
    pel, n_sub = args.pel, args.pel * args.pel
    gp = gsup.get_frame(0).props
    luma_planes = 1 if pel == 1 else 4  # at pel 4 the half-pel planes, the quarter ones computed
    image = pel == 4 and chroma and (xr > 1 or yr > 1)  # subsampled chroma at pel 4: its quarter-pel image
    chroma_planes = n_sub if image else luma_planes     # plane sizes per chroma plane: else kept as luma is
    top = gp['MVGPUtensilsSuperLevels'] - 1

    planes = 3 if chroma else 1
    cl0 = [core.std.PropToClip(csup, prop='MVUtensilsSuperLevel0', index=p) for p in range(planes)]
    gl0 = [core.std.GPUDownload(core.std.PropToClip(gsup, prop='MVGPUtensilsSuperLevel0', index=i)) for i in range(2 if chroma else 1)]
    gpyr = core.std.GPUDownload(core.std.PropToClip(gsup, prop='MVGPUtensilsSuperPyramid')) if top > 0 else None

    differ = compared = 0
    worst = None
    pyr_differ = pyr_compared = 0
    for n in range(args.frames):
        gframes = [c.get_frame(n) for c in gl0]
        g_real = []  # each plane's full-pel plane inside the padding, for the pyramid model
        for p in range(planes):
            W = (aw if p == 0 else aw // xr) + 2 * (padx if p == 0 else padx // xr)
            H = (ah if p == 0 else ah // yr) + 2 * (pady if p == 0 else pady // yr)
            mvu = np.asarray(cl0[p].get_frame(n)[0])
            g = np.asarray(gframes[0 if p == 0 else 1][0])
            stored = luma_planes if p == 0 else chroma_planes
            base = 0 if p == 0 else (p - 1) * chroma_planes
            if p > 0 and image:
                # The plane's quarter-pel image: 4 H rows of 4 strides, phase (fx, fy) of pixel (x, y)
                # at (4 x + fx, 4 y + fy); U's, then V's
                buf, stride = flat(gframes[1], dtype)
                img = buf[base * H * stride:(base + n_sub) * H * stride].reshape(4 * H, 4 * stride)
                gsub = lambda slot, img=img: img[slot >> 2::4, slot & 3::4][:H, :W]
            else:
                gsub = lambda slot: g[(base + slot) * H:(base + slot + 1) * H, :W]
            half = None
            if stored == 4 and pel == 4:
                half = {0: gsub(0), 2: gsub(1), 8: gsub(2), 10: gsub(3)}
                computed = quarters(half[0], half[2], half[8], half[10])
            for k in range(n_sub):
                expected = mvu[k * H:(k + 1) * H, :W]
                if stored == n_sub:
                    got = gsub(k)
                else:
                    got = computed[k]
                if pel == 4:
                    mask = unset_mask(k, H, W)
                else:
                    mask = np.ones((H, W), bool)
                bad = (bits_of(expected) != bits_of(got)) & mask
                compared += int(mask.sum())
                if bad.any():
                    differ += int(bad.sum())
                    if worst is None:
                        yy, xx = np.argwhere(bad)[0]
                        worst = (n, p, k, int(xx), int(yy), expected[yy, xx], got[yy, xx])
            ppx, ppy = (padx, pady) if p == 0 else (padx // xr, pady // yr)
            rw, rh = (w, h) if p == 0 else (w // xr, h // yr)
            g_real.append(gsub(0)[ppy:ppy + rh, ppx:ppx + rw])
        if gpyr is not None:
            buf, _ = flat(gpyr.get_frame(n), dtype)
            table = level_table(w, h, chroma, xr, yr, padx)
            src = g_real
            for (lw, lh, lwc, lhc, offY, offU, offV, sy, sc, bdy, bdc) in table:
                nxt = []
                for p in range(planes):
                    lvl = reduce_plane(src[p], args.rfilter)
                    dw, dh, off, stride, border = (lw, lh, offY, sy, bdy) if p == 0 else (lwc, lhc, offU if p == 1 else offV, sc, bdc)
                    assert lvl.shape == (dh, dw)
                    padded = lvl[np.clip(np.arange(-border, dh + border), 0, dh - 1)][:, np.clip(np.arange(-border, dw + border), 0, dw - 1)]
                    start = off - border * stride - border
                    got = np.lib.stride_tricks.as_strided(buf[start:], shape=padded.shape, strides=(stride * buf.itemsize, buf.itemsize))
                    bad = bits_of(np.ascontiguousarray(got)) != bits_of(padded)
                    pyr_compared += bad.size
                    pyr_differ += int(bad.sum())
                    nxt.append(lvl)
                src = nxt

    print(f'{args.format} pel {pel} sharp {args.sharp} rfilter {args.rfilter} blksize {args.blksize} overlap {args.overlap} pad {args.pad}'
          f'{" pelclip" if args.pelclip else ""}: level 0 {differ} of {compared} samples differ'
          + (f' (first: frame {worst[0]} plane {worst[1]} sub-plane {worst[2]} at ({worst[3]}, {worst[4]}): mvu {worst[5]}, mvgpu {worst[6]})' if worst else '')
          + (f'; coarse levels {pyr_differ} of {pyr_compared} samples differ from the model ({top} levels)' if top > 0 else '; no coarse levels'))
    sys.exit(0 if differ == 0 and pyr_differ == 0 and compared > 0 else 1)


if __name__ == '__main__':
    main()
