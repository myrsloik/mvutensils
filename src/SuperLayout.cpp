#include "SuperLayout.h"

#include <algorithm>
#include <stdexcept>

namespace {

// SuperGPULayout: what the kernels expect of the frames. 2: any format, pel 1, separate
// horizontal and vertical geometry. 3: chroma at pel 4 as one image of its quarter-pel grid. 4: only
// subsampled chroma; 4:4:4 keeps four half-pel planes, as luma does. 5: the coarse levels' chroma
// border covers chroma's reach in either direction (4:4:4's as far as luma's).
constexpr int kLayoutVersion = 5;

int AlignUp(int v, int a) {
    return (v + a - 1) / a * a;
}

int GetInt(const VSMap *props, const std::string &key, const VSAPI *vsapi) {
    int err = 0;
    const int v = vsapi->mapGetIntSaturated(props, key.c_str(), 0, &err);
    if (err)
        throw std::runtime_error("the super clip lacks the property " + key + "; it must come from mvgpu.Super with the same prefix");
    return v;
}

} // namespace

SuperFormat SuperFormat::Of(const VSVideoFormat &f) {
    if ((f.bitsPerSample > 16 && f.sampleType == stInteger) || (f.bitsPerSample != 32 && f.sampleType == stFloat) || f.subSamplingW > 1 ||
        f.subSamplingH > 1 || (f.colorFamily != cfYUV && f.colorFamily != cfGray))
        throw std::runtime_error("input clip must be GRAY, YUV420, YUV422, YUV440, or YUV444, up to 16 bits integer or 32 bit float, with constant dimensions");
    SuperFormat s;
    s.bits = f.bitsPerSample;
    s.chroma = f.colorFamily != cfGray;
    s.xr = s.chroma ? 1 << f.subSamplingW : 1;
    s.yr = s.chroma ? 1 << f.subSamplingH : 1;
    return s;
}

SuperLayout SuperLayout::Make(const SuperFormat &format, int width, int height, int blkX, int blkY, int overlapX, int overlapY, int padX, int padY, int pel,
                              bool pyramid) {
    SuperLayout s;
    s.width = width;
    s.height = height;
    s.format = format;
    s.pel = pel;
    s.blk = blkX;
    s.blkY = blkY;
    s.overlap = overlapX;
    s.overlapY = overlapY;
    s.step = blkX - overlapX;
    s.stepY = blkY - overlapY;
    // mvu.Super's BlockAlignedDimension, which matches mvu.Analyse's extended grid
    auto aligned = [](int size, int blk, int overlap) {
        const int step = blk - overlap;
        const int b = step * ((size - overlap) / step) + overlap;
        return b < size ? b + step : b;
    };
    s.aw = aligned(width, blkX, overlapX);
    s.ah = aligned(height, blkY, overlapY);
    s.nbx = std::max(0, (s.aw - overlapX) / s.step);
    s.nby = std::max(0, (s.ah - overlapY) / s.stepY);
    s.pad = padX;
    s.padY = padY;
    s.padc = padX / format.xr;
    s.padcY = padY / format.yr;
    s.wp = AlignUp(s.aw + 2 * padX, 4);
    s.hp = s.ah + 2 * padY;
    if (format.chroma) {
        s.wc = AlignUp(s.aw / format.xr + 2 * s.padc, 4);
        s.hc = s.ah / format.yr + 2 * s.padcY;
    }
    s.lumaPlanes = pel == 1 ? 1 : 4;

    if (pyramid)
        for (int cw = width; cw / 2 >= kTopWidth; cw = (cw + 1) / 2)
            ++s.topLevel;

    // Level L halves level L - 1 rounding up, luma and chroma separately, as the CPU reference's
    // Reduce does. Each plane sits inside a border of repeated edge pixels, so no level SAD clamps:
    // a block reaches up to pad >> L pixels past the luma edges, that divided by the subsampling
    // (rounded up) past the chroma edges, and LevelSadOf's words up to 3 bytes further on either
    // side of a row. Rows are whole words apart, planes start on 16 samples.
    s.levels.assign(s.topLevel + 1, LevelEntry{});
    int offset = 0;
    auto addPlane = [&](int w, int h, int border, int32_t &stride) {
        stride = AlignUp(w + 2 * border, 4);
        const int first = offset + border * stride + border;
        offset = AlignUp(offset + stride * (h + 2 * border), 16);
        return first;
    };
    int w = width, h = height, wc = format.chroma ? width / format.xr : 0, hc = format.chroma ? height / format.yr : 0;
    for (int L = 1; L <= s.topLevel; ++L) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        wc = (wc + 1) / 2;
        hc = (hc + 1) / 2;
        const int levelPad = std::max(1, padX >> L);
        LevelEntry &e = s.levels[L];
        e.w = w;
        e.h = h;
        e.borderY = levelPad + 3;
        e.offY = addPlane(w, h, e.borderY, e.strideY);
        if (format.chroma) {
            e.wc = wc;
            e.hc = hc;
            e.borderC = std::max((levelPad + format.xr - 1) / format.xr, (levelPad + format.yr - 1) / format.yr) + 4;
            e.offU = addPlane(wc, hc, e.borderC, e.strideC);
            e.offV = addPlane(wc, hc, e.borderC, e.strideC);
        }
        e.nbx = std::max(1, w / 8);
        e.nby = std::max(1, h / 8);
        e.pad = levelPad;
        e.lambdaOff = L * kLambdaEntries;
        if (L >= kFinest) {
            e.fieldOff = s.fieldTotal;
            s.fieldTotal += e.nbx * e.nby;
            // vectors stay within the padding, so |v| <= size + pad
            s.maxCoarseVector = std::max({s.maxCoarseVector, w + levelPad, h + levelPad});
        }
    }
    s.pyramidSamples = offset;
    LevelEntry &e0 = s.levels[0];
    e0.w = width;
    e0.h = height;
    e0.wc = format.chroma ? width / format.xr : 0;
    e0.hc = format.chroma ? height / format.yr : 0;
    e0.pad = padX;
    e0.frameSamples = offset;
    e0.fieldTotal = s.fieldTotal;
    return s;
}

std::string SuperLayout::Unsupported([[maybe_unused]] Use use) const {
    if (format.chroma && format.xr != format.yr)
        return "only Gray, 4:2:0 and 4:4:4 supers are implemented so far";
    if (pel != 1 && pel != 2 && pel != 4)
        return "only supers with pel=1, pel=2 or pel=4 are implemented so far";
    if (blk != blkY || (blk != 8 && blk != 16 && blk != 32))
        return "only supers with 8x8, 16x16 or 32x32 blocks are implemented so far";
    if (overlap != overlapY)
        return "only supers with the same overlap horizontally and vertically are implemented so far";
    if (pad != padY)
        return "only supers with the same padding horizontally and vertically are implemented so far";
    // Chroma's padding is luma's divided by the subsampling: an odd one leaves the reach of luma's
    // vectors half a pixel short, where mvu reads outside the planes
    if (pad % format.xr || padY % format.yr)
        return "with subsampled chroma the padding must be even, or chroma would be read outside its padding";
    if (nbx < 1 || nby < 1)
        return "the frame is too small to hold a single block at this block size and overlap";
    return {};
}

bool SuperLayout::operator==(const SuperLayout &o) const {
    return width == o.width && height == o.height && format == o.format && blk == o.blk && blkY == o.blkY && overlap == o.overlap && overlapY == o.overlapY &&
           pad == o.pad && padY == o.padY && pel == o.pel && topLevel == o.topLevel;
}

void SuperFrames::Free(const VSAPI *vsapi) {
    vsapi->freeFrame(luma);
    vsapi->freeFrame(chroma);
    vsapi->freeFrame(pyramid);
    luma = chroma = pyramid = nullptr;
}

void ExportSuper(VSFrame *dst, const SuperLayout &layout, const SuperFrames &frames, const std::string &prefix, const VSAPI *vsapi) {
    VSMap *props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetFrame(props, (prefix + "SuperLevel0").c_str(), frames.luma, maReplace);
    if (frames.chroma)
        vsapi->mapSetFrame(props, (prefix + "SuperLevel0").c_str(), frames.chroma, maAppend);
    if (frames.pyramid)
        vsapi->mapSetFrame(props, (prefix + "SuperPyramid").c_str(), frames.pyramid, maReplace);
    auto set = [&](const char *name, int64_t v) { vsapi->mapSetInt(props, (prefix + name).c_str(), v, maReplace); };
    set("SuperWidth", layout.aw);
    set("SuperHeight", layout.ah);
    set("SuperRealWidth", layout.width);
    set("SuperRealHeight", layout.height);
    set("SuperHPad", layout.pad);
    set("SuperVPad", layout.padY);
    set("SuperPel", layout.pel);
    set("SuperLevels", layout.topLevel + 1);
    set("SuperChroma", layout.format.chroma);
    set("SuperXRatioUV", layout.format.xr);
    set("SuperYRatioUV", layout.format.yr);
    set("SuperBitsPerSample", layout.format.bits);
    set("SuperBlkSizeX", layout.blk);
    set("SuperBlkSizeY", layout.blkY);
    set("SuperOverlapX", layout.overlap);
    set("SuperOverlapY", layout.overlapY);
    set("SuperGPULayout", kLayoutVersion);
}

SuperLayout ImportSuperLayout(VSNode *node, const std::string &prefix, const VSAPI *vsapi) {
    char err[1024] = {};
    const VSFrame *frame = vsapi->getFrame(0, node, err, sizeof(err));
    if (!frame)
        throw std::runtime_error(std::string("failed to get the super clip's first frame: ") + err);
    const VSMap *props = vsapi->getFramePropertiesRO(frame);
    try {
        if (GetInt(props, prefix + "SuperGPULayout", vsapi) != kLayoutVersion)
            throw std::runtime_error("the super clip comes from another version of mvgpu.Super");
        SuperFormat format;
        format.bits = GetInt(props, prefix + "SuperBitsPerSample", vsapi);
        format.chroma = GetInt(props, prefix + "SuperChroma", vsapi) != 0;
        format.xr = GetInt(props, prefix + "SuperXRatioUV", vsapi);
        format.yr = GetInt(props, prefix + "SuperYRatioUV", vsapi);
        const int pel = GetInt(props, prefix + "SuperPel", vsapi), levels = GetInt(props, prefix + "SuperLevels", vsapi);
        const int blkX = GetInt(props, prefix + "SuperBlkSizeX", vsapi), blkY = GetInt(props, prefix + "SuperBlkSizeY", vsapi);
        const int overlapX = GetInt(props, prefix + "SuperOverlapX", vsapi), overlapY = GetInt(props, prefix + "SuperOverlapY", vsapi);
        const int padX = GetInt(props, prefix + "SuperHPad", vsapi), padY = GetInt(props, prefix + "SuperVPad", vsapi);
        const int width = GetInt(props, prefix + "SuperRealWidth", vsapi), height = GetInt(props, prefix + "SuperRealHeight", vsapi);
        if ((format.bits < 8 || format.bits > 16) && format.bits != 32)
            throw std::runtime_error("the super clip's bit depth is invalid");
        if (format.xr < 1 || format.xr > 2 || format.yr < 1 || format.yr > 2 || (!format.chroma && (format.xr != 1 || format.yr != 1)))
            throw std::runtime_error("the super clip's subsampling is invalid");
        if (pel != 1 && pel != 2 && pel != 4)
            throw std::runtime_error("the super clip's pel isn't 1, 2 or 4");
        if (width < 1 || height < 1 || padX < 1 || padY < 1 || blkX < 2 || blkY < 2 || overlapX < 0 || overlapY < 0 || overlapX > blkX / 2 || overlapY > blkY / 2)
            throw std::runtime_error("the super clip's geometry is invalid");
        SuperLayout layout = SuperLayout::Make(format, width, height, blkX, blkY, overlapX, overlapY, padX, padY, pel, levels > 1);
        if (layout.aw != GetInt(props, prefix + "SuperWidth", vsapi) || layout.ah != GetInt(props, prefix + "SuperHeight", vsapi) ||
            layout.topLevel + 1 != levels)
            throw std::runtime_error("the super clip's layout doesn't match this version of mvgpu");
        vsapi->freeFrame(frame);
        return layout;
    } catch (...) {
        vsapi->freeFrame(frame);
        throw;
    }
}

bool GetSuperFrames(const VSFrame *frame, const SuperLayout &layout, const std::string &prefix, SuperFrames &out, const VSAPI *vsapi) {
    const VSMap *props = vsapi->getFramePropertiesRO(frame);
    int err = 0;
    out.luma = vsapi->mapGetFrame(props, (prefix + "SuperLevel0").c_str(), 0, &err);
    if (layout.format.chroma)
        out.chroma = vsapi->mapGetFrame(props, (prefix + "SuperLevel0").c_str(), 1, &err);
    if (layout.topLevel > 0)
        out.pyramid = vsapi->mapGetFrame(props, (prefix + "SuperPyramid").c_str(), 0, &err);
    if (!out.luma || (layout.format.chroma && !out.chroma) || (layout.topLevel > 0 && !out.pyramid)) {
        out.Free(vsapi);
        return false;
    }
    return true;
}

VectorInfo ReadVectorInfo(VSNode *node, const std::string &prefix, const VSAPI *vsapi) {
    char err[1024] = {};
    const VSFrame *frame = vsapi->getFrame(0, node, err, sizeof(err));
    if (!frame)
        throw std::runtime_error(std::string("failed to get a vector clip's first frame: ") + err);
    const VSMap *props = vsapi->getFramePropertiesRO(frame);
    auto get = [&](const char *name) {
        int e = 0;
        const int v = vsapi->mapGetIntSaturated(props, (prefix + name).c_str(), 0, &e);
        if (e) {
            vsapi->freeFrame(frame);
            throw std::runtime_error(std::string("a vector clip lacks the property ") + prefix + name + "; it must come from mvgpu.Analyse with the same prefix");
        }
        return v;
    };
    VectorInfo v;
    v.width = get("AnalysisWidth");
    v.height = get("AnalysisHeight");
    v.realWidth = get("AnalysisRealWidth");
    v.realHeight = get("AnalysisRealHeight");
    v.hpad = get("AnalysisHPad");
    v.vpad = get("AnalysisVPad");
    v.pel = get("AnalysisPel");
    v.blkX = get("AnalysisBlkSizeX");
    v.blkY = get("AnalysisBlkSizeY");
    v.overlapX = get("AnalysisOverlapX");
    v.overlapY = get("AnalysisOverlapY");
    v.nbx = get("AnalysisNBlkX");
    v.nby = get("AnalysisNBlkY");
    v.delta = get("AnalysisDeltaFrame");
    v.bits = get("AnalysisBitsPerSample");
    v.chroma = get("AnalysisChroma");
    v.xRatio = get("AnalysisXRatioUV");
    v.yRatio = get("AnalysisYRatioUV");
    vsapi->freeFrame(frame);
    return v;
}

bool SameStorage(const SuperLayout &a, const SuperLayout &b) {
    return SameGeometry(a, b) && a.format == b.format;
}

bool SameGeometry(const SuperLayout &a, const SuperLayout &b) {
    return a.width == b.width && a.height == b.height && a.format.chroma == b.format.chroma && a.format.xr == b.format.xr && a.format.yr == b.format.yr &&
           a.aw == b.aw && a.ah == b.ah && a.pad == b.pad && a.padY == b.padY && a.pel == b.pel;
}

void ExportAnalysis(VSFrame *dst, const SuperLayout &layout, int delta, bool chroma, const VSFrame *vectors, const std::string &prefix, const VSAPI *vsapi) {
    VSMap *props = vsapi->getFramePropertiesRW(dst);
    auto set = [&](const char *name, int64_t v) { vsapi->mapSetInt(props, (prefix + name).c_str(), v, maReplace); };
    set("AnalysisWidth", layout.aw);
    set("AnalysisHeight", layout.ah);
    set("AnalysisRealWidth", layout.width);
    set("AnalysisRealHeight", layout.height);
    set("AnalysisHPad", layout.pad);
    set("AnalysisVPad", layout.padY);
    set("AnalysisPel", layout.pel);
    set("AnalysisLevels", layout.topLevel + 1);
    set("AnalysisChroma", chroma && layout.format.chroma);
    set("AnalysisXRatioUV", layout.format.xr);
    set("AnalysisYRatioUV", layout.format.yr);
    set("AnalysisBlkSizeX", layout.blk);
    set("AnalysisBlkSizeY", layout.blkY);
    set("AnalysisOverlapX", layout.overlap);
    set("AnalysisOverlapY", layout.overlapY);
    set("AnalysisNBlkX", layout.nbx);
    set("AnalysisNBlkY", layout.nby);
    set("AnalysisDeltaFrame", delta);
    set("AnalysisBitsPerSample", layout.format.bits);
    if (vectors)
        vsapi->mapSetFrame(props, (prefix + "AnalysisVectors").c_str(), vectors, maReplace);
    else
        vsapi->mapDeleteKey(props, (prefix + "AnalysisVectors").c_str());
}

const VSFrame *GetAnalysisVectors(const VSFrame *frame, const std::string &prefix, const VSAPI *vsapi) {
    int err = 0;
    return vsapi->mapGetFrame(vsapi->getFramePropertiesRO(frame), (prefix + "AnalysisVectors").c_str(), 0, &err);
}

void GetPairArgument(int &h, int &v, const char *name, int defaultH, int defaultV, const VSMap *in, const VSAPI *vsapi) {
    int err = 0;
    const int numElems = vsapi->mapNumElements(in, name);
    if (numElems > 2)
        throw std::runtime_error(std::string("Too many values passed to ") + name);
    h = vsapi->mapGetIntSaturated(in, name, 0, &err);
    if (err)
        h = defaultH;
    v = vsapi->mapGetIntSaturated(in, name, 1, &err);
    if (err)
        v = (numElems == 1) ? h : defaultV;
}

void CheckBlockSize(int blkX, int blkY, int overlapX, int overlapY, int subSamplingW, int subSamplingH) {
    if ((blkX != 4 || blkY != 4) && (blkX != 8 || blkY != 4) && (blkX != 8 || blkY != 8) && (blkX != 16 || blkY != 2) && (blkX != 16 || blkY != 8) &&
        (blkX != 16 || blkY != 16) && (blkX != 32 || blkY != 16) && (blkX != 32 || blkY != 32) && (blkX != 64 || blkY != 32) && (blkX != 64 || blkY != 64) &&
        (blkX != 128 || blkY != 64) && (blkX != 128 || blkY != 128))
        throw std::runtime_error("the block size must be 4x4, 8x4, 8x8, 16x2, 16x8, 16x16, 32x16, 32x32, 64x32, 64x64, 128x64 or 128x128");
    if (overlapX < 0 || overlapX > blkX / 2 || overlapY < 0 || overlapY > blkY / 2)
        throw std::runtime_error("overlap must be between 0 and half of blksize");
    if (overlapX % (1 << subSamplingW) || overlapY % (1 << subSamplingH))
        throw std::runtime_error("the specified overlap is incompatible with the super clip's subsampling");
}
