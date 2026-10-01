#include "SuperLayout.h"

#include <algorithm>
#include <stdexcept>

namespace {

constexpr int kLayoutVersion = 1; // SuperGPULayout: what the kernels expect of the frames

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

SuperLayout SuperLayout::Make(int width, int height, int blk, int overlap, int pad, bool pyramid) {
    SuperLayout s;
    s.width = width;
    s.height = height;
    s.blk = blk;
    s.overlap = overlap;
    s.step = blk - overlap;
    // mvu.Super's BlockAlignedDimension, which matches mvu.Analyse's extended grid
    auto aligned = [&](int size) {
        const int b = s.step * ((size - overlap) / s.step) + overlap;
        return b < size ? b + s.step : b;
    };
    s.aw = aligned(width);
    s.ah = aligned(height);
    s.nbx = (s.aw - overlap) / s.step;
    s.nby = (s.ah - overlap) / s.step;
    if (s.nbx < 1 || s.nby < 1 || s.aw < blk || s.ah < blk)
        throw std::runtime_error("the frame is too small to hold a single block at this block size and overlap");
    s.pad = pad;
    s.padc = pad / 2;
    s.wp = AlignUp(s.aw + 2 * pad, 4);
    s.hp = s.ah + 2 * pad;
    s.wc = AlignUp(s.aw / 2 + 2 * s.padc, 4);
    s.hc = s.ah / 2 + 2 * s.padc;

    if (pyramid) {
        for (int cw = width; cw / 2 >= kTopWidth; cw = (cw + 1) / 2)
            ++s.topLevel;
        if (s.topLevel < kFinest)
            throw std::runtime_error("the frame must be at least " + std::to_string(2 * kTopWidth) + " pixels wide for the coarse search");
    }

    // Level L halves level L - 1 rounding up, luma and chroma separately, as the CPU reference's
    // Reduce does. Each plane sits inside a border of repeated edge pixels, so no level SAD clamps:
    // a block reaches up to pad >> L pixels past the luma edges, about half that past the chroma
    // edges, and LevelSadOf's words up to 3 bytes further on either side of a row. Rows are whole
    // words apart, planes start on 16 bytes.
    s.levels.assign(s.topLevel + 1, LevelEntry{});
    int offset = 0, maxVector = 0;
    auto addPlane = [&](int w, int h, int border, int32_t &stride) {
        stride = AlignUp(w + 2 * border, 4);
        const int first = offset + border * stride + border;
        offset = AlignUp(offset + stride * (h + 2 * border), 16);
        return first;
    };
    int w = width, h = height, wc = width / 2, hc = height / 2;
    for (int L = 1; L <= s.topLevel; ++L) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        wc = (wc + 1) / 2;
        hc = (hc + 1) / 2;
        const int levelPad = std::max(1, pad >> L);
        LevelEntry &e = s.levels[L];
        e.w = w;
        e.h = h;
        e.wc = wc;
        e.hc = hc;
        e.borderY = levelPad + 3;
        e.borderC = (levelPad + 1) / 2 + 4;
        e.offY = addPlane(w, h, e.borderY, e.strideY);
        e.offU = addPlane(wc, hc, e.borderC, e.strideC);
        e.offV = addPlane(wc, hc, e.borderC, e.strideC);
        e.nbx = std::max(1, w / 8);
        e.nby = std::max(1, h / 8);
        e.pad = levelPad;
        e.lambdaOff = L * kLambdaEntries;
        if (L >= kFinest) {
            e.fieldOff = s.fieldTotal;
            s.fieldTotal += e.nbx * e.nby;
            // vectors stay within the padding, so |v| <= size + pad
            maxVector = std::max({maxVector, w + levelPad, h + levelPad});
        }
    }
    s.pyramidBytes = offset;
    LevelEntry &e0 = s.levels[0];
    e0.w = width;
    e0.h = height;
    e0.wc = width / 2;
    e0.hc = height / 2;
    e0.pad = pad;
    e0.frameBytes = offset;
    e0.fieldTotal = s.fieldTotal;
    // median.comp's histogram
    if (maxVector >= 4064)
        throw std::runtime_error("the frame is too large for the coarse search");
    return s;
}

bool SuperLayout::operator==(const SuperLayout &o) const {
    return width == o.width && height == o.height && blk == o.blk && overlap == o.overlap && pad == o.pad && topLevel == o.topLevel;
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
    vsapi->mapSetFrame(props, (prefix + "SuperLevel0").c_str(), frames.chroma, maAppend);
    if (frames.pyramid)
        vsapi->mapSetFrame(props, (prefix + "SuperPyramid").c_str(), frames.pyramid, maReplace);
    auto set = [&](const char *name, int64_t v) { vsapi->mapSetInt(props, (prefix + name).c_str(), v, maReplace); };
    set("SuperWidth", layout.aw);
    set("SuperHeight", layout.ah);
    set("SuperRealWidth", layout.width);
    set("SuperRealHeight", layout.height);
    set("SuperHPad", layout.pad);
    set("SuperVPad", layout.pad);
    set("SuperPel", 2);
    set("SuperLevels", layout.topLevel + 1);
    set("SuperChroma", 1);
    set("SuperXRatioUV", 2);
    set("SuperYRatioUV", 2);
    set("SuperBitsPerSample", 8);
    set("SuperBlkSizeX", layout.blk);
    set("SuperBlkSizeY", layout.blk);
    set("SuperOverlapX", layout.overlap);
    set("SuperOverlapY", layout.overlap);
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
        const int blk = GetInt(props, prefix + "SuperBlkSizeX", vsapi), overlap = GetInt(props, prefix + "SuperOverlapX", vsapi);
        const int pad = GetInt(props, prefix + "SuperHPad", vsapi), levels = GetInt(props, prefix + "SuperLevels", vsapi);
        SuperLayout layout = SuperLayout::Make(GetInt(props, prefix + "SuperRealWidth", vsapi), GetInt(props, prefix + "SuperRealHeight", vsapi), blk, overlap,
                                               pad, levels > 1);
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
    out.chroma = vsapi->mapGetFrame(props, (prefix + "SuperLevel0").c_str(), 1, &err);
    if (layout.topLevel > 0)
        out.pyramid = vsapi->mapGetFrame(props, (prefix + "SuperPyramid").c_str(), 0, &err);
    if (!out.luma || !out.chroma || (layout.topLevel > 0 && !out.pyramid)) {
        out.Free(vsapi);
        return false;
    }
    return true;
}

void ExportAnalysis(VSFrame *dst, const SuperLayout &layout, int delta, const VSFrame *vectors, const std::string &prefix, const VSAPI *vsapi) {
    VSMap *props = vsapi->getFramePropertiesRW(dst);
    auto set = [&](const char *name, int64_t v) { vsapi->mapSetInt(props, (prefix + name).c_str(), v, maReplace); };
    set("AnalysisWidth", layout.aw);
    set("AnalysisHeight", layout.ah);
    set("AnalysisRealWidth", layout.width);
    set("AnalysisRealHeight", layout.height);
    set("AnalysisHPad", layout.pad);
    set("AnalysisVPad", layout.pad);
    set("AnalysisPel", 2);
    set("AnalysisLevels", layout.topLevel + 1);
    set("AnalysisChroma", 1);
    set("AnalysisXRatioUV", 2);
    set("AnalysisYRatioUV", 2);
    set("AnalysisBlkSizeX", layout.blk);
    set("AnalysisBlkSizeY", layout.blk);
    set("AnalysisOverlapX", layout.overlap);
    set("AnalysisOverlapY", layout.overlap);
    set("AnalysisNBlkX", layout.nbx);
    set("AnalysisNBlkY", layout.nby);
    set("AnalysisDeltaFrame", delta);
    set("AnalysisBitsPerSample", 8);
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
