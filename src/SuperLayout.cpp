#include "SuperLayout.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "Common.h"

namespace {

// SuperGPULayout: what the kernels expect of the frames. 2: any format, pel 1, separate
// horizontal and vertical geometry. 3: chroma at pel 4 as one image of its quarter-pel grid. 4: only
// subsampled chroma; 4:4:4 keeps four half-pel planes, as luma does. 5: the coarse levels' chroma
// border covers chroma's reach in either direction (4:4:4's as far as luma's). 6: the super frame is
// the storage, its parts in its one plane (Regions); vector clips' frames are the records.
constexpr int kLayoutVersion = 6;

int AlignUp(int v, int a) {
    return (v + a - 1) / a * a;
}

int64_t AlignUp64(int64_t v, int64_t a) {
    return (v + a - 1) / a * a;
}

// A property of a super's description, of the super clip's frame or of a vector clip's (vectors), which
// carries the description of the super it was analysed on
int GetInt(const VSMap *props, const std::string &key, const VSAPI *vsapi, bool vectors) {
    int err = 0;
    const int v = vsapi->mapGetIntSaturated(props, key.c_str(), 0, &err);
    if (err)
        throw std::runtime_error(vectors ? "a vector clip lacks the property " + key +
                                               ", the description of the super it was analysed on; it must come from mvgpu.Analyse, mvgpu.Recalculate or mvgpu.FromMVU with the same prefix"
                                         : "the super clip lacks the property " + key + "; it must come from mvgpu.Super with the same prefix");
    return v;
}

// What a frame holds that isn't a vector frame of this prefix, for the error saying so: mvgpu's
// vectors under another prefix, or mvu's, whose frames have the analysis description but no
// AnalysisHasVectors (their vectors are property arrays); empty when neither
std::string VectorsHint(const VSMap *props, const std::string &prefix, const VSAPI *vsapi) {
    static const std::string suffix = "AnalysisDeltaFrame";
    std::string mvgpuPrefix, mvuPrefix;
    bool mvgpu = false, mvu = false;
    const int keys = vsapi->mapNumKeys(props);
    for (int i = 0; i < keys; ++i) {
        const std::string key = vsapi->mapGetKey(props, i);
        if (key.size() < suffix.size() || key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        const std::string p = key.substr(0, key.size() - suffix.size());
        if (vsapi->mapNumElements(props, (p + "AnalysisHasVectors").c_str()) > 0) {
            if (!mvgpu && p != prefix) {
                mvgpuPrefix = p;
                mvgpu = true;
            }
        } else if (!mvu) {
            mvuPrefix = p;
            mvu = true;
        }
    }
    if (mvgpu)
        return "its frames have mvgpu's vector properties under the prefix '" + mvgpuPrefix + "'; pass prefix='" + mvgpuPrefix + "'";
    if (mvu)
        return "it holds mvu's vectors (" + mvuPrefix + "Analysis* properties); convert them with mvgpu.FromMVU" +
               (mvuPrefix != DEFAULT_MVUTENSILS_PREFIX ? " and mvuprefix='" + mvuPrefix + "'" : std::string());
    return std::string();
}

// The part of a super's description (ExportSuper) that every frame of the clip shares: all of it but
// where the parts are, which follows from the frame's stride
struct SuperProperty {
    const char *name;
    int64_t value;
};

std::array<SuperProperty, 17> SuperDescription(const SuperLayout &layout) {
    return {{
        {"SuperWidth", layout.aw},
        {"SuperHeight", layout.ah},
        {"SuperRealWidth", layout.width},
        {"SuperRealHeight", layout.height},
        {"SuperHPad", layout.pad},
        {"SuperVPad", layout.padY},
        {"SuperPel", layout.pel},
        {"SuperLevels", layout.topLevel + 1},
        {"SuperChroma", layout.format.chroma},
        {"SuperXRatioUV", layout.format.xr},
        {"SuperYRatioUV", layout.format.yr},
        {"SuperBitsPerSample", layout.format.bits},
        {"SuperBlkSizeX", layout.blk},
        {"SuperBlkSizeY", layout.blkY},
        {"SuperOverlapX", layout.overlap},
        {"SuperOverlapY", layout.overlapY},
        {"SuperGPULayout", kLayoutVersion},
    }};
}

// A vector frame's analysis description; returns the first property missing, empty when none is
std::string ReadVectorDescription(const VSMap *props, const std::string &prefix, VectorInfo &v, const VSAPI *vsapi) {
    struct Field {
        const char *name;
        int VectorInfo::*member;
    };
    static constexpr Field fields[] = {
        {"AnalysisWidth", &VectorInfo::width},       {"AnalysisHeight", &VectorInfo::height},     {"AnalysisRealWidth", &VectorInfo::realWidth},
        {"AnalysisRealHeight", &VectorInfo::realHeight}, {"AnalysisHPad", &VectorInfo::hpad},     {"AnalysisVPad", &VectorInfo::vpad},
        {"AnalysisPel", &VectorInfo::pel},           {"AnalysisBlkSizeX", &VectorInfo::blkX},     {"AnalysisBlkSizeY", &VectorInfo::blkY},
        {"AnalysisOverlapX", &VectorInfo::overlapX}, {"AnalysisOverlapY", &VectorInfo::overlapY}, {"AnalysisNBlkX", &VectorInfo::nbx},
        {"AnalysisNBlkY", &VectorInfo::nby},         {"AnalysisDeltaFrame", &VectorInfo::delta},  {"AnalysisBitsPerSample", &VectorInfo::bits},
        {"AnalysisChroma", &VectorInfo::chroma},     {"AnalysisXRatioUV", &VectorInfo::xRatio},   {"AnalysisYRatioUV", &VectorInfo::yRatio},
    };
    for (const Field &f : fields) {
        int err = 0;
        const int value = vsapi->mapGetIntSaturated(props, (prefix + f.name).c_str(), 0, &err);
        if (err)
            return prefix + f.name;
        v.*f.member = value;
    }
    return std::string();
}

// The layout a super clip's frames were made with, or the one of the super a vector clip's vectors
// were analysed on (vectors), read from its first frame
SuperLayout ImportLayout(VSNode *node, const std::string &prefix, const VSAPI *vsapi, bool vectors) {
    char err[1024] = {};
    const VSFrame *frame = vsapi->getFrame(0, node, err, sizeof(err));
    if (!frame)
        throw std::runtime_error(std::string(vectors ? "failed to get a vector clip's first frame: " : "failed to get the super clip's first frame: ") + err);
    const VSMap *props = vsapi->getFramePropertiesRO(frame);
    const std::string subject = vectors ? "a vector clip's super" : "the super clip's";
    auto get = [&](const char *name) { return GetInt(props, prefix + name, vsapi, vectors); };
    try {
        if (get("SuperGPULayout") != kLayoutVersion)
            throw std::runtime_error(vectors ? "a vector clip was analysed on a super from another version of mvgpu.Super" : "the super clip comes from another version of mvgpu.Super");
        SuperFormat format;
        format.bits = get("SuperBitsPerSample");
        format.chroma = get("SuperChroma") != 0;
        format.xr = get("SuperXRatioUV");
        format.yr = get("SuperYRatioUV");
        const int pel = get("SuperPel"), levels = get("SuperLevels");
        const int blkX = get("SuperBlkSizeX"), blkY = get("SuperBlkSizeY");
        const int overlapX = get("SuperOverlapX"), overlapY = get("SuperOverlapY");
        const int padX = get("SuperHPad"), padY = get("SuperVPad");
        const int width = get("SuperRealWidth"), height = get("SuperRealHeight");
        if ((format.bits < 8 || format.bits > 16) && format.bits != 32)
            throw std::runtime_error(subject + " bit depth is invalid");
        if (format.xr < 1 || format.xr > 2 || format.yr < 1 || format.yr > 2 || (!format.chroma && (format.xr != 1 || format.yr != 1)))
            throw std::runtime_error(subject + " subsampling is invalid");
        if (pel != 1 && pel != 2 && pel != 4)
            throw std::runtime_error(subject + " pel isn't 1, 2 or 4");
        if (width < 1 || height < 1 || padX < 1 || padY < 1 || blkX < 2 || blkY < 2 || overlapX < 0 || overlapY < 0 || overlapX > blkX / 2 || overlapY > blkY / 2)
            throw std::runtime_error(subject + " geometry is invalid");
        // Super refuses a padding the chroma subsampling doesn't divide, as mvu.Super does
        if (format.chroma && (padX % format.xr || padY % format.yr))
            throw std::runtime_error(subject + " padding isn't divisible by the chroma subsampling");
        SuperLayout layout = SuperLayout::Make(format, width, height, blkX, blkY, overlapX, overlapY, padX, padY, pel, levels > 1);
        if (layout.aw != get("SuperWidth") || layout.ah != get("SuperHeight") || layout.topLevel + 1 != levels)
            throw std::runtime_error(subject + " layout doesn't match this version of mvgpu");
        vsapi->freeFrame(frame);
        return layout;
    } catch (...) {
        vsapi->freeFrame(frame);
        throw;
    }
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
    if (nbx < 1 || nby < 1)
        return "the frame is too small to hold a single block at this block size and overlap";
    return {};
}

bool SuperLayout::operator==(const SuperLayout &o) const {
    return width == o.width && height == o.height && format == o.format && blk == o.blk && blkY == o.blkY && overlap == o.overlap && overlapY == o.overlapY &&
           pad == o.pad && padY == o.padY && pel == o.pel && topLevel == o.topLevel;
}

SuperRegions SuperLayout::Regions(int64_t stride) const {
    const int64_t bytes = format.Bytes();
    SuperRegions r;
    r.lumaStride = stride;
    r.lumaBytes = stride * LumaRows();
    int64_t end = r.lumaBytes;
    if (format.chroma) {
        r.chromaStride = AlignUp64(wc * bytes, 64);
        r.chroma = AlignUp64(end, 256);
        r.chromaBytes = r.chromaStride * ChromaRows();
        end = r.chroma + r.chromaBytes;
    }
    if (topLevel > 0) {
        r.pyramid = AlignUp64(end, 256);
        r.pyramidBytes = static_cast<int64_t>(PyramidWidth()) * PyramidRows() * bytes;
    }
    return r;
}

int SuperLayout::FrameRows() const {
    // The rows past luma's hold the other parts and their alignment at the least stride, so at any
    // greater one too
    const int64_t rowBytes = static_cast<int64_t>(wp) * format.Bytes();
    const SuperRegions r = Regions(rowBytes);
    const int64_t rest = (format.chroma ? r.chromaBytes + 255 : 0) + (topLevel > 0 ? r.pyramidBytes + 255 : 0);
    return LumaRows() + static_cast<int>((rest + rowBytes - 1) / rowBytes);
}

void ExportSuper(VSFrame *dst, const SuperLayout &layout, const SuperRegions &regions, const std::string &prefix, const VSAPI *vsapi) {
    VSMap *props = vsapi->getFramePropertiesRW(dst);
    auto set = [&](const char *name, int64_t v) { vsapi->mapSetInt(props, (prefix + name).c_str(), v, maReplace); };
    for (const SuperProperty &p : SuperDescription(layout))
        set(p.name, p.value);
    set("SuperChromaOffset", regions.chroma);
    set("SuperChromaStride", regions.chromaStride);
    set("SuperPyramidOffset", regions.pyramid);
}

std::string CheckSuperFrame(const VSFrame *frame, const SuperLayout &layout, const std::string &prefix, SuperRegions &out, const VSAPI *vsapi) {
    const VSMap *props = vsapi->getFramePropertiesRO(frame);
    for (const SuperProperty &p : SuperDescription(layout)) {
        int err = 0;
        if (vsapi->mapGetInt(props, (prefix + p.name).c_str(), 0, &err) != p.value || err)
            return "a super clip frame was made with different Super arguments than its first frame";
    }
    if (!GetSuperRegions(frame, layout, out, vsapi))
        return "the super clip's frames aren't its storage; it must come from mvgpu.Super with the same prefix";
    return std::string();
}

SuperLayout ImportSuperLayout(VSNode *node, const std::string &prefix, const VSAPI *vsapi) {
    return ImportLayout(node, prefix, vsapi, false);
}

SuperLayout ImportAnalysedLayout(VSNode *vectors, const std::string &prefix, const VSAPI *vsapi) {
    return ImportLayout(vectors, prefix, vsapi, true);
}

bool GetSuperRegions(const VSFrame *frame, const SuperLayout &layout, SuperRegions &out, const VSAPI *vsapi) {
    const VSVideoFormat *f = vsapi->getVideoFrameFormat(frame);
    if (!f || f->colorFamily != cfGray || f->bytesPerSample != layout.format.Bytes() || vsapi->getFrameWidth(frame, 0) != layout.FrameWidth() ||
        vsapi->getFrameHeight(frame, 0) != layout.FrameRows())
        return false;
    const int64_t stride = vsapi->getStride(frame, 0);
    out = layout.Regions(stride);
    const int64_t end = layout.topLevel > 0 ? out.pyramid + out.pyramidBytes : layout.format.chroma ? out.chroma + out.chromaBytes : out.lumaBytes;
    return end <= stride * layout.FrameRows();
}

VectorInfo ReadVectorInfo(VSNode *node, const std::string &prefix, const VSAPI *vsapi) {
    char err[1024] = {};
    const VSFrame *frame = vsapi->getFrame(0, node, err, sizeof(err));
    if (!frame)
        throw std::runtime_error(std::string("failed to get a vector clip's first frame: ") + err);
    const VSMap *props = vsapi->getFramePropertiesRO(frame);
    VectorInfo v;
    std::string missing = ReadVectorDescription(props, prefix, v, vsapi);
    // Every frame of mvgpu's vector clips says whether it has vectors; mvu's frames, whose vectors are
    // property arrays, don't, and would otherwise pass for frames without vectors
    if (missing.empty() && vsapi->mapNumElements(props, (prefix + "AnalysisHasVectors").c_str()) < 1)
        missing = prefix + "AnalysisHasVectors";
    if (!missing.empty()) {
        const std::string hint = VectorsHint(props, prefix, vsapi);
        vsapi->freeFrame(frame);
        throw std::runtime_error("a vector clip lacks the property " + missing + "; " +
                                 (hint.empty() ? "it must come from mvgpu.Analyse, mvgpu.Recalculate or mvgpu.FromMVU with the same prefix" : hint));
    }
    vsapi->freeFrame(frame);
    return v;
}

bool SameAnalysis(const VectorInfo &a, const VectorInfo &b) {
    return a.width == b.width && a.height == b.height && a.realWidth == b.realWidth && a.realHeight == b.realHeight && a.hpad == b.hpad &&
           a.vpad == b.vpad && a.pel == b.pel && a.bits == b.bits && a.blkX == b.blkX && a.blkY == b.blkY && a.overlapX == b.overlapX &&
           a.overlapY == b.overlapY && a.nbx == b.nbx && a.nby == b.nby && a.delta == b.delta && (a.chroma != 0) == (b.chroma != 0) &&
           (!a.chroma || (a.xRatio == b.xRatio && a.yRatio == b.yRatio));
}

bool SameStorage(const SuperLayout &a, const SuperLayout &b) {
    return SameGeometry(a, b) && a.format == b.format;
}

bool SameGeometry(const SuperLayout &a, const SuperLayout &b) {
    return a.width == b.width && a.height == b.height && a.format.chroma == b.format.chroma && a.format.xr == b.format.xr && a.format.yr == b.format.yr &&
           a.aw == b.aw && a.ah == b.ah && a.pad == b.pad && a.padY == b.padY && a.pel == b.pel;
}

void ExportAnalysis(VSFrame *dst, const SuperLayout &layout, int delta, bool chroma, bool hasVectors, const std::string &prefix, const VSAPI *vsapi) {
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
    set("AnalysisHasVectors", hasVectors);
}

const VSFrame *GetAnalysisVectors(const VSFrame *frame, const std::string &prefix, const VSAPI *vsapi) {
    int err = 0;
    const bool has = vsapi->mapGetInt(vsapi->getFramePropertiesRO(frame), (prefix + "AnalysisHasVectors").c_str(), 0, &err) != 0;
    return has && !err ? vsapi->addFrameRef(frame) : nullptr;
}

const VSFrame *GetAnalysisVectors(const VSFrame *frame, const VectorInfo &expected, const std::string &prefix, std::string &error, const VSAPI *vsapi) {
    const VSFrame *vectors = GetAnalysisVectors(frame, prefix, vsapi);
    if (vectors) {
        VectorInfo v;
        if (!ReadVectorDescription(vsapi->getFramePropertiesRO(frame), prefix, v, vsapi).empty() || !SameAnalysis(v, expected)) {
            vsapi->freeFrame(vectors);
            error = "a vector clip frame was made with different Analyse/Recalculate arguments than its first frame";
            return nullptr;
        }
    }
    return vectors;
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
