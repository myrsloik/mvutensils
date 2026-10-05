#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <VSVulkan4.h>

#include "Common.h"
#include "SuperLayout.h"
#include "VulkanContext.h"

// Vector clips between mvgpu and mvu. mvu's filters read a vector frame's properties and nothing else:
// the analysis description, and the vectors as two arrays, AnalysisVectors (x and y packed into one
// int64 per block, x in the low 32 bits) and AnalysisSAD, both absent on a frame without vectors.
// mvgpu's vector frames are the vectors' records themselves, on the GPU, with the description, whether
// they hold vectors (AnalysisHasVectors) and the description of the super they were analysed on
// (ExportAnalysis). ToMVU gives mvgpu.Analyse's and mvgpu.Recalculate's vectors mvu's form, FromMVU
// gives mvu.Analyse's and mvu.Recalculate's mvgpu's; the vectors and SADs stay as they are, so either
// plugin's filters make the same frames from them as the other's (the test suites run mvu's filters on
// mvgpu's vectors and mvgpu's on mvu's through these two).

namespace {

// mvu's analysis description, which every frame of a vector clip has, with vectors or without
namespace key {
enum : int {
    Width, Height, RealWidth, RealHeight, HPad, VPad, Pel, Levels, Chroma, XRatio, YRatio,
    BlkX, BlkY, OverlapX, OverlapY, NBlkX, NBlkY, Delta, Bits, Count
};
} // namespace key
constexpr const char *kDescriptionNames[key::Count] = {
    "AnalysisWidth", "AnalysisHeight", "AnalysisRealWidth", "AnalysisRealHeight", "AnalysisHPad", "AnalysisVPad", "AnalysisPel",
    "AnalysisLevels", "AnalysisChroma", "AnalysisXRatioUV", "AnalysisYRatioUV", "AnalysisBlkSizeX", "AnalysisBlkSizeY",
    "AnalysisOverlapX", "AnalysisOverlapY", "AnalysisNBlkX", "AnalysisNBlkY", "AnalysisDeltaFrame", "AnalysisBitsPerSample",
};
using Description = std::array<int64_t, key::Count>;

// A frame's description under prefix; returns the first property missing, empty when none is
std::string ReadDescription(const VSMap *props, const std::string &prefix, Description &desc, const VSAPI *vsapi) {
    for (int i = 0; i < key::Count; ++i) {
        int err = 0;
        desc[i] = vsapi->mapGetInt(props, (prefix + kDescriptionNames[i]).c_str(), 0, &err);
        if (err)
            return prefix + kDescriptionNames[i];
    }
    return std::string();
}

// mvu's test of a frame with vectors against the clip's first frame (MotionBlockPyramid::Geometry):
// the whole description but the levels, the subsampling only where the SADs count chroma
bool SameAnalysis(const Description &a, const Description &b) {
    for (int i = 0; i < key::Count; ++i)
        if (i != key::Levels && i != key::XRatio && i != key::YRatio && a[i] != b[i])
            return false;
    return !a[key::Chroma] || (a[key::XRatio] == b[key::XRatio] && a[key::YRatio] == b[key::YRatio]);
}

void DeleteKeysStartingWith(VSMap *props, const std::string &start, const VSAPI *vsapi) {
    std::vector<std::string> keys;
    const int count = vsapi->mapNumKeys(props);
    for (int i = 0; i < count; ++i) {
        const char *key = vsapi->mapGetKey(props, i);
        if (std::strncmp(key, start.c_str(), start.size()) == 0)
            keys.emplace_back(key);
    }
    for (const std::string &key : keys)
        vsapi->mapDeleteKey(props, key.c_str());
}

std::string Prefix(const VSMap *in, const char *name, const char *otherwise, const VSAPI *vsapi) {
    int err = 0;
    const char *prefix = vsapi->mapGetData(in, name, 0, &err);
    return prefix ? prefix : otherwise;
}

// Both filters take a list of vector clips, as AnalyseMany and Recalculate return them, and return one
// clip for each, in their order; create(i) appends clip i's to out or throws
template<typename Create>
void CreateEach(const char *name, const VSMap *in, VSMap *out, Create create, const VSAPI *vsapi) {
    const int clips = vsapi->mapNumElements(in, "vectors");
    for (int i = 0; i < clips; ++i) {
        std::string error;
        try {
            create(i);
        } catch (const std::exception &e) {
            error = e.what();
        }
        const char *created = vsapi->mapGetError(out);
        if (error.empty() && created)
            error = created;
        if (!error.empty()) {
            vsapi->clearMap(out);
            vsapi->mapSetError(out, (std::string(name) + ": " + (clips > 1 ? "vector clip " + std::to_string(i) + ": " : std::string()) + error).c_str());
            return;
        }
    }
}

} // namespace


// ToMVU: the records downloaded, with mvu's description and arrays added; the frames stay mvgpu's
// vector frames too, records and properties (mvgpu's filters take them back, uploaded)

struct ToMVUData {
    VSNode *node = nullptr; // the vector clip, on the CPU
    VSVideoInfo vi = {};
    std::string prefix, mvuprefix;

    const VSAPI *vsapi;

    ToMVUData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~ToMVUData() {
        vsapi->freeNode(node);
    }
};

static const VSFrame *VS_CC toMVUGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    ToMVUData *d = reinterpret_cast<ToMVUData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        auto fail = [&](const std::string &message) -> const VSFrame * {
            vsapi->freeFrame(src);
            vsapi->setFilterError(("ToMVU: frame " + std::to_string(n) + ": " + message).c_str(), frameCtx);
            return nullptr;
        };

        const VSMap *props = vsapi->getFramePropertiesRO(src);
        Description desc;
        int err = 0;
        if (const std::string missing = ReadDescription(props, d->prefix, desc, vsapi); !missing.empty())
            return fail("the frame lacks the property " + missing);
        const bool has = vsapi->mapGetInt(props, (d->prefix + "AnalysisHasVectors").c_str(), 0, &err) != 0;
        if (err)
            return fail("the frame lacks the property " + d->prefix + "AnalysisHasVectors");
        const int64_t nbx = desc[key::NBlkX], nby = desc[key::NBlkY];
        if (nbx < 1 || nby < 1 || vsapi->getFrameWidth(src, 0) != 4 * nbx || vsapi->getFrameHeight(src, 0) != nby)
            return fail("the frame isn't the records of its grid");

        VSFrame *dst = vsapi->copyFrame(src, core);
        VSMap *out = vsapi->getFramePropertiesRW(dst);
        for (int i = 0; i < key::Count; ++i)
            vsapi->mapSetInt(out, (d->mvuprefix + kDescriptionNames[i]).c_str(), desc[i], maReplace);
        const std::string vectorsKey = d->mvuprefix + "AnalysisVectors", sadKey = d->mvuprefix + "AnalysisSAD";
        if (has) {
            const size_t blocks = static_cast<size_t>(nbx * nby);
            std::vector<int64_t> packed(blocks), sads(blocks);
            const uint8_t *records = vsapi->getReadPtr(src, 0);
            const ptrdiff_t stride = vsapi->getStride(src, 0);
            for (int64_t y = 0; y < nby; ++y) {
                const int32_t *row = reinterpret_cast<const int32_t *>(records + y * stride);
                for (int64_t x = 0; x < nbx; ++x) {
                    const size_t i = static_cast<size_t>(y * nbx + x);
                    packed[i] = std::bit_cast<int64_t>(static_cast<uint64_t>(static_cast<uint32_t>(row[4 * x])) |
                                                       static_cast<uint64_t>(static_cast<uint32_t>(row[4 * x + 1])) << 32);
                    sads[i] = static_cast<uint32_t>(row[4 * x + 2]); // unsigned, as mvu's SADs are
                }
            }
            vsapi->mapSetIntArray(out, vectorsKey.c_str(), packed.data(), static_cast<int>(blocks));
            vsapi->mapSetIntArray(out, sadKey.c_str(), sads.data(), static_cast<int>(blocks));
        } else {
            // mvu's frames without vectors have no arrays
            vsapi->mapDeleteKey(out, vectorsKey.c_str());
            vsapi->mapDeleteKey(out, sadKey.c_str());
        }
        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC toMVUCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    const std::string prefix = Prefix(in, "prefix", DEFAULT_MVGPUTENSILS_PREFIX, vsapi);
    const std::string mvuprefix = Prefix(in, "mvuprefix", DEFAULT_MVUTENSILS_PREFIX, vsapi);
    VSPlugin *stdPlugin = vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core);

    CreateEach("ToMVU", in, out, [&](int i) {
        std::unique_ptr<ToMVUData> d = std::make_unique<ToMVUData>(vsapi);
        d->prefix = prefix;
        d->mvuprefix = mvuprefix;
        d->node = vsapi->mapGetNode(in, "vectors", i, nullptr);
        const VectorInfo v = ReadVectorInfo(d->node, prefix, vsapi);

        // The records on the CPU, through the core's transfer (a clip already there passes as it is)
        VSMap *args = vsapi->createMap();
        vsapi->mapSetNode(args, "clip", d->node, maReplace);
        VSMap *ret = vsapi->invoke(stdPlugin, "GPUDownload", args);
        vsapi->freeMap(args);
        if (const char *error = vsapi->mapGetError(ret)) {
            const std::string message = error;
            vsapi->freeMap(ret);
            throw std::runtime_error(message);
        }
        vsapi->freeNode(d->node);
        d->node = vsapi->mapGetNode(ret, "clip", 0, nullptr);
        vsapi->freeMap(ret);

        d->vi = *vsapi->getVideoInfo(d->node);
        const VSVideoFormat &f = d->vi.format;
        if (f.colorFamily != cfGray || f.sampleType != stInteger || f.bitsPerSample != 32 || d->vi.width != 4 * v.nbx || d->vi.height != v.nby)
            throw std::runtime_error("the vector clip isn't the records of its grid (Gray32, 4 * NBlkX x NBlkY)");

        VSFilterDependency deps[1] = {
            {d->node, rpStrictSpatial},
        };
        vsapi->createVideoFilter(out, "ToMVU", &d->vi, toMVUGetFrame, filterFree<ToMVUData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
        if (!vsapi->mapGetError(out))
            d.release();
    }, vsapi);
}


// FromMVU: the records built from mvu's arrays and copied to a GPU frame of their own, with mvgpu's
// description of them and of the super; the frame keeps the clip's other properties

struct FromMVUData {
    VSNode *node = nullptr; // mvu's vector clip
    VSVideoInfo vi = {};    // the records'
    SuperLayout layout;     // the super's
    SuperLayout grid;       // the same with the vectors' grid, its own or another (WithGrid)
    SuperRegions regions;   // where its frames keep their parts, for its description (ExportSuper)
    Description first;      // the vector clip's first frame's description
    std::string prefix, mvuprefix;
    VSVideoFormat gray32 = {};

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;

    const VSAPI *vsapi;

    FromMVUData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~FromMVUData() {
        // The pool drains the GPU first, so no copy still reads a staging buffer
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        vc.reset();
        vsapi->freeNode(node);
    }
};

static const VSFrame *VS_CC fromMVUGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    FromMVUData *d = reinterpret_cast<FromMVUData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VulkanContext &vc = *d->vc;
        const VSVULKANAPI *vkapi = vc.vkapi;
        const SuperLayout &L = d->layout, &G = d->grid; // the super's description, the vectors' grid

        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrame *dst = nullptr;
        VSGPUBuffer *staging = nullptr;
        VSGPUExecContext *ctx = nullptr;
        auto fail = [&](const std::string &message) -> const VSFrame * {
            if (ctx)
                vkapi->gpuExecAbandon(ctx);
            if (staging)
                vkapi->destroyGPUBuffer(staging);
            vsapi->freeFrame(dst);
            vsapi->freeFrame(src);
            vsapi->setFilterError(("FromMVU: frame " + std::to_string(n) + ": " + message).c_str(), frameCtx);
            return nullptr;
        };

        const VSMap *props = vsapi->getFramePropertiesRO(src);
        Description desc;
        if (const std::string missing = ReadDescription(props, d->mvuprefix, desc, vsapi); !missing.empty())
            return fail("the frame lacks the property " + missing);
        // mvu's rule: a frame has vectors when it has both arrays, a block each
        const std::string vectorsKey = d->mvuprefix + "AnalysisVectors", sadKey = d->mvuprefix + "AnalysisSAD";
        const int64_t numVectors = vsapi->mapNumElements(props, vectorsKey.c_str()), numSads = vsapi->mapNumElements(props, sadKey.c_str());
        const bool has = numVectors == numSads && numVectors == desc[key::NBlkX] * desc[key::NBlkY];
        if (has) {
            if (!SameAnalysis(desc, d->first))
                return fail("the vectors were made with other Analyse or Recalculate arguments than the clip's first frame's");
            if (vsapi->mapGetType(props, vectorsKey.c_str()) != ptInt || vsapi->mapGetType(props, sadKey.c_str()) != ptInt)
                return fail(vectorsKey + " and " + sadKey + " must be integers");
        }

        // The output: the records, with the properties of mvu's frame but for mvu's own, which make way
        // for mvgpu's: its analysis description, the vectors' arrays (now the records) and the
        // description of its super, whose planes its frames hold as property frames
        dst = vkapi->newGPUVideoFrame(&d->gray32, 4 * G.nbx, G.nby, src, core);
        if (!dst)
            return fail("failed to allocate the vector frame");
        VSMap *out = vsapi->getFramePropertiesRW(dst);
        DeleteKeysStartingWith(out, d->mvuprefix + "Analysis", vsapi);
        DeleteKeysStartingWith(out, d->mvuprefix + "Super", vsapi);
        ExportSuper(dst, L, d->regions, d->prefix, vsapi);
        ExportAnalysis(dst, G, static_cast<int>(desc[key::Delta]), desc[key::Chroma] != 0, has, d->prefix, vsapi);
        // As the vectors have them: the levels the search went through, and the bit depth, which may be
        // another than the super's (vectors analysed on an 8-bit copy of a clip serve it, as in mvu)
        vsapi->mapSetInt(out, (d->prefix + "AnalysisLevels").c_str(), desc[key::Levels], maReplace);
        vsapi->mapSetInt(out, (d->prefix + "AnalysisBitsPerSample").c_str(), desc[key::Bits], maReplace);
        if (!has) {
            // Records that don't count, as on mvgpu's own frames without vectors
            vsapi->freeFrame(src);
            return dst;
        }

        VSVulkanPlaneInfo plane;
        if (vkapi->getGPUPlane(dst, 0, &plane))
            return fail("the vector frame isn't GPU resident");
        char err[1024] = {};
        VSVulkanBufferInfo info = {};
        staging = vkapi->createGPUBuffer(core, plane.bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0,
                                         &info, err, sizeof(err));
        if (!staging)
            return fail(std::string("failed to allocate a staging buffer: ") + err);

        // The records, each vector checked as mvu checks the vectors it loads (ValidateVectors): every
        // block read at its vector stays inside the padded planes, and the SADs aren't negative, nor
        // too large for the records' unsigned 32 bits
        const int64_t *vectors = vsapi->mapGetIntArray(props, vectorsKey.c_str(), nullptr);
        const int64_t *sads = vsapi->mapGetIntArray(props, sadKey.c_str(), nullptr);
        const ptrdiff_t stride = vsapi->getStride(dst, 0);
        const int logPel = ilog2(L.pel);
        const int64_t paddedW = desc[key::Width] + 2 * desc[key::HPad], paddedH = desc[key::Height] + 2 * desc[key::VPad];
        const int64_t stepX = desc[key::BlkX] - desc[key::OverlapX], stepY = desc[key::BlkY] - desc[key::OverlapY];
        for (int by = 0; by < G.nby; ++by) {
            int32_t *row = reinterpret_cast<int32_t *>(static_cast<uint8_t *>(info.mapped) + by * stride);
            const int64_t y0 = desc[key::VPad] + stepY * by;
            const int64_t dyMin = -(y0 << logPel), dyMax = (paddedH - y0 - desc[key::BlkY]) << logPel;
            for (int bx = 0; bx < G.nbx; ++bx) {
                const size_t i = static_cast<size_t>(by) * G.nbx + bx;
                const int64_t x0 = desc[key::HPad] + stepX * bx;
                const int64_t dxMin = -(x0 << logPel), dxMax = (paddedW - x0 - desc[key::BlkX]) << logPel;
                const int32_t vx = static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(vectors[i])));
                const int32_t vy = static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(vectors[i]) >> 32));
                if (vx < dxMin || vx >= dxMax || vy < dyMin || vy >= dyMax)
                    return fail("the vector of block (" + std::to_string(bx) + ", " + std::to_string(by) + ") would read outside the padded frame");
                if (sads[i] < 0 || sads[i] > std::numeric_limits<uint32_t>::max())
                    return fail("the SAD of block (" + std::to_string(bx) + ", " + std::to_string(by) + ") is negative or doesn't fit 32 bits");
                row[4 * bx] = vx;
                row[4 * bx + 1] = vy;
                row[4 * bx + 2] = static_cast<int32_t>(static_cast<uint32_t>(sads[i]));
                row[4 * bx + 3] = 0;
            }
            std::memset(row + 4 * G.nbx, 0, static_cast<size_t>(stride) - 16 * static_cast<size_t>(G.nbx));
        }

        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        vkapi->gpuExecWritesPlane(ctx, dst, 0);
        VkBufferCopy2 region = {VK_STRUCTURE_TYPE_BUFFER_COPY_2};
        region.size = plane.bufferSize;
        VkCopyBufferInfo2 copy = {VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
        copy.srcBuffer = info.buffer;
        copy.dstBuffer = plane.buffer;
        copy.regionCount = 1;
        copy.pRegions = &region;
        vc.vk->vkCmdCopyBuffer2(vkapi->gpuExecCommandBuffer(ctx), &copy);
        vkapi->gpuExecUsesBuffer(ctx, staging);
        staging = nullptr; // the context's now, destroyed once the copy has run
        const int submitted = vkapi->gpuExecSubmit(ctx, nullptr, err, sizeof(err));
        ctx = nullptr;
        if (submitted)
            return fail(err);

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC fromMVUCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    const std::string prefix = Prefix(in, "prefix", DEFAULT_MVGPUTENSILS_PREFIX, vsapi);
    const std::string mvuprefix = Prefix(in, "mvuprefix", DEFAULT_MVUTENSILS_PREFIX, vsapi);

    // The super: its layout and where its frames keep their parts, which every output frame describes
    SuperLayout layout;
    SuperRegions regions;
    VSNode *super = vsapi->mapGetNode(in, "super", 0, nullptr);
    try {
        layout = ImportSuperLayout(super, prefix, vsapi);
        char err[1024] = {};
        const VSFrame *frame = vsapi->getFrame(0, super, err, sizeof(err));
        if (!frame)
            throw std::runtime_error(std::string("failed to get the super clip's first frame: ") + err);
        const bool storage = GetSuperRegions(frame, layout, regions, vsapi);
        vsapi->freeFrame(frame);
        if (!storage)
            throw std::runtime_error("the super clip's frames aren't its storage; it must come from mvgpu.Super with the same prefix");
    } catch (const std::exception &e) {
        vsapi->freeNode(super);
        vsapi->mapSetError(out, (std::string("FromMVU: ") + e.what()).c_str());
        return;
    }
    vsapi->freeNode(super);
    const SuperLayout &L = layout;

    CreateEach("FromMVU", in, out, [&](int i) {
        std::unique_ptr<FromMVUData> d = std::make_unique<FromMVUData>(vsapi);
        d->prefix = prefix;
        d->mvuprefix = mvuprefix;
        d->layout = layout;
        d->regions = regions;
        d->node = vsapi->mapGetNode(in, "vectors", i, nullptr);

        char err[1024] = {};
        const VSFrame *frame = vsapi->getFrame(0, d->node, err, sizeof(err));
        if (!frame)
            throw std::runtime_error(std::string("failed to get the vector clip's first frame: ") + err);
        const VSMap *props = vsapi->getFramePropertiesRO(frame);
        std::string missing = ReadDescription(props, mvuprefix, d->first, vsapi);
        const bool mvgpu = vsapi->mapNumElements(props, (prefix + "AnalysisHasVectors").c_str()) > 0;
        vsapi->freeFrame(frame);
        if (!missing.empty())
            throw std::runtime_error("the vector clip lacks the property " + missing + "; " +
                                     (mvgpu ? "its vectors are mvgpu's already" : "it must come from mvu.Analyse, mvu.AnalyseMany or mvu.Recalculate with mvuprefix as their prefix"));
        const Description &v = d->first;

        // The super must be one of the clip the vectors were analysed on, or of the clip at another bit
        // depth: the description mvgpu's filters check vector clips by, as the super they were
        // analysed on. Their grid is one of mvu's, the super's or another that fits its block-aligned
        // frame, as mvu.Analyse and mvu.Recalculate take one.
        if (v[key::BlkX] < 1 || v[key::BlkX] > 128 || v[key::BlkY] < 1 || v[key::BlkY] > 128 || v[key::OverlapX] < 0 || v[key::OverlapY] < 0)
            throw std::runtime_error("the vector clip's description is invalid");
        const int blkX = static_cast<int>(v[key::BlkX]), blkY = static_cast<int>(v[key::BlkY]);
        const int overlapX = static_cast<int>(v[key::OverlapX]), overlapY = static_cast<int>(v[key::OverlapY]);
        CheckBlockSize(blkX, blkY, overlapX, overlapY, L.format.xr > 1 ? 1 : 0, L.format.yr > 1 ? 1 : 0);
        if (v[key::RealWidth] != L.width || v[key::RealHeight] != L.height || v[key::HPad] != L.pad || v[key::VPad] != L.padY || v[key::Pel] != L.pel)
            throw std::runtime_error("the vectors were analysed on a " + std::to_string(v[key::RealWidth]) + "x" + std::to_string(v[key::RealHeight]) + " clip padded by " +
                                     std::to_string(v[key::HPad]) + "x" + std::to_string(v[key::VPad]) + " at pel " + std::to_string(v[key::Pel]) +
                                     "; super must be mvgpu.Super of that clip with the same pad and pel");
        d->grid = L.WithGrid(blkX, blkY, overlapX, overlapY);
        const SuperLayout &G = d->grid;
        if (v[key::Width] != L.aw || v[key::Height] != L.ah || v[key::NBlkX] != G.nbx || v[key::NBlkY] != G.nby)
            throw std::runtime_error("the vectors' " + std::to_string(v[key::NBlkX]) + "x" + std::to_string(v[key::NBlkY]) + " blocks covering " +
                                     std::to_string(v[key::Width]) + "x" + std::to_string(v[key::Height]) + " aren't their grid's on the super");
        if (v[key::Chroma] && (!L.format.chroma || v[key::XRatio] != L.format.xr || v[key::YRatio] != L.format.yr))
            throw std::runtime_error("the vectors' SADs count chroma subsampled " + std::to_string(v[key::XRatio]) + "x" + std::to_string(v[key::YRatio]) +
                                     ", which the super clip's format doesn't have");
        if (v[key::Delta] == 0 || v[key::Levels] < 1 || ((v[key::Bits] < 8 || v[key::Bits] > 16) && v[key::Bits] != 32))
            throw std::runtime_error("the vector clip's description is invalid");

        if (!vsapi->queryVideoFormat(&d->gray32, cfGray, stInteger, 32, 0, 0, core))
            throw std::runtime_error("failed to query the Gray32 format");
        d->vi = *vsapi->getVideoInfo(d->node);
        d->vi.format = d->gray32;
        d->vi.width = 4 * G.nbx;
        d->vi.height = G.nby;

        d->vc = VulkanContext::Get(core, vsapi);
        d->pool = d->vc->vkapi->createGPUExecPool(core, vqCompute, err, sizeof(err));
        if (!d->pool)
            throw std::runtime_error(err);

        VSFilterDependency deps[1] = {
            {d->node, rpStrictSpatial},
        };
        vsapi->createVideoFilterEx(out, "FromMVU", &d->vi, fromMVUGetFrame, filterFree<FromMVUData>, fmParallel, ffGPUOutput, deps, ARRAY_SIZE(deps), d.get(), core);
        if (!vsapi->mapGetError(out))
            d.release();
    }, vsapi);
}

void convertRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("ToMVU",
                             "vectors:vnode[]:all;"
                             "prefix:data:opt;"
                             "mvuprefix:data:opt;",
                             "clip:vnode[];", toMVUCreate, nullptr, plugin);
    vspapi->registerFunction("FromMVU",
                             "vectors:vnode[]:all;"
                             "super:vnode:gpu;"
                             "prefix:data:opt;"
                             "mvuprefix:data:opt;",
                             "clip:vnode[]:gpu;", fromMVUCreate, nullptr, plugin);
}
