#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <VSVulkan4.h>

#include "Common.h"
#include "FilterShared.h"
#include "SuperLayout.h"
#include "VulkanContext.h"

// mvu.VectorLengthMask, mvu.SADMask and mvu.OcclusionMask on the GPU: a Gray mask of the frame's
// size from a vector clip, each block's value spread over the pixels by zimg's bilinear resize of the
// whole plane (mvu's PlaneResizer), or scval where the vectors don't serve. Two kernels per frame:
// the blocks' values and the scene change test's count (mask_blocks.comp), then every pixel
// (mask_resize.comp). The masks are mvu's bit for bit given the same vectors, but for a gamma other
// than 1 (other than 1 or 2 for the vector length), which goes through the GPU's pow rather than the
// C library's, a block's value then possibly a step apart from mvu's.
//
// Implemented: all of their arguments, on vectors from mvgpu.Analyse.

namespace {

// mask_common.glsl's kinds and flags
constexpr int kLength = 0, kSad = 1, kOcclusion = 2;
constexpr int kNoVectors = 1;

// The _Range mvu sets on its masks: full range in API 4.2's VSRange, the API mvu is built for
// (VSConstants4.h gives the API 4.0 values under VS_USE_API_43)
constexpr int kRangeFull = 1;

} // namespace

struct MaskData {
    VSNode *node = nullptr; // the vectors
    VectorInfo info;        // their first frame's description, which every frame with vectors must have
    VSVideoInfo vi = {};    // the mask's
    int kind = kLength;
    std::string name, prefix;
    int nbx = 0, nby = 0;
    MaskParams base = {}; // the push constants of every frame, but for the vectors' and the mask's strides

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline blocks = VK_NULL_HANDLE, resize = VK_NULL_HANDLE;
    VSGPUBuffer *taps = nullptr; // the resize's taps (PlaneTaps), also bound where a frame has nothing
    VSVulkanBufferInfo tapsInfo = {};
    std::unique_ptr<StageProfiler> profile; // MVGPU_PROFILE

    const VSAPI *vsapi;

    MaskData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~MaskData() {
        // The pool drains the GPU first, so nothing still uses what follows
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        profile.reset();
        if (taps)
            vc->vkapi->destroyGPUBuffer(taps);
        vc.reset();
        vsapi->freeNode(node);
    }
};

static const VSFrame *VS_CC maskGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    MaskData *d = reinterpret_cast<MaskData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VulkanContext &vc = *d->vc;
        const VSVULKANAPI *vkapi = vc.vkapi;

        const VSFrame *frame = vsapi->getFrameFilter(n, d->node, frameCtx);
        std::string error;
        const VSFrame *vectors = GetAnalysisVectors(frame, d->info, d->prefix, error, vsapi);
        vsapi->freeFrame(frame);
        if (!error.empty()) {
            vsapi->setFilterError((d->name + ": " + error).c_str(), frameCtx);
            return nullptr;
        }

        VSFrame *dst = nullptr;
        VSGPUExecContext *ctx = nullptr;
        VSGPUBuffer *scratch = nullptr;
        auto fail = [&](const std::string &message) -> const VSFrame * {
            if (ctx)
                vkapi->gpuExecAbandon(ctx);
            else if (scratch)
                vkapi->destroyGPUBuffer(scratch);
            vsapi->freeFrame(vectors);
            vsapi->freeFrame(dst);
            vsapi->setFilterError((d->name + ": " + message).c_str(), frameCtx);
            return nullptr;
        };

        dst = vkapi->newGPUVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
        if (!dst)
            return fail("failed to allocate the output frame");
        vsapi->mapSetInt(vsapi->getFramePropertiesRW(dst), "_Range", kRangeFull, maAppend);

        VSVulkanPlaneInfo outPlane = {}, vecPlane = {};
        if (vkapi->getGPUPlane(dst, 0, &outPlane))
            return fail("the output frame isn't GPU resident");
        ptrdiff_t recBytes = 0;
        if (vectors) {
            if (vkapi->getGPUPlane(vectors, 0, &vecPlane))
                return fail("the vectors aren't GPU resident");
            recBytes = vsapi->getStride(vectors, 0);
            if (vsapi->getFrameWidth(vectors, 0) != 4 * d->nbx || vsapi->getFrameHeight(vectors, 0) != d->nby || recBytes % 16)
                return fail("a vector frame doesn't match the vectors' grid");
        }

        // Scratch: the scene change count, then the blocks' values
        const int nb = d->nbx * d->nby;
        const VkDeviceSize align = std::max<VkDeviceSize>(vc.limits.minStorageBufferOffsetAlignment, 16);
        const VkDeviceSize countBytes = (16 + align - 1) / align * align, valueBytes = static_cast<VkDeviceSize>(nb) * 4;
        char err[1024] = {};
        VSVulkanBufferInfo scratchInfo = {};
        if (vectors) {
            scratch = vkapi->createGPUBuffer(core, countBytes + valueBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &scratchInfo, err, sizeof(err));
            if (!scratch)
                return fail(err);
        }
        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        if (scratch)
            vkapi->gpuExecUsesBuffer(ctx, scratch);
        if (vectors)
            vkapi->gpuExecReadsFrame(ctx, vectors);
        vkapi->gpuExecWritesPlane(ctx, dst, 0);

        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        Recorder rec(vc, cmd, d->tapsInfo.buffer);
        const VkQueryPool queries = d->profile ? d->profile->Begin(vc, cmd) : VK_NULL_HANDLE;
        auto stamp = [&](int stage) {
            if (d->profile)
                d->profile->Stamp(vc, cmd, queries, stage);
        };

        MaskParams pc = d->base;
        pc.recStride = static_cast<int32_t>(recBytes / 16);
        pc.outStride = static_cast<int32_t>(vsapi->getStride(dst, 0) / d->vi.format.bytesPerSample);
        pc.flags = vectors ? 0 : kNoVectors;
        rec.Bind(kMkTaps, d->tapsInfo.buffer);
        rec.Bind(kMkOut, outPlane.buffer);
        if (vectors) {
            rec.Bind(kMkVectors, vecPlane.buffer);
            rec.Bind(kMkCounts, scratchInfo.buffer, 0, countBytes);
            rec.Bind(kMkValues, scratchInfo.buffer, countBytes, valueBytes);
            // The count and the occlusion mask's largest values start at zero
            rec.Fill(scratchInfo.buffer, 0, countBytes + valueBytes, 0);
            rec.TransferToCompute();
            rec.Dispatch(d->blocks, pc, static_cast<uint32_t>((nb + 63) / 64), 1);
            rec.ComputeBarrier();
        }
        stamp(1);
        rec.Dispatch(d->resize, pc, static_cast<uint32_t>((d->vi.width + 63) / 64), static_cast<uint32_t>(d->vi.height));
        stamp(2);

        uint64_t signaled = 0;
        const int submitted = vkapi->gpuExecSubmit(ctx, &signaled, err, sizeof(err));
        ctx = nullptr;
        scratch = nullptr; // the context owned it
        if (submitted)
            return fail(err);
        if (d->profile)
            d->profile->Finish(vc, d->pool, signaled, queries);
        vsapi->freeFrame(vectors);
        return dst;
    }

    return nullptr;
}

// userData: the mask's kind (kLength, kSad, kOcclusion)
static void VS_CC maskCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<MaskData> d = std::make_unique<MaskData>(vsapi);
    d->kind = static_cast<int>(reinterpret_cast<intptr_t>(userData));
    d->name = d->kind == kLength ? "VectorLengthMask" : d->kind == kSad ? "SADMask" : "OcclusionMask";

    try {
        int err;

        float ml = vsapi->mapGetFloatSaturated(in, "ml", 0, &err);
        if (err)
            ml = 100.0f;
        float gamma = vsapi->mapGetFloatSaturated(in, "gamma", 0, &err);
        if (err)
            gamma = 1.0f;
        double time = vsapi->mapGetFloat(in, "time", 0, &err);
        if (err)
            time = 100.0;
        const float scval = vsapi->mapGetFloatSaturated(in, "scval", 0, &err);
        int64_t thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
        if (err)
            thscd1 = MV_DEFAULT_SCD1;
        float thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
        if (err)
            thscd2 = MV_DEFAULT_SCD2;

        if (!std::isfinite(gamma) || gamma < 0.0f)
            throw std::runtime_error("gamma must be a finite non-negative value");
        if (!std::isfinite(ml) || ml <= 0.0f)
            throw std::runtime_error("ml must be a finite value greater than 0");
        if (!std::isfinite(time) || time < 0.0 || time > 100.0)
            throw std::runtime_error("time must be between 0.0 and 100.0");

        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        d->prefix = prefix ? prefix : DEFAULT_MVGPUTENSILS_PREFIX;

        d->node = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        const VectorInfo v = d->info = ReadVectorInfo(d->node, d->prefix, vsapi);
        const SuperLayout analysed = ImportAnalysedLayout(d->node, d->prefix, vsapi);
        if (v.blkX != analysed.blk || v.blkY != analysed.blkY || v.overlapX != analysed.overlap || v.overlapY != analysed.overlapY || v.nbx != analysed.nbx ||
            v.nby != analysed.nby)
            throw std::runtime_error("the vectors' grid isn't their super's; they must come from mvgpu.Analyse");
        d->nbx = v.nbx;
        d->nby = v.nby;

        // A Gray mask of the frame's size at the vectors' bit depth
        d->vi = *vsapi->getVideoInfo(d->node);
        d->vi.width = v.realWidth;
        d->vi.height = v.realHeight;
        const bool isFloat = v.bits == 32;
        if (!vsapi->queryVideoFormat(&d->vi.format, cfGray, isFloat ? stFloat : stInteger, v.bits, 0, 0, core))
            throw std::runtime_error("the vectors' bit depth makes no Gray format");
        const int maxVal = isFloat ? 1 : (1 << v.bits) - 1;

        MaskParams &pc = d->base;
        if (!isFloat) {
            if (!std::isfinite(scval))
                throw std::runtime_error("scval must be a finite value for integer formats");
            pc.scval = static_cast<int>(scval + 0.5f); // round to nearest integer
            const int maxScval = (1 << std::min(16, v.bits)) - 1;
            if (pc.scval < 0 || pc.scval > maxScval)
                throw std::runtime_error("scval must be between 0 and " + std::to_string(maxScval));
        }
        pc.scvalF = scval;

        const SceneChange scd = ScaleSceneChange(v, thscd1, thscd2);
        pc.nbx = v.nbx;
        pc.nby = v.nby;
        pc.kind = d->kind;
        pc.width = d->vi.width;
        pc.height = d->vi.height;
        pc.thscd1 = scd.thscd1;
        pc.scdLimit = scd.limit;
        pc.pel = v.pel;
        pc.maxVal = maxVal;

        // The masks' arithmetic as mvu sets it up (Mask.cpp, MotionBlockPyramid.cpp's Make*Mask)
        const float normFactor = 1.0f / ml;
        const int time256 = static_cast<int>(time * 256 / 100);
        const int stepX = v.blkX - v.overlapX, stepY = v.blkY - v.overlapY;
        if (d->kind == kLength) {
            pc.norm = normFactor * normFactor;
            pc.normY = 1.0f / static_cast<float>(v.pel * v.pel); // exact
            pc.gamma = gamma / 2;
        } else if (d->kind == kSad) {
            pc.norm = 4.0f * normFactor / (v.blkX * v.blkY);
            pc.gamma = gamma;
            pc.time4096X = (256 - time256) * 16 / (stepX * v.pel);
            pc.time4096Y = (256 - time256) * 16 / (stepY * v.pel);
            pc.sadShift = std::min(16, v.bits) - 8;
        } else {
            pc.norm = (80.0f * normFactor) / (stepX * v.pel);
            pc.normY = (80.0f * normFactor) / (stepY * v.pel);
            pc.gamma = gamma;
            pc.time4096X = time256 * 16 / (stepX * v.pel);
            pc.time4096Y = time256 * 16 / (stepY * v.pel);
            pc.backward = v.delta > 0;
        }

        // The resize's taps, PlaneResizer's: the columns', then the rows'
        const int coverW = stepX * v.nbx + v.overlapX, coverH = stepY * v.nby + v.overlapY;
        if (!PlaneHorizontalFirst(v.nbx, d->vi.width, coverW, v.nby, d->vi.height, coverH))
            throw std::runtime_error("zimg would resize the mask vertically first, which isn't implemented");
        const std::vector<ResizeTap> cols = PlaneTaps(v.nbx, d->vi.width, coverW), rows = PlaneTaps(v.nby, d->vi.height, coverH);
        std::vector<int32_t> tables;
        auto append = [&](const std::vector<ResizeTap> &taps) {
            for (const ResizeTap &t : taps) {
                int32_t f0, f1;
                memcpy(&f0, &t.f0, 4);
                memcpy(&f1, &t.f1, 4);
                tables.insert(tables.end(), {t.left, t.c, f0, f1});
            }
        };
        pc.colOff = 0;
        append(cols);
        pc.rowOff = static_cast<int32_t>(tables.size() / 4);
        append(rows);
        tables.resize(std::max<size_t>(tables.size(), 64)); // a dummy for the bindings a frame doesn't use too

        d->vc = VulkanContext::Get(core, vsapi);
        VulkanContext &vc = *d->vc;
        // Specialization constant 6: the mask's samples
        const int sampleKind = v.bits == 8 ? 0 : isFloat ? 2 : 1;
        d->blocks = vc.Pipeline(Kernel::MaskBlocks, 0, 0, sampleKind);
        d->resize = vc.Pipeline(Kernel::MaskResize, 0, 0, sampleKind);
        if (static_cast<VkDeviceSize>(v.nbx) * v.nby * 4 > vc.limits.maxStorageBufferRange)
            throw std::runtime_error("the frame is too large for the device's storage buffers");

        char errMsg[1024] = {};
        d->pool = vc.vkapi->createGPUExecPool(core, vqCompute, errMsg, sizeof(errMsg));
        if (!d->pool)
            throw std::runtime_error(errMsg);
        d->taps = vc.Upload(core, d->pool, tables.data(), tables.size() * sizeof(int32_t), d->tapsInfo);
        if (StageProfiler::Requested())
            d->profile = std::make_unique<StageProfiler>(d->name, std::vector<std::string>{"blocks", "pixels"});
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, (d->name + ": " + e.what()).c_str());
        return;
    }

    VSFilterDependency deps[1] = {
        {d->node, rpStrictSpatial},
    };
    vsapi->createVideoFilterEx(out, d->name.c_str(), &d->vi, maskGetFrame, filterFree<MaskData>, fmParallel, ffGPUOutput, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void maskRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    static constexpr char args[] = "vectors:vnode:gpu;"
                                   "ml:float:opt;"
                                   "gamma:float:opt;"
                                   "time:float:opt;"
                                   "scval:float:opt;"
                                   "thscd1:int:opt;"
                                   "thscd2:float:opt;"
                                   "prefix:data:opt;";
    vspapi->registerFunction("VectorLengthMask", args, "clip:vnode:gpu;", maskCreate, reinterpret_cast<void *>(static_cast<intptr_t>(kLength)), plugin);
    vspapi->registerFunction("SADMask", args, "clip:vnode:gpu;", maskCreate, reinterpret_cast<void *>(static_cast<intptr_t>(kSad)), plugin);
    vspapi->registerFunction("OcclusionMask", args, "clip:vnode:gpu;", maskCreate, reinterpret_cast<void *>(static_cast<intptr_t>(kOcclusion)), plugin);
}
