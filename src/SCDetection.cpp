#include <algorithm>
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

// mvu.SCDetection with mvgpu's vectors: the clip's frames, GPU resident or not, with
// _SceneChangePrev (vectors pointing back) or _SceneChangeNext (forward) set to 1 where the frame's
// vectors don't serve, absent or at a scene change. The scene change test counts the blocks above
// thscd1 on the GPU (mask_blocks.comp), and the host waits for the count, since the property is the
// host's: the one filter here whose frames wait on their submission.

struct SCDetectionData {
    VSNode *node = nullptr;    // the clip
    VSNode *vectors = nullptr;
    const VSVideoInfo *vi = nullptr;
    std::string prefix;
    const char *prop = nullptr; // the property set
    int nbx = 0, nby = 0;
    SceneChange scd;

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline count = VK_NULL_HANDLE;
    VSGPUBuffer *dummy = nullptr; // bound where the kernel reads nothing
    VSVulkanBufferInfo dummyInfo = {};

    const VSAPI *vsapi;

    SCDetectionData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~SCDetectionData() {
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        if (dummy)
            vc->vkapi->destroyGPUBuffer(dummy);
        vc.reset();
        vsapi->freeNode(vectors);
        vsapi->freeNode(node);
    }
};

// The count of the vector frame's blocks above thscd1, counted on the GPU and waited for
static int64_t CountBadBlocks(const SCDetectionData *d, const VSFrame *vectors, VSCore *core, const VSAPI *vsapi) {
    const VulkanContext &vc = *d->vc;
    const VSVULKANAPI *vkapi = vc.vkapi;
    VSVulkanPlaneInfo vecPlane = {};
    if (vkapi->getGPUPlane(vectors, 0, &vecPlane))
        throw std::runtime_error("the vectors aren't GPU resident");
    const ptrdiff_t recBytes = vsapi->getStride(vectors, 0);
    if (vsapi->getFrameWidth(vectors, 0) != 4 * d->nbx || vsapi->getFrameHeight(vectors, 0) != d->nby || recBytes % 16)
        throw std::runtime_error("a vector frame doesn't match the vectors' grid");

    // The count comes back through host visible memory, zeroed here: host writes before the submit
    // are visible to it
    char err[1024] = {};
    VSVulkanBufferInfo info = {};
    VSGPUBuffer *readback = vkapi->createGPUBuffer(core, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &info, err, sizeof(err));
    if (!readback)
        throw std::runtime_error(err);
    memset(info.mapped, 0, 16);
    VSGPUExecContext *ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
    if (!ctx) {
        vkapi->destroyGPUBuffer(readback);
        throw std::runtime_error(err);
    }
    vkapi->gpuExecReadsFrame(ctx, vectors);
    Recorder rec(vc, vkapi->gpuExecCommandBuffer(ctx), d->dummyInfo.buffer);
    RecordBadBlockCount(rec, d->count, vecPlane.buffer, static_cast<int>(recBytes / 16), d->nbx, d->nby, d->scd.thscd1, info.buffer, 0);
    rec.ComputeToHost();
    uint64_t signaled = 0;
    if (vkapi->gpuExecSubmit(ctx, &signaled, err, sizeof(err))) {
        vkapi->destroyGPUBuffer(readback); // never submitted
        throw std::runtime_error(err);
    }
    const int drained = vkapi->gpuExecWaitValue(d->pool, signaled, err, sizeof(err));
    if (drained != gdDrained) {
        // Still queued, it may yet write the buffer: leave it be rather than free memory in use
        if (vsGPUDrainSafeToDestroy(drained))
            vkapi->destroyGPUBuffer(readback);
        throw std::runtime_error(std::string("waiting for the scene change count failed: ") + err);
    }
    uint32_t count = 0;
    memcpy(&count, info.mapped, 4);
    vkapi->destroyGPUBuffer(readback);
    return count;
}

static const VSFrame *VS_CC scdetectionGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    SCDetectionData *d = reinterpret_cast<SCDetectionData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        vsapi->requestFrameFilter(n, d->vectors, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrame *dst = vsapi->copyFrame(src, core);
        vsapi->freeFrame(src);

        const VSFrame *frame = vsapi->getFrameFilter(n, d->vectors, frameCtx);
        const VSFrame *vectors = GetAnalysisVectors(frame, d->prefix, vsapi);
        vsapi->freeFrame(frame);
        try {
            // mvu's !IsUsable: no vectors, or more blocks above thscd1 than the limit (never with as
            // many blocks as the limit)
            bool change = true;
            if (vectors)
                change = d->scd.limit < d->nbx * d->nby && CountBadBlocks(d, vectors, core, vsapi) > d->scd.limit;
            vsapi->mapSetInt(vsapi->getFramePropertiesRW(dst), d->prop, change, maReplace);
        } catch (const std::exception &e) {
            vsapi->freeFrame(vectors);
            vsapi->freeFrame(dst);
            vsapi->setFilterError((std::string("SCDetection: ") + e.what()).c_str(), frameCtx);
            return nullptr;
        }
        vsapi->freeFrame(vectors);
        return dst;
    }

    return nullptr;
}

static void VS_CC scdetectionCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<SCDetectionData> d = std::make_unique<SCDetectionData>(vsapi);

    try {
        int err;
        int64_t thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
        if (err)
            thscd1 = MV_DEFAULT_SCD1;
        float thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
        if (err)
            thscd2 = MV_DEFAULT_SCD2;
        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        d->prefix = prefix ? prefix : DEFAULT_MVGPUTENSILS_PREFIX;

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vectors = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        const VectorInfo v = ReadVectorInfo(d->vectors, d->prefix, vsapi);
        const SuperLayout analysed = ImportSuperLayout(d->vectors, d->prefix, vsapi);
        if (v.nbx != analysed.nbx || v.nby != analysed.nby)
            throw std::runtime_error("the vectors' grid isn't their super's; they must come from mvgpu.Analyse");
        d->nbx = v.nbx;
        d->nby = v.nby;
        d->scd = ScaleSceneChange(v, thscd1, thscd2);
        d->prop = v.delta > 0 ? "_SceneChangeNext" : "_SceneChangePrev";

        d->vc = VulkanContext::Get(core, vsapi);
        VulkanContext &vc = *d->vc;
        d->count = vc.Pipeline(Kernel::MaskBlocks, 0, 0, 0);
        char errMsg[1024] = {};
        d->pool = vc.vkapi->createGPUExecPool(core, vqCompute, errMsg, sizeof(errMsg));
        if (!d->pool)
            throw std::runtime_error(errMsg);
        const std::vector<int32_t> zeros(64, 0);
        d->dummy = vc.Upload(core, d->pool, zeros.data(), zeros.size() * sizeof(int32_t), d->dummyInfo);
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, (std::string("SCDetection: ") + e.what()).c_str());
        return;
    }

    VSFilterDependency deps[2] = {
        {d->node, rpStrictSpatial},
        {d->vectors, rpStrictSpatial},
    };
    // The clip's frames pass through, wherever they live
    const int flags = vsapi->getNodeResidency(d->node) == nrGPU ? ffGPUOutput : 0;
    vsapi->createVideoFilterEx(out, "SCDetection", d->vi, scdetectionGetFrame, filterFree<SCDetectionData>, fmParallel, flags, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void scdetectionRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("SCDetection",
                             "clip:vnode:all;"
                             "vectors:vnode:gpu;"
                             "thscd1:int:opt;"
                             "thscd2:float:opt;"
                             "prefix:data:opt;",
                             "clip:vnode:all;", scdetectionCreate, nullptr, plugin);
}
