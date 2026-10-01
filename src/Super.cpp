#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <VSVulkan4.h>

#include "Common.h"
#include "SuperLayout.h"
#include "VulkanContext.h"

// mvu.Super on the GPU: the frame padded to the block-aligned size the grid covers, its half-pel
// planes and the coarse levels the search starts from, built by super.comp and pyr_reduce.comp
// into the frames SuperLayout.h describes and attached to the output frame, which is the input
// frame itself. Implemented: 8-bit 4:2:0, pel 2, sharp 2, rfilter 1, square blocks of 8 or 16, the
// same padding horizontally and vertically.

struct SuperData {
    VSNode *node = nullptr;
    VSVideoInfo vi = {};

    SuperLayout layout;
    std::string prefix;

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline superPipeline = VK_NULL_HANDLE;
    VkPipeline reducePipeline = VK_NULL_HANDLE;
    VSGPUBuffer *constants = nullptr; // the level table
    VSVulkanBufferInfo constantsInfo = {};
    VSVideoFormat gray8 = {};

    const VSAPI *vsapi;

    SuperData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~SuperData() {
        // The pool drains the GPU first, so nothing still uses what follows
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        if (constants)
            vc->vkapi->destroyGPUBuffer(constants);
        vc.reset();
        vsapi->freeNode(node);
    }
};

static const VSFrame *VS_CC superGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    SuperData *d = reinterpret_cast<SuperData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSVULKANAPI *vkapi = d->vc->vkapi;
        const SuperLayout &L = d->layout;
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrame *luma = vkapi->newGPUVideoFrame(&d->gray8, L.wp, L.LumaRows(), nullptr, core);
        VSFrame *chroma = vkapi->newGPUVideoFrame(&d->gray8, L.wc, L.ChromaRows(), nullptr, core);
        VSFrame *pyramid = L.topLevel > 0 ? vkapi->newGPUVideoFrame(&d->gray8, L.PyramidWidth(), L.PyramidRows(), nullptr, core) : nullptr;
        VSGPUExecContext *ctx = nullptr;
        auto fail = [&](const std::string &message) -> const VSFrame * {
            if (ctx)
                vkapi->gpuExecAbandon(ctx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(luma);
            vsapi->freeFrame(chroma);
            vsapi->freeFrame(pyramid);
            vsapi->setFilterError(("Super: " + message).c_str(), frameCtx);
            return nullptr;
        };
        if (!luma || !chroma || (L.topLevel > 0 && !pyramid))
            return fail("failed to allocate the GPU frames");

        VSVulkanPlaneInfo srcPlanes[3], lumaPlane, chromaPlane, pyramidPlane = {};
        for (int p = 0; p < 3; ++p)
            if (vkapi->getGPUPlane(src, p, &srcPlanes[p]))
                return fail("the input frame isn't GPU resident");
        if (vkapi->getGPUPlane(luma, 0, &lumaPlane) || vkapi->getGPUPlane(chroma, 0, &chromaPlane) || (pyramid && vkapi->getGPUPlane(pyramid, 0, &pyramidPlane)))
            return fail("the storage frames aren't GPU resident");
        const ptrdiff_t lumaStride = vsapi->getStride(luma, 0), chromaStride = vsapi->getStride(chroma, 0);
        if (lumaStride % 4 || chromaStride % 4)
            return fail("the storage frames' rows don't start on whole words");

        char err[1024] = {};
        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        vkapi->gpuExecReadsFrame(ctx, src);
        vkapi->gpuExecWritesPlane(ctx, luma, 0);
        vkapi->gpuExecWritesPlane(ctx, chroma, 0);
        if (pyramid)
            vkapi->gpuExecWritesPlane(ctx, pyramid, 0);

        Recorder rec(*d->vc, vkapi->gpuExecCommandBuffer(ctx), d->constantsInfo.buffer);
        rec.Bind(kCurLuma, lumaPlane.buffer);
        rec.Bind(kCurChroma, chromaPlane.buffer);
        if (pyramid)
            rec.Bind(kCurPyramid, pyramidPlane.buffer);
        rec.Bind(kLevels, d->constantsInfo.buffer);
        rec.Bind(kRawY, srcPlanes[0].buffer);
        rec.Bind(kRawU, srcPlanes[1].buffer);
        rec.Bind(kRawV, srcPlanes[2].buffer);

        Params q = {};
        q.w = L.width;
        q.h = L.height;
        q.nbx = L.nbx;
        q.nby = L.nby;
        q.step = L.step;
        q.pad = L.pad;
        q.padc = L.padc;
        q.wp = static_cast<int32_t>(lumaStride);
        q.hp = L.hp;
        q.wc = static_cast<int32_t>(chromaStride);
        q.hc = L.hc;
        q.blockRows = L.blk;
        q.srcStrideY = static_cast<int32_t>(vsapi->getStride(src, 0));
        q.srcStrideC = static_cast<int32_t>(vsapi->getStride(src, 1));
        // The full-pel planes from the frame, then the x + 1/2 and y + 1/2 planes from them, then
        // the diagonal plane from the y + 1/2 plane; 256 pixels of a padded row per workgroup
        const uint32_t padded = static_cast<uint32_t>(L.aw + 2 * L.pad), rows = static_cast<uint32_t>(L.hp);
        for (int step = 0; step < 3; ++step) {
            q.level = step;
            rec.Dispatch(d->superPipeline, q, (padded + 255) / 256, rows, 3);
            rec.ComputeBarrier();
        }
        // Each coarse level from the one below it, borders included
        for (int level = 1; level <= L.topLevel; ++level) {
            const LevelEntry &e = L.levels[level];
            q.level = level;
            rec.Dispatch(d->reducePipeline, q, static_cast<uint32_t>((e.w + 2 * e.borderY + 63) / 64), static_cast<uint32_t>(e.h + 2 * e.borderY), 3);
            if (level < L.topLevel)
                rec.ComputeBarrier();
        }
        const int submitted = vkapi->gpuExecSubmit(ctx, nullptr, err, sizeof(err));
        ctx = nullptr;
        if (submitted)
            return fail(err);

        VSFrame *dst = vsapi->copyFrame(src, core);
        SuperFrames frames;
        frames.luma = luma;
        frames.chroma = chroma;
        frames.pyramid = pyramid;
        ExportSuper(dst, L, frames, d->prefix, vsapi);
        frames.Free(vsapi);
        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC superCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<SuperData> d = std::make_unique<SuperData>(vsapi);
    int err;

    try {
        int hpad, vpad;
        GetPairArgument(hpad, vpad, "pad", 16, 16, in, vsapi);

        if (hpad <= 0 || vpad <= 0)
            throw std::runtime_error("pad must be positive");

        int pel = vsapi->mapGetIntSaturated(in, "pel", 0, &err);
        if (err)
            pel = 2;

        const bool onelevel = !!vsapi->mapGetIntSaturated(in, "onelevel", 0, &err);

        int sharp = vsapi->mapGetIntSaturated(in, "sharp", 0, &err);
        if (err)
            sharp = 2;

        int rfilter = vsapi->mapGetIntSaturated(in, "rfilter", 0, &err);
        if (err)
            rfilter = 1;

        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        if (prefix)
            d->prefix = prefix;
        else
            d->prefix = DEFAULT_MVGPUTENSILS_PREFIX;

        if ((pel != 1) && (pel != 2) && (pel != 4))
            throw std::runtime_error("pel must be 1, 2, or 4");

        if (sharp < 0 || sharp > 2)
            throw std::runtime_error("sharp must be between 0 and 2");

        if (rfilter < 0 || rfilter > 2)
            throw std::runtime_error("rfilter must be between 0 and 2");

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(&d->vi) || (d->vi.format.bitsPerSample > 16 && d->vi.format.sampleType == stInteger) || (d->vi.format.bitsPerSample != 32 && d->vi.format.sampleType == stFloat) ||
            d->vi.format.subSamplingW > 1 || d->vi.format.subSamplingH > 1 || (d->vi.format.colorFamily != cfYUV && d->vi.format.colorFamily != cfGray))
            throw std::runtime_error("input clip must be GRAY, YUV420, YUV422, YUV440, or YUV444, up to 16 bits integer or 32 bit float, with constant dimensions");

        int blkX, blkY, overlapX, overlapY;
        GetPairArgument(blkX, blkY, "blksize", 8, 8, in, vsapi);
        GetPairArgument(overlapX, overlapY, "overlap", 0, 0, in, vsapi);

        CheckBlockSize(blkX, blkY, overlapX, overlapY, d->vi.format.subSamplingW, d->vi.format.subSamplingH);

        VSNode *pelclipNode = vsapi->mapGetNode(in, "pelclip", 0, &err);
        const bool pelclip = pelclipNode != nullptr;
        vsapi->freeNode(pelclipNode);

        // What the GPU path implements so far
        if (d->vi.format.colorFamily != cfYUV || d->vi.format.sampleType != stInteger || d->vi.format.bitsPerSample != 8 || d->vi.format.subSamplingW != 1 ||
            d->vi.format.subSamplingH != 1)
            throw std::runtime_error("only 8-bit YUV420 input is implemented so far");
        if (pel != 2)
            throw std::runtime_error("only pel=2 is implemented so far");
        if (sharp != 2)
            throw std::runtime_error("only sharp=2 is implemented so far");
        if (rfilter != 1)
            throw std::runtime_error("only rfilter=1 is implemented so far");
        if (pelclip)
            throw std::runtime_error("pelclip isn't implemented yet");
        if (blkX != blkY || (blkX != 8 && blkX != 16))
            throw std::runtime_error("only 8x8 and 16x16 blocks are implemented so far");
        if (overlapX != overlapY)
            throw std::runtime_error("only the same overlap horizontally and vertically is implemented so far");
        if (hpad != vpad)
            throw std::runtime_error("only the same padding horizontally and vertically is implemented so far");

        d->layout = SuperLayout::Make(d->vi.width, d->vi.height, blkX, overlapX, hpad, !onelevel);

        d->vc = VulkanContext::Get(core, vsapi);
        d->superPipeline = d->vc->Pipeline(Kernel::Super, blkX);
        if (d->layout.topLevel > 0)
            d->reducePipeline = d->vc->Pipeline(Kernel::PyrReduce, blkX);

        char errMsg[1024] = {};
        d->pool = d->vc->vkapi->createGPUExecPool(core, vqCompute, errMsg, sizeof(errMsg));
        if (!d->pool)
            throw std::runtime_error(errMsg);
        const std::vector<LevelEntry> &table = d->layout.levels;
        d->constants = d->vc->Upload(core, d->pool, table.data(), table.size() * sizeof(LevelEntry), d->constantsInfo);

        if (!vsapi->queryVideoFormat(&d->gray8, cfGray, stInteger, 8, 0, 0, core))
            throw std::runtime_error("failed to query the Gray8 format");
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("Super: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[1] = {
        { d->node, rpStrictSpatial }
    };

    vsapi->createVideoFilterEx(out, "Super", &d->vi, superGetFrame, filterFree<SuperData>, fmParallel, ffGPUOutput, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void superRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Super",
                 "clip:vnode:gpu;"
                 "blksize:int[];"
                 "overlap:int[];"
                 "pad:int[]:opt;"
                 "onelevel:int:opt;"
                 "sharp:int:opt;"
                 "rfilter:int:opt;"
                 "pel:int:opt;"
                 "pelclip:vnode:gpu:opt;"
                 "prefix:data:opt;",
                 "clip:vnode:gpu;",
                 superCreate, nullptr, plugin);
}
