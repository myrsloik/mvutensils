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

// mvu.Super on the GPU: the frame padded to the block-aligned size the grid covers, its sub-pel
// planes and the coarse levels the search starts from, built by super.comp, super_qpel.comp and
// pyr_reduce.comp into the frames SuperLayout.h describes and attached to the output frame, which
// is the input frame itself. All of mvu.Super's arguments: GRAY and YUV 4:2:0, 4:2:2, 4:4:0 and
// 4:4:4 at 8 to 16 bits or float, pel 1, 2 and 4, sharp, rfilter, onelevel, pelclip, every block
// size, overlap and padding mvu takes. Level 0 is mvu.Super's bit for bit (test/check_super.py),
// at pel 4 luma's quarter planes computed where they're read. That leaves no room for a pelclip's
// own quarter samples, so a pelclip at pel 4 is refused: a divergence from mvu.Super, which saves
// half of a pel 4 super's memory. The coarse levels are the GPU search's own (SuperLayout.h),
// reduced with rfilter's filter.

struct SuperData {
    VSNode *node = nullptr;
    VSNode *pelclip = nullptr; // with usePelClip, the half-pel planes come from it
    VSVideoInfo vi = {};    // the clip's
    VSVideoInfo outVi = {}; // the super clip's: the storage frames (SuperLayout.h)

    SuperLayout layout;
    std::string prefix;
    int sharp = 2, rfilter = 1;
    bool usePelClip = false;

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline planesPipeline = VK_NULL_HANDLE;
    VkPipeline quarterPipeline = VK_NULL_HANDLE; // pel 4 chroma: its quarter-pel images, whole
    VkPipeline reducePipeline = VK_NULL_HANDLE;  // with coarse levels
    VSGPUBuffer *constants = nullptr;            // the level table
    VSVulkanBufferInfo constantsInfo = {};
    VSVideoFormat storage = {}; // the storage frame's format: Gray of the clip's samples
    std::unique_ptr<StageProfiler> profile; // MVGPU_PROFILE

    const VSAPI *vsapi;

    SuperData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~SuperData() {
        // The pool drains the GPU first, so nothing still uses what follows
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        profile.reset();
        if (constants)
            vc->vkapi->destroyGPUBuffer(constants);
        vc.reset();
        vsapi->freeNode(node);
        vsapi->freeNode(pelclip);
    }
};

static const VSFrame *VS_CC superGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    SuperData *d = reinterpret_cast<SuperData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (d->usePelClip)
            vsapi->requestFrameFilter(n, d->pelclip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSVULKANAPI *vkapi = d->vc->vkapi;
        const SuperLayout &L = d->layout;
        const SuperFormat &F = L.format;
        const int planes = F.chroma ? 3 : 1;
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame *pel = d->usePelClip ? vsapi->getFrameFilter(n, d->pelclip, frameCtx) : nullptr;
        // The output frame is the storage, the clip's frame's properties carried over
        VSFrame *dst = vkapi->newGPUVideoFrame(&d->storage, L.FrameWidth(), L.FrameRows(), src, core);
        VSGPUExecContext *ctx = nullptr;
        auto fail = [&](const std::string &message) -> const VSFrame * {
            if (ctx)
                vkapi->gpuExecAbandon(ctx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(pel);
            vsapi->freeFrame(dst);
            vsapi->setFilterError(("Super: " + message).c_str(), frameCtx);
            return nullptr;
        };
        if (!dst)
            return fail("failed to allocate the GPU frame");

        VSVulkanPlaneInfo srcPlanes[3] = {}, pelPlanes[3] = {}, storagePlane = {};
        for (int p = 0; p < planes; ++p) {
            if (vkapi->getGPUPlane(src, p, &srcPlanes[p]))
                return fail("the input frame isn't GPU resident");
            if (pel && vkapi->getGPUPlane(pel, p, &pelPlanes[p]))
                return fail("the pelclip frame isn't GPU resident");
        }
        if (vkapi->getGPUPlane(dst, 0, &storagePlane))
            return fail("the storage frame isn't GPU resident");
        SuperRegions regions;
        if (!GetSuperRegions(dst, L, regions, vsapi))
            return fail("the storage frame doesn't hold the storage");
        const ptrdiff_t bytes = F.Bytes();
        const ptrdiff_t lumaStride = regions.lumaStride, chromaStride = regions.chromaStride;
        if (lumaStride % (4 * bytes) || chromaStride % (4 * bytes))
            return fail("the storage frame's rows don't start on groups of four samples");

        char err[1024] = {};
        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        vkapi->gpuExecReadsFrame(ctx, src);
        if (pel)
            vkapi->gpuExecReadsFrame(ctx, pel);
        vkapi->gpuExecWritesPlane(ctx, dst, 0);

        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        Recorder rec(*d->vc, cmd, d->constantsInfo.buffer);
        const VkQueryPool queries = d->profile ? d->profile->Begin(*d->vc, cmd) : VK_NULL_HANDLE;
        auto stamp = [&](int stage) {
            if (d->profile)
                d->profile->Stamp(*d->vc, cmd, queries, stage);
        };
        rec.Bind(kCurLuma, storagePlane.buffer, regions.luma, regions.lumaBytes);
        if (F.chroma)
            rec.Bind(kCurChroma, storagePlane.buffer, regions.chroma, regions.chromaBytes);
        if (L.topLevel > 0)
            rec.Bind(kCurPyramid, storagePlane.buffer, regions.pyramid, regions.pyramidBytes);
        rec.Bind(kLevels, d->constantsInfo.buffer);
        const int raw[3] = {kRawY, kRawU, kRawV}, pelBindings[3] = {kPelY, kPelU, kPelV};
        for (int p = 0; p < planes; ++p) {
            rec.Bind(raw[p], srcPlanes[p].buffer);
            if (pel)
                rec.Bind(pelBindings[p], pelPlanes[p].buffer);
        }

        SuperParams q = {};
        q.w = L.width;
        q.h = L.height;
        q.aw = L.aw;
        q.ah = L.ah;
        q.padX = L.pad;
        q.padY = L.padY;
        q.padcX = L.padc;
        q.padcY = L.padcY;
        q.wp = static_cast<int32_t>(lumaStride / bytes);
        q.hp = L.hp;
        q.wc = static_cast<int32_t>(chromaStride / (2 * bytes)); // chroma's pixels per row, U and V interleaved
        q.hc = L.hc;
        q.lumaPlanes = L.lumaPlanes;
        q.chromaPlanes = L.ChromaPlanes();
        q.xr = F.xr;
        q.yr = F.yr;
        q.planes = planes;
        q.sharp = d->sharp;
        q.pixelMax = F.Float() ? 0 : (1 << F.bits) - 1;
        q.srcStrideY = static_cast<int32_t>(vsapi->getStride(src, 0) / bytes);
        q.srcStrideC = F.chroma ? static_cast<int32_t>(vsapi->getStride(src, 1) / bytes) : 0;
        q.pelStrideY = pel ? static_cast<int32_t>(vsapi->getStride(pel, 0) / bytes) : 0;
        q.pelStrideC = pel && F.chroma ? static_cast<int32_t>(vsapi->getStride(pel, 1) / bytes) : 0;
        q.rfilter = d->rfilter;
        q.pel = L.pel;

        // The full-pel planes from the frame; then the half-pel planes from them (the diagonal one
        // after the y + 1/2 one it filters), or from the pelclip. At pel 4 with subsampled chroma
        // those are luma's alone, and super_qpel.comp makes chroma's quarter-pel image whole from the
        // frame; 4:4:4 chroma gets the same planes as luma. Four pixels of a padded row per lane,
        // luma's (z 0) or U's and V's together (z 1), whose planes are interleaved.
        const uint32_t groups = static_cast<uint32_t>((L.aw + 2 * L.pad + 255) / 256), rows = static_cast<uint32_t>(L.hp);
        const uint32_t stepPlanes = F.chroma && !d->quarterPipeline ? 2u : 1u;
        auto planesStep = [&](int step) {
            q.step = step;
            rec.Dispatch(d->planesPipeline, q, groups, rows, stepPlanes);
            rec.ComputeBarrier();
        };
        planesStep(0);
        stamp(1);
        if (L.pel > 1) {
            if (d->usePelClip) {
                planesStep(3);
            } else {
                planesStep(1);
                planesStep(2);
            }
        }
        stamp(2);
        if (d->quarterPipeline) {
            rec.Dispatch(d->quarterPipeline, q, static_cast<uint32_t>((L.aw / F.xr + 2 * L.padc + 255) / 256), static_cast<uint32_t>(L.hc), 1);
            rec.ComputeBarrier();
        }
        stamp(3);
        // Each coarse level from the one below it, borders included; chroma's border can be the wider
        // one, and without subsampling its plane as large as luma's
        for (int level = 1; level <= L.topLevel; ++level) {
            const LevelEntry &e = L.levels[level];
            q.level = level;
            const int cols = std::max(e.w + 2 * e.borderY, F.chroma ? e.wc + 2 * e.borderC : 0);
            const int rows = std::max(e.h + 2 * e.borderY, F.chroma ? e.hc + 2 * e.borderC : 0);
            // Four pixels per lane where that still leaves enough lanes to fill the GPU; narrow
            // levels are latency bound, and a lane making four would take four times as long
            q.quad = cols >= 640;
            const int perGroup = q.quad ? 256 : 64;
            rec.Dispatch(d->reducePipeline, q, static_cast<uint32_t>((cols + perGroup - 1) / perGroup), static_cast<uint32_t>(rows), static_cast<uint32_t>(planes));
            if (level < L.topLevel)
                rec.ComputeBarrier();
        }
        stamp(4);
        uint64_t signaled = 0;
        const int submitted = vkapi->gpuExecSubmit(ctx, &signaled, err, sizeof(err));
        ctx = nullptr;
        if (submitted)
            return fail(err);
        if (d->profile)
            d->profile->Finish(*d->vc, d->pool, signaled, queries);

        ExportSuper(dst, L, regions, d->prefix, vsapi);
        vsapi->freeFrame(src);
        vsapi->freeFrame(pel);
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

        d->sharp = vsapi->mapGetIntSaturated(in, "sharp", 0, &err);
        if (err)
            d->sharp = 2;

        d->rfilter = vsapi->mapGetIntSaturated(in, "rfilter", 0, &err);
        if (err)
            d->rfilter = 1;

        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        if (prefix)
            d->prefix = prefix;
        else
            d->prefix = DEFAULT_MVGPUTENSILS_PREFIX;

        if ((pel != 1) && (pel != 2) && (pel != 4))
            throw std::runtime_error("pel must be 1, 2, or 4");

        if (d->sharp < 0 || d->sharp > 2)
            throw std::runtime_error("sharp must be between 0 and 2");

        if (d->rfilter < 0 || d->rfilter > 2)
            throw std::runtime_error("rfilter must be between 0 and 2");

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(&d->vi))
            throw std::runtime_error("input clip must be GRAY, YUV420, YUV422, YUV440, or YUV444, up to 16 bits integer or 32 bit float, with constant dimensions");
        const SuperFormat format = SuperFormat::Of(d->vi.format);

        // The chroma planes are padded by pad / ratio. Rounded down, a luma vector reaching -pad would map
        // below the chroma padding in the filters that shift vectors to chroma (as mvu.Super refuses)
        if (hpad % format.xr || vpad % format.yr)
            throw std::runtime_error("pad must be divisible by the chroma subsampling: the horizontal pad must be even for 4:2:0 and 4:2:2, the vertical pad for 4:2:0 and 4:4:0");

        int blkX, blkY, overlapX, overlapY;
        GetPairArgument(blkX, blkY, "blksize", 8, 8, in, vsapi);
        GetPairArgument(overlapX, overlapY, "overlap", 0, 0, in, vsapi);

        CheckBlockSize(blkX, blkY, overlapX, overlapY, d->vi.format.subSamplingW, d->vi.format.subSamplingH);

        d->pelclip = vsapi->mapGetNode(in, "pelclip", 0, &err);
        const VSVideoInfo *pelvi = d->pelclip ? vsapi->getVideoInfo(d->pelclip) : nullptr;

        if (pelvi && (!vsh::isConstantVideoFormat(pelvi) || !vsh::isSameVideoFormat(&pelvi->format, &d->vi.format)))
            throw std::runtime_error("pelclip must have the same format as the input clip, and it must have constant dimensions");

        // mvu.Super takes a pelclip at pel 4 too, but this super keeps luma's half-pel planes only
        // and computes its quarter samples as averages of them, which a pelclip's needn't be
        if (pelvi && pel == 4)
            throw std::runtime_error("a pelclip isn't supported at pel=4: mvgpu.Super stores only the half-pel planes and computes the quarter-pel samples "
                                     "from them, to save memory; use pel=2 with a pelclip, or pel=4 without one");

        if (pelvi && pel >= 2) {
            if (pelvi->width != d->vi.width * pel || pelvi->height != d->vi.height * pel)
                throw std::runtime_error("pelclip's dimensions must be pel times the input clip's dimensions");

            if (pelvi->numFrames != d->vi.numFrames)
                throw std::runtime_error("pelclip's length must match the input clip's length");

            d->usePelClip = true;
        }

        d->layout = SuperLayout::Make(format, d->vi.width, d->vi.height, blkX, blkY, overlapX, overlapY, hpad, vpad, pel, !onelevel);
        const SuperLayout &L = d->layout;

        d->vc = VulkanContext::Get(core, vsapi);
        VulkanContext &vc = *d->vc;
        // The super kernels don't use the grid's block size, so their pipelines don't depend on it
        d->planesPipeline = vc.Pipeline(Kernel::Super, 0, pel, format.Kind());
        if (L.ChromaImage())
            d->quarterPipeline = vc.Pipeline(Kernel::SuperQuarter, 0, pel, format.Kind());
        if (L.topLevel > 0)
            d->reducePipeline = vc.Pipeline(Kernel::PyrReduce, 0, pel, format.Kind());
        const VkDeviceSize lumaBytes = static_cast<VkDeviceSize>(L.LumaRows()) * L.wp * format.Bytes();
        const VkDeviceSize chromaBytes = static_cast<VkDeviceSize>(L.ChromaRows()) * 2 * L.wc * format.Bytes();
        if (lumaBytes > vc.limits.maxStorageBufferRange || (format.chroma && chromaBytes > vc.limits.maxStorageBufferRange))
            throw std::runtime_error("the frame is too large for the device's storage buffers");

        char errMsg[1024] = {};
        d->pool = vc.vkapi->createGPUExecPool(core, vqCompute, errMsg, sizeof(errMsg));
        if (!d->pool)
            throw std::runtime_error(errMsg);
        const std::vector<LevelEntry> &table = L.levels;
        d->constants = vc.Upload(core, d->pool, table.data(), table.size() * sizeof(LevelEntry), d->constantsInfo);

        if (!vsapi->queryVideoFormat(&d->storage, cfGray, d->vi.format.sampleType, d->vi.format.bitsPerSample, 0, 0, core))
            throw std::runtime_error("failed to query the storage frame's format");
        d->outVi = d->vi;
        d->outVi.format = d->storage;
        d->outVi.width = L.FrameWidth();
        d->outVi.height = L.FrameRows();
        if (StageProfiler::Requested())
            d->profile = std::make_unique<StageProfiler>("Super (pel " + std::to_string(pel) + ")",
                                                         std::vector<std::string>{"full-pel planes", "sub-pel planes", "pel 4 chroma images", "coarse levels"});
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("Super: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[2] = {
        { d->node, rpStrictSpatial },
        { d->pelclip, rpStrictSpatial }
    };

    vsapi->createVideoFilterEx(out, "Super", &d->outVi, superGetFrame, filterFree<SuperData>, fmParallel, ffGPUOutput, deps, d->usePelClip ? 2 : 1, d.get(), core);
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
