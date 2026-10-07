#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

// mvu.Compensate on the GPU: every block of the frame replaced by the reference frame's block its
// vector, scaled to the time, points at (or kept where its SAD isn't under thsad), the blocks blended
// through their overlap windows. Two kernels per frame: the scene change test's count of badly matched
// blocks (mask_blocks.comp), then every pixel of every plane (compensate.comp), which copies the clip's
// pixel where the vectors are at a scene change, since the host can't see the count. The blocks whose
// SAD isn't under thsad come from the frame's own super, as in mvu. The result is mvu's bit for bit
// given the same super and vectors.
//
// Implemented: all of mvu.Compensate's arguments, on mvgpu.Super's Gray and YUV supers of any
// subsampling and 8 to 16-bit or float samples at any pel with any of mvu's block sizes, overlaps and
// paddings, and vectors made from such supers, of any of those bit depths, as mvgpu.Analyse makes
// them.

struct CompensateData {
    VSNode *node = nullptr; // the clip
    VSNode *super = nullptr;
    VSNode *vectors = nullptr;
    const VSVideoInfo *vi = nullptr;
    SuperLayout layout;
    VectorInfo info; // the vectors' first frame's description, which every frame with vectors must have
    std::string prefix;

    int delta = 0; // the vectors' reference frame is n + delta
    int time256 = 0;
    uint32_t thsad = 0; // scaled
    int nbx = 0, nby = 0, step = 0, stepY = 0, overlap = 0, overlapY = 0;
    SceneChange scd;

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline count = VK_NULL_HANDLE, pixels = VK_NULL_HANDLE;
    VSGPUBuffer *windows = nullptr; // the overlap windows, also bound where a frame has nothing
    VSVulkanBufferInfo windowsInfo = {};
    int winOff[2] = {}; // where luma's and chroma's are in it
    std::unique_ptr<StageProfiler> profile; // MVGPU_PROFILE

    const VSAPI *vsapi;

    CompensateData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~CompensateData() {
        // The pool drains the GPU first, so nothing still uses what follows
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        profile.reset();
        if (windows)
            vc->vkapi->destroyGPUBuffer(windows);
        vc.reset();
        vsapi->freeNode(vectors);
        vsapi->freeNode(super);
        vsapi->freeNode(node);
    }
};

static const VSFrame *VS_CC compensateGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    CompensateData *d = reinterpret_cast<CompensateData *>(instanceData);
    const int nref = n + d->delta;
    const bool load = nref >= 0 && nref < d->vi->numFrames;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (load) {
            vsapi->requestFrameFilter(n, d->vectors, frameCtx);
            vsapi->requestFrameFilter(std::min(n, nref), d->super, frameCtx);
            vsapi->requestFrameFilter(std::max(n, nref), d->super, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const VulkanContext &vc = *d->vc;
        const VSVULKANAPI *vkapi = vc.vkapi;
        const SuperLayout &L = d->layout;

        // Everything this frame holds a reference to, released on every way out
        std::vector<const VSFrame *> held;
        VSFrame *dst = nullptr;
        VSGPUExecContext *ctx = nullptr;
        VSGPUBuffer *scratch = nullptr;
        auto release = [&]() {
            for (const VSFrame *f : held)
                vsapi->freeFrame(f);
            held.clear();
        };
        auto fail = [&](const std::string &message) -> const VSFrame * {
            if (ctx)
                vkapi->gpuExecAbandon(ctx);
            else if (scratch)
                vkapi->destroyGPUBuffer(scratch);
            release();
            vsapi->freeFrame(dst);
            vsapi->setFilterError((std::string("Compensate: ") + message).c_str(), frameCtx);
            return nullptr;
        };
        auto hold = [&](const VSFrame *f) {
            if (f)
                held.push_back(f);
            return f;
        };

        std::string error;
        const VSFrame *vec = load ? hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(n, d->vectors, frameCtx)), d->info, d->prefix, error, vsapi)) : nullptr;
        if (!error.empty())
            return fail(error);
        const VSFrame *src = hold(vsapi->getFrameFilter(n, d->node, frameCtx));
        if (!vec) {
            // The clip's frame, as mvu returns it where the vectors don't serve
            const VSFrame *f = vsapi->addFrameRef(src);
            release();
            return f;
        }

        // The reference frame's super, and the frame's own, which the blocks not under thsad come from
        const VSFrame *ref = hold(vsapi->getFrameFilter(nref, d->super, frameCtx));
        const VSFrame *own = hold(vsapi->getFrameFilter(n, d->super, frameCtx));
        SuperRegions regions, ownRegions;
        if (const std::string e = CheckSuperFrame(ref, L, d->prefix, regions, vsapi); !e.empty())
            return fail(e);
        if (const std::string e = CheckSuperFrame(own, L, d->prefix, ownRegions, vsapi); !e.empty())
            return fail(e);
        if (ownRegions.lumaStride != regions.lumaStride || ownRegions.chromaStride != regions.chromaStride)
            return fail("the super frames' storage strides differ");

        dst = vkapi->newGPUVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core);
        if (!dst)
            return fail("failed to allocate the output frame");

        const int numPlanes = d->vi->format.numPlanes;
        const bool chroma = L.format.chroma;
        VSVulkanPlaneInfo clipPlanes[3] = {}, outPlanes[3] = {}, superPlane = {}, ownPlane = {}, vecPlane = {};
        for (int p = 0; p < numPlanes; ++p)
            if (vkapi->getGPUPlane(src, p, &clipPlanes[p]) || vkapi->getGPUPlane(dst, p, &outPlanes[p]))
                return fail("the clip's frames aren't GPU resident");
        if (vkapi->getGPUPlane(ref, 0, &superPlane) || vkapi->getGPUPlane(own, 0, &ownPlane))
            return fail("the super's frames aren't GPU resident");
        if (vkapi->getGPUPlane(vec, 0, &vecPlane))
            return fail("the vectors aren't GPU resident");
        // Chroma's U and V are made together, one dispatch taking both planes
        if (const std::string e = CheckChromaStrides({src, dst}, vsapi); !e.empty())
            return fail(e);
        const ptrdiff_t recBytes = vsapi->getStride(vec, 0);
        if (vsapi->getFrameWidth(vec, 0) != 4 * d->nbx || vsapi->getFrameHeight(vec, 0) != d->nby || recBytes % 16)
            return fail("a vector frame doesn't match the super's grid");

        // Scratch: the scene change count
        char err[1024] = {};
        VSVulkanBufferInfo scratchInfo = {};
        scratch = vkapi->createGPUBuffer(core, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                         &scratchInfo, err, sizeof(err));
        if (!scratch)
            return fail(err);
        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        vkapi->gpuExecUsesBuffer(ctx, scratch);
        vkapi->gpuExecReadsFrame(ctx, src);
        vkapi->gpuExecReadsFrame(ctx, ref);
        vkapi->gpuExecReadsFrame(ctx, own);
        vkapi->gpuExecReadsFrame(ctx, vec);
        for (int p = 0; p < numPlanes; ++p)
            vkapi->gpuExecWritesPlane(ctx, dst, p);

        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        Recorder rec(vc, cmd, d->windowsInfo.buffer);
        const VkQueryPool queries = d->profile ? d->profile->Begin(vc, cmd) : VK_NULL_HANDLE;
        auto stamp = [&](int stage) {
            if (d->profile)
                d->profile->Stamp(vc, cmd, queries, stage);
        };

        rec.Fill(scratchInfo.buffer, 0, 16, 0);
        rec.TransferToCompute();
        RecordBadBlockCount(rec, d->count, vecPlane.buffer, static_cast<int>(recBytes / 16), d->nbx, d->nby, d->scd.thscd1, scratchInfo.buffer, 0);
        rec.ComputeBarrier();
        stamp(1);

        const ptrdiff_t bytes = L.format.Bytes(); // strides in samples
        FlowParams pc = {};
        pc.nbx = d->nbx;
        pc.nby = d->nby;
        pc.recStride = static_cast<int32_t>(recBytes / 16);
        pc.pad = L.pad;
        pc.padY = L.padY;
        pc.padc = L.padc;
        pc.padcY = L.padcY;
        pc.wp = static_cast<int32_t>(regions.lumaStride / bytes);
        pc.hp = L.hp;
        pc.wc = regions.KernelWc(bytes);
        pc.hc = L.hc;
        pc.time256 = d->time256;
        pc.scdLimit = d->scd.limit;
        // compensate.comp's slots: thsad, the grid's step and overlap, the clip's largest value
        pc.thscd1 = static_cast<int32_t>(d->thsad); // compensate.comp reads it unsigned
        pc.time4096FX = d->step;
        pc.time4096FY = d->overlap;
        pc.stepY = d->stepY;
        pc.overlapY = d->overlapY;
        pc.time4096BX = L.format.Float() ? 0 : (1 << L.format.bits) - 1;

        rec.Bind(kFlTaps, d->windowsInfo.buffer);
        rec.Bind(kFlRefLuma, superPlane.buffer, regions.luma, regions.lumaBytes);
        rec.Bind(kFlSrcLuma, ownPlane.buffer, ownRegions.luma, ownRegions.lumaBytes);
        if (chroma) {
            rec.Bind(kFlRefChroma, superPlane.buffer, regions.chroma, regions.chromaBytes);
            rec.Bind(kFlSrcChroma, ownPlane.buffer, ownRegions.chroma, ownRegions.chromaBytes);
        }
        rec.Bind(kFlVecF, vecPlane.buffer);
        rec.Bind(kFlCounts, scratchInfo.buffer, 0, 16);
        // Luma, then chroma: U and V in one dispatch, V's output and clip planes at their own bindings
        for (int p = 0; p < (numPlanes > 1 ? 2 : 1); ++p) {
            pc.plane = p;
            pc.width = vsapi->getFrameWidth(dst, p);
            pc.height = vsapi->getFrameHeight(dst, p);
            pc.outStride = static_cast<int32_t>(vsapi->getStride(dst, p) / bytes);
            pc.clipStride = static_cast<int32_t>(vsapi->getStride(src, p) / bytes);
            pc.colOff = d->winOff[p]; // compensate.comp's windows
            rec.Bind(kFlClipSrc, clipPlanes[p].buffer);
            rec.Bind(kFlOut, outPlanes[p].buffer);
            if (p) {
                rec.Bind(kFlClipSrcV, clipPlanes[2].buffer);
                rec.Bind(kFlOutV, outPlanes[2].buffer);
            }
            // Four pixels per lane where they share their blocks: 8-bit samples at pel 1 or 2, the
            // plane's step and overlap multiples of 4 (compensate.comp's kQuad)
            const int lx = p ? L.format.xr >> 1 : 0;
            pc.time4096BY = bytes == 1 && L.pel != 4 && ((d->step >> lx) & 3) == 0 && ((d->overlap >> lx) & 3) == 0;
            rec.Dispatch(d->pixels, pc, static_cast<uint32_t>(pc.time4096BY ? (pc.width + 255) / 256 : (pc.width + 63) / 64), static_cast<uint32_t>(pc.height));
        }
        stamp(2);

        uint64_t signaled = 0;
        const int submitted = vkapi->gpuExecSubmit(ctx, &signaled, err, sizeof(err));
        ctx = nullptr;
        scratch = nullptr; // the context owned it
        if (submitted)
            return fail(err);
        if (d->profile)
            d->profile->Finish(vc, d->pool, signaled, queries);
        release();
        return dst;
    }

    return nullptr;
}

static void VS_CC compensateCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<CompensateData> d = std::make_unique<CompensateData>(vsapi);

    try {
        int err;
        int64_t thsad = vsapi->mapGetIntSaturated(in, "thsad", 0, &err); // saturated so the scaling below can't overflow int64
        if (err)
            thsad = 10000;
        double time = vsapi->mapGetFloat(in, "time", 0, &err);
        if (err)
            time = 100.0;
        int64_t thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
        if (err)
            thscd1 = MV_DEFAULT_SCD1;
        float thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
        if (err)
            thscd2 = MV_DEFAULT_SCD2;
        if (!std::isfinite(time) || time < 0.0 || time > 100.0)
            throw std::runtime_error("time must be between 0.0 and 100.0");
        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        d->prefix = prefix ? prefix : DEFAULT_MVGPUTENSILS_PREFIX;

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);
        d->vectors = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        d->layout = ImportSuperLayout(d->super, d->prefix, vsapi);
        const SuperLayout &L = d->layout;
        if (const std::string unsupported = L.Unsupported(SuperLayout::Use::Compensation); !unsupported.empty())
            throw std::runtime_error(unsupported);
        if (!vsh::isConstantVideoFormat(d->vi) || d->vi->format.colorFamily != (L.format.chroma ? cfYUV : cfGray) ||
            d->vi->format.sampleType != (L.format.Kind() == 2 ? stFloat : stInteger) || d->vi->format.bitsPerSample != L.format.bits ||
            (1 << d->vi->format.subSamplingW) != L.format.xr || (1 << d->vi->format.subSamplingH) != L.format.yr || d->vi->width != L.width ||
            d->vi->height != L.height)
            throw std::runtime_error("source clip isn't compatible with super clip");

        CheckClipLength(d->super, "super", d->vi->numFrames, "clip", vsapi);
        CheckClipLength(d->vectors, "vectors", d->vi->numFrames, "clip", vsapi);

        // The vectors: mvgpu.Analyse's of a super with the same level 0 as this one but for the bit
        // depth (mvu's IsCompatibleWithAnalysis). The grid is theirs, as in mvu; the super only
        // supplies the planes.
        const VectorInfo v = ReadVectorInfo(d->vectors, d->prefix, vsapi);
        const SuperLayout analysed = ImportAnalysedLayout(d->vectors, d->prefix, vsapi);
        if (!SameGeometry(analysed, L) || v.width != L.aw || v.height != L.ah || v.realWidth != L.width || v.realHeight != L.height || v.hpad != L.pad ||
            v.vpad != L.padY || v.pel != L.pel || (v.chroma && (v.xRatio != L.format.xr || v.yRatio != L.format.yr)))
            throw std::runtime_error("wrong source or super clip frame size");
        d->info = v;
        d->delta = v.delta;
        d->nbx = v.nbx;
        d->nby = v.nby;
        d->step = v.blkX - v.overlapX;
        d->stepY = v.blkY - v.overlapY;
        d->overlap = v.overlapX;
        d->overlapY = v.overlapY;
        d->scd = ScaleSceneChange(v, thscd1, thscd2);
        // thsad scaled as thscd1 is, truncated as mvu's int64 is, then unsigned, as a SAD is: a SAD is
        // below 2^32 - 1, so a larger threshold takes every vector as that does, and one of 0 or less none
        const double scaled = static_cast<double>(thsad) * ThSCDScale(v) + 0.5;
        constexpr double kMax = static_cast<double>(std::numeric_limits<uint32_t>::max());
        d->thsad = scaled >= kMax ? std::numeric_limits<uint32_t>::max() : scaled <= 0.0 ? 0u : static_cast<uint32_t>(static_cast<int64_t>(scaled));
        d->time256 = static_cast<int>(time * 256 / 100);

        d->vc = VulkanContext::Get(core, vsapi);
        VulkanContext &vc = *d->vc;
        // Specialization constant 6: chroma's subsampling, bit 0 horizontal, bit 1 vertical; bit 2
        // 16-bit samples, bit 3 float
        const int chromaLog = (L.format.xr > 1 ? 1 : 0) | (L.format.yr > 1 ? 2 : 0);
        d->count = vc.Pipeline(Kernel::MaskBlocks, 0, 0, 0);
        d->pixels = vc.Pipeline(Kernel::Compensate, 0, L.pel, chromaLog | (L.format.Kind() == 1 ? 4 : L.format.Kind() == 2 ? 8 : 0));

        // The overlap windows, luma's and chroma's (mvu's OverWins and OverWinsUV)
        std::vector<int32_t> tables;
        if (d->overlap > 0 || d->overlapY > 0) {
            const int xr = L.format.xr, yr = L.format.yr;
            const std::vector<int32_t> lumaWin = MakeOverlapWindows(v.blkX, v.blkY, d->overlap, d->overlapY);
            const std::vector<int32_t> chromaWin = MakeOverlapWindows(v.blkX / xr, v.blkY / yr, d->overlap / xr, d->overlapY / yr);
            d->winOff[0] = static_cast<int>(tables.size());
            tables.insert(tables.end(), lumaWin.begin(), lumaWin.end());
            d->winOff[1] = static_cast<int>(tables.size());
            tables.insert(tables.end(), chromaWin.begin(), chromaWin.end());
        }
        tables.resize(std::max<size_t>(tables.size(), 64)); // a dummy for the bindings a frame doesn't use too

        char errMsg[1024] = {};
        d->pool = vc.vkapi->createGPUExecPool(core, vqCompute, errMsg, sizeof(errMsg));
        if (!d->pool)
            throw std::runtime_error(errMsg);
        d->windows = vc.Upload(core, d->pool, tables.data(), tables.size() * sizeof(int32_t), d->windowsInfo);
        if (StageProfiler::Requested())
            d->profile = std::make_unique<StageProfiler>("Compensate (pel " + std::to_string(L.pel) + ")", std::vector<std::string>{"count", "pixels"});
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, (std::string("Compensate: ") + e.what()).c_str());
        return;
    }

    VSFilterDependency deps[3] = {
        {d->node, rpStrictSpatial},
        {d->super, rpGeneral},
        {d->vectors, rpStrictSpatial},
    };
    vsapi->createVideoFilterEx(out, "Compensate", d->vi, compensateGetFrame, filterFree<CompensateData>, fmParallel, ffGPUOutput, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void compensateRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Compensate",
                             "clip:vnode:gpu;"
                             "super:vnode:gpu;"
                             "vectors:vnode:gpu;"
                             "thsad:int:opt;"
                             "time:float:opt;"
                             "thscd1:int:opt;"
                             "thscd2:float:opt;"
                             "prefix:data:opt;",
                             "clip:vnode:gpu;", compensateCreate, nullptr, plugin);
}
