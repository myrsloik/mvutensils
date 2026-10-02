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
// pixel where the vectors are at a scene change, since the host can't see the count. The result is
// mvu's bit for bit given the same super and vectors.
//
// Implemented: all of mvu.Compensate's arguments but fields, on mvgpu.Super's Gray, 4:2:0 and 4:4:4
// supers of 8 to 16-bit or float samples at any pel with square blocks of 8, 16 or 32, and vectors
// made from such supers, of any of those bit depths, as mvgpu.Analyse makes them.

struct CompensateData {
    VSNode *node = nullptr; // the clip
    VSNode *super = nullptr;
    VSNode *vectors = nullptr;
    const VSVideoInfo *vi = nullptr;
    SuperLayout layout;
    std::string prefix;

    int delta = 0; // the vectors' reference frame is n + delta
    int time256 = 0;
    int thsad = 0; // scaled
    int nbx = 0, nby = 0, step = 0, overlap = 0;
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
            vsapi->requestFrameFilter(nref, d->super, frameCtx);
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

        const VSFrame *vec = load ? hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(n, d->vectors, frameCtx)), d->prefix, vsapi)) : nullptr;
        const VSFrame *src = hold(vsapi->getFrameFilter(n, d->node, frameCtx));
        if (!vec) {
            // The clip's frame, as mvu returns it where the vectors don't serve
            const VSFrame *f = vsapi->addFrameRef(src);
            release();
            return f;
        }

        SuperFrames ref;
        if (!GetSuperFrames(hold(vsapi->getFrameFilter(nref, d->super, frameCtx)), L, d->prefix, ref, vsapi))
            return fail("the super clip's frames lack their GPU planes; it must come from mvgpu.Super with the same prefix");
        for (const VSFrame *f : {ref.luma, ref.chroma, ref.pyramid})
            hold(f);

        dst = vkapi->newGPUVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core);
        if (!dst)
            return fail("failed to allocate the output frame");

        const int numPlanes = d->vi->format.numPlanes;
        const bool chroma = L.format.chroma;
        VSVulkanPlaneInfo clipPlanes[3] = {}, outPlanes[3] = {}, superPlanes[2] = {}, vecPlane = {};
        for (int p = 0; p < numPlanes; ++p)
            if (vkapi->getGPUPlane(src, p, &clipPlanes[p]) || vkapi->getGPUPlane(dst, p, &outPlanes[p]))
                return fail("the clip's frames aren't GPU resident");
        if (vkapi->getGPUPlane(ref.luma, 0, &superPlanes[0]) || (chroma && vkapi->getGPUPlane(ref.chroma, 0, &superPlanes[1])))
            return fail("the super's planes aren't GPU resident");
        if (vkapi->getGPUPlane(vec, 0, &vecPlane))
            return fail("the vectors aren't GPU resident");
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
        for (const VSFrame *f : {ref.luma, ref.chroma})
            if (f)
                vkapi->gpuExecReadsFrame(ctx, f);
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
        pc.wp = static_cast<int32_t>(vsapi->getStride(ref.luma, 0) / bytes);
        pc.hp = L.hp;
        pc.wc = chroma ? static_cast<int32_t>(vsapi->getStride(ref.chroma, 0) / bytes) : 0;
        pc.hc = L.hc;
        pc.time256 = d->time256;
        pc.scdLimit = d->scd.limit;
        // compensate.comp's slots: thsad, the grid's step and overlap, the clip's largest value
        pc.thscd1 = d->thsad;
        pc.time4096FX = d->step;
        pc.time4096FY = d->overlap;
        pc.time4096BX = L.format.Float() ? 0 : (1 << L.format.bits) - 1;

        rec.Bind(kFlTaps, d->windowsInfo.buffer);
        rec.Bind(kFlRefLuma, superPlanes[0].buffer);
        rec.Bind(kFlRefChroma, chroma ? superPlanes[1].buffer : d->windowsInfo.buffer);
        rec.Bind(kFlVecF, vecPlane.buffer);
        rec.Bind(kFlCounts, scratchInfo.buffer, 0, 16);
        for (int p = 0; p < numPlanes; ++p) {
            pc.plane = p;
            pc.width = vsapi->getFrameWidth(dst, p);
            pc.height = vsapi->getFrameHeight(dst, p);
            pc.outStride = static_cast<int32_t>(vsapi->getStride(dst, p) / bytes);
            pc.clipStride = static_cast<int32_t>(vsapi->getStride(src, p) / bytes);
            pc.colOff = d->winOff[p ? 1 : 0]; // compensate.comp's windows
            rec.Bind(kFlClipSrc, clipPlanes[p].buffer);
            rec.Bind(kFlOut, outPlanes[p].buffer);
            rec.Dispatch(d->pixels, pc, static_cast<uint32_t>((pc.width + 63) / 64), static_cast<uint32_t>(pc.height));
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
        int64_t thsad = vsapi->mapGetInt(in, "thsad", 0, &err);
        if (err)
            thsad = 10000;
        if (vsapi->mapGetInt(in, "fields", 0, &err) && !err)
            throw std::runtime_error("fields=True isn't implemented");
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

        // The vectors: mvgpu.Analyse's of a super with the same level 0 as this one but for the bit
        // depth (mvu's IsCompatibleWithAnalysis). The grid is theirs, as in mvu; the super only
        // supplies the planes.
        const VectorInfo v = ReadVectorInfo(d->vectors, d->prefix, vsapi);
        const SuperLayout analysed = ImportSuperLayout(d->vectors, d->prefix, vsapi);
        if (!SameGeometry(analysed, L) || v.width != L.aw || v.height != L.ah || v.realWidth != L.width || v.realHeight != L.height || v.hpad != L.pad ||
            v.vpad != L.padY || v.pel != L.pel || (v.chroma && (v.xRatio != L.format.xr || v.yRatio != L.format.yr)))
            throw std::runtime_error("wrong source or super clip frame size");
        if (v.blkX != analysed.blk || v.blkY != analysed.blk || v.overlapX != analysed.overlap || v.overlapY != analysed.overlap || v.nbx != analysed.nbx ||
            v.nby != analysed.nby)
            throw std::runtime_error("the vectors' grid isn't their super's; they must come from mvgpu.Analyse");
        d->delta = v.delta;
        d->nbx = v.nbx;
        d->nby = v.nby;
        d->step = v.blkX - v.overlapX;
        d->overlap = v.overlapX;
        d->scd = ScaleSceneChange(v, thscd1, thscd2);
        // thsad scaled as thscd1 is, truncated as mvu's int64 is; a SAD is never above int's range, so
        // a larger threshold takes every vector
        const double scaled = static_cast<double>(thsad) * ThSCDScale(v) + 0.5;
        constexpr int kMax = std::numeric_limits<int>::max(), kMin = std::numeric_limits<int>::min();
        d->thsad = scaled >= kMax ? kMax : scaled <= kMin ? kMin : static_cast<int>(static_cast<int64_t>(scaled));
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
        if (d->overlap > 0) {
            const int xr = L.format.xr, yr = L.format.yr, blk = v.blkX;
            const std::vector<int32_t> lumaWin = MakeOverlapWindows(blk, blk, d->overlap, d->overlap);
            const std::vector<int32_t> chromaWin = MakeOverlapWindows(blk / xr, blk / yr, d->overlap / xr, d->overlap / yr);
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
                             "fields:int:opt;"
                             "time:float:opt;"
                             "thscd1:int:opt;"
                             "thscd2:float:opt;"
                             "tff:int:opt;"
                             "prefix:data:opt;",
                             "clip:vnode:gpu;", compensateCreate, nullptr, plugin);
}
