#include <algorithm>
#include <cmath>
#include <cstdint>
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

// mvu.Flow and mvu.FlowBlur on the GPU, from one source: every pixel of the frame fetched from the
// reference frame along its own vector (Flow), or averaged with the samples of its own frame along its
// forward and backward vectors (FlowBlur), the blocks' vectors resized to the pixels by zimg's bilinear
// resize in 64 x 64 tiles as mvu does it (TileTaps, as for FlowInter). Two kernels per frame: the scene
// change test's count of badly matched blocks per vector frame (mask_blocks.comp), then every pixel of
// every plane (flow_fetch.comp, flow_blur.comp), which copies the clip's pixel where the vectors are at
// a scene change, since the host can't see the count. The result is mvu's bit for bit given the same
// super and vectors.
//
// Implemented: all of their arguments but Flow's fields, on mvgpu.Super's Gray, 4:2:0 and 4:4:4 supers
// of 8 to 16-bit or float samples at any pel with square blocks of 8, 16 or 32, and vectors made from
// such supers, of any of those bit depths, as mvgpu.Analyse makes them.

struct FlowFetchData {
    VSNode *node = nullptr;  // the clip
    VSNode *super = nullptr;
    VSNode *vectors[2] = {}; // Flow: the vectors; FlowBlur: mvfw (delta -off) and mvbw (delta off)
    VectorInfo infos[2];     // their first frames' descriptions, which their frames with vectors must have
    const VSVideoInfo *vi = nullptr;
    SuperLayout layout;
    std::string prefix, name;

    bool blur = false; // FlowBlur, else Flow
    int delta = 0;     // Flow: the vectors' reference frame is n + delta; FlowBlur: mvfw's delta, -off
    int time256 = 0;   // Flow's time, FlowBlur's blur256
    int prec = 1;      // FlowBlur
    int nbx = 0, nby = 0;
    SceneChange scd;

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline count = VK_NULL_HANDLE, pixels = VK_NULL_HANDLE;
    VSGPUBuffer *taps = nullptr; // the resize's taps (TileTaps), also bound where a frame has nothing
    VSVulkanBufferInfo tapsInfo = {};
    int colOff[2] = {}, rowOff[2] = {}; // where luma's and chroma's are in it
    std::unique_ptr<StageProfiler> profile; // MVGPU_PROFILE

    const VSAPI *vsapi;

    FlowFetchData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~FlowFetchData() {
        // The pool drains the GPU first, so nothing still uses what follows
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        profile.reset();
        if (taps)
            vc->vkapi->destroyGPUBuffer(taps);
        vc.reset();
        vsapi->freeNode(vectors[0]);
        vsapi->freeNode(vectors[1]);
        vsapi->freeNode(super);
        vsapi->freeNode(node);
    }
};

static const VSFrame *VS_CC flowGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    FlowFetchData *d = reinterpret_cast<FlowFetchData *>(instanceData);

    // The vector frames and the super frame the pixels come from: Flow's vectors of frame n and the
    // super of its reference frame; FlowBlur's mvfw of frame n - delta and mvbw of frame n + delta, both
    // pointing at frame n, and frame n's super
    const int frames = d->vi->numFrames;
    const int vecFrame[2] = {d->blur ? n - d->delta : n, n + d->delta};
    const int superFrame = d->blur ? n : n + d->delta;
    const bool load = d->blur ? n + d->delta >= 0 && n - d->delta < frames : superFrame >= 0 && superFrame < frames;
    const int vectorClips = d->blur ? 2 : 1;

    if (activationReason == arInitial) {
        if (load) {
            for (int i = 0; i < vectorClips; ++i)
                vsapi->requestFrameFilter(vecFrame[i], d->vectors[i], frameCtx);
            vsapi->requestFrameFilter(superFrame, d->super, frameCtx);
        }
        vsapi->requestFrameFilter(n, d->node, frameCtx);
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
            vsapi->setFilterError((d->name + ": " + message).c_str(), frameCtx);
            return nullptr;
        };
        auto hold = [&](const VSFrame *f) {
            if (f)
                held.push_back(f);
            return f;
        };

        // The vectors that have vectors, mvu's HasMotionVectors; their scene change test is the GPU's
        const VSFrame *vec[2] = {};
        bool have = load;
        for (int i = 0; i < vectorClips && have; ++i) {
            std::string error;
            vec[i] = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(vecFrame[i], d->vectors[i], frameCtx)), d->infos[i], d->prefix, error, vsapi));
            if (!error.empty())
                return fail(error);
            have = vec[i] != nullptr;
        }
        const VSFrame *src = hold(vsapi->getFrameFilter(n, d->node, frameCtx));
        if (!have) {
            // The clip's frame, as mvu returns it
            const VSFrame *f = vsapi->addFrameRef(src);
            release();
            return f;
        }

        const VSFrame *sup = hold(vsapi->getFrameFilter(superFrame, d->super, frameCtx));
        SuperRegions regions;
        if (const std::string e = CheckSuperFrame(sup, L, d->prefix, regions, vsapi); !e.empty())
            return fail(e);

        dst = vkapi->newGPUVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core);
        if (!dst)
            return fail("failed to allocate the output frame");

        const int numPlanes = d->vi->format.numPlanes;
        const bool chroma = L.format.chroma;
        VSVulkanPlaneInfo clipPlanes[3] = {}, outPlanes[3] = {}, superPlane = {}, vecPlanes[2] = {};
        for (int p = 0; p < numPlanes; ++p)
            if (vkapi->getGPUPlane(src, p, &clipPlanes[p]) || vkapi->getGPUPlane(dst, p, &outPlanes[p]))
                return fail("the clip's frames aren't GPU resident");
        if (vkapi->getGPUPlane(sup, 0, &superPlane))
            return fail("the super's frames aren't GPU resident");
        const ptrdiff_t lumaStride = regions.lumaStride, chromaStride = regions.chromaStride;
        ptrdiff_t recBytes = 0;
        for (int i = 0; i < vectorClips; ++i) {
            if (vkapi->getGPUPlane(vec[i], 0, &vecPlanes[i]))
                return fail("the vectors aren't GPU resident");
            const ptrdiff_t stride = vsapi->getStride(vec[i], 0);
            if (vsapi->getFrameWidth(vec[i], 0) != 4 * d->nbx || vsapi->getFrameHeight(vec[i], 0) != d->nby || stride % 16 || (recBytes && stride != recBytes))
                return fail("a vector frame doesn't match the super's grid");
            recBytes = stride;
        }

        // Scratch: the scene change counts
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
        vkapi->gpuExecReadsFrame(ctx, sup);
        for (int i = 0; i < vectorClips; ++i)
            vkapi->gpuExecReadsFrame(ctx, vec[i]);
        for (int p = 0; p < numPlanes; ++p)
            vkapi->gpuExecWritesPlane(ctx, dst, p);

        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        Recorder rec(vc, cmd, d->tapsInfo.buffer);
        const VkQueryPool queries = d->profile ? d->profile->Begin(vc, cmd) : VK_NULL_HANDLE;
        auto stamp = [&](int stage) {
            if (d->profile)
                d->profile->Stamp(vc, cmd, queries, stage);
        };

        rec.Fill(scratchInfo.buffer, 0, 16, 0);
        rec.TransferToCompute();
        for (int i = 0; i < vectorClips; ++i)
            RecordBadBlockCount(rec, d->count, vecPlanes[i].buffer, static_cast<int>(recBytes / 16), d->nbx, d->nby, d->scd.thscd1, scratchInfo.buffer, i);
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
        pc.wp = static_cast<int32_t>(lumaStride / bytes);
        pc.hp = L.hp;
        pc.wc = static_cast<int32_t>(chromaStride / bytes);
        pc.hc = L.hc;
        pc.time256 = d->time256;
        pc.scdLimit = d->scd.limit;
        pc.time4096FX = d->prec; // flow_blur.comp's prec

        // Flow reads the reference frame's super as ref, FlowBlur its own frame's as src
        const int lumaBinding = d->blur ? kFlSrcLuma : kFlRefLuma, chromaBinding = d->blur ? kFlSrcChroma : kFlRefChroma;
        rec.Bind(kFlTaps, d->tapsInfo.buffer);
        rec.Bind(lumaBinding, superPlane.buffer, regions.luma, regions.lumaBytes);
        if (chroma)
            rec.Bind(chromaBinding, superPlane.buffer, regions.chroma, regions.chromaBytes);
        rec.Bind(kFlVecF, vecPlanes[0].buffer);
        rec.Bind(kFlVecB, d->blur ? vecPlanes[1].buffer : d->tapsInfo.buffer);
        rec.Bind(kFlCounts, scratchInfo.buffer, 0, 16);
        for (int p = 0; p < numPlanes; ++p) {
            pc.plane = p;
            pc.width = vsapi->getFrameWidth(dst, p);
            pc.height = vsapi->getFrameHeight(dst, p);
            pc.outStride = static_cast<int32_t>(vsapi->getStride(dst, p) / bytes);
            pc.clipStride = static_cast<int32_t>(vsapi->getStride(src, p) / bytes);
            pc.colOff = d->colOff[p ? 1 : 0];
            pc.rowOff = d->rowOff[p ? 1 : 0];
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

// FlowBlur takes userData 1, Flow null
static void VS_CC flowCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<FlowFetchData> d = std::make_unique<FlowFetchData>(vsapi);
    d->blur = userData != nullptr;
    d->name = d->blur ? "FlowBlur" : "Flow";

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

        if (d->blur) {
            float blur = vsapi->mapGetFloatSaturated(in, "blur", 0, &err);
            if (err)
                blur = 50.0f;
            d->prec = vsapi->mapGetIntSaturated(in, "prec", 0, &err);
            if (err)
                d->prec = 1;
            if (!std::isfinite(blur) || blur < 0.0f || blur > 200.0f)
                throw std::runtime_error("blur must be between 0 and 200");
            if (d->prec < 1)
                throw std::runtime_error("prec must be at least 1");
            d->time256 = static_cast<int>(blur * 256.0f / 200.0f);
        } else {
            double time = vsapi->mapGetFloat(in, "time", 0, &err);
            if (err)
                time = 100.0;
            if (vsapi->mapGetInt(in, "fields", 0, &err) && !err)
                throw std::runtime_error("fields=True isn't implemented");
            if (!std::isfinite(time) || time < 0.0 || time > 100.0)
                throw std::runtime_error("time must be between 0 and 100%");
            d->time256 = static_cast<int>(time * 256.0 / 100.0);
        }

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);
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

        if (d->blur) {
            if (vsapi->mapNumElements(in, "vectors") != 2)
                throw std::runtime_error("vectors must have exactly 2 elements");
            d->vectors[1] = vsapi->mapGetNode(in, "vectors", 0, nullptr); // mvbw
            d->vectors[0] = vsapi->mapGetNode(in, "vectors", 1, nullptr); // mvfw
        } else {
            d->vectors[0] = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        }
        CheckClipLength(d->super, "super", d->vi->numFrames, "clip", vsapi);
        for (int i = 0; i < (d->blur ? 2 : 1); ++i)
            CheckClipLength(d->vectors[i], "vectors", d->vi->numFrames, "clip", vsapi);

        // The vectors: mvgpu.Analyse's of a super with the same level 0 as this one but for the bit
        // depth (mvu's IsCompatibleWithAnalysis); FlowBlur's both on one grid with opposite deltas. The
        // grid is theirs, as in mvu; the super only supplies the planes.
        VectorInfo info[2];
        for (int i = 0; i < (d->blur ? 2 : 1); ++i) {
            const VectorInfo &v = info[i] = d->infos[i] = ReadVectorInfo(d->vectors[i], d->prefix, vsapi);
            const SuperLayout analysed = ImportAnalysedLayout(d->vectors[i], d->prefix, vsapi);
            if (i == 0 && (!SameGeometry(analysed, L) || v.width != L.aw || v.height != L.ah || v.realWidth != L.width || v.realHeight != L.height ||
                           v.hpad != L.pad || v.vpad != L.padY || v.pel != L.pel || (v.chroma && (v.xRatio != L.format.xr || v.yRatio != L.format.yr))))
                throw std::runtime_error("wrong source or super clip frame size");
            if (v.blkX != analysed.blk || v.blkY != analysed.blk || v.overlapX != analysed.overlap || v.overlapY != analysed.overlap || v.nbx != analysed.nbx ||
                v.nby != analysed.nby)
                throw std::runtime_error("the vectors' grid isn't their super's; they must come from mvgpu.Analyse");
        }
        const VectorInfo &v = info[0];
        if (d->blur) {
            // One analysis but for the delta (mvu's IsCompatible)
            VectorInfo bw = info[1];
            const int bwDelta = bw.delta;
            bw.delta = v.delta;
            if (!SameAnalysis(bw, v) || bwDelta != -v.delta || v.delta > 0 || bwDelta < 0)
                throw std::runtime_error("mvfw and mvbw must be compatible with each other and have opposite sign delta");
        }
        d->delta = v.delta;
        d->nbx = v.nbx;
        d->nby = v.nby;
        d->scd = ScaleSceneChange(v, thscd1, thscd2);

        d->vc = VulkanContext::Get(core, vsapi);
        VulkanContext &vc = *d->vc;
        // Specialization constant 6: chroma's subsampling, bit 0 horizontal, bit 1 vertical; bit 2
        // 16-bit samples, bit 3 float
        const int chromaLog = (L.format.xr > 1 ? 1 : 0) | (L.format.yr > 1 ? 2 : 0);
        d->count = vc.Pipeline(Kernel::MaskBlocks, 0, 0, 0);
        d->pixels = vc.Pipeline(d->blur ? Kernel::FlowBlur : Kernel::FlowFetch, 0, L.pel, chromaLog | (L.format.Kind() == 1 ? 4 : L.format.Kind() == 2 ? 8 : 0));

        // The resize's taps: luma's columns and rows, then chroma's, its blocks luma's shifted by the
        // subsampling, as mvu's MaskResizer has them (4:4:4 chroma takes luma's)
        const int blk = v.blkX, overlap = v.overlapX;
        std::vector<int32_t> tables;
        for (int c = 0; c < (L.format.xr > 1 || L.format.yr > 1 ? 2 : 1); ++c) {
            const int lx = c ? d->vi->format.subSamplingW : 0, ly = c ? d->vi->format.subSamplingH : 0;
            const int stepX = (blk >> lx) - (overlap >> lx), stepY = (blk >> ly) - (overlap >> ly);
            const int w = d->vi->width >> lx, h = d->vi->height >> ly;
            if (!TilesHorizontalFirst(d->nbx, w, stepX, overlap >> lx, d->nby, h, stepY, overlap >> ly))
                throw std::runtime_error("zimg would resize the vectors vertically first, which isn't implemented");
            d->colOff[c] = static_cast<int>(tables.size());
            const std::vector<int32_t> cols = TileTaps(d->nbx, w, stepX, overlap >> lx);
            tables.insert(tables.end(), cols.begin(), cols.end());
            d->rowOff[c] = static_cast<int>(tables.size());
            const std::vector<int32_t> rows = TileTaps(d->nby, h, stepY, overlap >> ly);
            tables.insert(tables.end(), rows.begin(), rows.end());
        }
        if (L.format.xr == 1 && L.format.yr == 1) {
            d->colOff[1] = d->colOff[0];
            d->rowOff[1] = d->rowOff[0];
        }
        tables.resize(std::max<size_t>(tables.size(), 64)); // a dummy for the bindings a frame doesn't use too

        char errMsg[1024] = {};
        d->pool = vc.vkapi->createGPUExecPool(core, vqCompute, errMsg, sizeof(errMsg));
        if (!d->pool)
            throw std::runtime_error(errMsg);
        d->taps = vc.Upload(core, d->pool, tables.data(), tables.size() * sizeof(int32_t), d->tapsInfo);
        if (StageProfiler::Requested())
            d->profile = std::make_unique<StageProfiler>(d->name + " (pel " + std::to_string(L.pel) + ")", std::vector<std::string>{"counts", "pixels"});
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, (d->name + ": " + e.what()).c_str());
        return;
    }

    std::vector<VSFilterDependency> deps = {{d->node, rpStrictSpatial}, {d->super, rpGeneral}};
    for (VSNode *v : d->vectors)
        if (v)
            deps.push_back({v, d->blur ? rpGeneral : rpStrictSpatial});
    vsapi->createVideoFilterEx(out, d->name.c_str(), d->vi, flowGetFrame, filterFree<FlowFetchData>, fmParallel, ffGPUOutput, deps.data(), static_cast<int>(deps.size()), d.get(),
                               core);
    d.release();
}

void flowFetchRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Flow",
                             "clip:vnode:gpu;"
                             "super:vnode:gpu;"
                             "vectors:vnode:gpu;"
                             "time:float:opt;"
                             "fields:int:opt;"
                             "thscd1:int:opt;"
                             "thscd2:float:opt;"
                             "tff:int:opt;"
                             "prefix:data:opt;",
                             "clip:vnode:gpu;", flowCreate, nullptr, plugin);
    vspapi->registerFunction("FlowBlur",
                             "clip:vnode:gpu;"
                             "super:vnode:gpu;"
                             "vectors:vnode[]:gpu;"
                             "blur:float:opt;"
                             "prec:int:opt;"
                             "thscd1:int:opt;"
                             "thscd2:float:opt;"
                             "prefix:data:opt;",
                             "clip:vnode:gpu;", flowCreate, reinterpret_cast<void *>(1), plugin);
}
