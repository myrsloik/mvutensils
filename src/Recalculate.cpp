#include <algorithm>
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
#include "SuperLayout.h"
#include "VulkanContext.h"

// mvu.Recalculate on the GPU: vectors for the super's grid from another analysis (of another super of
// the same clip, a prefiltered one say), each block starting from the old vectors interpolated at its
// centre and searched around where that start matches badly (recalc.comp). The vectors are mvu's bit for
// bit given the same supers and old vectors, float supers' those of mvu's AVX-512 build, whose float SADs
// round apart from its AVX2 and SSE2 builds' (a SAD past 32 bits, of float samples far outside their
// range, is stored saturated).
//
// Implemented: all of mvu.Recalculate's arguments, on the supers mvgpu.Analyse takes, for their own
// grid or another that fits their block-aligned frame (SuperLayout::WithGrid), from old vectors of
// mvgpu's of the same bit depth, on any grid and at any pel (rescaled to the super's); satd makes
// luma's SAD mvu's SATD (16x2 blocks refused, as mvu refuses them).

struct RecalcData {
    VSNode *super = nullptr;
    VSNode *vectors = nullptr;
    const VSVideoInfo *vi = nullptr; // the super clip's
    VSVideoInfo outVi = {};          // the output's: the vectors' records
    SuperLayout layout;
    SuperLayout grid;                // the same with the grid recalculated, the super's or another
    std::string prefix;
    int delta = 0;
    bool chroma = true;
    int nbxOld = 0, nbyOld = 0;
    VectorInfo info;        // the old vectors' first frame's description, which every frame with vectors must have
    RecalcParams base = {}; // the push constants of every frame, but for the records' strides
    VSVideoFormat gray32 = {};

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline search = VK_NULL_HANDLE;
    VSGPUBuffer *zeros = nullptr; // bound where a Gray super has no chroma
    VSVulkanBufferInfo zerosInfo = {};
    std::unique_ptr<StageProfiler> profile; // MVGPU_PROFILE

    const VSAPI *vsapi;

    RecalcData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~RecalcData() {
        // The pool drains the GPU first, so nothing still uses what follows
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        profile.reset();
        if (zeros)
            vc->vkapi->destroyGPUBuffer(zeros);
        vc.reset();
        vsapi->freeNode(vectors);
        vsapi->freeNode(super);
    }
};

// The analysis description mvu.Recalculate attaches: the super's, with the vectors' delta, and one
// level, the recalculated one
static void ExportRecalculated(VSFrame *dst, const RecalcData *d, bool hasVectors, const VSAPI *vsapi) {
    ExportAnalysis(dst, d->grid, d->delta, d->chroma, hasVectors, d->prefix, vsapi, 1);
}

static const VSFrame *VS_CC recalculateGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    RecalcData *d = reinterpret_cast<RecalcData *>(instanceData);
    // As in mvu, a frame whose reference frame lies outside the super clip gets no vectors, also when
    // the vector clip, which may be longer, has some for it: they belong to a reference this super
    // doesn't have
    const int nref = n + d->delta;
    const bool hasRef = nref >= 0 && nref < d->vi->numFrames;

    if (activationReason == arInitial) {
        if (hasRef) {
            vsapi->requestFrameFilter(n, d->vectors, frameCtx);
            vsapi->requestFrameFilter(std::min(n, nref), d->super, frameCtx);
            vsapi->requestFrameFilter(std::max(n, nref), d->super, frameCtx);
        } else {
            vsapi->requestFrameFilter(n, d->super, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const VulkanContext &vc = *d->vc;
        const VSVULKANAPI *vkapi = vc.vkapi;
        const SuperLayout &L = d->layout, &G = d->grid; // the super's frames, the new vectors' grid

        // Everything this frame holds a reference to, released on every way out
        std::vector<const VSFrame *> held;
        VSFrame *vectors = nullptr;
        VSGPUExecContext *ctx = nullptr;
        auto release = [&]() {
            for (const VSFrame *f : held)
                vsapi->freeFrame(f);
            held.clear();
        };
        auto fail = [&](const std::string &message) -> const VSFrame * {
            if (ctx)
                vkapi->gpuExecAbandon(ctx);
            release();
            vsapi->freeFrame(vectors);
            vsapi->setFilterError((std::string("Recalculate: ") + message).c_str(), frameCtx);
            return nullptr;
        };
        auto hold = [&](const VSFrame *f) {
            if (f)
                held.push_back(f);
            return f;
        };

        const VSFrame *src = hold(vsapi->getFrameFilter(n, d->super, frameCtx));
        SuperRegions cur, rf;
        if (const std::string e = CheckSuperFrame(src, L, d->prefix, cur, vsapi); !e.empty())
            return fail(e);
        std::string error;
        const VSFrame *old = hasRef ? hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(n, d->vectors, frameCtx)), d->info, d->prefix, error, vsapi)) : nullptr;
        if (!error.empty())
            return fail(error);
        if (!old) {
            // A frame without old vectors or without a reference frame stays without vectors, as in
            // mvu: records that don't count, with the new grid's description
            VSFrame *out = vkapi->newGPUVideoFrame(&d->gray32, 4 * G.nbx, G.nby, src, core);
            if (!out)
                return fail("failed to allocate the vector frame");
            ExportRecalculated(out, d, false, vsapi);
            release();
            return out;
        }
        const VSFrame *refSuper = hold(vsapi->getFrameFilter(nref, d->super, frameCtx));
        if (const std::string e = CheckSuperFrame(refSuper, L, d->prefix, rf, vsapi); !e.empty())
            return fail(e);

        // The output: the records, with the super frame's properties
        vectors = vkapi->newGPUVideoFrame(&d->gray32, 4 * G.nbx, G.nby, src, core);
        if (!vectors)
            return fail("failed to allocate the vector frame");
        const ptrdiff_t recBytes = vsapi->getStride(vectors, 0), oldBytes = vsapi->getStride(old, 0);
        if (recBytes % 16 || oldBytes % 16 || vsapi->getFrameWidth(old, 0) != 4 * d->nbxOld || vsapi->getFrameHeight(old, 0) != d->nbyOld)
            return fail("a vector frame doesn't match its grid");

        const bool chroma = L.format.chroma;
        const ptrdiff_t bytes = L.format.Bytes();
        const ptrdiff_t lumaStride = cur.lumaStride, chromaStride = cur.chromaStride;
        if (lumaStride != rf.lumaStride || chromaStride != rf.chromaStride)
            return fail("the super frames' storage strides differ");
        VSVulkanPlaneInfo curPlane, refPlane, oldRec, outRec;
        if (vkapi->getGPUPlane(src, 0, &curPlane) || vkapi->getGPUPlane(refSuper, 0, &refPlane) || vkapi->getGPUPlane(old, 0, &oldRec) ||
            vkapi->getGPUPlane(vectors, 0, &outRec))
            return fail("a frame the search reads isn't GPU resident");

        char err[1024] = {};
        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        for (const VSFrame *f : {src, refSuper, old})
            vkapi->gpuExecReadsFrame(ctx, f);
        vkapi->gpuExecWritesPlane(ctx, vectors, 0);

        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        Recorder rec(vc, cmd, d->zerosInfo.buffer);
        const VkQueryPool queries = d->profile ? d->profile->Begin(vc, cmd) : VK_NULL_HANDLE;
        RecalcParams pc = d->base;
        pc.recStride = static_cast<int32_t>(recBytes / 16);
        pc.recStrideOld = static_cast<int32_t>(oldBytes / 16);
        pc.wp = static_cast<int32_t>(lumaStride / bytes);
        pc.wc = cur.KernelWc(bytes);
        rec.Bind(0, curPlane.buffer, cur.luma, cur.lumaBytes);
        rec.Bind(2, refPlane.buffer, rf.luma, rf.lumaBytes);
        if (chroma) {
            rec.Bind(1, curPlane.buffer, cur.chroma, cur.chromaBytes);
            rec.Bind(3, refPlane.buffer, rf.chroma, rf.chromaBytes);
        }
        rec.Bind(4, oldRec.buffer);
        rec.Bind(5, outRec.buffer);
        rec.Dispatch(d->search, pc, static_cast<uint32_t>(G.nbx), static_cast<uint32_t>(G.nby));
        if (d->profile)
            d->profile->Stamp(vc, cmd, queries, 1);

        uint64_t signaled = 0;
        const int submitted = vkapi->gpuExecSubmit(ctx, &signaled, err, sizeof(err));
        ctx = nullptr;
        if (submitted)
            return fail(err);
        if (d->profile)
            d->profile->Finish(vc, d->pool, signaled, queries);

        ExportRecalculated(vectors, d, true, vsapi);
        VSFrame *out = vectors;
        vectors = nullptr; // returned, not released
        release();
        return out;
    }

    return nullptr;
}

static void VS_CC recalculateCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<RecalcData> d = std::make_unique<RecalcData>(vsapi);

    try {
        int err;
        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        d->prefix = prefix ? prefix : DEFAULT_MVGPUTENSILS_PREFIX;

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->super);
        d->layout = ImportSuperLayout(d->super, d->prefix, vsapi);
        const SuperLayout &L = d->layout;
        if (const std::string unsupported = L.Unsupported(SuperLayout::Use::Search); !unsupported.empty())
            throw std::runtime_error(unsupported);

        int blkX, blkY, overlapX, overlapY;
        GetPairArgument(blkX, blkY, "blksize", L.blk, L.blkY, in, vsapi);
        GetPairArgument(overlapX, overlapY, "overlap", L.overlap, L.overlapY, in, vsapi);
        const bool useSatd = !!vsapi->mapGetInt(in, "satd", 0, &err);
        CheckBlockSize(blkX, blkY, overlapX, overlapY, L.format.xr > 1 ? 1 : 0, L.format.yr > 1 ? 1 : 0, useSatd);
        d->grid = L.WithGrid(blkX, blkY, overlapX, overlapY);
        const SuperLayout &G = d->grid;

        int64_t thsad = vsapi->mapGetIntSaturated(in, "thsad", 0, &err); // saturated, so that the scaling below can't overflow
        if (err)
            thsad = 200;
        bool smooth = !!vsapi->mapGetIntSaturated(in, "smooth", 0, &err);
        if (err)
            smooth = true;
        int search = vsapi->mapGetIntSaturated(in, "search", 0, &err);
        if (err)
            search = 2;
        int searchParam = std::max(vsapi->mapGetIntSaturated(in, "searchparam", 0, &err), 1);
        if (err)
            searchParam = 2;
        d->chroma = !!vsapi->mapGetInt(in, "chroma", 0, &err);
        if (err)
            d->chroma = true;
        if (!L.format.chroma)
            d->chroma = false;
        int64_t lambda = vsapi->mapGetIntSaturated(in, "mvlambda", 0, &err);
        if (err)
            lambda = 1000;
        if (lambda < 0)
            throw std::runtime_error("mvlambda must be non-negative");
        // Multiplied before dividing so that small blocks keep a lambda, as mvu does
        lambda = lambda * (blkX * blkY) / 64;
        int pnew = vsapi->mapGetIntSaturated(in, "pnew", 0, &err);
        if (err)
            pnew = 25;
        vsapi->mapGetInt(in, "meander", 0, &err); // the order the CPU took the blocks in; the GPU takes them all at once
        if (search < 0 || search > 5)
            throw std::runtime_error("search must be between 0 and 5");
        if (pnew < 0 || pnew > 256)
            throw std::runtime_error("pnew must be between 0 and 256");

        d->vectors = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        CheckClipLength(d->vectors, "vectors", d->vi->numFrames, "super", vsapi); // the output is as long as super
        const VectorInfo v = d->info = ReadVectorInfo(d->vectors, d->prefix, vsapi);
        if (v.bits != L.format.bits)
            throw std::runtime_error("Incompatible frame format for motion vector recalculation, bitdepth must match");
        d->delta = v.delta;
        d->nbxOld = v.nbx;
        d->nbyOld = v.nby;

        // thsad and mvlambda for 8-bit luma of 8x8 blocks scaled to the bit depth (float on the 16-bit
        // scale), the block size and chroma, as mvu scales them
        const int pixelMax = (1 << std::min(16, L.format.bits)) - 1;
        thsad = static_cast<int64_t>(static_cast<double>(thsad) * pixelMax / 255.0 + 0.5);
        lambda = static_cast<int64_t>(static_cast<double>(lambda) * pixelMax / 255.0 + 0.5);
        thsad = thsad * (blkX * blkY) / 64;
        if (d->chroma)
            thsad += thsad / (L.format.xr * L.format.yr) * 2;
        const int64_t lambdaLevel = lambda / (L.pel * L.pel);

        RecalcParams &pc = d->base;
        pc.nbx = G.nbx;
        pc.nby = G.nby;
        pc.step = G.step;
        pc.stepY = G.stepY;
        pc.nbxOld = v.nbx;
        pc.nbyOld = v.nby;
        pc.blkOld = v.blkX;
        pc.blkOldY = v.blkY;
        pc.stepOld = v.blkX - v.overlapX;
        pc.stepOldY = v.blkY - v.overlapY;
        pc.pad = L.pad;
        pc.padY = L.padY;
        pc.padc = L.padc;
        pc.padcY = L.padcY;
        pc.hp = L.hp;
        pc.hc = L.hc;
        pc.aw = L.aw;
        pc.ah = L.ah;
        // lambda capped at int's largest; thsad unsigned, as a SAD is, and 64 bits wide, as float SADs add up
        // past 32 bits (each plane's up to 2^32 - 1); a negative one as 0 (a block of SAD 0 keeps its vector
        // anyway: nothing costs less)
        pc.lambda = static_cast<int32_t>(std::min<int64_t>(lambdaLevel, std::numeric_limits<int32_t>::max()));
        const uint64_t th = static_cast<uint64_t>(std::max<int64_t>(thsad, 0));
        pc.thsad = static_cast<int32_t>(static_cast<uint32_t>(th));
        pc.thsadHi = static_cast<int32_t>(static_cast<uint32_t>(th >> 32));
        pc.pnew = pnew;
        pc.search = search;
        pc.searchParam = searchParam;
        pc.smoothing = smooth;
        // The new vectors take the super's pel, and the old ones are rescaled to it, as mvu does
        pc.oldPelShift = ilog2(L.pel) - ilog2(v.pel);

        if (!vsapi->queryVideoFormat(&d->gray32, cfGray, stInteger, 32, 0, 0, core))
            throw std::runtime_error("failed to query the Gray32 format");
        d->outVi = *d->vi;
        d->outVi.format = d->gray32;
        d->outVi.width = 4 * G.nbx;
        d->outVi.height = G.nby;

        d->vc = VulkanContext::Get(core, vsapi);
        VulkanContext &vc = *d->vc;
        // Specialization constant 6: chroma's subsampling, bit 0 horizontal, bit 1 vertical; bit 2
        // 16-bit samples, bit 3 float; bit 4 the SADs count chroma; bit 5 luma's SAD is its SATD
        const int variant = (L.format.xr > 1 ? 1 : 0) | (L.format.yr > 1 ? 2 : 0) | (L.format.Kind() == 1 ? 4 : L.format.Kind() == 2 ? 8 : 0) | (d->chroma ? 16 : 0) |
                            (useSatd ? 32 : 0);
        d->search = vc.Pipeline(Kernel::Recalculate, G.blk, L.pel, variant, G.blkY);

        char errMsg[1024] = {};
        d->pool = vc.vkapi->createGPUExecPool(core, vqCompute, errMsg, sizeof(errMsg));
        if (!d->pool)
            throw std::runtime_error(errMsg);
        const std::vector<int32_t> zeros(64, 0);
        d->zeros = vc.Upload(core, d->pool, zeros.data(), zeros.size() * sizeof(int32_t), d->zerosInfo);
        if (StageProfiler::Requested())
            d->profile = std::make_unique<StageProfiler>("Recalculate (pel " + std::to_string(L.pel) + ")", std::vector<std::string>{"search"});
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, (std::string("Recalculate: ") + e.what()).c_str());
        return;
    }

    VSFilterDependency deps[2] = {
        {d->super, rpGeneral},
        {d->vectors, rpStrictSpatial},
    };
    vsapi->createVideoFilterEx(out, "Recalculate", &d->outVi, recalculateGetFrame, filterFree<RecalcData>, fmParallel, ffGPUOutput, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

// mvu's: one Recalculate per vector clip, the clips returned in their order
static void VS_CC recalculateCreateMany(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    const int numVectors = vsapi->mapNumElements(in, "vectors");
    if (numVectors == 1) {
        recalculateCreate(in, out, userData, core, vsapi);
        return;
    }
    VSMap *args = vsapi->createMap();
    vsapi->copyMap(in, args);
    for (int i = 0; i < numVectors; ++i) {
        vsapi->mapConsumeNode(args, "vectors", vsapi->mapGetNode(in, "vectors", i, nullptr), maReplace);
        recalculateCreate(args, out, userData, core, vsapi);
        if (const char *error = vsapi->mapGetError(out)) {
            const std::string message = "Recalculate: Error when recalculating vector " + std::to_string(i) + ": " + error;
            vsapi->clearMap(out);
            vsapi->mapSetError(out, message.c_str());
            break;
        }
    }
    vsapi->freeMap(args);
}

void recalculateRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Recalculate",
                             "super:vnode:gpu;"
                             "vectors:vnode[]:gpu;"
                             "thsad:int:opt;"
                             "smooth:int:opt;"
                             "blksize:int[]:opt;"
                             "search:int:opt;"
                             "searchparam:int:opt;"
                             "mvlambda:int:opt;"
                             "chroma:int:opt;"
                             "pnew:int:opt;"
                             "overlap:int[]:opt;"
                             "meander:int:opt;"
                             "satd:int:opt;"
                             "prefix:data:opt;",
                             "clip:vnode[]:gpu;", recalculateCreateMany, nullptr, plugin);
}
