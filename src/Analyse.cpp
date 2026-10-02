#include <algorithm>
#include <cstdint>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSVulkan4.h>

#include "Common.h"
#include "SuperLayout.h"
#include "VulkanContext.h"

// mvu.Analyse on the GPU. A field (n, d), the vectors that match the blocks of frame n in frame
// n + d, comes from two nodes:
//
// - the coarse node, shared by every field Analyse or AnalyseMany makes: frame n holds the coarse
//   searches of all of frame n's fields, run as one batch, the field in the dispatches' z: an
//   exhaustive search at the super's smallest level, then at every finer level down to the finest
//   seeds from the level above and checkerboard passes (pyr_top/pyr_seed/pyr_pass/median.comp);
// - the field's node: a seed list per block of the full-size grid, from zero, the field's median
//   and the finest level's vectors around the block, and with AnalyseMany the chained and inverted
//   vectors of fields refined before (seed_scatter/seed_build.comp); then the refinement: the
//   seeds measured, a pair of checkerboard passes under mvu.Analyse's cost, the wide fallback
//   search for the blocks still above badsad and one more pair, then the 8 half-pel positions
//   around each block's vector (refine_*.comp), and at pel 4 the 8 quarter-pel positions around
//   that. At pel 4 the chained and inverted seeds are rounded to the half-pel grid, where the passes
//   and the half-pel step stay, and the fallback stays on the full-pel grid; the quarter samples are
//   computed from the super's half-pel planes (SuperLayout.h).
//
// The result is the CPU reference's (test/reference/reference.cpp) bit for bit. The output frame is
// the super frame with mvu.Analyse's properties and a vector frame (SuperLayout.h's ExportAnalysis).
//
// AnalyseMany wires each field to the fields its seeds come from: (n, d) chains (n, u) and
// (n + u, d - u), u the step toward d, and for d > 0 inverts (n + d, -d); Analyse on its own
// seeds from the coarse search only.
//
// Implemented: chroma, mvlambda, lsad, plevel (lambda * 2^(plevel * L) at coarse level L, as mvu
// scales it per level), badsad and badrange (the fallback's threshold and radius), delta, prefix,
// plus badstep, the fallback's step, which mvu doesn't have. The fallback's defaults are the tested
// ones (badsad 1000, badrange 40, badstep 2) rather than mvu's (badsad 10000, which this search would
// almost never reach, and badrange 24). search, searchparam, pelsearch, levels, pnew, pzero,
// pglobal, globalmv, meander and trymany tune mvu's search, which this one doesn't use; they are
// checked and otherwise ignored. satd and fields aren't implemented yet, and blksize and overlap
// must be the super's.

namespace {

constexpr int kMaxSeeds = 10;     // seed slots per block: zero, the median, chained, inverted and 5 coarse, at most
constexpr int kCounterSlots = 64; // the kernels' statistics, two per step
constexpr int kMedianSlots = 32, kGlobalSlot = 31;
constexpr int kPairs = 1;         // checkerboard pass pairs before the fallback
constexpr int kTopRadius = 32;    // the exhaustive search's radius at the top level, px
constexpr int kLevelPairs = 2;    // checkerboard pass pairs at every coarse level

// Entries of the full-size grid's lambda table: (largest SAD of a block) >> 1, plus one
constexpr int LambdaEntries(int blk, bool chroma) {
    return blk * blk * (chroma ? 3 : 2) / 2 * 255 / 2 + 1;
}

// The search kernels' variant (specialization constant 6, refine_common.glsl): bit 0 for SADs of
// luma alone
int SearchVariant(bool chroma) {
    return chroma ? 0 : 1;
}

struct Region {
    VkDeviceSize offset = 0, size = 0;
};

// Regions of one buffer, each aligned for binding at its offset
class Regions {
public:
    explicit Regions(const VulkanContext &vc) : align(std::max<VkDeviceSize>(vc.limits.minStorageBufferOffsetAlignment, 16)) {}
    void Place(Region &r, VkDeviceSize size) {
        r.offset = total;
        r.size = std::max<VkDeviceSize>(size, 16);
        total = (total + r.size + align - 1) / align * align;
    }
    VkDeviceSize Total() const { return total; }

private:
    VkDeviceSize align, total = 0;
};

// mvu.Analyse's scaling for 8-bit and the block size, lambda divided by pel squared at full size,
// exactly as the CPU reference computes it
struct Scaled {
    int64_t lambda0, lsad, lambdaBlock;
};

// The ints of one row of the coarse node's frames: a field's global median, then its finest level
int CoarseRowInts(const SuperLayout &L) {
    const LevelEntry &finest = L.levels[SuperLayout::kFinest];
    return 2 * (1 + finest.nbx * finest.nby);
}

} // namespace

// ---- The coarse node

// Frame n holds the coarse search of field (n, deltas[s]) in row s, where n + deltas[s] is a frame:
// its global median, then the finest coarse level's vectors, full pels of that level; rows the
// frame's stride apart. The fields of a frame are searched in batches of up to kMaxBatch, in one
// submission.
struct CoarseData {
    VSNode *node = nullptr; // the super clip
    VSVideoInfo vi = {};    // Gray32, a row per delta
    int numFrames = 0;
    SuperLayout layout;
    std::vector<int> deltas;
    std::string prefix;

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline pyrTop = VK_NULL_HANDLE, pyrPass = VK_NULL_HANDLE, pyrSeed = VK_NULL_HANDLE, median = VK_NULL_HANDLE;

    // The tables, uploaded once: lambda per coarse level, the level table
    VSGPUBuffer *constants = nullptr;
    VSVulkanBufferInfo constantsInfo = {};
    Region levelLambda, levels;

    // A batch's working buffers, regions of one buffer allocated per frame
    int batch = 1;
    Region levelVec, levelSad, medians;
    VkDeviceSize scratchBytes = 0;

    std::unique_ptr<StageProfiler> profile; // MVGPU_PROFILE

    const VSAPI *vsapi;

    CoarseData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~CoarseData() {
        // The pool drains the GPU first, so nothing still uses what follows
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        profile.reset();
        if (constants)
            vc->vkapi->destroyGPUBuffer(constants);
        vc.reset();
        vsapi->freeNode(node);
    }
};

// The coarse searches of a batch's `count` fields, the field in z, as the prototype's vkrefine.cpp
// records them: the top level's exhaustive search and passes, then each finer level seeded from the
// one above, down to the finest; then the finest level's median
class CoarseRecorder {
public:
    CoarseRecorder(const CoarseData &d, Recorder &rec, uint32_t count) : d(d), L(d.layout), rec(rec), count(count) {}

    void Search() {
        const int T = L.topLevel, F = SuperLayout::kFinest;
        rec.Dispatch(d.pyrTop, LevelParams(T), static_cast<uint32_t>(L.levels[T].nbx), static_cast<uint32_t>(L.levels[T].nby), count);
        rec.ComputeBarrier();
        LevelPasses(T);
        for (int level = T - 1; level >= F; --level) {
            Median(level + 1, 2, level);
            rec.Dispatch(d.pyrSeed, LevelParams(level), static_cast<uint32_t>((L.levels[level].nbx + 7) / 8), static_cast<uint32_t>(L.levels[level].nby), count);
            rec.ComputeBarrier();
            LevelPasses(level);
        }
        Median(F, L.pel << F, kGlobalSlot);
    }

private:
    // The coarse kernels' parameters at a level; the levels use 8x8 blocks
    Params LevelParams(int level, int colour = 0) const {
        Params q = {};
        q.colour = colour;
        q.level = level;
        q.topRadius = kTopRadius;
        q.medianScale = 1;
        q.finest = SuperLayout::kFinest;
        q.blockRows = 8;
        return q;
    }

    // Checkerboard pass pairs at a coarse level
    void LevelPasses(int level) {
        const LevelEntry &e = L.levels[level];
        for (int p = 0; p < 2 * kLevelPairs; ++p) {
            rec.Dispatch(d.pyrPass, LevelParams(level, p & 1), static_cast<uint32_t>(((e.nbx + 1) / 2 + 7) / 8), static_cast<uint32_t>(e.nby), count);
            rec.ComputeBarrier();
        }
    }

    // The median of a level's coarse field, times scale, into each field's medians at slot
    void Median(int level, int scale, int slot) {
        Params q = LevelParams(level);
        q.medianScale = scale;
        q.medianSlot = slot;
        rec.Dispatch(d.median, q, 1, 1, count);
        rec.ComputeBarrier();
    }

    const CoarseData &d;
    const SuperLayout &L;
    Recorder &rec;
    uint32_t count;
};

static const VSFrame *VS_CC coarseGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    CoarseData *d = reinterpret_cast<CoarseData *>(instanceData);

    // The rows with a field, and the reference frames, each once
    std::vector<int> slots, refs;
    for (int s = 0; s < static_cast<int>(d->deltas.size()); ++s) {
        const int nref = n + d->deltas[s];
        if (nref < 0 || nref >= d->numFrames)
            continue;
        slots.push_back(s);
        if (std::find(refs.begin(), refs.end(), nref) == refs.end())
            refs.push_back(nref);
    }

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        for (int nref : refs)
            vsapi->requestFrameFilter(nref, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSVULKANAPI *vkapi = d->vc->vkapi;
        const SuperLayout &L = d->layout;
        VSFrame *out = vkapi->newGPUVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);
        if (!out) {
            vsapi->setFilterError("Analyse: failed to allocate the coarse search's frame", frameCtx);
            return nullptr;
        }
        if (slots.empty())
            return out; // no field of this frame has a reference frame, so nothing reads it

        std::vector<SuperFrames> held;
        VSGPUExecContext *ctx = nullptr;
        VSGPUBuffer *scratch = nullptr;
        auto fail = [&](const std::string &message) -> const VSFrame * {
            if (ctx)
                vkapi->gpuExecAbandon(ctx);
            else if (scratch)
                vkapi->destroyGPUBuffer(scratch);
            for (SuperFrames &f : held)
                f.Free(vsapi);
            vsapi->freeFrame(out);
            vsapi->setFilterError(("Analyse: " + message).c_str(), frameCtx);
            return nullptr;
        };

        // The current frame's pyramid, then the references'
        std::vector<VSVulkanPlaneInfo> pyramids;
        for (int f = -1; f < static_cast<int>(refs.size()); ++f) {
            const VSFrame *frame = vsapi->getFrameFilter(f < 0 ? n : refs[f], d->node, frameCtx);
            SuperFrames sf;
            const bool found = GetSuperFrames(frame, L, d->prefix, sf, vsapi);
            vsapi->freeFrame(frame);
            if (!found)
                return fail("the super clip's frames lack their GPU planes; it must come from mvgpu.Super with the same prefix");
            held.push_back(sf);
            VSVulkanPlaneInfo info;
            if (vkapi->getGPUPlane(sf.pyramid, 0, &info))
                return fail("the super clip's pyramid isn't GPU resident");
            pyramids.push_back(info);
        }
        VSVulkanPlaneInfo outPlane;
        if (vkapi->getGPUPlane(out, 0, &outPlane))
            return fail("the coarse search's frame isn't GPU resident");
        const VkDeviceSize rowBytes = static_cast<VkDeviceSize>(vsapi->getStride(out, 0));

        char err[1024] = {};
        VSVulkanBufferInfo scratchInfo = {};
        scratch = vkapi->createGPUBuffer(core, d->scratchBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &scratchInfo, err, sizeof(err));
        if (!scratch)
            return fail(err);
        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        vkapi->gpuExecUsesBuffer(ctx, scratch);
        for (const SuperFrames &f : held)
            vkapi->gpuExecReadsFrame(ctx, f.pyramid);
        vkapi->gpuExecWritesPlane(ctx, out, 0);

        const VkBuffer constants = d->constantsInfo.buffer, work = scratchInfo.buffer;
        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        Recorder rec(*d->vc, cmd, constants, Layout::Coarse);
        const VkQueryPool queries = d->profile ? d->profile->Begin(*d->vc, cmd) : VK_NULL_HANDLE;
        rec.Bind(kCurPyramid, pyramids[0].buffer);
        rec.Bind(kLevels, constants, d->levels.offset, d->levels.size);
        rec.Bind(kLevelLambda, constants, d->levelLambda.offset, d->levelLambda.size);
        rec.Bind(kLevelVec, work, d->levelVec.offset, d->levelVec.size);
        rec.Bind(kLevelSad, work, d->levelSad.offset, d->levelSad.size);
        rec.Bind(kMedians, work, d->medians.offset, d->medians.size);

        const LevelEntry &finest = L.levels[SuperLayout::kFinest];
        const VkDeviceSize finestBytes = static_cast<VkDeviceSize>(finest.nbx) * finest.nby * 8;
        for (size_t start = 0; start < slots.size(); start += d->batch) {
            const uint32_t count = static_cast<uint32_t>(std::min<size_t>(d->batch, slots.size() - start));
            for (uint32_t z = 0; z < count; ++z) {
                const int nref = n + d->deltas[slots[start + z]];
                const size_t r = std::find(refs.begin(), refs.end(), nref) - refs.begin();
                rec.Bind(kRefPyramid, pyramids[1 + r].buffer, 0, VK_WHOLE_SIZE, static_cast<int>(z));
            }
            // The batch before has copied out what this one overwrites
            if (start > 0)
                rec.TransferToCompute();
            CoarseRecorder(*d, rec, count).Search();
            // Each field's global median and finest level to its row
            rec.ComputeToTransfer();
            for (uint32_t z = 0; z < count; ++z) {
                const VkDeviceSize row = rowBytes * slots[start + z];
                rec.Copy(work, d->medians.offset + (static_cast<VkDeviceSize>(z) * kMedianSlots + kGlobalSlot) * 8, outPlane.buffer, row, 8);
                rec.Copy(work, d->levelVec.offset + (static_cast<VkDeviceSize>(z) * L.fieldTotal + finest.fieldOff) * 8, outPlane.buffer, row + 8, finestBytes);
            }
        }
        if (d->profile)
            d->profile->Stamp(*d->vc, cmd, queries, 1);

        uint64_t signaled = 0;
        const int submitted = vkapi->gpuExecSubmit(ctx, &signaled, err, sizeof(err));
        ctx = nullptr;
        scratch = nullptr; // the context owned it
        if (submitted)
            return fail(err);
        if (d->profile)
            d->profile->Finish(*d->vc, d->pool, signaled, queries);
        for (SuperFrames &f : held)
            f.Free(vsapi);
        return out;
    }

    return nullptr;
}

// ---- The field's node

struct AnalyseData {
    VSNode *node = nullptr;
    VSNode *coarseNode = nullptr; // and the row of its frames this field's coarse search is in
    int coarseRow = 0;
    // The fields this one's seeds chain (stepNode at n, restNode at n + unit) and invert (invNode at n + delta), or null
    VSNode *stepNode = nullptr;
    VSNode *restNode = nullptr;
    VSNode *invNode = nullptr;
    int unit = 0;

    VSVideoInfo vi = {};
    SuperLayout layout;

    int deltaFrame = 1;
    bool chroma = true; // the SADs count chroma
    int badSad = 0;
    int fallbackRadius = 0;
    int fallbackStep = 1;

    std::string prefix;

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline seedScatter = VK_NULL_HANDLE, seedBuild = VK_NULL_HANDLE;
    VkPipeline init = VK_NULL_HANDLE, pass = VK_NULL_HANDLE, flag = VK_NULL_HANDLE, fallback = VK_NULL_HANDLE, apply = VK_NULL_HANDLE, halfpel = VK_NULL_HANDLE;
    VkPipeline quarter = VK_NULL_HANDLE; // pel 4

    // The tables, uploaded once: lambda for the full-size grid, the level table
    VSGPUBuffer *constants = nullptr;
    VSVulkanBufferInfo constantsInfo = {};
    Region lambda, levels;

    // A field's working buffers, regions of one buffer allocated per frame
    Region seeds, seedCount, seedSad, vecA, sadA, vecB, sadB, lastChange, lastEval, counters, flagged, invKey;
    VkDeviceSize scratchBytes = 0;

    VSVideoFormat gray32 = {};

    std::unique_ptr<StageProfiler> profile; // MVGPU_PROFILE

    const VSAPI *vsapi;

    AnalyseData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~AnalyseData() {
        // The pool drains the GPU first, so nothing still uses what follows
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        profile.reset();
        if (constants)
            vc->vkapi->destroyGPUBuffer(constants);
        vc.reset();
        vsapi->freeNode(node);
        vsapi->freeNode(coarseNode);
        vsapi->freeNode(stepNode);
        vsapi->freeNode(restNode);
        vsapi->freeNode(invNode);
    }
};

// What a field's submission records, the order and the barriers as the prototype's vkrefine.cpp
class FieldRecorder {
public:
    // stamp(i), when given, marks the end of stage i (MVGPU_PROFILE): 1 the seeds, 2 init, 3 the
    // passes before the fallback, 4 the fallback, 5 the passes after it, 6 the half-pel step, 7 the
    // quarter-pel step
    FieldRecorder(const AnalyseData &d, Recorder &rec, VkBuffer scratch, int lumaStride, int chromaStride, int recStride, int coarseBase,
                  std::function<void(int)> stamp)
        : d(d), L(d.layout), rec(rec), scratch(scratch), lumaStride(lumaStride), chromaStride(chromaStride), recStride(recStride), coarseBase(coarseBase),
          stamp(std::move(stamp)) {}

    // flags is the caller's to set where a kernel uses it
    Params MakeParams(int colour, int stamp) const {
        Params q = {};
        q.w = L.width;
        q.h = L.height;
        q.nbx = L.nbx;
        q.nby = L.nby;
        q.step = L.step;
        q.pad = L.pad;
        q.padc = L.padc;
        q.colour = colour;
        q.wp = lumaStride;
        q.hp = L.hp;
        q.wc = chromaStride;
        q.hc = L.hc;
        q.fallbackRadius = d.fallbackRadius;
        q.fallbackStep = d.fallbackStep;
        q.badSad = d.badSad;
        q.maxSeeds = kMaxSeeds;
        q.stamp = stamp;
        q.topRadius = kTopRadius;
        q.medianScale = 1;
        q.finest = SuperLayout::kFinest;
        q.blockRows = L.blk;
        q.recStride = recStride;
        q.coarseBase = coarseBase;
        return q;
    }

    // The full-size seed lists; flags: 1 the chained fields are bound, 2 the inverted field is
    void SeedLists(int flags) {
        const uint32_t groups = static_cast<uint32_t>((static_cast<int64_t>(L.nbx) * L.nby + 63) / 64);
        Params q = MakeParams(0, 0);
        q.flags = flags;
        if (flags & 2) {
            rec.Dispatch(d.seedScatter, q, groups, 1);
            rec.ComputeBarrier();
        }
        rec.Dispatch(d.seedBuild, q, groups, 1);
        rec.ComputeBarrier();
        Stamp(1);
    }

    // The seeds measured (init), the checkerboard passes, the fallback and its extra pass pair, then
    // the half-pel step, which writes the output; at pel 4 the half-pel step writes field B and the
    // quarter-pel step, reading it as its field A, writes the output
    void Refinement() {
        const int64_t nb = static_cast<int64_t>(L.nbx) * L.nby;
        const uint32_t gx = static_cast<uint32_t>(L.nbx), gy = static_cast<uint32_t>(L.nby);
        const uint32_t gx8 = (gx + 7) / 8, gxHalf8 = ((gx + 1) / 2 + 7) / 8;
        const bool withFallback = d.fallbackRadius > 0;
        const int totalPairs = kPairs + (withFallback ? 1 : 0);
        int step = 0; // init is step 0, every pass and the fallback one more
        rec.Dispatch(d.init, MakeParams(0, step), gx8, gy);
        rec.ComputeBarrier();
        Stamp(2);
        if (!withFallback) {
            Stamp(3);
            Stamp(4);
        }
        for (int p = 0; p < 2 * totalPairs; ++p) {
            if (withFallback && p == 2 * kPairs) {
                Stamp(3);
                // List the blocks above badSad (init emptied the list); search those, reading A and
                // writing their results to B by list entry; then copy them to A
                rec.Dispatch(d.flag, MakeParams(0, step + 1), static_cast<uint32_t>((nb + 63) / 64), 1);
                rec.IndirectBarrier();
                ++step;
                rec.DispatchIndirect(d.fallback, MakeParams(0, step), scratch, d.flagged.offset);
                rec.ComputeBarrier();
                rec.DispatchIndirect(d.apply, MakeParams(0, step), scratch, d.flagged.offset);
                rec.ComputeBarrier();
                Stamp(4);
            }
            rec.Dispatch(d.pass, MakeParams(p & 1, ++step), gxHalf8, gy);
            rec.ComputeBarrier();
        }
        Stamp(5);
        rec.Dispatch(d.halfpel, MakeParams(0, step), gx8, gy);
        if (L.pel == 4) {
            rec.ComputeBarrier();
            Stamp(6);
            rec.Bind(kVecA, scratch, d.vecB.offset, d.vecB.size);
            rec.Bind(kSadA, scratch, d.sadB.offset, d.sadB.size);
            rec.Dispatch(d.quarter, MakeParams(0, step), gx8, gy);
        } else {
            Stamp(6);
        }
        Stamp(7);
    }

private:
    void Stamp(int stage) {
        if (stamp)
            stamp(stage);
    }

    const AnalyseData &d;
    const SuperLayout &L;
    Recorder &rec;
    VkBuffer scratch;
    int lumaStride, chromaStride, recStride, coarseBase;
    std::function<void(int)> stamp;
};

static const VSFrame *VS_CC analyseGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    AnalyseData *d = reinterpret_cast<AnalyseData *>(instanceData);

    const int nref = n + d->deltaFrame;
    const bool hasRef = nref >= 0 && nref < d->vi.numFrames;

    if (activationReason == arInitial) {
        if (hasRef) {
            vsapi->requestFrameFilter(std::min(n, nref), d->node, frameCtx);
            vsapi->requestFrameFilter(std::max(n, nref), d->node, frameCtx);
            vsapi->requestFrameFilter(n, d->coarseNode, frameCtx);
            if (d->stepNode && d->restNode) {
                vsapi->requestFrameFilter(n, d->stepNode, frameCtx);
                vsapi->requestFrameFilter(n + d->unit, d->restNode, frameCtx);
            }
            if (d->invNode)
                vsapi->requestFrameFilter(nref, d->invNode, frameCtx);
        } else {
            // too close to beginning/end of clip
            vsapi->requestFrameFilter(n, d->node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const VSVULKANAPI *vkapi = d->vc->vkapi;
        const SuperLayout &L = d->layout;
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);

        if (!hasRef) {
            VSFrame *dst = vsapi->copyFrame(src, core);
            ExportAnalysis(dst, L, d->deltaFrame, d->chroma, nullptr, d->prefix, vsapi);
            vsapi->freeFrame(src);
            return dst;
        }

        // Everything this frame holds a reference to, released on every way out
        std::vector<const VSFrame *> held = {src};
        VSFrame *vectors = nullptr;
        VSGPUExecContext *ctx = nullptr;
        VSGPUBuffer *scratch = nullptr;
        auto release = [&]() {
            for (const VSFrame *f : held)
                vsapi->freeFrame(f);
            vsapi->freeFrame(vectors);
        };
        auto fail = [&](const std::string &message) -> const VSFrame * {
            if (ctx)
                vkapi->gpuExecAbandon(ctx);
            else if (scratch)
                vkapi->destroyGPUBuffer(scratch);
            release();
            vsapi->setFilterError(("Analyse: " + message).c_str(), frameCtx);
            return nullptr;
        };
        auto hold = [&](const VSFrame *f) {
            if (f)
                held.push_back(f);
            return f;
        };

        const VSFrame *ref = hold(vsapi->getFrameFilter(nref, d->node, frameCtx));
        const VSFrame *coarse = hold(vsapi->getFrameFilter(n, d->coarseNode, frameCtx));
        SuperFrames cur, rf;
        if (!GetSuperFrames(src, L, d->prefix, cur, vsapi) || !GetSuperFrames(ref, L, d->prefix, rf, vsapi)) {
            cur.Free(vsapi);
            return fail("the super clip's frames lack their GPU planes; it must come from mvgpu.Super with the same prefix");
        }
        for (const VSFrame *f : {cur.luma, cur.chroma, cur.pyramid, rf.luma, rf.chroma, rf.pyramid})
            hold(f);

        // The fields the seeds chain and invert, refined by the other nodes AnalyseMany made
        const VSFrame *stepVec = nullptr, *restVec = nullptr, *invVec = nullptr;
        if (d->stepNode && d->restNode) {
            stepVec = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(n, d->stepNode, frameCtx)), d->prefix, vsapi));
            restVec = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(n + d->unit, d->restNode, frameCtx)), d->prefix, vsapi));
        }
        if (d->invNode)
            invVec = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(nref, d->invNode, frameCtx)), d->prefix, vsapi));

        vectors = vkapi->newGPUVideoFrame(&d->gray32, 4 * L.nbx, L.nby, nullptr, core);
        if (!vectors)
            return fail("failed to allocate the vector frame");
        const ptrdiff_t recBytes = vsapi->getStride(vectors, 0);
        if (recBytes % 16)
            return fail("the vector frame's rows don't start on whole records");
        auto sameShape = [&](const VSFrame *f) {
            return f && vsapi->getFrameWidth(f, 0) == 4 * L.nbx && vsapi->getFrameHeight(f, 0) == L.nby && vsapi->getStride(f, 0) == recBytes;
        };
        const int flags = (sameShape(stepVec) && sameShape(restVec) ? 1 : 0) | (sameShape(invVec) ? 2 : 0);

        const ptrdiff_t lumaStride = vsapi->getStride(cur.luma, 0), chromaStride = vsapi->getStride(cur.chroma, 0);
        if (lumaStride != vsapi->getStride(rf.luma, 0) || chromaStride != vsapi->getStride(rf.chroma, 0) || lumaStride % 4 || chromaStride % 4)
            return fail("the super frames' storage strides differ");
        const ptrdiff_t coarseRowBytes = vsapi->getStride(coarse, 0);
        if (coarseRowBytes % 8)
            return fail("the coarse search's rows don't start on whole vectors");

        // The buffers every binding names
        VSVulkanPlaneInfo curLuma, curChroma, refLuma, refChroma, coarsePlane, outRec, stepRec = {}, restRec = {}, invRec = {};
        if (vkapi->getGPUPlane(cur.luma, 0, &curLuma) || vkapi->getGPUPlane(cur.chroma, 0, &curChroma) || vkapi->getGPUPlane(rf.luma, 0, &refLuma) ||
            vkapi->getGPUPlane(rf.chroma, 0, &refChroma) || vkapi->getGPUPlane(coarse, 0, &coarsePlane) || vkapi->getGPUPlane(vectors, 0, &outRec) ||
            ((flags & 1) && (vkapi->getGPUPlane(stepVec, 0, &stepRec) || vkapi->getGPUPlane(restVec, 0, &restRec))) ||
            ((flags & 2) && vkapi->getGPUPlane(invVec, 0, &invRec)))
            return fail("a frame the analysis reads isn't GPU resident");

        char err[1024] = {};
        VSVulkanBufferInfo scratchInfo = {};
        scratch = vkapi->createGPUBuffer(core, d->scratchBytes,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &scratchInfo, err, sizeof(err));
        if (!scratch)
            return fail(err);
        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        vkapi->gpuExecUsesBuffer(ctx, scratch);
        for (const VSFrame *f : {cur.luma, cur.chroma, rf.luma, rf.chroma, coarse})
            vkapi->gpuExecReadsFrame(ctx, f);
        if (flags & 1) {
            vkapi->gpuExecReadsFrame(ctx, stepVec);
            vkapi->gpuExecReadsFrame(ctx, restVec);
        }
        if (flags & 2)
            vkapi->gpuExecReadsFrame(ctx, invVec);
        vkapi->gpuExecWritesPlane(ctx, vectors, 0);

        const VkBuffer constants = d->constantsInfo.buffer, work = scratchInfo.buffer;
        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        Recorder rec(*d->vc, cmd, constants);
        const VkQueryPool queries = d->profile ? d->profile->Begin(*d->vc, cmd) : VK_NULL_HANDLE;
        rec.Bind(kCurLuma, curLuma.buffer);
        rec.Bind(kRefLuma, refLuma.buffer);
        rec.Bind(kCurChroma, curChroma.buffer);
        rec.Bind(kRefChroma, refChroma.buffer);
        rec.Bind(kCoarse, coarsePlane.buffer);
        rec.Bind(kLambda, constants, d->lambda.offset, d->lambda.size);
        rec.Bind(kLevels, constants, d->levels.offset, d->levels.size);
        const std::pair<int, const Region *> regions[] = {
            {kSeeds, &d->seeds}, {kSeedCount, &d->seedCount}, {kSeedSad, &d->seedSad}, {kVecA, &d->vecA}, {kSadA, &d->sadA}, {kVecB, &d->vecB},
            {kSadB, &d->sadB}, {kLastChange, &d->lastChange}, {kLastEval, &d->lastEval}, {kCounters, &d->counters}, {kFlagged, &d->flagged},
            {kInvKey, &d->invKey}};
        for (const auto &[binding, region] : regions)
            rec.Bind(binding, work, region->offset, region->size);
        rec.Bind(kOutRec, outRec.buffer);
        if (flags & 1) {
            rec.Bind(kStepRec, stepRec.buffer);
            rec.Bind(kRestRec, restRec.buffer);
        }
        if (flags & 2) {
            rec.Bind(kInvRec, invRec.buffer);
            // No inverted vector lands anywhere yet
            rec.Fill(work, d->invKey.offset, d->invKey.size, 0xFFFFFFFFu);
            rec.TransferToCompute();
        }

        std::function<void(int)> stamp;
        if (d->profile)
            stamp = [&](int stage) { d->profile->Stamp(*d->vc, cmd, queries, stage); };
        FieldRecorder field(*d, rec, work, static_cast<int>(lumaStride), static_cast<int>(chromaStride), static_cast<int>(recBytes / 16),
                            static_cast<int>(d->coarseRow * coarseRowBytes / 8), std::move(stamp));
        field.SeedLists(flags);
        field.Refinement();

        uint64_t signaled = 0;
        const int submitted = vkapi->gpuExecSubmit(ctx, &signaled, err, sizeof(err));
        ctx = nullptr;
        scratch = nullptr; // the context owned it
        if (submitted)
            return fail(err);
        if (d->profile)
            d->profile->Finish(*d->vc, d->pool, signaled, queries);

        VSFrame *dst = vsapi->copyFrame(src, core);
        ExportAnalysis(dst, L, d->deltaFrame, d->chroma, vectors, d->prefix, vsapi);
        release();
        return dst;
    }

    return nullptr;
}

namespace {

// The arguments of Analyse and AnalyseMany, read and checked once; AnalyseMany creates every
// field's node from them
struct AnalyseArgs {
    VSNode *node = nullptr; // the super clip, a new reference
    SuperLayout layout;
    std::string prefix;
    int deltaFrame = 1;
    bool chroma = true;
    int plevel = 1;
    int64_t mvlambda = 1000, lsad = 400, badsad = 1000;
    int badrange = 40, badstep = 2;

    Scaled Scale() const {
        const int64_t area = static_cast<int64_t>(layout.blk) * layout.blk;
        return {mvlambda * area / 64 / (layout.pel * layout.pel), lsad * area / 64, mvlambda * area / 64};
    }
};

AnalyseArgs ParseAnalyseArgs(const VSMap *in, const VSAPI *vsapi) {
    AnalyseArgs a;
    int err;

    const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
    if (prefix)
        a.prefix = prefix;
    else
        a.prefix = DEFAULT_MVGPUTENSILS_PREFIX;

    a.node = vsapi->mapGetNode(in, "super", 0, nullptr);
    try {
        a.layout = ImportSuperLayout(a.node, a.prefix, vsapi);
        if (const std::string unsupported = a.layout.Unsupported(SuperLayout::Use::Search); !unsupported.empty())
            throw std::runtime_error(unsupported);
        if (a.layout.topLevel < 1)
            throw std::runtime_error("the super clip has no coarse levels, which the search starts from: it was made with onelevel=True, or the frame is narrower than " +
                                     std::to_string(2 * SuperLayout::kTopWidth) + " pixels");
        // median.comp's histogram
        if (a.layout.maxCoarseVector >= 4064)
            throw std::runtime_error("the frame is too large for the coarse search");

        int blkX, blkY, overlapX, overlapY;
        GetPairArgument(blkX, blkY, "blksize", a.layout.blk, a.layout.blk, in, vsapi);
        GetPairArgument(overlapX, overlapY, "overlap", a.layout.overlap, a.layout.overlap, in, vsapi);

        const bool useSatd = !!vsapi->mapGetInt(in, "satd", 0, &err);

        CheckBlockSize(blkX, blkY, overlapX, overlapY, 1, 1);

        // levels, search, searchparam and pelsearch steer mvu's hierarchical search
        vsapi->mapGetIntSaturated(in, "levels", 0, &err);

        int searchType = vsapi->mapGetIntSaturated(in, "search", 0, &err);
        if (err)
            searchType = 2;

        vsapi->mapGetIntSaturated(in, "searchparam", 0, &err);

        int pelSearch = vsapi->mapGetIntSaturated(in, "pelsearch", 0, &err);
        if (err)
            pelSearch = 2;

        if (pelSearch <= 0)
            throw std::runtime_error("pelsearch must be positive");

        a.chroma = !!vsapi->mapGetInt(in, "chroma", 0, &err);
        if (err)
            a.chroma = true;

        a.deltaFrame = vsapi->mapGetIntSaturated(in, "delta", 0, &err);
        if (err)
            a.deltaFrame = 1;

        a.mvlambda = vsapi->mapGetIntSaturated(in, "mvlambda", 0, &err);
        if (err)
            a.mvlambda = 1000;
        if (a.mvlambda < 0)
            throw std::runtime_error("mvlambda must be non-negative");

        a.lsad = vsapi->mapGetIntSaturated(in, "lsad", 0, &err);
        if (err)
            a.lsad = 400;

        a.plevel = vsapi->mapGetIntSaturated(in, "plevel", 0, &err);
        if (err)
            a.plevel = 1;

        vsapi->mapGetInt(in, "globalmv", 0, &err);

        int pnew = vsapi->mapGetIntSaturated(in, "pnew", 0, &err);
        if (err)
            pnew = 25;

        int pzero = vsapi->mapGetIntSaturated(in, "pzero", 0, &err);
        if (err)
            pzero = pnew;

        const int pglobal = vsapi->mapGetIntSaturated(in, "pglobal", 0, &err);

        a.badsad = vsapi->mapGetIntSaturated(in, "badsad", 0, &err);
        if (err)
            a.badsad = 1000;

        a.badrange = vsapi->mapGetIntSaturated(in, "badrange", 0, &err);
        if (err)
            a.badrange = 40;

        a.badstep = vsapi->mapGetIntSaturated(in, "badstep", 0, &err);
        if (err)
            a.badstep = 2;

        vsapi->mapGetInt(in, "meander", 0, &err);

        const int tryMany = vsapi->mapGetIntSaturated(in, "trymany", 0, &err);
        if (tryMany < 0 || tryMany > 2)
            throw std::runtime_error("trymany must be between 0 and 2");

        const bool fields = !!vsapi->mapGetInt(in, "fields", 0, &err);

        if (searchType < 0 || searchType > 5)
            throw std::runtime_error("search must be between 0 and 5");

        if (a.plevel < 0 || a.plevel > 2)
            throw std::runtime_error("plevel must be between 0 and 2");

        if (pnew < 0 || pnew > 256)
            throw std::runtime_error("pnew must be between 0 and 256");

        if (pzero < 0 || pzero > 256)
            throw std::runtime_error("pzero must be between 0 and 256");

        if (pglobal < 0 || pglobal > 256)
            throw std::runtime_error("pglobal must be between 0 and 256");

        if (a.deltaFrame == 0)
            throw std::runtime_error("delta can't be 0");

        if (a.badstep < 1 || a.badstep > 8)
            throw std::runtime_error("badstep must be between 1 and 8");

        if (std::abs(a.badrange) > 1024)
            throw std::runtime_error("badrange must be between -1024 and 1024");

        // What the GPU path implements so far
        if (blkX != a.layout.blk || blkY != a.layout.blk || overlapX != a.layout.overlap || overlapY != a.layout.overlap)
            throw std::runtime_error("blksize and overlap must be the super clip's; analysing another grid isn't implemented yet");
        if (useSatd)
            throw std::runtime_error("satd isn't implemented yet");
        if (fields)
            throw std::runtime_error("fields isn't implemented yet");
    } catch (...) {
        vsapi->freeNode(a.node);
        throw;
    }
    return a;
}

// The coarse node of the fields (n, deltas[s]); takes no reference of the caller's
VSNode *CreateCoarse(const AnalyseArgs &a, const std::vector<int> &deltas, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<CoarseData> d = std::make_unique<CoarseData>(vsapi);
    d->node = vsapi->addNodeRef(a.node);
    const VSVideoInfo *superVi = vsapi->getVideoInfo(a.node);
    d->numFrames = superVi->numFrames;
    d->layout = a.layout;
    d->deltas = deltas;
    d->prefix = a.prefix;
    const SuperLayout &L = d->layout;
    const Scaled s = a.Scale();

    d->vc = VulkanContext::Get(core, vsapi);
    VulkanContext &vc = *d->vc;
    const int variant = SearchVariant(a.chroma);
    d->pyrTop = vc.Pipeline(Kernel::PyrTop, L.blk, 2, variant);
    d->pyrPass = vc.Pipeline(Kernel::PyrPass, L.blk, 2, variant);
    d->pyrSeed = vc.Pipeline(Kernel::PyrSeed, L.blk, 2, variant);
    d->median = vc.Pipeline(Kernel::Median, L.blk, 2);

    // lambda for every (worst neighbour SAD) >> 1 per coarse level, mvlambda * 2^(plevel * level)
    // relaxed by lsad as the CPU reference does in double precision
    std::vector<int64_t> levelLambda(static_cast<size_t>(L.topLevel + 1) * SuperLayout::kLambdaEntries);
    for (int level = 0; level <= L.topLevel; ++level)
        for (int i = 0; i < SuperLayout::kLambdaEntries; ++i) {
            const double sc = static_cast<double>(s.lsad) / std::max<int64_t>(s.lsad + i, 1);
            levelLambda[static_cast<size_t>(level) * SuperLayout::kLambdaEntries + i] = static_cast<int64_t>((s.lambdaBlock << (a.plevel * level)) * sc * sc);
        }
    Regions constants(vc);
    constants.Place(d->levelLambda, levelLambda.size() * 8);
    constants.Place(d->levels, L.levels.size() * sizeof(LevelEntry));
    std::vector<uint8_t> tables(constants.Total());
    memcpy(tables.data() + d->levelLambda.offset, levelLambda.data(), levelLambda.size() * 8);
    memcpy(tables.data() + d->levels.offset, L.levels.data(), L.levels.size() * sizeof(LevelEntry));

    d->batch = std::min<int>(VulkanContext::kMaxBatch, static_cast<int>(deltas.size()));
    Regions scratch(vc);
    scratch.Place(d->levelVec, static_cast<VkDeviceSize>(d->batch) * L.fieldTotal * 8);
    scratch.Place(d->levelSad, static_cast<VkDeviceSize>(d->batch) * L.fieldTotal * 4);
    scratch.Place(d->medians, static_cast<VkDeviceSize>(d->batch) * kMedianSlots * 8);
    d->scratchBytes = scratch.Total();

    char err[1024] = {};
    d->pool = vc.vkapi->createGPUExecPool(core, vqCompute, err, sizeof(err));
    if (!d->pool)
        throw std::runtime_error(err);
    d->constants = vc.Upload(core, d->pool, tables.data(), tables.size(), d->constantsInfo);

    d->vi = *superVi;
    if (!vsapi->queryVideoFormat(&d->vi.format, cfGray, stInteger, 32, 0, 0, core))
        throw std::runtime_error("failed to query the Gray32 format");
    d->vi.width = CoarseRowInts(L);
    d->vi.height = static_cast<int>(deltas.size());

    if (StageProfiler::Requested())
        d->profile = std::make_unique<StageProfiler>("coarse searches", std::vector<std::string>{"all fields of a frame, batched"});

    VSFilterDependency deps[1] = {{d->node, rpGeneral}};
    VSNode *out = vsapi->createVideoFilterEx2("AnalyseCoarse", &d->vi, coarseGetFrame, filterFree<CoarseData>, fmParallel, ffGPUOutput, deps, 1, d.get(), core);
    if (!out)
        throw std::runtime_error("failed to create the coarse search's filter");
    d.release();
    return out;
}

// The node of field delta, its coarse search in row coarseRow of coarseNode's frames, its seeds
// chaining stepNode and restNode (with the step unit) and inverting invNode where those aren't
// null; takes no reference of the caller's
VSNode *CreateAnalyse(const AnalyseArgs &a, int delta, VSNode *coarseNode, int coarseRow, VSNode *stepNode, VSNode *restNode, int unit, VSNode *invNode,
                      VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<AnalyseData> d = std::make_unique<AnalyseData>(vsapi);
    d->node = vsapi->addNodeRef(a.node);
    d->coarseNode = vsapi->addNodeRef(coarseNode);
    d->coarseRow = coarseRow;
    d->vi = *vsapi->getVideoInfo(a.node);
    d->layout = a.layout;
    d->prefix = a.prefix;
    d->deltaFrame = delta;
    d->chroma = a.chroma;
    if (stepNode && restNode) {
        d->stepNode = vsapi->addNodeRef(stepNode);
        d->restNode = vsapi->addNodeRef(restNode);
        d->unit = unit;
    }
    if (invNode)
        d->invNode = vsapi->addNodeRef(invNode);

    const SuperLayout &L = d->layout;
    const int blk = L.blk, area = blk * blk;
    const Scaled s = a.Scale();
    d->badSad = static_cast<int>(std::min<int64_t>(a.badsad * area / 64, INT32_MAX));
    d->fallbackRadius = std::abs(a.badrange);
    d->fallbackStep = a.badstep;

    d->vc = VulkanContext::Get(core, vsapi);
    VulkanContext &vc = *d->vc;
    const int pel = L.pel, variant = SearchVariant(a.chroma);
    d->seedScatter = vc.Pipeline(Kernel::SeedScatter, blk, pel);
    d->seedBuild = vc.Pipeline(Kernel::SeedBuild, blk, pel);
    d->init = vc.Pipeline(Kernel::RefineInit, blk, pel, variant);
    d->pass = vc.Pipeline(Kernel::RefinePass, blk, pel, variant);
    d->flag = vc.Pipeline(Kernel::RefineFlag, blk, pel);
    d->fallback = vc.Pipeline(Kernel::RefineFallback, blk, pel, variant);
    d->apply = vc.Pipeline(Kernel::RefineApply, blk, pel);
    d->halfpel = vc.Pipeline(Kernel::RefineHalfpel, blk, pel, variant);
    if (pel == 4)
        d->quarter = vc.Pipeline(Kernel::RefineQuarter, blk, pel, variant);

    // lambda for every (worst neighbour SAD) >> 1, relaxed by lsad as the CPU reference does in
    // double precision
    std::vector<int64_t> lambda(LambdaEntries(blk, a.chroma));
    for (size_t i = 0; i < lambda.size(); ++i) {
        const double scale = static_cast<double>(s.lsad) / std::max<int64_t>(s.lsad + static_cast<int64_t>(i), 1);
        lambda[i] = static_cast<int64_t>(s.lambda0 * scale * scale);
    }
    Regions constants(vc);
    constants.Place(d->lambda, lambda.size() * 8);
    constants.Place(d->levels, L.levels.size() * sizeof(LevelEntry));
    std::vector<uint8_t> tables(constants.Total());
    memcpy(tables.data() + d->lambda.offset, lambda.data(), lambda.size() * 8);
    memcpy(tables.data() + d->levels.offset, L.levels.data(), L.levels.size() * sizeof(LevelEntry));

    const VkDeviceSize nb = static_cast<VkDeviceSize>(L.nbx) * L.nby;
    Regions scratch(vc);
    scratch.Place(d->seeds, nb * kMaxSeeds * 8);
    scratch.Place(d->seedCount, nb * 4);
    scratch.Place(d->seedSad, nb * kMaxSeeds * 4);
    scratch.Place(d->vecA, nb * 8);
    scratch.Place(d->sadA, nb * 4);
    scratch.Place(d->vecB, nb * 8);
    scratch.Place(d->sadB, nb * 4);
    scratch.Place(d->lastChange, nb * 4);
    scratch.Place(d->lastEval, nb * 4);
    scratch.Place(d->counters, kCounterSlots * 4);
    scratch.Place(d->flagged, 12 + nb * 4); // the fallback's dispatch size (x, y, z), then its list of blocks
    scratch.Place(d->invKey, nb * 8);
    d->scratchBytes = scratch.Total();
    if (d->scratchBytes > vc.limits.maxStorageBufferRange || static_cast<VkDeviceSize>(L.LumaRows()) * L.wp > vc.limits.maxStorageBufferRange)
        throw std::runtime_error("the frame is too large for the device's storage buffers");

    char err[1024] = {};
    d->pool = vc.vkapi->createGPUExecPool(core, vqCompute, err, sizeof(err));
    if (!d->pool)
        throw std::runtime_error(err);
    d->constants = vc.Upload(core, d->pool, tables.data(), tables.size(), d->constantsInfo);

    if (!vsapi->queryVideoFormat(&d->gray32, cfGray, stInteger, 32, 0, 0, core))
        throw std::runtime_error("failed to query the Gray32 format");

    if (StageProfiler::Requested())
        d->profile = std::make_unique<StageProfiler>("field " + std::to_string(delta) + " (pel " + std::to_string(pel) + ")",
                                                     std::vector<std::string>{"seeds", "init", "passes before the fallback", "fallback", "passes after it",
                                                                              "half-pel step", "quarter-pel step"});

    VSFilterDependency deps[5] = {{d->node, rpGeneral}, {d->coarseNode, rpGeneral}};
    int numDeps = 2;
    for (VSNode *dep : {d->stepNode, d->restNode, d->invNode})
        if (dep)
            deps[numDeps++] = {dep, rpGeneral};

    VSNode *out = vsapi->createVideoFilterEx2("Analyse", &d->vi, analyseGetFrame, filterFree<AnalyseData>, fmParallel, ffGPUOutput, deps, numDeps, d.get(), core);
    if (!out)
        throw std::runtime_error("failed to create the filter");
    d.release();
    return out;
}

} // namespace

static void VS_CC analyseCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    VSNode *coarse = nullptr;
    try {
        AnalyseArgs a = ParseAnalyseArgs(in, vsapi);
        try {
            coarse = CreateCoarse(a, {a.deltaFrame}, core, vsapi);
            vsapi->mapConsumeNode(out, "clip", CreateAnalyse(a, a.deltaFrame, coarse, 0, nullptr, nullptr, 0, nullptr, core, vsapi), maAppend);
        } catch (...) {
            vsapi->freeNode(a.node);
            throw;
        }
        vsapi->freeNode(a.node);
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("Analyse: " + std::string(e.what())).c_str());
    }
    vsapi->freeNode(coarse);
}

// Every field of radius frames either side, in mvu.AnalyseMany's order (delta, -delta,
// 2 * delta, -2 * delta, ...). They are created in the order their seeds need each other, -k
// before +k and k before k + 1, and each node gets the nodes of the fields it chains and inverts.
// One coarse node searches all of a frame's fields, row 2 (r - 1) for -r * delta and the next for
// r * delta.
static void VS_CC analyseManyCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    int err;

    int radius = vsapi->mapGetIntSaturated(in, "radius", 0, &err);
    if (err)
        radius = 1;
    if (radius < 1) {
        vsapi->mapSetError(out, "AnalyseMany: radius must be a positive number");
        return;
    }

    int delta = vsapi->mapGetIntSaturated(in, "delta", 0, &err);
    if (err)
        delta = 1;

    if (delta < 1) {
        vsapi->mapSetError(out, "AnalyseMany: delta must be a positive number");
        return;
    }

    std::vector<VSNode *> negative(radius + 1, nullptr), positive(radius + 1, nullptr); // index r: field -r * delta, r * delta
    VSNode *coarse = nullptr;
    try {
        AnalyseArgs a = ParseAnalyseArgs(in, vsapi);
        try {
            std::vector<int> deltas;
            for (int r = 1; r <= radius; ++r) {
                deltas.push_back(-r * delta);
                deltas.push_back(r * delta);
            }
            coarse = CreateCoarse(a, deltas, core, vsapi);
            for (int r = 1; r <= radius; ++r) {
                // (n, -r) chains (n, -1) and (n - 1, -(r - 1)); (n, r) chains (n, 1) and (n + 1, r - 1)
                // and inverts (n + r, -r)
                const bool chain = r >= 2;
                negative[r] = CreateAnalyse(a, -r * delta, coarse, 2 * (r - 1), chain ? negative[1] : nullptr, chain ? negative[r - 1] : nullptr, -delta, nullptr,
                                            core, vsapi);
                positive[r] = CreateAnalyse(a, r * delta, coarse, 2 * (r - 1) + 1, chain ? positive[1] : nullptr, chain ? positive[r - 1] : nullptr, delta,
                                            negative[r], core, vsapi);
            }
        } catch (...) {
            vsapi->freeNode(a.node);
            throw;
        }
        vsapi->freeNode(a.node);
        for (int r = 1; r <= radius; ++r) {
            vsapi->mapConsumeNode(out, "clip", positive[r], maAppend);
            vsapi->mapConsumeNode(out, "clip", negative[r], maAppend);
            positive[r] = negative[r] = nullptr;
        }
    } catch (const std::exception &e) {
        for (int r = 1; r <= radius; ++r) {
            vsapi->freeNode(negative[r]);
            vsapi->freeNode(positive[r]);
        }
        vsapi->mapSetError(out, ("AnalyseMany: " + std::string(e.what())).c_str());
    }
    vsapi->freeNode(coarse);
}

void analyseRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Analyse",
                 "super:vnode:gpu;"
                 "blksize:int[]:opt;"
                 "levels:int:opt;"
                 "search:int:opt;"
                 "searchparam:int:opt;"
                 "pelsearch:int:opt;"
                 "mvlambda:int:opt;"
                 "chroma:int:opt;"
                 "delta:int:opt;"
                 "lsad:int:opt;"
                 "plevel:int:opt;"
                 "globalmv:int:opt;"
                 "pnew:int:opt;"
                 "pzero:int:opt;"
                 "pglobal:int:opt;"
                 "overlap:int[]:opt;"
                 "badsad:int:opt;"
                 "badrange:int:opt;"
                 "meander:int:opt;"
                 "trymany:int:opt;"
                 "fields:int:opt;"
                 "tff:int:opt;"
                 "satd:int:opt;"
                 "prefix:data:opt;"
                 "badstep:int:opt;",
                 "clip:vnode:gpu;",
                 analyseCreate, nullptr, plugin);
    vspapi->registerFunction("AnalyseMany",
                 "super:vnode:gpu;"
                 "blksize:int[]:opt;"
                 "levels:int:opt;"
                 "search:int:opt;"
                 "searchparam:int:opt;"
                 "pelsearch:int:opt;"
                 "mvlambda:int:opt;"
                 "chroma:int:opt;"
                 "delta:int:opt;"
                 "lsad:int:opt;"
                 "plevel:int:opt;"
                 "globalmv:int:opt;"
                 "pnew:int:opt;"
                 "pzero:int:opt;"
                 "pglobal:int:opt;"
                 "overlap:int[]:opt;"
                 "badsad:int:opt;"
                 "badrange:int:opt;"
                 "meander:int:opt;"
                 "trymany:int:opt;"
                 "fields:int:opt;"
                 "tff:int:opt;"
                 "satd:int:opt;"
                 "radius:int:opt;"
                 "prefix:data:opt;"
                 "badstep:int:opt;",
                 "clip:vnode[]:gpu;",
                 analyseManyCreate, nullptr, plugin);
}
