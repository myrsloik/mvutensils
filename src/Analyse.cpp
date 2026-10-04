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
// seeds from the coarse search only. A super without coarse levels (onelevel=True, or a frame
// narrower than 192 pixels) has no coarse node: its fields seed from zero and the chained and inverted
// vectors alone.
//
// Implemented, on Gray and YUV supers of any subsampling (4:2:0, 4:2:2, 4:4:0, 4:4:4) and 8 to
// 16-bit or float samples (mvlambda, lsad and badsad scaled to the depth as mvu scales them, floats
// searched as 16-bit samples, a Gray super's SADs luma's): chroma, mvlambda, lsad, plevel (lambda *
// 2^(plevel * L) at coarse level L, as mvu scales it per level), badsad and badrange (the
// fallback's threshold and radius), delta, prefix, plus badstep, the fallback's step, which mvu
// doesn't have. The fallback's defaults are the tested ones (badsad 1000, badrange 40, badstep 2)
// rather than mvu's (badsad 10000, which this search would almost never reach, and badrange 24).
// search, searchparam, pelsearch, levels, pnew, pzero, pglobal, globalmv, meander and trymany tune
// mvu's search, which this one doesn't use; they are checked and otherwise ignored. satd makes luma's
// SAD its SATD, mvu's, at the full-size grid (the coarse search keeps the SAD; 16x2 blocks are refused,
// as mvu refuses them). fields and tff shift the zero and median seeds of a field of an odd delta at
// pel 2 or 4 by mvu's field shift, as mvu shifts its zero and global predictors. Blocks whose SAD can
// pass 2^31 (128x128 with chroma at 16 bits or float) are refused. blksize and overlap may be other
// than the super's, as in mvu: the grid then has to fit the super's block-aligned frame
// (SuperLayout::WithGrid).

namespace {

constexpr int kMaxSeeds = 10;     // seed slots per block: zero, the median, chained, inverted and 5 coarse, at most
constexpr int kCounterSlots = 64; // the kernels' statistics, two per step
constexpr int kMedianSlots = 32, kGlobalSlot = 31;
constexpr int kPairs = 1;         // checkerboard pass pairs before the fallback
constexpr int kTopRadius = 32;    // the exhaustive search's radius at the top level, px
constexpr int kLevelPairs = 2;    // checkerboard pass pairs at every coarse level

// The pixels of a block's SAD: luma's, and U's and V's unless it is luma's alone
int BlockPixels(const SuperLayout &L, bool chroma) {
    return L.blk * L.blkY + (chroma ? 2 * (L.blk / L.format.xr) * (L.blkY / L.format.yr) : 0);
}

// The largest SAD of a block, in largest samples: its pixels, luma's twice with satd, whose SATD reaches
// at most twice the SAD (refine_common.glsl's kBlockPixels)
int64_t SadPixels(const SuperLayout &L, bool chroma, bool satd) {
    return BlockPixels(L, chroma) + (satd ? static_cast<int64_t>(L.blk) * L.blkY : 0);
}

// The bit depth the search works at: the samples', but 16 for floats, which it searches as the
// 16-bit samples they stand for (refine_common.glsl's Quantize), as mvu scales float SADs
int SearchBits(const SuperLayout &L) {
    return L.format.Kind() == 2 ? 16 : L.format.bits;
}

// The full-size grid's lambda table takes the worst neighbour SAD / 2 in steps of 2^(bits - 8), and
// for blocks of more than 32x32 pixels in steps 2^AreaShift times that (refine_common.glsl's
// kAreaShift), so that it stays the size of 32x32 blocks'
int AreaShift(const SuperLayout &L) {
    const int area = L.blk * L.blkY;
    return area > 8192 ? 4 : area > 4096 ? 3 : area > 2048 ? 2 : area > 1024 ? 1 : 0;
}

// Entries of the full-size grid's lambda table: (largest SAD of a block) >> (1 + bits - 8 + AreaShift),
// plus one
int LambdaEntries(const SuperLayout &L, bool chroma, bool satd) {
    const int bits = SearchBits(L);
    return static_cast<int>(((SadPixels(L, chroma, satd) * ((1 << bits) - 1)) >> (bits - 7 + AreaShift(L))) + 1);
}

// The search kernels' variant (specialization constant 6, refine_common.glsl): bit 0 for SADs of
// luma alone, bits 1 and 9 for chroma subsampled horizontally and vertically, bits 4 to 7 the
// search's bit depth less 8, bit 8 for float samples; bit 10 for luma's SATD, which only the
// full-size grid's kernels take (satd)
int SearchVariant(const SuperLayout &L, bool chroma, bool satd = false) {
    return (chroma ? 0 : 1) | (L.format.xr > 1 ? 2 : 0) | (L.format.yr > 1 ? 512 : 0) | ((SearchBits(L) - 8) << 4) | (L.format.Kind() == 2 ? 256 : 0) |
           (satd ? 1024 : 0);
}

// Lanes per candidate in the kernels that measure 8 candidates per block, init, the passes and the
// half- and quarter-pel steps (refine_common.glsl's kSplit, log2 of it their variant's bits 2 and
// 3): 2, and 4 blocks per workgroup instead of 8, for blocks whose current pixels take kSplitBytes
// or more; 4, and 2 blocks, for those taking kSplit4Bytes or more (32x32 blocks with 4:4:4 chroma, or
// of 16-bit samples). Measured (RX 6900 XT, radius 3, serialized profiles), the fields' GPU time
// against 8 blocks per workgroup: 4:4:4 32x32 -31..-32%, 4:2:0 32x32 -14..-21%, luma-only 32x32
// -3..-7%, 4:4:4 16x16 -2..-4%, 16-bit 4:4:4 16x16 -17%, 16-bit 4:2:0 16x16 +-3%; split 4 against 2:
// 4:4:4 32x32 -12..-13%, 16-bit 4:2:0 32x32 -18..-21%, 16-bit 4:4:4 32x32 -35..-36%, but blocks of
// 1.5 KB (4:2:0 32x32, 16-bit 4:4:4 16x16) +6..7%.
constexpr int kSplitBytes = 768, kSplit4Bytes = 3072;
int RefineSplit(const SuperLayout &L, bool chroma) {
    const int bytes = BlockPixels(L, chroma) * (L.format.Kind() == 0 ? 1 : 2); // floats are cached as 16-bit samples
    return bytes >= kSplit4Bytes ? 4 : bytes >= kSplitBytes ? 2 : 1;
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

// mvu.Analyse's scaling: mvlambda, lsad and badsad to the bit depth, by (2^bits - 1) / 255 rounded,
// then to the block size, lambda divided by pel squared at full size, exactly as the CPU reference
// computes it
struct Scaled {
    int64_t lambda0, lsad, lambdaBlock, badSad;
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

        std::vector<const VSFrame *> held;
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
            vsapi->freeFrame(out);
            vsapi->setFilterError(("Analyse: " + message).c_str(), frameCtx);
            return nullptr;
        };

        // The current frame's pyramid, then the references': each its super frame's region
        struct Pyramid {
            VkBuffer buffer;
            VkDeviceSize offset, size;
        };
        std::vector<Pyramid> pyramids;
        for (int f = -1; f < static_cast<int>(refs.size()); ++f) {
            const VSFrame *frame = vsapi->getFrameFilter(f < 0 ? n : refs[f], d->node, frameCtx);
            held.push_back(frame);
            SuperRegions regions;
            VSVulkanPlaneInfo info;
            if (const std::string e = CheckSuperFrame(frame, L, d->prefix, regions, vsapi); !e.empty())
                return fail(e);
            if (vkapi->getGPUPlane(frame, 0, &info))
                return fail("the super clip's frames aren't GPU resident");
            pyramids.push_back({info.buffer, static_cast<VkDeviceSize>(regions.pyramid), static_cast<VkDeviceSize>(regions.pyramidBytes)});
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
        for (const VSFrame *f : held)
            vkapi->gpuExecReadsFrame(ctx, f);
        vkapi->gpuExecWritesPlane(ctx, out, 0);

        const VkBuffer constants = d->constantsInfo.buffer, work = scratchInfo.buffer;
        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        Recorder rec(*d->vc, cmd, constants, Layout::Coarse);
        const VkQueryPool queries = d->profile ? d->profile->Begin(*d->vc, cmd) : VK_NULL_HANDLE;
        rec.Bind(kCurPyramid, pyramids[0].buffer, pyramids[0].offset, pyramids[0].size);
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
                rec.Bind(kRefPyramid, pyramids[1 + r].buffer, pyramids[1 + r].offset, pyramids[1 + r].size, static_cast<int>(z));
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
        release();
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
    SuperLayout layout; // the super's
    SuperLayout grid;   // the same with the grid analysed

    int deltaFrame = 1;
    bool chroma = true; // the SADs count chroma
    bool satd = false;  // luma's SAD is its SATD
    bool fields = false, tff = false, tffExists = false; // the frames are fields, of these parities (GetTopField)
    int split = 1;      // RefineSplit
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
        : d(d), L(d.grid), rec(rec), scratch(scratch), lumaStride(lumaStride), chromaStride(chromaStride), recStride(recStride), coarseBase(coarseBase),
          stamp(std::move(stamp)) {}

    // flags is the caller's to set where a kernel uses it
    Params MakeParams(int colour, int stamp) const {
        Params q = {};
        q.nbx = L.nbx;
        q.nby = L.nby;
        q.step = L.step;
        q.stepY = L.stepY;
        q.pad = L.pad;
        q.padY = L.padY;
        q.padc = L.padc;
        q.padcY = L.padcY;
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
        q.blockRows = L.blkY;
        q.recStride = recStride;
        q.coarseBase = coarseBase;
        q.aw = L.aw;
        q.ah = L.ah;
        return q;
    }

    // The full-size seed lists; flags: 1 the chained fields are bound, 2 the inverted field is;
    // fieldShift: the field shift of zero and the median
    void SeedLists(int flags, int fieldShift) {
        const uint32_t groups = static_cast<uint32_t>((static_cast<int64_t>(L.nbx) * L.nby + 63) / 64);
        Params q = MakeParams(0, 0);
        q.flags = flags;
        q.fieldShift = fieldShift;
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
        // a workgroup's blocks in a row, all of them or those of one colour
        const uint32_t blocks = static_cast<uint32_t>(8 / d.split);
        const uint32_t gxAll = (gx + blocks - 1) / blocks, gxHalf = ((gx + 1) / 2 + blocks - 1) / blocks;
        const bool withFallback = d.fallbackRadius > 0;
        const int totalPairs = kPairs + (withFallback ? 1 : 0);
        int step = 0; // init is step 0, every pass and the fallback one more
        rec.Dispatch(d.init, MakeParams(0, step), gxAll, gy);
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
            rec.Dispatch(d.pass, MakeParams(p & 1, ++step), gxHalf, gy);
            rec.ComputeBarrier();
        }
        Stamp(5);
        rec.Dispatch(d.halfpel, MakeParams(0, step), gxAll, gy);
        if (L.pel == 4) {
            rec.ComputeBarrier();
            Stamp(6);
            rec.Bind(kVecA, scratch, d.vecB.offset, d.vecB.size);
            rec.Bind(kSadA, scratch, d.sadB.offset, d.sadB.size);
            rec.Dispatch(d.quarter, MakeParams(0, step), gxAll, gy);
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
            if (d->coarseNode)
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
        const SuperLayout &L = d->layout, &G = d->grid; // the super's frames, the vectors' grid
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);

        if (!hasRef) {
            // No vectors: a frame of records that don't count, with the description. The super frame
            // is checked all the same, as mvu checks it, its parity too with fields.
            SuperRegions regions;
            std::string e = CheckSuperFrame(src, L, d->prefix, regions, vsapi);
            if (e.empty() && d->fields) {
                try {
                    (void)GetTopField(src, n, d->tffExists, d->tff, true, vsapi);
                } catch (const std::exception &x) {
                    e = x.what();
                }
            }
            if (!e.empty()) {
                vsapi->freeFrame(src);
                vsapi->setFilterError(("Analyse: " + e).c_str(), frameCtx);
                return nullptr;
            }
            VSFrame *dst = vkapi->newGPUVideoFrame(&d->gray32, 4 * G.nbx, G.nby, src, core);
            vsapi->freeFrame(src);
            if (!dst) {
                vsapi->setFilterError("Analyse: failed to allocate the vector frame", frameCtx);
                return nullptr;
            }
            ExportAnalysis(dst, G, d->deltaFrame, d->chroma, false, d->prefix, vsapi);
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
        const VSFrame *coarse = d->coarseNode ? hold(vsapi->getFrameFilter(n, d->coarseNode, frameCtx)) : nullptr;
        SuperRegions cur, rf;
        if (const std::string e = CheckSuperFrame(src, L, d->prefix, cur, vsapi); !e.empty())
            return fail(e);
        if (const std::string e = CheckSuperFrame(ref, L, d->prefix, rf, vsapi); !e.empty())
            return fail(e);

        // mvu.Analyse's field shift: with fields, at pel 2 or 4 and an odd delta, the shift between the
        // frames' parities (their _Field, or tff), which zero and the median take (seed_build.comp)
        int fieldShift = 0;
        if (d->fields) {
            try {
                const bool srcTop = GetTopField(src, n, d->tffExists, d->tff, true, vsapi), refTop = GetTopField(ref, nref, d->tffExists, d->tff, true, vsapi);
                if (L.pel > 1 && d->deltaFrame % 2 != 0)
                    fieldShift = ComputeFieldShift(srcTop, refTop, L.pel);
            } catch (const std::exception &e) {
                return fail(e.what());
            }
        }

        // The fields the seeds chain and invert, refined by the other nodes AnalyseMany made
        const VSFrame *stepVec = nullptr, *restVec = nullptr, *invVec = nullptr;
        if (d->stepNode && d->restNode) {
            stepVec = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(n, d->stepNode, frameCtx)), d->prefix, vsapi));
            restVec = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(n + d->unit, d->restNode, frameCtx)), d->prefix, vsapi));
        }
        if (d->invNode)
            invVec = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(nref, d->invNode, frameCtx)), d->prefix, vsapi));

        // The output: the records, with the super frame's properties
        vectors = vkapi->newGPUVideoFrame(&d->gray32, 4 * G.nbx, G.nby, src, core);
        if (!vectors)
            return fail("failed to allocate the vector frame");
        const ptrdiff_t recBytes = vsapi->getStride(vectors, 0);
        if (recBytes % 16)
            return fail("the vector frame's rows don't start on whole records");
        auto sameShape = [&](const VSFrame *f) {
            return f && vsapi->getFrameWidth(f, 0) == 4 * G.nbx && vsapi->getFrameHeight(f, 0) == G.nby && vsapi->getStride(f, 0) == recBytes;
        };
        const int flags = (sameShape(stepVec) && sameShape(restVec) ? 1 : 0) | (sameShape(invVec) ? 2 : 0) | (coarse ? 0 : 4);

        // A Gray super has no chroma
        const bool chroma = L.format.chroma;
        const ptrdiff_t lumaStride = cur.lumaStride, chromaStride = cur.chromaStride;
        const ptrdiff_t bytes = L.format.Bytes();
        if (lumaStride != rf.lumaStride || chromaStride != rf.chromaStride || lumaStride % (4 * bytes) || chromaStride % (4 * bytes))
            return fail("the super frames' storage strides differ");
        const ptrdiff_t coarseRowBytes = coarse ? vsapi->getStride(coarse, 0) : 0;
        if (coarseRowBytes % 8)
            return fail("the coarse search's rows don't start on whole vectors");

        // The buffers every binding names
        VSVulkanPlaneInfo curPlane, refPlane, coarsePlane = {}, outRec, stepRec = {}, restRec = {}, invRec = {};
        if (vkapi->getGPUPlane(src, 0, &curPlane) || vkapi->getGPUPlane(ref, 0, &refPlane) || (coarse && vkapi->getGPUPlane(coarse, 0, &coarsePlane)) ||
            vkapi->getGPUPlane(vectors, 0, &outRec) ||
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
        for (const VSFrame *f : {src, ref, coarse})
            if (f)
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
        rec.Bind(kCurLuma, curPlane.buffer, cur.luma, cur.lumaBytes);
        rec.Bind(kRefLuma, refPlane.buffer, rf.luma, rf.lumaBytes);
        if (chroma) { // never read without
            rec.Bind(kCurChroma, curPlane.buffer, cur.chroma, cur.chromaBytes);
            rec.Bind(kRefChroma, refPlane.buffer, rf.chroma, rf.chromaBytes);
        }
        if (coarse) // else seed_build.comp takes no coarse seeds
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
        FieldRecorder field(*d, rec, work, static_cast<int>(lumaStride / bytes), static_cast<int>(chromaStride / bytes), static_cast<int>(recBytes / 16),
                            static_cast<int>(d->coarseRow * coarseRowBytes / 8), std::move(stamp));
        field.SeedLists(flags, fieldShift);
        field.Refinement();

        uint64_t signaled = 0;
        const int submitted = vkapi->gpuExecSubmit(ctx, &signaled, err, sizeof(err));
        ctx = nullptr;
        scratch = nullptr; // the context owned it
        if (submitted)
            return fail(err);
        if (d->profile)
            d->profile->Finish(*d->vc, d->pool, signaled, queries);

        ExportAnalysis(vectors, G, d->deltaFrame, d->chroma, true, d->prefix, vsapi);
        VSFrame *dst = vectors;
        vectors = nullptr; // returned, not released
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
    SuperLayout grid;       // the same with the grid analysed, the super's or another (WithGrid)
    std::string prefix;
    int deltaFrame = 1;
    bool chroma = true;
    bool satd = false;
    bool fields = false, tff = false, tffExists = false;
    int plevel = 1;
    int64_t mvlambda = 1000, lsad = 400, badsad = 1000;
    int badrange = 40, badstep = 2;

    Scaled Scale() const {
        const int64_t area = static_cast<int64_t>(grid.blk) * grid.blkY;
        const int pixelMax = (1 << SearchBits(layout)) - 1;
        auto toDepth = [&](int64_t v) { return static_cast<int64_t>(static_cast<double>(v) * pixelMax / 255.0 + 0.5); };
        const int64_t mvl = toDepth(mvlambda), ls = toDepth(lsad), bs = toDepth(badsad);
        return {mvl * area / 64 / (layout.pel * layout.pel), ls * area / 64, mvl * area / 64, std::min<int64_t>(bs * area / 64, INT32_MAX)};
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
        // median.comp's histogram (a super without coarse levels has none to search: onelevel=True, or
        // a frame narrower than 2 * kTopWidth pixels)
        if (a.layout.maxCoarseVector >= 65024)
            throw std::runtime_error("the frame is too large for the coarse search");

        int blkX, blkY, overlapX, overlapY;
        GetPairArgument(blkX, blkY, "blksize", a.layout.blk, a.layout.blkY, in, vsapi);
        GetPairArgument(overlapX, overlapY, "overlap", a.layout.overlap, a.layout.overlapY, in, vsapi);

        a.satd = !!vsapi->mapGetInt(in, "satd", 0, &err);

        CheckBlockSize(blkX, blkY, overlapX, overlapY, a.layout.format.xr > 1 ? 1 : 0, a.layout.format.yr > 1 ? 1 : 0, a.satd);
        a.grid = a.layout.WithGrid(blkX, blkY, overlapX, overlapY);

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
        if (!a.layout.format.chroma) // a Gray super's SADs are luma's alone, as in mvu
            a.chroma = false;
        // The kernels keep a block's SAD in an int
        if (SadPixels(a.grid, a.chroma, a.satd) * ((1 << SearchBits(a.grid)) - 1) > INT32_MAX)
            throw std::runtime_error("the blocks' SADs can pass 2^31 (128x128 blocks with chroma at 16 bits or float), which isn't implemented");

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

        a.fields = !!vsapi->mapGetInt(in, "fields", 0, &err);

        a.tff = !!vsapi->mapGetInt(in, "tff", 0, &err);
        a.tffExists = !err;

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
    const int variant = SearchVariant(L, a.chroma);
    d->pyrTop = vc.Pipeline(Kernel::PyrTop, L.blk, 2, variant);
    d->pyrPass = vc.Pipeline(Kernel::PyrPass, L.blk, 2, variant);
    d->pyrSeed = vc.Pipeline(Kernel::PyrSeed, L.blk, 2, variant);
    d->median = vc.Pipeline(Kernel::Median, L.blk, 2);

    // lambda for every (worst neighbour SAD) >> (1 + bits - 8) per coarse level, mvlambda *
    // 2^(plevel * level) relaxed by lsad as the CPU reference does in double precision
    const int depthShift = SearchBits(L) - 8;
    std::vector<int64_t> levelLambda(static_cast<size_t>(L.topLevel + 1) * SuperLayout::kLambdaEntries);
    for (int level = 0; level <= L.topLevel; ++level)
        for (int i = 0; i < SuperLayout::kLambdaEntries; ++i) {
            const double sc = static_cast<double>(s.lsad) / std::max<int64_t>(s.lsad + (static_cast<int64_t>(i) << depthShift), 1);
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
    d->coarseNode = coarseNode ? vsapi->addNodeRef(coarseNode) : nullptr;
    d->coarseRow = coarseRow;
    d->vi = *vsapi->getVideoInfo(a.node);
    d->layout = a.layout;
    d->grid = a.grid;
    d->prefix = a.prefix;
    d->deltaFrame = delta;
    d->chroma = a.chroma;
    d->satd = a.satd;
    d->fields = a.fields;
    d->tff = a.tff;
    d->tffExists = a.tffExists;
    if (stepNode && restNode) {
        d->stepNode = vsapi->addNodeRef(stepNode);
        d->restNode = vsapi->addNodeRef(restNode);
        d->unit = unit;
    }
    if (invNode)
        d->invNode = vsapi->addNodeRef(invNode);

    // The kernels work on the grid analysed, in the super's planes, which the grid copies
    const SuperLayout &L = d->grid;
    const int blk = L.blk, blkY = L.blkY;
    const Scaled s = a.Scale();
    d->badSad = static_cast<int>(s.badSad);
    d->fallbackRadius = std::abs(a.badrange);
    d->fallbackStep = a.badstep;

    d->vc = VulkanContext::Get(core, vsapi);
    VulkanContext &vc = *d->vc;
    d->split = RefineSplit(L, a.chroma);
    const int pel = L.pel, variant = SearchVariant(L, a.chroma, a.satd), split = variant | (d->split == 4 ? 8 : d->split == 2 ? 4 : 0);
    d->seedScatter = vc.Pipeline(Kernel::SeedScatter, blk, pel, 0, blkY);
    d->seedBuild = vc.Pipeline(Kernel::SeedBuild, blk, pel, 0, blkY);
    d->init = vc.Pipeline(Kernel::RefineInit, blk, pel, split, blkY);
    d->pass = vc.Pipeline(Kernel::RefinePass, blk, pel, split, blkY);
    d->flag = vc.Pipeline(Kernel::RefineFlag, blk, pel, 0, blkY);
    d->fallback = vc.Pipeline(Kernel::RefineFallback, blk, pel, variant, blkY);
    d->apply = vc.Pipeline(Kernel::RefineApply, blk, pel, 0, blkY);
    d->halfpel = vc.Pipeline(Kernel::RefineHalfpel, blk, pel, split, blkY);
    if (pel == 4)
        d->quarter = vc.Pipeline(Kernel::RefineQuarter, blk, pel, split, blkY);

    // lambda for every (worst neighbour SAD) >> (1 + bits - 8 + AreaShift), relaxed by lsad as the CPU
    // reference does in double precision
    std::vector<int64_t> lambda(LambdaEntries(L, a.chroma, a.satd));
    for (size_t i = 0; i < lambda.size(); ++i) {
        const double scale = static_cast<double>(s.lsad) / std::max<int64_t>(s.lsad + (static_cast<int64_t>(i) << (SearchBits(L) - 8 + AreaShift(L))), 1);
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
    // The field's clip: its vectors' records
    d->vi.format = d->gray32;
    d->vi.width = 4 * L.nbx;
    d->vi.height = L.nby;

    if (StageProfiler::Requested())
        d->profile = std::make_unique<StageProfiler>("field " + std::to_string(delta) + " (pel " + std::to_string(pel) + ")",
                                                     std::vector<std::string>{"seeds", "init", "passes before the fallback", "fallback", "passes after it",
                                                                              "half-pel step", "quarter-pel step"});

    VSFilterDependency deps[5] = {{d->node, rpGeneral}};
    int numDeps = 1;
    for (VSNode *dep : {d->coarseNode, d->stepNode, d->restNode, d->invNode})
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
            if (a.layout.topLevel >= 1)
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
            if (a.layout.topLevel >= 1)
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
