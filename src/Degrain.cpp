#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <VSVulkan4.h>

#include "Common.h"
#include "SuperLayout.h"
#include "VulkanContext.h"

// mvu.Degrain on the GPU: each block of the frame averaged with its motion compensated blocks in
// radius frames either side, weighted by how well they match, the blocks blended through their
// overlap windows. Three kernels per frame: the scene change test's count of badly matched blocks
// per reference (degrain_count.comp), every block's weights (degrain_weights.comp), then every pixel
// of every processed plane from its blocks (degrain.comp). The result is mvu.Degrain's bit for bit
// given the same super and vectors, the weights' double precision arithmetic reproduced exactly in
// integers (degrain_weights.comp).
//
// The bindings come as one descriptor set per frame, allocated here and freed when the frame's
// submission completes: 2 * radius references, each with its super's two planes and its vectors,
// don't fit a push descriptor set.
//
// Implemented: all of mvu.Degrain's arguments, on mvgpu.Super's 8-bit 4:2:0 supers at pel 2 and 4
// and the vectors mvgpu.Analyse makes from them.

namespace {

constexpr int kMaxRadius = kMaxDegrainRefs / 2;

// mvu's OverlapWindows::Init: the 9 windows of nx x ny blocks overlapping by ox, oy (top left, top
// middle, top right, middle left, ..., bottom right, nx * ny values each, 0 .. 2048), in the same
// float arithmetic
std::vector<int32_t> MakeOverlapWindows(int nx, int ny, int ox, int oy) {
    std::vector<float> fWin1UVx(nx), fWin1UVxfirst(nx), fWin1UVxlast(nx);
    for (int i = 0; i < ox; i++) {
        fWin1UVx[i] = cosf(std::numbers::pi_v<float> * (i - ox + 0.5f) / (ox * 2));
        fWin1UVx[i] = fWin1UVx[i] * fWin1UVx[i];
        fWin1UVxfirst[i] = 1;
        fWin1UVxlast[i] = fWin1UVx[i];
    }
    for (int i = ox; i < nx - ox; i++) {
        fWin1UVx[i] = 1;
        fWin1UVxfirst[i] = 1;
        fWin1UVxlast[i] = 1;
    }
    for (int i = nx - ox; i < nx; i++) {
        fWin1UVx[i] = cosf(std::numbers::pi_v<float> * (i - nx + ox + 0.5f) / (ox * 2));
        fWin1UVx[i] = fWin1UVx[i] * fWin1UVx[i];
        fWin1UVxfirst[i] = fWin1UVx[i];
        fWin1UVxlast[i] = 1;
    }
    std::vector<float> fWin1UVy(ny), fWin1UVyfirst(ny), fWin1UVylast(ny);
    for (int i = 0; i < oy; i++) {
        fWin1UVy[i] = cosf(std::numbers::pi_v<float> * (i - oy + 0.5f) / (oy * 2));
        fWin1UVy[i] = fWin1UVy[i] * fWin1UVy[i];
        fWin1UVyfirst[i] = 1;
        fWin1UVylast[i] = fWin1UVy[i];
    }
    for (int i = oy; i < ny - oy; i++) {
        fWin1UVy[i] = 1;
        fWin1UVyfirst[i] = 1;
        fWin1UVylast[i] = 1;
    }
    for (int i = ny - oy; i < ny; i++) {
        fWin1UVy[i] = cosf(std::numbers::pi_v<float> * (i - ny + oy + 0.5f) / (oy * 2));
        fWin1UVy[i] = fWin1UVy[i] * fWin1UVy[i];
        fWin1UVyfirst[i] = fWin1UVy[i];
        fWin1UVylast[i] = 1;
    }
    const std::vector<float> *ys[3] = {&fWin1UVyfirst, &fWin1UVy, &fWin1UVylast}, *xs[3] = {&fWin1UVxfirst, &fWin1UVx, &fWin1UVxlast};
    std::vector<int32_t> windows(static_cast<size_t>(9) * nx * ny);
    for (int w = 0; w < 9; ++w)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                windows[(static_cast<size_t>(w) * ny + j) * nx + i] = static_cast<int16_t>((int)((*ys[w / 3])[j] * (*xs[w % 3])[i] * 2048 + 0.5f));
    return windows;
}

// mvu's interpolateThSAD: thsad at distance 1, thsad2 at distance tr, a raised cosine between
int64_t InterpolateThSAD(int64_t thsad, int64_t thsad2, int d, int tr) {
    if (d <= 1 || tr <= 1)
        return thsad;
    constexpr double kPi = 3.14159265358979323846;
    const double x = (d - 1) * kPi / (tr - 1);
    const double lerp = (1.0 - std::cos(x)) * 0.5;
    return static_cast<int64_t>(std::floor(thsad + lerp * static_cast<double>(thsad2 - thsad) + 0.5));
}

// A vector clip's description, as mvu.Degrain reads it from the first frame
struct VectorInfo {
    int width = 0, height = 0, realWidth = 0, realHeight = 0, hpad = 0, vpad = 0, pel = 0;
    int blkX = 0, blkY = 0, overlapX = 0, overlapY = 0, nbx = 0, nby = 0;
    int delta = 0, bits = 0, chroma = 0, xRatio = 0, yRatio = 0;
};

VectorInfo ReadVectorInfo(VSNode *node, const std::string &prefix, const VSAPI *vsapi) {
    char err[1024] = {};
    const VSFrame *frame = vsapi->getFrame(0, node, err, sizeof(err));
    if (!frame)
        throw std::runtime_error(std::string("failed to get a vector clip's first frame: ") + err);
    const VSMap *props = vsapi->getFramePropertiesRO(frame);
    auto get = [&](const char *name) {
        int e = 0;
        const int v = vsapi->mapGetIntSaturated(props, (prefix + name).c_str(), 0, &e);
        if (e) {
            vsapi->freeFrame(frame);
            throw std::runtime_error(std::string("a vector clip lacks the property ") + prefix + name + "; it must come from mvgpu.Analyse with the same prefix");
        }
        return v;
    };
    VectorInfo v;
    v.width = get("AnalysisWidth");
    v.height = get("AnalysisHeight");
    v.realWidth = get("AnalysisRealWidth");
    v.realHeight = get("AnalysisRealHeight");
    v.hpad = get("AnalysisHPad");
    v.vpad = get("AnalysisVPad");
    v.pel = get("AnalysisPel");
    v.blkX = get("AnalysisBlkSizeX");
    v.blkY = get("AnalysisBlkSizeY");
    v.overlapX = get("AnalysisOverlapX");
    v.overlapY = get("AnalysisOverlapY");
    v.nbx = get("AnalysisNBlkX");
    v.nby = get("AnalysisNBlkY");
    v.delta = get("AnalysisDeltaFrame");
    v.bits = get("AnalysisBitsPerSample");
    v.chroma = get("AnalysisChroma");
    v.xRatio = get("AnalysisXRatioUV");
    v.yRatio = get("AnalysisYRatioUV");
    vsapi->freeFrame(frame);
    return v;
}

// The same level-0 storage, whatever the levels above it and the grid: the planes the kernels read
bool SameStorage(const SuperLayout &a, const SuperLayout &b) {
    return a.width == b.width && a.height == b.height && a.format == b.format && a.aw == b.aw && a.ah == b.ah && a.pad == b.pad && a.padY == b.padY &&
           a.pel == b.pel;
}

// mvu's FramePyramid::IsCompatible: the same storage, padded for the same block size
bool SameLevel0(const SuperLayout &a, const SuperLayout &b) {
    return SameStorage(a, b) && a.blk == b.blk && a.blkY == b.blkY;
}

// mvu's argument helper for float lists: absent -> the defaults, one value -> v = h
void GetFloatPair(float &h, float &v, const char *name, float defaultH, float defaultV, const VSMap *in, const VSAPI *vsapi) {
    int err = 0;
    const int numElems = vsapi->mapNumElements(in, name);
    if (numElems > 2)
        throw std::runtime_error(std::string("Too many values passed to ") + name);
    h = vsapi->mapGetFloatSaturated(in, name, 0, &err);
    if (err)
        h = defaultH;
    v = vsapi->mapGetFloatSaturated(in, name, 1, &err);
    if (err)
        v = (numElems == 1) ? h : defaultV;
}

// Descriptor sets of the Degrain layout, one per submission, from pools added as they fill up; a
// set goes back to its pool when its submission completes (ReleaseSet)
class DescriptorSets {
public:
    explicit DescriptorSets(const VulkanContext &vc) : vc(vc) {
        for (const auto &[binding, count] : vc.LayoutBindings(Layout::Degrain))
            perSet += count;
    }
    ~DescriptorSets() {
        for (VkDescriptorPool p : pools)
            vc.vk->vkDestroyDescriptorPool(vc.device, p, nullptr);
    }
    DescriptorSets(const DescriptorSets &) = delete;
    DescriptorSets &operator=(const DescriptorSets &) = delete;

    bool Allocate(VkDescriptorPool &pool, VkDescriptorSet &set) {
        std::lock_guard<std::mutex> guard(lock);
        const VkDescriptorSetLayout layout = vc.SetLayout(Layout::Degrain);
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        for (VkDescriptorPool p : pools) {
            ai.descriptorPool = p;
            if (vc.vk->vkAllocateDescriptorSets(vc.device, &ai, &set) == VK_SUCCESS) {
                pool = p;
                return true;
            }
        }
        const VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSetsPerPool * perSet};
        VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        ci.maxSets = kSetsPerPool;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &size;
        VkDescriptorPool p = VK_NULL_HANDLE;
        if (vc.vk->vkCreateDescriptorPool(vc.device, &ci, nullptr, &p) != VK_SUCCESS)
            return false;
        pools.push_back(p);
        ai.descriptorPool = p;
        if (vc.vk->vkAllocateDescriptorSets(vc.device, &ai, &set) != VK_SUCCESS)
            return false;
        pool = p;
        return true;
    }

    void Free(VkDescriptorPool pool, VkDescriptorSet set) {
        std::lock_guard<std::mutex> guard(lock);
        vc.vk->vkFreeDescriptorSets(vc.device, pool, 1, &set);
    }

private:
    static constexpr uint32_t kSetsPerPool = 16; // an exec pool has at most 8 submissions in flight
    const VulkanContext &vc;
    uint32_t perSet = 0;
    std::mutex lock;
    std::vector<VkDescriptorPool> pools;
};

struct SetRelease {
    DescriptorSets *owner;
    VkDescriptorPool pool;
    VkDescriptorSet set;
};

void VS_CC ReleaseSet(void *object) {
    std::unique_ptr<SetRelease> r(static_cast<SetRelease *>(object));
    r->owner->Free(r->pool, r->set);
}

} // namespace

struct DegrainData {
    VSNode *node = nullptr;        // the clip
    VSNode *super = nullptr;       // the references, and the centre unless centerSuper is given
    VSNode *centerSuper = nullptr; // optional
    std::vector<VSNode *> vectors; // mvu order: delta 1, -1, 2, -2, ...
    std::vector<int> deltas;
    VSVideoInfo vi = {};
    SuperLayout layout;       // the super's: the planes the references come from
    SuperLayout centreLayout; // centerSuper's, which may lack the levels above 0
    std::string prefix, name;
    int radius = 0;
    // The vectors' grid, which mvu.Degrain takes from them rather than from the super: blk x blk
    // blocks, step apart, overlapping by overlap, nbx x nby of them
    int blk = 0, overlap = 0, step = 0, nbx = 0, nby = 0;

    bool process[3] = {};
    int limit[3] = {-1, -1, -1}; // per plane, -1 for none
    int thscd1 = 0;              // scaled
    int scdLimit = 0;            // a reference with more blocks above thscd1 is at a scene change
    int winOff[2] = {};          // tables: luma's windows, chroma's
    int thOff = 0, uwOff = 0;    // tables: each reference's thsad (luma's, then chroma's), the user weights

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline count = VK_NULL_HANDLE, weights = VK_NULL_HANDLE, pixels = VK_NULL_HANDLE;
    VSGPUBuffer *constants = nullptr; // the tables
    VSVulkanBufferInfo constantsInfo = {};
    std::unique_ptr<DescriptorSets> sets;
    std::unique_ptr<StageProfiler> profile; // MVGPU_PROFILE

    const VSAPI *vsapi;

    DegrainData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~DegrainData() {
        // The pool drains the GPU and runs every release callback first, so no set is in use
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        sets.reset();
        profile.reset();
        if (constants)
            vc->vkapi->destroyGPUBuffer(constants);
        vc.reset();
        for (VSNode *v : vectors)
            vsapi->freeNode(v);
        vsapi->freeNode(centerSuper);
        vsapi->freeNode(super);
        vsapi->freeNode(node);
    }
};

static const VSFrame *VS_CC degrainGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    DegrainData *d = reinterpret_cast<DegrainData *>(instanceData);
    const int refs = 2 * d->radius;

    if (activationReason == arInitial) {
        for (int r = 0; r < refs; ++r) {
            vsapi->requestFrameFilter(n, d->vectors[r], frameCtx);
            const int nref = n + d->deltas[r];
            if (nref >= 0 && nref < d->vi.numFrames)
                vsapi->requestFrameFilter(nref, d->super, frameCtx);
        }
        vsapi->requestFrameFilter(n, d->centerSuper ? d->centerSuper : d->super, frameCtx);
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

        const VSFrame *src = hold(vsapi->getFrameFilter(n, d->node, frameCtx));
        SuperFrames cur;
        if (!GetSuperFrames(hold(vsapi->getFrameFilter(n, d->centerSuper ? d->centerSuper : d->super, frameCtx)), d->centreLayout, d->prefix, cur, vsapi))
            return fail("the super clip's frames lack their GPU planes; it must come from mvgpu.Super with the same prefix");
        for (const VSFrame *f : {cur.luma, cur.chroma, cur.pyramid})
            hold(f);

        // The references with vectors and in the clip; mvu.Degrain's isUsable, but for the scene
        // change test, which degrain_count.comp makes
        std::vector<const VSFrame *> refLuma(refs, nullptr), refChroma(refs, nullptr), refVec(refs, nullptr);
        uint64_t usable = 0;
        for (int r = 0; r < refs; ++r) {
            const VSFrame *analysis = hold(vsapi->getFrameFilter(n, d->vectors[r], frameCtx));
            const VSFrame *vec = hold(GetAnalysisVectors(analysis, d->prefix, vsapi));
            if (!vec)
                continue;
            int err = 0;
            const int delta = vsapi->mapGetIntSaturated(vsapi->getFramePropertiesRO(analysis), (d->prefix + "AnalysisDeltaFrame").c_str(), 0, &err);
            if (err || delta != d->deltas[r])
                return fail("vector clip " + std::to_string(r) + " reports delta " + std::to_string(delta) + " at frame " + std::to_string(n) +
                            " but was created with delta " + std::to_string(d->deltas[r]) + "; the delta must be constant for the whole clip");
            const int nref = n + d->deltas[r];
            if (nref < 0 || nref >= d->vi.numFrames)
                continue;
            SuperFrames rf;
            if (!GetSuperFrames(hold(vsapi->getFrameFilter(nref, d->super, frameCtx)), L, d->prefix, rf, vsapi))
                return fail("the super clip's frames lack their GPU planes; it must come from mvgpu.Super with the same prefix");
            for (const VSFrame *f : {rf.luma, rf.chroma, rf.pyramid})
                hold(f);
            refLuma[r] = rf.luma;
            refChroma[r] = rf.chroma;
            refVec[r] = vec;
            usable |= uint64_t(1) << r;
        }

        const ptrdiff_t lumaStride = vsapi->getStride(cur.luma, 0), chromaStride = vsapi->getStride(cur.chroma, 0);
        ptrdiff_t recBytes = 0;
        for (int r = 0; r < refs; ++r) {
            if (!refVec[r])
                continue;
            if (vsapi->getStride(refLuma[r], 0) != lumaStride || vsapi->getStride(refChroma[r], 0) != chromaStride)
                return fail("the super frames' storage strides differ");
            const ptrdiff_t stride = vsapi->getStride(refVec[r], 0);
            if (vsapi->getFrameWidth(refVec[r], 0) != 4 * d->nbx || vsapi->getFrameHeight(refVec[r], 0) != d->nby || stride % 16 || (recBytes && stride != recBytes))
                return fail("a vector frame doesn't match the super's grid");
            recBytes = stride;
        }

        // The output: the processed planes written here, the others the clip's
        const bool all = d->process[0] && d->process[1] && d->process[2];
        if (all) {
            dst = vkapi->newGPUVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
        } else {
            const VSFrame *planeSrc[3] = {d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src};
            const int planes[3] = {0, 1, 2};
            dst = vsapi->newVideoFrame2(&d->vi.format, d->vi.width, d->vi.height, planeSrc, planes, src, core);
        }
        if (!dst)
            return fail("failed to allocate the output frame; the clip must be GPU resident");

        VSVulkanPlaneInfo curLuma, curChroma, outPlanes[3] = {};
        if (vkapi->getGPUPlane(cur.luma, 0, &curLuma) || vkapi->getGPUPlane(cur.chroma, 0, &curChroma))
            return fail("the super's planes aren't GPU resident");
        for (int p = 0; p < 3; ++p)
            if (d->process[p] && vkapi->getGPUPlane(dst, p, &outPlanes[p]))
                return fail("the output frame isn't GPU resident; the clip must be");
        std::vector<VSVulkanPlaneInfo> rl(refs), rc(refs), rv(refs);
        for (int r = 0; r < refs; ++r)
            if (refVec[r] && (vkapi->getGPUPlane(refLuma[r], 0, &rl[r]) || vkapi->getGPUPlane(refChroma[r], 0, &rc[r]) || vkapi->getGPUPlane(refVec[r], 0, &rv[r])))
                return fail("a reference's planes or vectors aren't GPU resident");

        // Scratch: the counts, then the blocks' weights
        const int nb = d->nbx * d->nby;
        const VkDeviceSize align = std::max<VkDeviceSize>(vc.limits.minStorageBufferOffsetAlignment, 16);
        const VkDeviceSize countBytes = (std::max<VkDeviceSize>(refs * 4, 16) + align - 1) / align * align;
        const VkDeviceSize metaBytes = static_cast<VkDeviceSize>(refs + 1) * nb * 4;
        char err[1024] = {};
        VSVulkanBufferInfo scratchInfo = {};
        scratch = vkapi->createGPUBuffer(core, countBytes + metaBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &scratchInfo, err, sizeof(err));
        if (!scratch)
            return fail(err);
        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        vkapi->gpuExecUsesBuffer(ctx, scratch);
        vkapi->gpuExecReadsFrame(ctx, cur.luma);
        vkapi->gpuExecReadsFrame(ctx, cur.chroma);
        for (int r = 0; r < refs; ++r) {
            if (refVec[r]) {
                vkapi->gpuExecReadsFrame(ctx, refLuma[r]);
                vkapi->gpuExecReadsFrame(ctx, refChroma[r]);
                vkapi->gpuExecReadsFrame(ctx, refVec[r]);
            }
        }
        for (int p = 0; p < 3; ++p)
            if (d->process[p])
                vkapi->gpuExecWritesPlane(ctx, dst, p);

        VkDescriptorPool setPool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (!d->sets->Allocate(setPool, set))
            return fail("failed to allocate a descriptor set");
        vkapi->gpuExecRetain(ctx, ReleaseSet, new SetRelease{d->sets.get(), setPool, set}, 0);

        // The bindings, in degrain_common.glsl's order; what a frame doesn't use is a dummy
        const VkBuffer dummy = d->constantsInfo.buffer;
        std::vector<VkDescriptorBufferInfo> infos;
        infos.push_back({curLuma.buffer, 0, VK_WHOLE_SIZE});
        infos.push_back({curChroma.buffer, 0, VK_WHOLE_SIZE});
        infos.push_back({scratchInfo.buffer, countBytes, metaBytes});
        infos.push_back({d->constantsInfo.buffer, 0, VK_WHOLE_SIZE});
        infos.push_back({scratchInfo.buffer, 0, countBytes});
        for (int p = 0; p < 3; ++p)
            infos.push_back({d->process[p] ? outPlanes[p].buffer : dummy, 0, VK_WHOLE_SIZE});
        for (const std::vector<VSVulkanPlaneInfo> *list : {&rl, &rc, &rv})
            for (int r = 0; r < kMaxDegrainRefs; ++r)
                infos.push_back({r < refs && refVec[r] ? (*list)[r].buffer : dummy, 0, VK_WHOLE_SIZE});
        std::vector<VkWriteDescriptorSet> writes;
        size_t at = 0;
        for (const auto &[binding, count] : vc.LayoutBindings(Layout::Degrain)) {
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = set;
            w.dstBinding = binding;
            w.descriptorCount = count;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &infos[at];
            at += count;
            writes.push_back(w);
        }
        vc.vk->vkUpdateDescriptorSets(vc.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        const VkPipelineLayout layout = vc.PipelineLayout(Layout::Degrain);
        VkBindDescriptorSetsInfo bind = {VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO};
        bind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bind.layout = layout;
        bind.descriptorSetCount = 1;
        bind.pDescriptorSets = &set;
        vc.vk->vkCmdBindDescriptorSets2(cmd, &bind);
        const VkQueryPool queries = d->profile ? d->profile->Begin(vc, cmd) : VK_NULL_HANDLE;
        auto stamp = [&](int stage) {
            if (d->profile)
                d->profile->Stamp(vc, cmd, queries, stage);
        };
        auto barrier = [&](VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess) {
            VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            mb.srcStageMask = srcStage;
            mb.srcAccessMask = srcAccess;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo dep = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &mb;
            vc.vk->vkCmdPipelineBarrier2(cmd, &dep);
        };
        auto dispatch = [&](VkPipeline pipeline, const DegrainParams &pc, uint32_t x, uint32_t y) {
            vc.vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            VkPushConstantsInfo push = {VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO};
            push.layout = layout;
            push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            push.size = sizeof(DegrainParams);
            push.pValues = &pc;
            vc.vk->vkCmdPushConstants2(cmd, &push);
            vc.vk->vkCmdDispatch(cmd, x, y, 1);
        };

        DegrainParams pc = {};
        pc.nbx = d->nbx;
        pc.nby = d->nby;
        pc.step = d->step;
        pc.overlap = d->overlap;
        pc.pad = L.pad;
        pc.padc = L.padc;
        pc.wp = static_cast<int32_t>(lumaStride);
        pc.hp = L.hp;
        pc.wc = static_cast<int32_t>(chromaStride);
        pc.hc = L.hc;
        pc.recStride = static_cast<int32_t>(recBytes / 16);
        pc.refs = refs;
        pc.usable0 = static_cast<uint32_t>(usable);
        pc.usable1 = static_cast<uint32_t>(usable >> 32);
        pc.thscd1 = d->thscd1;
        pc.scdLimit = d->scdLimit;
        pc.thOff = d->thOff;
        pc.uwOff = d->uwOff;
        pc.nb = nb;

        // The scene change test's counts start at zero, and stay there when no count can exceed
        // the limit
        vc.vk->vkCmdFillBuffer(cmd, scratchInfo.buffer, 0, countBytes, 0);
        barrier(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        const uint32_t blockGroups = static_cast<uint32_t>((nb + 63) / 64);
        if (usable && d->scdLimit < nb) {
            dispatch(d->count, pc, blockGroups, static_cast<uint32_t>(refs));
            barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        stamp(1);
        dispatch(d->weights, pc, blockGroups, 1);
        barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        stamp(2);
        for (int p = 0; p < 3; ++p) {
            if (!d->process[p])
                continue;
            pc.plane = p;
            pc.width = vsapi->getFrameWidth(dst, p);
            pc.height = vsapi->getFrameHeight(dst, p);
            pc.outStride = static_cast<int32_t>(vsapi->getStride(dst, p));
            pc.limit = d->limit[p];
            pc.winOff = d->winOff[p ? 1 : 0];
            dispatch(d->pixels, pc, static_cast<uint32_t>((pc.width + 63) / 64), static_cast<uint32_t>(d->nby));
        }
        stamp(3);

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

// Degrain1 .. Degrain25 take their radius as userData, Degrain (null) takes it from the vectors
static void VS_CC degrainCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<DegrainData> d = std::make_unique<DegrainData>(vsapi);
    const int fixedRadius = static_cast<int>(reinterpret_cast<intptr_t>(userData));
    d->name = fixedRadius ? "Degrain" + std::to_string(fixedRadius) : "Degrain";

    try {
        int err;

        const int numVectors = vsapi->mapNumElements(in, "vectors");
        if (fixedRadius) {
            d->radius = fixedRadius;
            if (numVectors != 2 * d->radius)
                throw std::runtime_error("the number of vector clips must be exactly " + std::to_string(2 * d->radius));
        } else {
            if (numVectors % 2 != 0)
                throw std::runtime_error("number of vectors must be even");
            d->radius = numVectors / 2;
            if (d->radius < 1 || d->radius > kMaxRadius)
                throw std::runtime_error("number of vector pairs must be between 1 and " + std::to_string(kMaxRadius));
        }
        const int refs = 2 * d->radius;

        int64_t thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
        if (err)
            thscd1 = MV_DEFAULT_SCD1;
        float thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
        if (err)
            thscd2 = MV_DEFAULT_SCD2;

        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        d->prefix = prefix ? prefix : DEFAULT_MVGPUTENSILS_PREFIX;

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);
        d->centerSuper = vsapi->mapGetNode(in, "centersuper", 0, &err);
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);
        for (int r = 0; r < refs; ++r)
            d->vectors.push_back(vsapi->mapGetNode(in, "vectors", r, nullptr));

        // planes
        const int m = vsapi->mapNumElements(in, "planes");
        for (int i = 0; i < 3; i++)
            d->process[i] = (m <= 0);
        for (int i = 0; i < m; i++) {
            const int64_t o = vsapi->mapGetInt(in, "planes", i, nullptr);
            if (o < 0 || o >= 3)
                throw std::runtime_error("plane index out of range");
            if (d->process[o])
                throw std::runtime_error("plane specified twice");
            d->process[o] = true;
        }

        // weights, given in temporal order [bw_radius, ..., bw_1, centre, fw_1, ..., fw_radius],
        // kept as [centre, bw_1, fw_1, bw_2, fw_2, ...] as normaliseWeights takes them
        std::vector<int> userWeights(refs + 1, 1);
        const int weightCount = vsapi->mapNumElements(in, "weights");
        if (weightCount != -1) {
            if (weightCount != refs + 1)
                throw std::runtime_error("weights, if given, must have exactly " + std::to_string(refs + 1) + " elements");
            userWeights[0] = vsapi->mapGetIntSaturated(in, "weights", d->radius, nullptr);
            for (int r = 0; r < d->radius; r++) {
                userWeights[r * 2 + 1] = vsapi->mapGetIntSaturated(in, "weights", d->radius - (r + 1), nullptr);
                userWeights[r * 2 + 2] = vsapi->mapGetIntSaturated(in, "weights", d->radius + (r + 1), nullptr);
            }
            const int maxWeight = (std::numeric_limits<int>::max() - 1) / (256 * (refs + 1));
            for (int w : userWeights)
                if (w < 0 || w > maxWeight)
                    throw std::runtime_error("weights must be between 0 and " + std::to_string(maxWeight));
        }

        d->layout = ImportSuperLayout(d->super, d->prefix, vsapi);
        const SuperLayout &L = d->layout;
        if (const std::string unsupported = L.Unsupported(); !unsupported.empty())
            throw std::runtime_error(unsupported);

        // The vectors: mvgpu.Analyse's of a super with the same level 0 as this one (mvu's
        // IsCompatibleWithAnalysis), all on one grid (IsCompatible), deltas in mvu's order. The grid
        // is theirs, as in mvu.Degrain; the super only supplies the planes.
        VectorInfo first;
        for (int r = 0; r < refs; ++r) {
            const VectorInfo v = ReadVectorInfo(d->vectors[r], d->prefix, vsapi);
            if (r == 0)
                first = v;
            const SuperLayout analysed = ImportSuperLayout(d->vectors[r], d->prefix, vsapi);
            if (!SameStorage(analysed, L) || v.width != L.aw || v.height != L.ah || v.realWidth != L.width || v.realHeight != L.height || v.hpad != L.pad ||
                v.vpad != L.padY || v.pel != L.pel || v.bits != L.format.bits || (v.chroma && (v.xRatio != L.format.xr || v.yRatio != L.format.yr)))
                throw std::runtime_error("The motion vectors passed are not compatible with the super clip");
            if (v.blkX != first.blkX || v.blkY != first.blkY || v.overlapX != first.overlapX || v.overlapY != first.overlapY || v.nbx != first.nbx ||
                v.nby != first.nby || v.chroma != first.chroma)
                throw std::runtime_error("The motion vectors passed are not compatible with each other");
            if (v.blkX != analysed.blk || v.blkY != analysed.blk || v.overlapX != analysed.overlap || v.overlapY != analysed.overlap || v.nbx != analysed.nbx ||
                v.nby != analysed.nby)
                throw std::runtime_error("the vectors' grid isn't their super's; they must come from mvgpu.Analyse");
            d->deltas.push_back(v.delta);
            if (r % 2 == 1) {
                if (d->deltas[r] != -d->deltas[r - 1])
                    throw std::runtime_error("forward and backward vector clips must be symmetric in their delta frame");
                if (r >= 2 && std::abs(d->deltas[r - 2]) >= std::abs(d->deltas[r]))
                    throw std::runtime_error("vector clips must have increasing number of delta frames");
            }
        }
        d->blk = first.blkX;
        d->overlap = first.overlapX;
        d->step = d->blk - d->overlap;
        d->nbx = first.nbx;
        d->nby = first.nby;

        if (!vsh::isConstantVideoFormat(&d->vi) || d->vi.format.colorFamily != cfYUV || d->vi.format.sampleType != stInteger || d->vi.format.bitsPerSample != 8 ||
            d->vi.format.subSamplingW != 1 || d->vi.format.subSamplingH != 1 || d->vi.width != L.width || d->vi.height != L.height)
            throw std::runtime_error("super clip is not compatible with the source clip");

        d->centreLayout = d->centerSuper ? ImportSuperLayout(d->centerSuper, d->prefix, vsapi) : L;
        if (!SameLevel0(d->centreLayout, L))
            throw std::runtime_error("centersuper must be created with the same Super arguments as super");

        int thsadRaw[2], thsad2Raw[2];
        GetPairArgument(thsadRaw[0], thsadRaw[1], "thsad", 400, 400, in, vsapi);
        GetPairArgument(thsad2Raw[0], thsad2Raw[1], "thsad2", thsadRaw[0], thsadRaw[1], in, vsapi);

        // mvu's ScaleThSCD and GetThSCDScaleFactor, for the vectors' block size, chroma and 8 bits
        constexpr int maxSAD = 8 * 8 * 255;
        if (thscd1 < 0 || thscd1 > maxSAD)
            throw std::runtime_error("thscd1 must be between 0 and " + std::to_string(maxSAD));
        if (!std::isfinite(thscd2) || thscd2 < 0.0f || thscd2 > 100.0f)
            throw std::runtime_error("thscd2 must be a percentage between 0 and 100");
        const double scale = static_cast<double>(first.blkX * first.blkY) / (8.0 * 8.0) * (first.chroma ? (1.0 + 2.0 / (first.xRatio * first.yRatio)) : 1.0) *
                             (((1 << std::min(16, first.bits)) - 1) / 255.0);
        d->thscd1 = static_cast<int>(static_cast<int64_t>(thscd1 * scale + 0.5));
        const float blocksTh2 = static_cast<float>(static_cast<double>(thscd2) * first.nbx * first.nby / 100.0);
        d->scdLimit = static_cast<int>(std::floor(blocksTh2)); // a count above the float limit is above its floor

        int64_t thsadScaled[2], thsad2Scaled[2];
        for (int p = 0; p < 2; p++) {
            thsadScaled[p] = static_cast<int64_t>(thsadRaw[p] * scale + .5);
            thsad2Scaled[p] = static_cast<int64_t>(thsad2Raw[p] * scale + .5);
            if (thsadScaled[p] >= std::numeric_limits<int>::max() || thsad2Scaled[p] >= std::numeric_limits<int>::max()) {
                const int64_t maximum = static_cast<int64_t>((std::numeric_limits<int>::max() - 1) / scale);
                throw std::runtime_error("with this block size and video format, the " + std::string(p ? "thsad/thsad2 chroma values" : "thsad/thsad2 luma values") +
                                         " must not exceed " + std::to_string(maximum) + " or some calculations would overflow");
            }
        }

        float fLimit[3];
        GetFloatPair(fLimit[0], fLimit[1], "limit", std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), in, vsapi);
        fLimit[2] = fLimit[1];
        for (int i = 0; i < 3; i++) {
            if (!std::isfinite(fLimit[i]))
                continue;
            if (fLimit[i] <= 0.0f)
                throw std::runtime_error("limit must be non-negative");
            if (fLimit[i] < 255.0f)
                d->limit[i] = static_cast<int>(fLimit[i] + 0.5f);
        }

        // The tables: the overlap windows of luma's and chroma's blocks, each reference's thsad,
        // luma's then chroma's, and the user weights
        std::vector<int32_t> tables;
        if (d->overlap > 0) {
            const std::vector<int32_t> lumaWin = MakeOverlapWindows(d->blk, d->blk, d->overlap, d->overlap);
            const std::vector<int32_t> chromaWin = MakeOverlapWindows(d->blk / 2, d->blk / 2, d->overlap / 2, d->overlap / 2);
            d->winOff[0] = static_cast<int>(tables.size());
            tables.insert(tables.end(), lumaWin.begin(), lumaWin.end());
            d->winOff[1] = static_cast<int>(tables.size());
            tables.insert(tables.end(), chromaWin.begin(), chromaWin.end());
        }
        d->thOff = static_cast<int>(tables.size());
        for (int p = 0; p < 2; ++p)
            for (int r = 0; r < refs; ++r)
                tables.push_back(static_cast<int32_t>(InterpolateThSAD(thsadScaled[p], thsad2Scaled[p], r / 2 + 1, d->radius)));
        d->uwOff = static_cast<int>(tables.size());
        tables.insert(tables.end(), userWeights.begin(), userWeights.end());

        d->vc = VulkanContext::Get(core, vsapi);
        VulkanContext &vc = *d->vc;
        vc.RequireDegrain();
        d->count = vc.Pipeline(Kernel::DegrainCount, d->blk, L.pel);
        d->weights = vc.Pipeline(Kernel::DegrainWeights, d->blk, L.pel);
        d->pixels = vc.Pipeline(Kernel::DegrainPixels, d->blk, L.pel);
        const VkDeviceSize metaBytes = static_cast<VkDeviceSize>(refs + 1) * d->nbx * d->nby * 4;
        if (metaBytes > vc.limits.maxStorageBufferRange)
            throw std::runtime_error("the frame is too large for the device's storage buffers");

        char errMsg[1024] = {};
        d->pool = vc.vkapi->createGPUExecPool(core, vqCompute, errMsg, sizeof(errMsg));
        if (!d->pool)
            throw std::runtime_error(errMsg);
        d->constants = vc.Upload(core, d->pool, tables.data(), tables.size() * sizeof(int32_t), d->constantsInfo);
        d->sets = std::make_unique<DescriptorSets>(vc);
        if (StageProfiler::Requested())
            d->profile = std::make_unique<StageProfiler>(d->name + " (pel " + std::to_string(L.pel) + ")",
                                                         std::vector<std::string>{"scene change counts", "weights", "pixels"});
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, (d->name + ": " + e.what()).c_str());
        return;
    }

    std::vector<VSFilterDependency> deps;
    deps.push_back({d->node, rpStrictSpatial});
    deps.push_back({d->super, rpGeneral});
    if (d->centerSuper)
        deps.push_back({d->centerSuper, rpStrictSpatial});
    for (VSNode *v : d->vectors)
        deps.push_back({v, rpStrictSpatial});

    vsapi->createVideoFilterEx(out, d->name.c_str(), &d->vi, degrainGetFrame, filterFree<DegrainData>, fmParallel, ffGPUOutput, deps.data(),
                               static_cast<int>(deps.size()), d.get(), core);
    d.release();
}

void degrainRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    static constexpr const char *args =
        "clip:vnode:gpu;"
        "super:vnode:gpu;"
        "vectors:vnode[]:gpu;"
        "thsad:int[]:opt;"
        "thsad2:int[]:opt;"
        "planes:int[]:opt;"
        "limit:float[]:opt;"
        "thscd1:int:opt;"
        "thscd2:float:opt;"
        "weights:int[]:opt;"
        "centersuper:vnode:gpu:opt;"
        "prefix:data:opt;";
    for (int r = 1; r <= kMaxRadius; ++r)
        vspapi->registerFunction(("Degrain" + std::to_string(r)).c_str(), args, "clip:vnode:gpu;", degrainCreate, reinterpret_cast<void *>(static_cast<intptr_t>(r)), plugin);
    vspapi->registerFunction("Degrain", args, "clip:vnode:gpu;", degrainCreate, nullptr, plugin);
}
