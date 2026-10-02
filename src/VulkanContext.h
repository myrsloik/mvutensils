#pragma once

// The Vulkan side every mvgpu filter shares: the core's device and dispatch table, the
// descriptor layouts the kernels use (storage buffers pushed with every dispatch, or for Degrain a
// descriptor set per frame, plus the push constants), the kernels' pipelines, compiled once per
// core and variant, and a recorder for the dispatches of one submission. Filters keep a shared_ptr to it, so the pipelines live until the
// last instance on the core is freed.

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <VapourSynth4.h>
#include <VSVulkan4.h>

// The kernels in src/shaders
enum class Kernel {
    Super,
    SuperQuarter,
    PyrReduce,
    PyrTop,
    PyrPass,
    PyrSeed,
    Median,
    SeedScatter,
    SeedBuild,
    RefineInit,
    RefinePass,
    RefineFlag,
    RefineFallback,
    RefineApply,
    RefineHalfpel,
    RefineQuarter,
    DegrainCount,
    DegrainWeights,
    DegrainPixels,
};

// The bindings, as the kernels declare them (refine_common.glsl, pyr_common.glsl, super.comp,
// refine_halfpel.comp)
enum Binding {
    kCurLuma, kRefLuma, kCurChroma, kRefChroma,
    kSeeds, kSeedCount, kSeedSad,
    kVecA, kSadA, kVecB, kSadB,
    kLambda, kLastChange, kLastEval, kCounters, kFlagged,
    kCurPyramid, kRefPyramid, kLevelVec, kLevelSad, kLevels, kLevelLambda, kMedians,
    kStepRec, kRestRec, kInvRec, kOutRec, kInvKey, kCoarse,
    kRawY, kRawU, kRawV,
    kBindings,
    // Super's kernels (super_common.glsl) take the pelclip's planes where the search has its seeds
    kPelY = kSeeds, kPelU = kSeedCount, kPelV = kSeedSad,
};

// The descriptor layouts. Main: bindings 0 .. 31, one buffer each, for every kernel but the
// coarse search's and Degrain's. Coarse: the coarse search's bindings 16 .. 22 (pyr_common.glsl),
// binding 17 an array of the batch's reference pyramids, kMaxBatch of them. Both together would
// exceed the 32 descriptors a push descriptor set is guaranteed, so they are two. Degrain: a
// descriptor set allocated per frame rather than pushed (DegrainBinding), since every reference
// brings three buffers.
enum class Layout {
    Main,
    Coarse,
    Degrain,
};

// The Degrain layout's bindings (degrain_common.glsl): the centre's super planes, the blocks'
// weights, the tables, the scene change counts, the output planes (3), and each reference's super
// planes and vectors, kMaxDegrainRefs of each
enum DegrainBinding {
    kDgCurLuma, kDgCurChroma, kDgMeta, kDgTables, kDgCounts, kDgOut, kDgRefLuma, kDgRefChroma, kDgRefVec,
    kDgBindings
};
constexpr int kMaxDegrainRefs = 50; // 2 * the largest radius, degrain_common.glsl's kMaxRefs

// Push constants, matching refine_common.glsl's Params
struct Params {
    int32_t w, h, nbx, nby;
    int32_t step, pad, padc, colour;
    int32_t wp, hp, wc, hc;
    int32_t fallbackRadius, fallbackStep, badSad, maxSeeds;
    int32_t stamp, level, flags, topRadius;
    int32_t medianScale, medianSlot, finest, blockRows;
    int32_t srcStrideY, srcStrideC, recStride, coarseBase;
};
static_assert(sizeof(Params) == 112, "refine_common.glsl's Params is 28 ints");

// Super's push constants (super.comp, super_qpel.comp, pyr_reduce.comp), matching
// super_common.glsl's Params; they share the Main layout and its push constant range
struct SuperParams {
    int32_t w, h, aw, ah;
    int32_t padX, padY, padcX, padcY;
    int32_t wp, hp, wc, hc;
    int32_t lumaPlanes, chromaPlanes, xr, yr;
    int32_t planes, step, sharp, pixelMax;
    int32_t srcStrideY, srcStrideC, pelStrideY, pelStrideC;
    int32_t level, rfilter, pel, reserved0;
};
static_assert(sizeof(SuperParams) <= 112, "super_common.glsl's Params must fit the Main layout's push constants");

// Degrain's push constants, matching degrain_common.glsl's Params
struct DegrainParams {
    int32_t nbx, nby, step, overlap;
    int32_t pad, padc, wp, hp;
    int32_t wc, hc, recStride, refs;
    uint32_t usable0, usable1;
    int32_t thscd1, scdLimit;
    int32_t plane, width, height, outStride;
    int32_t limit, winOff, thOff, uwOff;
    int32_t nb, reserved0, reserved1, reserved2;
};
static_assert(sizeof(DegrainParams) == 112, "degrain_common.glsl's Params is 28 ints");

class VulkanContext {
public:
    // The context of the core's device, created with the first filter that asks; throws when the
    // core has no usable Vulkan device or the device lacks what the kernels need
    static std::shared_ptr<VulkanContext> Get(VSCore *core, const VSAPI *vsapi);
    ~VulkanContext();
    VulkanContext(const VulkanContext &) = delete;
    VulkanContext &operator=(const VulkanContext &) = delete;

    // The kernel's pipeline for blocks of blk x blk and vectors of 1 / pel pixels, compiled on first
    // use; filters ask for theirs when they are created, not while producing frames. variant is
    // specialization constant 6, the super kernels' sample kind (SuperFormat::Kind).
    VkPipeline Pipeline(Kernel kernel, int blk, int pel, int variant = 0);

    // A device-local storage buffer holding size bytes of data, copied there through the pool and
    // waited for: the tables a filter reads for its whole life, uploaded when it is created.
    // Throws on failure.
    VSGPUBuffer *Upload(VSCore *core, VSGPUExecPool *pool, const void *data, VkDeviceSize size, VSVulkanBufferInfo &info) const;

    // A layout's bindings, (binding, descriptor count) in binding order
    const std::vector<std::pair<uint32_t, uint32_t>> &LayoutBindings(Layout layout) const { return bindings[static_cast<int>(layout)]; }
    VkPipelineLayout PipelineLayout(Layout layout) const { return layouts[static_cast<int>(layout)]; }
    VkDescriptorSetLayout SetLayout(Layout layout) const { return setLayouts[static_cast<int>(layout)]; }
    // Throws when the device can't hold the Degrain layout's descriptors
    void RequireDegrain() const;

    const VSVULKANAPI *vkapi = nullptr;
    const VSVulkanFunctions *vk = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceLimits limits = {};
    std::string deviceName;
    uint32_t subgroup = 32;      // lanes per subgroup, every pipeline's required subgroup size
    uint32_t medianLanes = 1024; // median.comp's workgroup, within the device's subgroups per workgroup

    static constexpr int kFallbackSubgroups = 4; // subgroups searching each block the fallback flags
    static constexpr int kMaxBatch = 16;         // fields per coarse search batch: pyr_common.glsl's kMaxBatch

private:
    VulkanContext(VSCore *core, const VSAPI *vsapi);
    void Destroy();

    VSCore *core;
    std::mutex lock;
    std::map<std::tuple<int, int, int, int>, VkPipeline> pipelines;
    static constexpr int kLayouts = 3;
    std::vector<std::pair<uint32_t, uint32_t>> bindings[kLayouts];
    VkDescriptorSetLayout setLayouts[kLayouts] = {};
    VkPipelineLayout layouts[kLayouts] = {};
    std::string degrainError; // why the Degrain layout couldn't be made, empty when it was
};

// GPU time per stage of a filter's submissions, with the environment variable MVGPU_PROFILE set:
// every submission gets a query pool and a timestamp after each stage, and is waited for once
// submitted, which serializes the frames, so it is for measuring only. The stages' times add up
// and go to stderr when the filter is freed. Stamps for a stage that didn't run come back to back.
class StageProfiler {
public:
    StageProfiler(std::string name, std::vector<std::string> stages);
    ~StageProfiler();

    static bool Requested();

    // Before the first stage; then Stamp after stage i = 1 .. stages.size()
    VkQueryPool Begin(const VulkanContext &vc, VkCommandBuffer cmd) const;
    void Stamp(const VulkanContext &vc, VkCommandBuffer cmd, VkQueryPool pool, int stage) const;
    // After the submit: waits for it, adds up its stages, frees the pool
    void Finish(const VulkanContext &vc, VSGPUExecPool *execPool, uint64_t signaled, VkQueryPool pool);

private:
    std::string name;
    std::vector<std::string> stages;
    std::mutex lock;
    std::vector<double> sums; // ms
    int64_t runs = 0;
};

// The dispatches of one submission into an exec context's command buffer, all with pipelines of
// one layout. Every binding holds a buffer, a dummy until Bind says otherwise, and all of them are
// pushed before a dispatch whenever one changed; push constants go with every dispatch.
class Recorder {
public:
    Recorder(const VulkanContext &vc, VkCommandBuffer cmd, VkBuffer dummy, Layout layout = Layout::Main);

    // element: the place in an array binding
    void Bind(int binding, VkBuffer buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE, int element = 0);
    void Dispatch(VkPipeline pipeline, const Params &pc, uint32_t x, uint32_t y, uint32_t z = 1);
    void Dispatch(VkPipeline pipeline, const SuperParams &pc, uint32_t x, uint32_t y, uint32_t z = 1);
    void DispatchIndirect(VkPipeline pipeline, const Params &pc, VkBuffer args, VkDeviceSize offset);

    void Fill(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t value);
    void Copy(VkBuffer src, VkDeviceSize srcOffset, VkBuffer dst, VkDeviceSize dstOffset, VkDeviceSize size);

    // Compute writes before compute reads and writes
    void ComputeBarrier();
    // Compute writes before an indirect dispatch reads its arguments, and before compute
    void IndirectBarrier();
    // Compute reads and writes before transfers, and transfer writes before compute
    void ComputeToTransfer();
    void TransferToCompute();

private:
    void Barrier(VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);
    void Prepare(VkPipeline pipeline, const void *pc, uint32_t size);

    const VulkanContext &vc;
    VkCommandBuffer cmd;
    VkPipelineLayout layout;
    const std::vector<std::pair<uint32_t, uint32_t>> &bindings;
    std::vector<VkDescriptorBufferInfo> infos; // every binding's descriptors, in binding order
    int first[kBindings];                      // where each binding starts in infos, -1 when the layout lacks it
    bool dirty = true;
};
