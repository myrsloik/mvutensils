#pragma once

// The Vulkan side every mvgpu filter shares: the core's device and dispatch table, the one
// descriptor layout all kernels use (32 storage buffers pushed with every dispatch, plus the push
// constants), the kernels' pipelines, compiled once per core and variant, and a recorder for the
// dispatches of one submission. Filters keep a shared_ptr to it, so the pipelines live until the
// last instance on the core is freed.

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <VapourSynth4.h>
#include <VSVulkan4.h>

// The kernels in src/shaders
enum class Kernel {
    Super,
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
    kBindings
};

// Push constants, matching refine_common.glsl's Params
struct Params {
    int32_t w, h, nbx, nby;
    int32_t step, pad, padc, colour;
    int32_t wp, hp, wc, hc;
    int32_t fallbackRadius, fallbackStep, badSad, maxSeeds;
    int32_t stamp, level, flags, topRadius;
    int32_t medianScale, medianSlot, finest, blockRows;
    int32_t srcStrideY, srcStrideC, recStride, reserved;
};
static_assert(sizeof(Params) == 112, "refine_common.glsl's Params is 28 ints");

class VulkanContext {
public:
    // The context of the core's device, created with the first filter that asks; throws when the
    // core has no usable Vulkan device or the device lacks what the kernels need
    static std::shared_ptr<VulkanContext> Get(VSCore *core, const VSAPI *vsapi);
    ~VulkanContext();
    VulkanContext(const VulkanContext &) = delete;
    VulkanContext &operator=(const VulkanContext &) = delete;

    // The kernel's pipeline for blocks of blk x blk, compiled on first use; filters ask for theirs
    // when they are created, not while producing frames
    VkPipeline Pipeline(Kernel kernel, int blk);

    // A device-local storage buffer holding size bytes of data, copied there through the pool and
    // waited for: the tables a filter reads for its whole life, uploaded when it is created.
    // Throws on failure.
    VSGPUBuffer *Upload(VSCore *core, VSGPUExecPool *pool, const void *data, VkDeviceSize size, VSVulkanBufferInfo &info) const;

    const VSVULKANAPI *vkapi = nullptr;
    const VSVulkanFunctions *vk = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceLimits limits = {};
    std::string deviceName;
    uint32_t subgroup = 32;      // lanes per subgroup, every pipeline's required subgroup size
    uint32_t medianLanes = 1024; // median.comp's workgroup, within the device's subgroups per workgroup
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;

    static constexpr int kFallbackSubgroups = 4; // subgroups searching each block the fallback flags

private:
    VulkanContext(VSCore *core, const VSAPI *vsapi);

    VSCore *core;
    std::mutex lock;
    std::map<std::pair<int, int>, VkPipeline> pipelines;
};

// The dispatches of one submission into an exec context's command buffer. Every binding holds a
// buffer, a dummy until Bind says otherwise, and all of them are pushed before a dispatch whenever
// one changed; push constants go with every dispatch.
class Recorder {
public:
    Recorder(const VulkanContext &vc, VkCommandBuffer cmd, VkBuffer dummy);

    void Bind(int binding, VkBuffer buffer, VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);
    void Dispatch(VkPipeline pipeline, const Params &pc, uint32_t x, uint32_t y, uint32_t z = 1);
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
    void Prepare(VkPipeline pipeline, const Params &pc);

    const VulkanContext &vc;
    VkCommandBuffer cmd;
    VkDescriptorBufferInfo infos[kBindings] = {};
    bool dirty = true;
};
