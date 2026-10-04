#include "VulkanContext.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "ShaderSources.h"

namespace {

const char *KernelFile(Kernel kernel) {
    switch (kernel) {
    case Kernel::Super: return "super.comp";
    case Kernel::SuperQuarter: return "super_qpel.comp";
    case Kernel::PyrReduce: return "pyr_reduce.comp";
    case Kernel::PyrTop: return "pyr_top.comp";
    case Kernel::PyrPass: return "pyr_pass.comp";
    case Kernel::PyrSeed: return "pyr_seed.comp";
    case Kernel::Median: return "median.comp";
    case Kernel::SeedScatter: return "seed_scatter.comp";
    case Kernel::SeedBuild: return "seed_build.comp";
    case Kernel::RefineInit: return "refine_init.comp";
    case Kernel::RefinePass: return "refine_pass.comp";
    case Kernel::RefineFlag: return "refine_flag.comp";
    case Kernel::RefineFallback: return "refine_fallback.comp";
    case Kernel::RefineApply: return "refine_apply.comp";
    case Kernel::RefineHalfpel: return "refine_halfpel.comp";
    case Kernel::RefineQuarter: return "refine_qpel.comp";
    case Kernel::DegrainCount: return "degrain_count.comp";
    case Kernel::DegrainWeights: return "degrain_weights.comp";
    case Kernel::DegrainPixels: return "degrain.comp";
    case Kernel::FlowPrep: return "flow_prep.comp";
    case Kernel::FlowInter: return "flow_inter.comp";
    case Kernel::FlowFetch: return "flow_fetch.comp";
    case Kernel::FlowBlur: return "flow_blur.comp";
    case Kernel::MaskBlocks: return "mask_blocks.comp";
    case Kernel::MaskResize: return "mask_resize.comp";
    case Kernel::Compensate: return "compensate.comp";
    case Kernel::Recalculate: return "recalc.comp";
    }
    return "";
}

// The coarse search's kernels and Degrain's have their own layouts
Layout KernelLayout(Kernel kernel) {
    switch (kernel) {
    case Kernel::PyrTop:
    case Kernel::PyrPass:
    case Kernel::PyrSeed:
    case Kernel::Median:
        return Layout::Coarse;
    case Kernel::DegrainCount:
    case Kernel::DegrainWeights:
    case Kernel::DegrainPixels:
        return Layout::Degrain;
    default:
        return Layout::Main;
    }
}

const char *FindSource(const std::string &name) {
    for (const ShaderSource &s : kShaderSources)
        if (name == s.name)
            return s.text;
    return nullptr;
}

// A kernel's text with its #include lines replaced by the files they name, recursively, and the
// include extension dropped: the core's compiler has no include handler
std::string Expand(const std::string &name, int depth = 0) {
    if (depth > 8)
        throw std::runtime_error("shader includes nest too deeply at " + name);
    const char *text = FindSource(name);
    if (!text)
        throw std::runtime_error("no shader source named " + name);
    std::istringstream in(text);
    std::string out, line;
    while (std::getline(in, line)) {
        if (line.rfind("#extension GL_GOOGLE_include_directive", 0) == 0)
            continue;
        if (line.rfind("#include \"", 0) == 0) {
            const size_t end = line.find('"', 10);
            if (end == std::string::npos)
                throw std::runtime_error("malformed #include in " + name);
            out += Expand(line.substr(10, end - 10), depth + 1);
            continue;
        }
        out += line;
        out += '\n';
    }
    return out;
}

} // namespace

std::shared_ptr<VulkanContext> VulkanContext::Get(VSCore *core, const VSAPI *vsapi) {
    static std::mutex mutex;
    static std::map<VSCore *, std::weak_ptr<VulkanContext>> contexts;
    std::lock_guard<std::mutex> guard(mutex);
    for (auto it = contexts.begin(); it != contexts.end();)
        it = it->second.expired() ? contexts.erase(it) : std::next(it);
    std::weak_ptr<VulkanContext> &slot = contexts[core];
    if (std::shared_ptr<VulkanContext> existing = slot.lock())
        return existing;
    std::shared_ptr<VulkanContext> created(new VulkanContext(core, vsapi));
    slot = created;
    return created;
}

VulkanContext::VulkanContext(VSCore *core, const VSAPI *vsapi) : core(core) {
    if (vsapi->getAPIVersion() < VS_MAKE_VERSION(4, 3))
        throw std::runtime_error("VapourSynth API 4.3 or later is required for GPU frames");
    vkapi = vsapi->getVulkanAPI();
    char err[1024] = {};
    vk = vkapi->getVulkanFunctions(core, err, sizeof(err));
    if (!vk)
        throw std::runtime_error(std::string("no usable Vulkan device: ") + err);
    VSVulkanCoreHandles handles = {};
    if (vkapi->getVulkanHandles(core, &handles, err, sizeof(err)))
        throw std::runtime_error(std::string("no usable Vulkan device: ") + err);
    device = handles.device;

    VkPhysicalDeviceVulkan14Properties p14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties p13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, &p14};
    VkPhysicalDeviceVulkan11Properties p11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &p13};
    VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p11};
    vk->vkGetPhysicalDeviceProperties2(handles.physicalDevice, &props);
    limits = props.properties.limits;
    deviceName = props.properties.deviceName;

    // The kernels run 32-lane subgroups, or 64 where the device offers only that, always full
    if (!(p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
        throw std::runtime_error(deviceName + " can't set the subgroup size of compute shaders");
    auto offers = [&](uint32_t s) { return p13.minSubgroupSize <= s && s <= p13.maxSubgroupSize; };
    if (!offers(32) && !offers(64))
        throw std::runtime_error(deviceName + " runs neither 32- nor 64-lane subgroups");
    subgroup = offers(32) ? 32 : 64;
    const VkSubgroupFeatureFlags ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                       VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
    if ((p11.subgroupSupportedOperations & ops) != ops || !(p11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
        throw std::runtime_error(deviceName + " lacks the subgroup operations the kernels use (basic, arithmetic, ballot, shuffle, clustered)");
    medianLanes = std::min({1024u, p13.maxComputeWorkgroupSubgroups * subgroup, limits.maxComputeWorkGroupInvocations, limits.maxComputeWorkGroupSize[0]});
    medianLanes -= medianLanes % subgroup;
    if (medianLanes < subgroup * kFallbackSubgroups)
        throw std::runtime_error(deviceName + " can't run workgroups of " + std::to_string(subgroup * kFallbackSubgroups) + " lanes");
    // Main: every binding once; coarse: bindings 16 .. 22, the reference pyramids (17) kMaxBatch times
    for (uint32_t i = 0; i < kBindings; ++i)
        bindings[static_cast<int>(Layout::Main)].push_back({i, 1});
    for (uint32_t i = kCurPyramid; i <= kMedians; ++i)
        bindings[static_cast<int>(Layout::Coarse)].push_back({i, i == kRefPyramid ? kMaxBatch : 1u});
    for (Layout pushed : {Layout::Main, Layout::Coarse}) {
        uint32_t count = 0;
        for (const auto &[binding, n] : bindings[static_cast<int>(pushed)])
            count += n;
        if (p14.maxPushDescriptors < count)
            throw std::runtime_error(deviceName + " pushes fewer than " + std::to_string(count) + " descriptors");
    }
    // Degrain: the output planes 3 times, each reference's buffers kMaxDegrainRefs times
    for (uint32_t i = 0; i < kDgBindings; ++i)
        bindings[static_cast<int>(Layout::Degrain)].push_back({i, i == kDgOut ? 3u : i >= kDgRefLuma ? static_cast<uint32_t>(kMaxDegrainRefs) : 1u});
    uint32_t degrainCount = 0;
    for (const auto &[binding, n] : bindings[static_cast<int>(Layout::Degrain)])
        degrainCount += n;
    if (limits.maxPerStageDescriptorStorageBuffers < degrainCount || limits.maxDescriptorSetStorageBuffers < degrainCount ||
        limits.maxPerStageResources < degrainCount)
        degrainError = deviceName + " binds fewer than " + std::to_string(degrainCount) + " storage buffers to a kernel, which Degrain needs";

    // The core enables these where the device has them; the kernels need them
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
    vk->vkGetPhysicalDeviceFeatures2(handles.physicalDevice, &features);
    if (!features.features.shaderInt64)
        throw std::runtime_error(deviceName + " lacks 64-bit integers in shaders");
    if (!f12.shaderBufferInt64Atomics)
        throw std::runtime_error(deviceName + " lacks 64-bit buffer atomics");
    float64 = features.features.shaderFloat64;

    for (int l = 0; l < kLayouts; ++l) {
        const bool degrain = l == static_cast<int>(Layout::Degrain);
        if (degrain && !degrainError.empty())
            continue;
        std::vector<VkDescriptorSetLayoutBinding> list;
        for (const auto &[binding, count] : bindings[l])
            list.push_back({binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.flags = degrain ? 0 : VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
        dslci.bindingCount = static_cast<uint32_t>(list.size());
        dslci.pBindings = list.data();
        if (vk->vkCreateDescriptorSetLayout(device, &dslci, nullptr, &setLayouts[l]) != VK_SUCCESS) {
            Destroy();
            throw std::runtime_error("vkCreateDescriptorSetLayout failed");
        }
        const VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
        VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &setLayouts[l];
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &range;
        if (vk->vkCreatePipelineLayout(device, &plci, nullptr, &layouts[l]) != VK_SUCCESS) {
            Destroy();
            throw std::runtime_error("vkCreatePipelineLayout failed");
        }
    }
}

void VulkanContext::Destroy() {
    for (const auto &[key, pipeline] : pipelines)
        vk->vkDestroyPipeline(device, pipeline, nullptr);
    pipelines.clear();
    for (int l = 0; l < kLayouts; ++l) {
        if (layouts[l])
            vk->vkDestroyPipelineLayout(device, layouts[l], nullptr);
        if (setLayouts[l])
            vk->vkDestroyDescriptorSetLayout(device, setLayouts[l], nullptr);
        layouts[l] = VK_NULL_HANDLE;
        setLayouts[l] = VK_NULL_HANDLE;
    }
}

VulkanContext::~VulkanContext() {
    Destroy();
}

void VulkanContext::RequireDegrain() const {
    if (!degrainError.empty())
        throw std::runtime_error(degrainError);
}

VkPipeline VulkanContext::Pipeline(Kernel kernel, int blk, int pel, int variant, int blkY) {
    if (KernelLayout(kernel) == Layout::Degrain)
        RequireDegrain();
    if (!blkY)
        blkY = blk;
    std::lock_guard<std::mutex> guard(lock);
    const std::tuple<int, int, int, int, int> key(static_cast<int>(kernel), blk, pel, variant, blkY);
    if (auto it = pipelines.find(key); it != pipelines.end())
        return it->second;

    // MVGPU_FLOAT64 after the #version line: whether the kernel may use doubles, which a module
    // declaring them can't be made into a pipeline without
    std::string source = Expand(KernelFile(kernel));
    const size_t version = source.find('\n');
    source.insert(version == std::string::npos ? source.size() : version + 1, std::string("#define MVGPU_FLOAT64 ") + (float64 ? "1" : "0") + "\n");
    std::vector<char> log(16384);
    VSGPUShader *shader = vkapi->compileGPUShader(core, slGLSL, source.c_str(), log.data(), static_cast<int>(log.size()));
    if (!shader)
        throw std::runtime_error(std::string("compiling ") + KernelFile(kernel) + " failed: " + log.data());
    size_t bytes = 0;
    const uint32_t *code = vkapi->getGPUShaderCode(shader, &bytes);
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = bytes;
    smci.pCode = code;
    VkShaderModule module = VK_NULL_HANDLE;
    const VkResult made = vk->vkCreateShaderModule(device, &smci, nullptr, &module);
    vkapi->freeGPUShader(shader);
    if (made != VK_SUCCESS)
        throw std::runtime_error(std::string("vkCreateShaderModule failed for ") + KernelFile(kernel));

    // refine_common.glsl's kSubgroup (0), the workgroup size of the kernels that spread a block over
    // one subgroup (1), median.comp's workgroup size (2), the target grid's block width (3) and height
    // (7), the fallback's workgroup size (4), the vectors' units per pixel (5) and the variant (6)
    const uint32_t constants[8] = {subgroup, subgroup, medianLanes, static_cast<uint32_t>(blk), subgroup * kFallbackSubgroups, static_cast<uint32_t>(pel),
                                   static_cast<uint32_t>(variant), static_cast<uint32_t>(blkY)};
    const VkSpecializationMapEntry entries[8] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}, {4, 16, 4}, {5, 20, 4}, {6, 24, 4}, {7, 28, 4}};
    const VkSpecializationInfo spec = {8, entries, sizeof(constants), constants};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo size = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    size.requiredSubgroupSize = subgroup;
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.pNext = &size;
    cpci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = &spec;
    cpci.layout = PipelineLayout(KernelLayout(kernel));
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult created = vk->vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline);
    vk->vkDestroyShaderModule(device, module, nullptr);
    if (created != VK_SUCCESS)
        throw std::runtime_error(std::string("vkCreateComputePipelines failed for ") + KernelFile(kernel));
    pipelines[key] = pipeline;
    return pipeline;
}

VSGPUBuffer *VulkanContext::Upload(VSCore *core, VSGPUExecPool *pool, const void *data, VkDeviceSize size, VSVulkanBufferInfo &info) const {
    char err[1024] = {};
    VSVulkanBufferInfo stagingInfo = {};
    VSGPUBuffer *staging = vkapi->createGPUBuffer(core, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0,
                                                  &stagingInfo, err, sizeof(err));
    if (!staging)
        throw std::runtime_error(std::string("failed to allocate a staging buffer: ") + err);
    memcpy(stagingInfo.mapped, data, size);
    VSGPUBuffer *buffer = vkapi->createGPUBuffer(core, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                                 &info, err, sizeof(err));
    if (!buffer) {
        vkapi->destroyGPUBuffer(staging);
        throw std::runtime_error(std::string("failed to allocate a GPU buffer: ") + err);
    }
    VSGPUExecContext *ctx = vkapi->gpuExecAcquire(pool, err, sizeof(err));
    if (!ctx) {
        vkapi->destroyGPUBuffer(staging);
        vkapi->destroyGPUBuffer(buffer);
        throw std::runtime_error(err);
    }
    VkBufferCopy2 region = {VK_STRUCTURE_TYPE_BUFFER_COPY_2};
    region.size = size;
    VkCopyBufferInfo2 copy = {VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
    copy.srcBuffer = stagingInfo.buffer;
    copy.dstBuffer = info.buffer;
    copy.regionCount = 1;
    copy.pRegions = &region;
    vk->vkCmdCopyBuffer2(vkapi->gpuExecCommandBuffer(ctx), &copy);
    vkapi->gpuExecUsesBuffer(ctx, staging);
    if (vkapi->gpuExecSubmit(ctx, nullptr, err, sizeof(err))) {
        // The context released the staging buffer; the copy never ran, so the buffer is unused
        vkapi->destroyGPUBuffer(buffer);
        throw std::runtime_error(err);
    }
    const int drained = vkapi->gpuExecPoolWaitIdle(pool, err, sizeof(err));
    if (drained != gdDrained) {
        if (vsGPUDrainSafeToDestroy(drained))
            vkapi->destroyGPUBuffer(buffer);
        throw std::runtime_error(std::string("uploading a table failed: ") + err);
    }
    return buffer;
}

Recorder::Recorder(const VulkanContext &vc, VkCommandBuffer cmd, VkBuffer dummy, Layout layout)
    : vc(vc), cmd(cmd), layout(vc.PipelineLayout(layout)), bindings(vc.LayoutBindings(layout)) {
    std::fill(std::begin(first), std::end(first), -1);
    for (const auto &[binding, count] : bindings) {
        first[binding] = static_cast<int>(infos.size());
        infos.insert(infos.end(), count, VkDescriptorBufferInfo{dummy, 0, VK_WHOLE_SIZE});
    }
}

void Recorder::Bind(int binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range, int element) {
    if (first[binding] < 0)
        throw std::logic_error("binding " + std::to_string(binding) + " isn't in the recorder's layout");
    infos[first[binding] + element] = {buffer, offset, range};
    dirty = true;
}

void Recorder::Prepare(VkPipeline pipeline, const void *pc, uint32_t size) {
    vc.vk->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    // Every pipeline the recorder binds has its layout, so pushed descriptors stay bound across them
    if (dirty) {
        std::vector<VkWriteDescriptorSet> writes;
        for (const auto &[binding, count] : bindings) {
            VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstBinding = binding;
            w.descriptorCount = count;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &infos[first[binding]];
            writes.push_back(w);
        }
        vc.vk->vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, static_cast<uint32_t>(writes.size()), writes.data());
        dirty = false;
    }
    VkPushConstantsInfo push = {VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO};
    push.layout = layout;
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = size;
    push.pValues = pc;
    vc.vk->vkCmdPushConstants2(cmd, &push);
}

void Recorder::Dispatch(VkPipeline pipeline, const Params &pc, uint32_t x, uint32_t y, uint32_t z) {
    if (!x || !y || !z)
        return;
    Prepare(pipeline, &pc, sizeof(pc));
    vc.vk->vkCmdDispatch(cmd, x, y, z);
}

void Recorder::Dispatch(VkPipeline pipeline, const SuperParams &pc, uint32_t x, uint32_t y, uint32_t z) {
    if (!x || !y || !z)
        return;
    Prepare(pipeline, &pc, sizeof(pc));
    vc.vk->vkCmdDispatch(cmd, x, y, z);
}

void Recorder::Dispatch(VkPipeline pipeline, const FlowParams &pc, uint32_t x, uint32_t y, uint32_t z) {
    if (!x || !y || !z)
        return;
    Prepare(pipeline, &pc, sizeof(pc));
    vc.vk->vkCmdDispatch(cmd, x, y, z);
}

void Recorder::Dispatch(VkPipeline pipeline, const MaskParams &pc, uint32_t x, uint32_t y, uint32_t z) {
    if (!x || !y || !z)
        return;
    Prepare(pipeline, &pc, sizeof(pc));
    vc.vk->vkCmdDispatch(cmd, x, y, z);
}

void Recorder::Dispatch(VkPipeline pipeline, const RecalcParams &pc, uint32_t x, uint32_t y, uint32_t z) {
    if (!x || !y || !z)
        return;
    Prepare(pipeline, &pc, sizeof(pc));
    vc.vk->vkCmdDispatch(cmd, x, y, z);
}

void RecordBadBlockCount(Recorder &rec, VkPipeline count, VkBuffer vectors, int recStride, int nbx, int nby, int thscd1, VkBuffer counts, int slot) {
    MaskParams pc = {};
    pc.nbx = nbx;
    pc.nby = nby;
    pc.recStride = recStride;
    pc.kind = 3; // mask_common.glsl's kCount
    pc.thscd1 = thscd1;
    pc.countSlot = slot;
    rec.Bind(kMkVectors, vectors);
    rec.Bind(kMkCounts, counts);
    rec.Dispatch(count, pc, static_cast<uint32_t>((nbx * nby + 63) / 64), 1);
}

void Recorder::DispatchIndirect(VkPipeline pipeline, const Params &pc, VkBuffer args, VkDeviceSize offset) {
    Prepare(pipeline, &pc, sizeof(pc));
    vc.vk->vkCmdDispatchIndirect(cmd, args, offset);
}

void Recorder::Fill(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t value) {
    vc.vk->vkCmdFillBuffer(cmd, buffer, offset, size, value);
}

void Recorder::Copy(VkBuffer src, VkDeviceSize srcOffset, VkBuffer dst, VkDeviceSize dstOffset, VkDeviceSize size) {
    VkBufferCopy2 region = {VK_STRUCTURE_TYPE_BUFFER_COPY_2};
    region.srcOffset = srcOffset;
    region.dstOffset = dstOffset;
    region.size = size;
    VkCopyBufferInfo2 info = {VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
    info.srcBuffer = src;
    info.dstBuffer = dst;
    info.regionCount = 1;
    info.pRegions = &region;
    vc.vk->vkCmdCopyBuffer2(cmd, &info);
}

void Recorder::Barrier(VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkMemoryBarrier2 barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    VkDependencyInfo dep = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vc.vk->vkCmdPipelineBarrier2(cmd, &dep);
}

void Recorder::ComputeBarrier() {
    Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

void Recorder::IndirectBarrier() {
    Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

void Recorder::ComputeToTransfer() {
    Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

void Recorder::TransferToCompute() {
    Barrier(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

void Recorder::ComputeToHost() {
    Barrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
}

StageProfiler::StageProfiler(std::string name, std::vector<std::string> stages) : name(std::move(name)), stages(std::move(stages)), sums(this->stages.size(), 0.0) {}

StageProfiler::~StageProfiler() {
    if (!runs)
        return;
    double total = 0;
    for (double s : sums)
        total += s;
    fprintf(stderr, "mvgpu profile, %s: %lld submissions, %.3f ms each on the GPU\n", name.c_str(), static_cast<long long>(runs), total / runs);
    for (size_t i = 0; i < stages.size(); ++i)
        fprintf(stderr, "  %-34s %.3f ms\n", stages[i].c_str(), sums[i] / runs);
}

bool StageProfiler::Requested() {
    const char *v = getenv("MVGPU_PROFILE");
    return v && *v && strcmp(v, "0") != 0;
}

VkQueryPool StageProfiler::Begin(const VulkanContext &vc, VkCommandBuffer cmd) const {
    VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = static_cast<uint32_t>(stages.size() + 1);
    VkQueryPool pool = VK_NULL_HANDLE;
    if (vc.vk->vkCreateQueryPool(vc.device, &qpci, nullptr, &pool) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    vc.vk->vkCmdResetQueryPool(cmd, pool, 0, qpci.queryCount);
    vc.vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool, 0);
    return pool;
}

void StageProfiler::Stamp(const VulkanContext &vc, VkCommandBuffer cmd, VkQueryPool pool, int stage) const {
    if (pool)
        vc.vk->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool, static_cast<uint32_t>(stage));
}

void StageProfiler::Finish(const VulkanContext &vc, VSGPUExecPool *execPool, uint64_t signaled, VkQueryPool pool) {
    if (!pool)
        return;
    char err[256] = {};
    if (vc.vkapi->gpuExecWaitValue(execPool, signaled, err, sizeof(err)) == gdDrained) {
        std::vector<uint64_t> ts(stages.size() + 1);
        if (vc.vk->vkGetQueryPoolResults(vc.device, pool, 0, static_cast<uint32_t>(ts.size()), ts.size() * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            std::lock_guard<std::mutex> guard(lock);
            for (size_t i = 0; i < stages.size(); ++i)
                sums[i] += (ts[i + 1] - ts[i]) * vc.limits.timestampPeriod * 1e-6;
            ++runs;
        }
    }
    vc.vk->vkDestroyQueryPool(vc.device, pool, nullptr);
}
