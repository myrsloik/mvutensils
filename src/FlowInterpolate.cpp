#include <algorithm>
#include <cfloat>
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
#include "SuperLayout.h"
#include "VulkanContext.h"

// mvu.FlowInter and mvu.FlowFPS on the GPU: the frame at some time between two frames, each pixel
// of it fetched from both along their vectors scaled to that time and blended through the blocks'
// occlusion masks (or, where the extra vectors serve, the median of four fetches). Two kernels per
// frame: the scene change test's counts and the occlusion masks, per block (flow_prep.comp), then
// every pixel of every plane (flow_inter.comp), which also falls back to blending the clip's frames
// (or copying the first) when the vectors are at a scene change. The result is mvu's bit for bit
// given the same super and vectors: the blocks' vectors and masks reach the pixels through zimg's
// bilinear resize in mvu, in 64 x 64 tiles, which flow_inter.comp reproduces with the taps zimg
// computes for each tile (ResizeTaps).
//
// Implemented: all of their arguments, on mvgpu.Super's 4:2:0 and 4:4:4 supers of 8 to 16-bit
// samples at pel 2 and 4 with square blocks of 8, 16 or 32, and vectors made from such supers, of
// any of those bit depths, as mvgpu.Analyse makes them.

namespace {

// FlowParams' flags, flow_common.glsl's: F and B have vectors, FF and BB have, blend the clip's
// frames where F and B don't serve
constexpr int kHaveFB = 1, kHaveExtra = 2, kBlend = 4;

// The taps of zimg's bilinear filter for one tile of dstDim samples from srcDim, mvu's MaskResizer
// asking for the source's samples from shift on, width of them (the active region): compute_filter
// and matrix_to_filter in zimg's resize/filter.cpp, in the same double arithmetic, with the matrix's
// sparse rows as RowMatrix keeps them. Each output sample takes source samples left and left + 1,
// weighted c and (1 << 14) - c (an upscale has two taps, which matrix_to_filter's correction makes
// add up to 1 << 14); out gets (left << 15) | c.
void TileTaps(unsigned srcDim, unsigned dstDim, double shift, double width, int32_t *out) {
    const double scale = static_cast<double>(dstDim) / width;
    if (scale < 1.0)
        throw std::runtime_error("the blocks are too small for the resize's two taps");
    const double step = std::min(scale, 1.0);
    const double support = static_cast<double>(1) / step;
    const unsigned filterSize = std::max(static_cast<unsigned>(std::ceil(support)) * 2U, 1U);
    auto filter = [](double x) { return std::max(1.0 - std::abs(x), 0.0); };
    auto roundHalfup = [](double x) { return x < 0 ? std::floor(x + 0.5) : std::floor(x + 0.49999999999999994); };

    // RowMatrix's rows: an entry is allocated when a value other than the one there is stored
    struct Row {
        size_t left = 0;
        std::vector<double> data;
        double Get(size_t j) const { return j < left || j >= left + data.size() ? 0.0 : data[j - left]; }
        void Set(size_t j, double v) {
            if (Get(j) == v)
                return;
            if (data.empty()) {
                data.assign(1, 0.0);
                left = j;
            } else if (j < left) {
                data.insert(data.begin(), left - j, 0.0);
                left = j;
            } else if (j >= left + data.size()) {
                data.insert(data.end(), j - (left + data.size()) + 1, 0.0);
            }
            data[j - left] = v;
        }
    };
    std::vector<Row> m(dstDim);
    for (unsigned i = 0; i < dstDim; ++i) {
        const double pos = (i + 0.5) / scale + shift;
        const double beginPos = roundHalfup(pos - filterSize / 2.0) + 0.5;
        double total = 0.0;
        for (unsigned j = 0; j < filterSize; ++j) {
            const double xpos = beginPos + j;
            total += filter((xpos - pos) * step);
        }
        size_t left = SIZE_MAX;
        for (unsigned j = 0; j < filterSize; ++j) {
            const double xpos = beginPos + j;
            double realPos = xpos < 0.0 ? -xpos : xpos >= srcDim ? 2.0 * srcDim - xpos : xpos;
            realPos = std::clamp(realPos, 0.0, std::nextafter(static_cast<double>(srcDim), -INFINITY));
            const size_t idx = static_cast<size_t>(std::floor(realPos));
            m[i].Set(idx, m[i].Get(idx) + filter((xpos - pos) * step) / total);
            left = std::min(left, idx);
        }
        if (m[i].Get(left) == 0.0) {
            m[i].Set(left, DBL_EPSILON);
            m[i].Set(left, 0.0);
        }
    }

    size_t taps = 0;
    for (const Row &r : m)
        taps = std::max(taps, r.data.size());
    if (taps > 2)
        throw std::runtime_error("the resize has more than two taps");
    for (unsigned i = 0; i < dstDim; ++i) {
        const size_t left = std::min(m[i].left, static_cast<size_t>(srcDim) - taps);
        double err = 0;
        int16_t sum = 0, greatest = 0;
        size_t greatestIdx = 0;
        int16_t c[2] = {};
        for (size_t j = 0; j < taps; ++j) {
            const double expected = m[i].Get(left + j) * (1 << 14) - err;
            const int16_t coeff = static_cast<int16_t>(std::lrint(expected));
            err = static_cast<double>(coeff) - expected;
            if (std::abs(coeff) > greatest) {
                greatest = coeff;
                greatestIdx = j;
            }
            sum += coeff;
            c[j] = coeff;
        }
        c[greatestIdx] += (1 << 14) - sum;
        if (c[0] < 0 || c[0] + (taps > 1 ? c[1] : 0) != (1 << 14))
            throw std::runtime_error("the resize's taps don't add up");
        out[i] = static_cast<int32_t>((left << 15) | static_cast<unsigned>(c[0]));
    }
}

// mvu's MaskResizer (MaskResize.cpp) for one dimension: the srcDim blocks laid over the samples
// they cover, the blocks blk apart (their size less the overlap) plus one overlap, resized to dstDim
// samples in tiles of 64; for each output sample its taps, as TileTaps packs them
std::vector<int32_t> ResizeTaps(int srcDim, int dstDim, int step, int overlap) {
    const double srcScale = static_cast<double>(srcDim) / (step * srcDim + overlap);
    std::vector<int32_t> taps(dstDim);
    for (int t = 0; t < dstDim; t += 64) {
        const int n = std::min(64, dstDim - t);
        // The tile's active region, which zimg's graph builder turns into the resize's shift and width
        const double left = t * srcScale, width = n * srcScale;
        const double scaleW = static_cast<double>(n) / width;
        TileTaps(static_cast<unsigned>(srcDim), static_cast<unsigned>(n), left - 0.0 / scaleW, width * (static_cast<double>(n) / static_cast<double>(n)),
                 taps.data() + t);
    }
    return taps;
}

// zimg's choice of the order of the passes (resize/resize.cpp's resize_h_first), for every size of
// tile mvu's MaskResizer makes of a dstW x dstH plane
bool HorizontalFirst(int srcW, int dstW, int stepX, int overlapX, int srcH, int dstH, int stepY, int overlapY) {
    const double sx = static_cast<double>(srcW) / (stepX * srcW + overlapX), sy = static_cast<double>(srcH) / (stepY * srcH + overlapY);
    for (int w : {std::min(64, dstW), (dstW - 1) % 64 + 1}) {
        for (int h : {std::min(64, dstH), (dstH - 1) % 64 + 1}) {
            const double xscale = static_cast<double>(w) / (w * sx), yscale = static_cast<double>(h) / (h * sy);
            const double hFirstCost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
            const double vFirstCost = std::max(yscale, 1.0) + yscale * std::max(xscale, 1.0) * 2.0;
            if (!(hFirstCost < vFirstCost))
                return false;
        }
    }
    return true;
}

void SetFpsDuration(VSFrame *frame, int64_t fpsNum, int64_t fpsDen, const VSAPI *vsapi) {
    VSMap *props = vsapi->getFramePropertiesRW(frame);
    vsapi->mapSetInt(props, "_DurationNum", fpsDen, maReplace);
    vsapi->mapSetInt(props, "_DurationDen", fpsNum, maReplace);
}

} // namespace

struct FlowData {
    VSNode *node = nullptr; // the clip
    VSNode *super = nullptr;
    VSNode *mvbw = nullptr; // vectors[0], delta off
    VSNode *mvfw = nullptr; // vectors[1], delta -off
    VSVideoInfo vi = {};    // the output's
    int clipFrames = 0;     // the clip's
    SuperLayout layout;
    std::string prefix, name;

    bool fps = false;      // FlowFPS, else FlowInter
    bool blend = true;     // blend the clip's frames where the vectors don't serve, else copy the first
    bool extraMask = true; // FlowInterExtra where the extra vectors serve (always for FlowInter)
    float mlInv = 0;       // 1 / ml
    int time256 = 0;       // FlowInter's time
    int64_t fa = 0, fb = 0; // FlowFPS: output frame n lies at input frame n * fa / fb
    int off = 0;           // the frames the vectors span
    // The vectors' grid: blk x blk blocks, step apart, overlapping by overlap, nbx x nby of them
    int blk = 0, overlap = 0, step = 0, nbx = 0, nby = 0;
    int thscd1 = 0;   // scaled
    int scdLimit = 0; // vectors with more blocks above thscd1 are at a scene change

    std::shared_ptr<VulkanContext> vc;
    VSGPUExecPool *pool = nullptr;
    VkPipeline prep = VK_NULL_HANDLE, pixels = VK_NULL_HANDLE;
    VSGPUBuffer *taps = nullptr; // the resize's taps (ResizeTaps), also bound where a frame has nothing
    VSVulkanBufferInfo tapsInfo = {};
    int colOff[2] = {}, rowOff[2] = {}; // where luma's and chroma's are in it
    std::unique_ptr<StageProfiler> profile; // MVGPU_PROFILE

    const VSAPI *vsapi;

    FlowData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~FlowData() {
        // The pool drains the GPU first, so nothing still uses what follows
        if (pool)
            vc->vkapi->freeGPUExecPool(pool);
        profile.reset();
        if (taps)
            vc->vkapi->destroyGPUBuffer(taps);
        vc.reset();
        vsapi->freeNode(mvfw);
        vsapi->freeNode(mvbw);
        vsapi->freeNode(super);
        vsapi->freeNode(node);
    }
};

static const VSFrame *VS_CC flowGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    FlowData *d = reinterpret_cast<FlowData *>(instanceData);

    // The frames the output frame lies between, nleft and nright = nleft + off, and its time
    // between them, as mvu computes them
    int nleft = n, time256 = d->time256;
    if (d->fps) {
        nleft = static_cast<int>(n * d->fa / d->fb);
        time256 = static_cast<int>((static_cast<double>(n) * d->fa / d->fb - nleft) * 256 + 0.5);
        if (d->off > 1)
            time256 = time256 / d->off;
    }
    const int nright = nleft + d->off, last = d->clipFrames - 1;
    // FlowFPS passes a frame through at time 0 or 256
    const int passThrough = !d->fps ? -1 : time256 == 0 ? nleft : time256 == 256 ? nright : -1;
    const bool loadVectors = nleft < d->clipFrames && nright < d->clipFrames;

    if (activationReason == arInitial) {
        if (passThrough >= 0) {
            vsapi->requestFrameFilter(std::min(passThrough, last), d->node, frameCtx);
            return nullptr;
        }
        if (loadVectors) {
            vsapi->requestFrameFilter(nright, d->mvfw, frameCtx);
            vsapi->requestFrameFilter(nleft, d->mvbw, frameCtx);
            if (d->extraMask) {
                vsapi->requestFrameFilter(nleft, d->mvfw, frameCtx);
                vsapi->requestFrameFilter(nright, d->mvbw, frameCtx);
            }
            vsapi->requestFrameFilter(nleft, d->super, frameCtx);
            vsapi->requestFrameFilter(nright, d->super, frameCtx);
        }
        vsapi->requestFrameFilter(std::min(nleft, last), d->node, frameCtx);
        if (d->blend)
            vsapi->requestFrameFilter(std::min(nright, last), d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VulkanContext &vc = *d->vc;
        const VSVULKANAPI *vkapi = vc.vkapi;
        const SuperLayout &L = d->layout;

        if (passThrough >= 0) {
            const VSFrame *src = vsapi->getFrameFilter(std::min(passThrough, last), d->node, frameCtx);
            VSFrame *dst = vsapi->copyFrame(src, core);
            vsapi->freeFrame(src);
            SetFpsDuration(dst, d->vi.fpsNum, d->vi.fpsDen, vsapi);
            return dst;
        }

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

        // The vectors that have vectors, mvu's HasMotionVectors; their scene change test is
        // flow_prep.comp's. F: the frame after's back to the frame before, B: the frame before's
        // forward, FF: the frame before's back, BB: the frame after's forward.
        const VSFrame *vec[4] = {};
        if (loadVectors) {
            vec[0] = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(nright, d->mvfw, frameCtx)), d->prefix, vsapi));
            vec[1] = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(nleft, d->mvbw, frameCtx)), d->prefix, vsapi));
            if (d->extraMask) {
                vec[2] = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(nleft, d->mvfw, frameCtx)), d->prefix, vsapi));
                vec[3] = hold(GetAnalysisVectors(hold(vsapi->getFrameFilter(nright, d->mvbw, frameCtx)), d->prefix, vsapi));
            }
        }
        const bool haveFB = vec[0] && vec[1], haveExtra = haveFB && vec[2] && vec[3];

        const VSFrame *src = hold(vsapi->getFrameFilter(std::min(nleft, last), d->node, frameCtx));
        if (!haveFB && !d->blend) {
            // The frame before, as mvu returns it
            if (!d->fps) {
                const VSFrame *f = vsapi->addFrameRef(src);
                release();
                return f;
            }
            VSFrame *f = vsapi->copyFrame(src, core);
            release();
            SetFpsDuration(f, d->vi.fpsNum, d->vi.fpsDen, vsapi);
            return f;
        }
        const VSFrame *ref = d->blend ? hold(vsapi->getFrameFilter(std::min(nright, last), d->node, frameCtx)) : src;

        SuperFrames before, after;
        if (haveFB) {
            if (!GetSuperFrames(hold(vsapi->getFrameFilter(nleft, d->super, frameCtx)), L, d->prefix, before, vsapi) ||
                !GetSuperFrames(hold(vsapi->getFrameFilter(nright, d->super, frameCtx)), L, d->prefix, after, vsapi))
                return fail("the super clip's frames lack their GPU planes; it must come from mvgpu.Super with the same prefix");
            for (const VSFrame *f : {before.luma, before.chroma, before.pyramid, after.luma, after.chroma, after.pyramid})
                hold(f);
        }

        dst = vkapi->newGPUVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
        if (!dst)
            return fail("failed to allocate the output frame");
        if (d->fps)
            SetFpsDuration(dst, d->vi.fpsNum, d->vi.fpsDen, vsapi);

        VSVulkanPlaneInfo clipSrc[3] = {}, clipRef[3] = {}, outPlanes[3] = {}, superPlanes[4] = {}, vecPlanes[4] = {};
        for (int p = 0; p < 3; ++p)
            if (vkapi->getGPUPlane(src, p, &clipSrc[p]) || vkapi->getGPUPlane(ref, p, &clipRef[p]) || vkapi->getGPUPlane(dst, p, &outPlanes[p]))
                return fail("the clip's frames aren't GPU resident");
        ptrdiff_t lumaStride = 0, chromaStride = 0, recBytes = 0;
        if (haveFB) {
            const VSFrame *planes[4] = {before.luma, before.chroma, after.luma, after.chroma};
            for (int i = 0; i < 4; ++i)
                if (vkapi->getGPUPlane(planes[i], 0, &superPlanes[i]))
                    return fail("the super's planes aren't GPU resident");
            lumaStride = vsapi->getStride(before.luma, 0);
            chromaStride = vsapi->getStride(before.chroma, 0);
            if (vsapi->getStride(after.luma, 0) != lumaStride || vsapi->getStride(after.chroma, 0) != chromaStride)
                return fail("the super frames' storage strides differ");
            for (int i = 0; i < 4; ++i) {
                if (!vec[i] || (i >= 2 && !haveExtra))
                    continue;
                if (vkapi->getGPUPlane(vec[i], 0, &vecPlanes[i]))
                    return fail("the vectors aren't GPU resident");
                const ptrdiff_t stride = vsapi->getStride(vec[i], 0);
                if (vsapi->getFrameWidth(vec[i], 0) != 4 * d->nbx || vsapi->getFrameHeight(vec[i], 0) != d->nby || stride % 16 || (recBytes && stride != recBytes))
                    return fail("a vector frame doesn't match the super's grid");
                recBytes = stride;
            }
        }
        for (int p = 0; p < 3; ++p)
            if (vsapi->getStride(ref, p) != vsapi->getStride(src, p))
                return fail("the clip's frames' strides differ");

        // Scratch: the scene change counts, then F's and B's occlusion masks
        const int nb = d->nbx * d->nby;
        const VkDeviceSize align = std::max<VkDeviceSize>(vc.limits.minStorageBufferOffsetAlignment, 16);
        const VkDeviceSize countBytes = (16 + align - 1) / align * align, maskBytes = static_cast<VkDeviceSize>(2) * nb * 4;
        char err[1024] = {};
        VSVulkanBufferInfo scratchInfo = {};
        if (haveFB) {
            scratch = vkapi->createGPUBuffer(core, countBytes + maskBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &scratchInfo, err, sizeof(err));
            if (!scratch)
                return fail(err);
        }
        ctx = vkapi->gpuExecAcquire(d->pool, err, sizeof(err));
        if (!ctx)
            return fail(err);
        if (scratch)
            vkapi->gpuExecUsesBuffer(ctx, scratch);
        vkapi->gpuExecReadsFrame(ctx, src);
        if (ref != src)
            vkapi->gpuExecReadsFrame(ctx, ref);
        if (haveFB) {
            for (const VSFrame *f : {before.luma, before.chroma, after.luma, after.chroma})
                vkapi->gpuExecReadsFrame(ctx, f);
            for (int i = 0; i < (haveExtra ? 4 : 2); ++i)
                vkapi->gpuExecReadsFrame(ctx, vec[i]);
        }
        for (int p = 0; p < 3; ++p)
            vkapi->gpuExecWritesPlane(ctx, dst, p);

        const VkCommandBuffer cmd = vkapi->gpuExecCommandBuffer(ctx);
        Recorder rec(vc, cmd, d->tapsInfo.buffer);
        const VkQueryPool queries = d->profile ? d->profile->Begin(vc, cmd) : VK_NULL_HANDLE;
        auto stamp = [&](int stage) {
            if (d->profile)
                d->profile->Stamp(vc, cmd, queries, stage);
        };

        const ptrdiff_t bytes = L.format.Bytes(); // strides in samples
        FlowParams pc = {};
        pc.nbx = d->nbx;
        pc.nby = d->nby;
        pc.recStride = static_cast<int32_t>(recBytes / 16);
        pc.flags = (haveFB ? kHaveFB : 0) | (haveExtra ? kHaveExtra : 0) | (d->blend ? kBlend : 0);
        pc.pad = L.pad;
        pc.padY = L.padY;
        pc.padc = L.padc;
        pc.padcY = L.padcY;
        pc.wp = static_cast<int32_t>(lumaStride / bytes);
        pc.hp = L.hp;
        pc.wc = static_cast<int32_t>(chromaStride / bytes);
        pc.hc = L.hc;
        pc.time256 = time256;
        pc.thscd1 = d->thscd1;
        pc.scdLimit = d->scdLimit;
        // MakeVectorOcclusionMask's, F's at the time, B's at 256 - time
        pc.occnormX = (80.0f * d->mlInv) / static_cast<float>(d->step * L.pel);
        pc.occnormY = pc.occnormX;
        pc.time4096FX = time256 * 16 / (d->step * L.pel);
        pc.time4096FY = pc.time4096FX;
        pc.time4096BX = (256 - time256) * 16 / (d->step * L.pel);
        pc.time4096BY = pc.time4096BX;

        rec.Bind(kFlTaps, d->tapsInfo.buffer);
        if (haveFB) {
            rec.Bind(kFlSrcLuma, superPlanes[0].buffer);
            rec.Bind(kFlSrcChroma, superPlanes[1].buffer);
            rec.Bind(kFlRefLuma, superPlanes[2].buffer);
            rec.Bind(kFlRefChroma, superPlanes[3].buffer);
            const int vecBindings[4] = {kFlVecF, kFlVecB, kFlVecFF, kFlVecBB};
            for (int i = 0; i < (haveExtra ? 4 : 2); ++i)
                rec.Bind(vecBindings[i], vecPlanes[i].buffer);
            rec.Bind(kFlCounts, scratchInfo.buffer, 0, countBytes);
            rec.Bind(kFlMasks, scratchInfo.buffer, countBytes, maskBytes);

            // The masks gather the largest value each block gets, from zero, as the counts do
            rec.Fill(scratchInfo.buffer, 0, countBytes + maskBytes, 0);
            rec.TransferToCompute();
            rec.Dispatch(d->prep, pc, static_cast<uint32_t>((nb + 63) / 64), 1);
            rec.ComputeBarrier();
        }
        stamp(1);
        for (int p = 0; p < 3; ++p) {
            pc.plane = p;
            pc.width = vsapi->getFrameWidth(dst, p);
            pc.height = vsapi->getFrameHeight(dst, p);
            pc.outStride = static_cast<int32_t>(vsapi->getStride(dst, p) / bytes);
            pc.clipStride = static_cast<int32_t>(vsapi->getStride(src, p) / bytes);
            pc.colOff = d->colOff[p ? 1 : 0];
            pc.rowOff = d->rowOff[p ? 1 : 0];
            rec.Bind(kFlClipSrc, clipSrc[p].buffer);
            rec.Bind(kFlClipRef, clipRef[p].buffer);
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

// FlowFPS takes userData 1, FlowInter null
static void VS_CC flowCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<FlowData> d = std::make_unique<FlowData>(vsapi);
    d->fps = userData != nullptr;
    d->name = d->fps ? "FlowFPS" : "FlowInter";

    try {
        int err;

        float time = vsapi->mapGetFloatSaturated(in, "time", 0, &err);
        if (err)
            time = 50.0f;
        int64_t num = vsapi->mapGetInt(in, "num", 0, &err);
        if (err)
            num = 25;
        int64_t den = vsapi->mapGetInt(in, "den", 0, &err);
        if (err)
            den = 1;
        if (d->fps) {
            d->extraMask = !!vsapi->mapGetIntSaturated(in, "extramask", 0, &err);
            if (err)
                d->extraMask = true;
        }
        float ml = vsapi->mapGetFloatSaturated(in, "ml", 0, &err);
        if (err)
            ml = 100.0f;
        d->blend = !!vsapi->mapGetInt(in, "blend", 0, &err);
        if (err)
            d->blend = true;
        int64_t thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
        if (err)
            thscd1 = MV_DEFAULT_SCD1;
        float thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
        if (err)
            thscd2 = MV_DEFAULT_SCD2;
        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        d->prefix = prefix ? prefix : DEFAULT_MVGPUTENSILS_PREFIX;

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);
        d->clipFrames = d->vi.numFrames;
        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);

        if (!d->fps && (!std::isfinite(time) || time < 0.0f || time > 100.0f))
            throw std::runtime_error("time must be between 0 and 100%");
        if (!std::isfinite(ml) || ml <= 0.0f)
            throw std::runtime_error("ml must be a finite value greater than 0");
        d->mlInv = 1.0f / ml;
        d->time256 = static_cast<int>(time * 256.0f / 100.0f);

        d->layout = ImportSuperLayout(d->super, d->prefix, vsapi);
        const SuperLayout &L = d->layout;
        if (const std::string unsupported = L.Unsupported(SuperLayout::Use::Compensation); !unsupported.empty())
            throw std::runtime_error(unsupported);
        if (!vsh::isConstantVideoFormat(&d->vi) || d->vi.format.colorFamily != cfYUV || d->vi.format.sampleType != stInteger || d->vi.format.bitsPerSample != L.format.bits ||
            (1 << d->vi.format.subSamplingW) != L.format.xr || (1 << d->vi.format.subSamplingH) != L.format.yr || d->vi.width != L.width || d->vi.height != L.height)
            throw std::runtime_error("super clip is not compatible with source clip");

        if (vsapi->mapNumElements(in, "vectors") != 2)
            throw std::runtime_error("vectors must have exactly 2 elements");
        d->mvbw = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        d->mvfw = vsapi->mapGetNode(in, "vectors", 1, nullptr);

        // The vectors: mvgpu.Analyse's of a super with the same level 0 as this one but for the bit
        // depth (mvu's IsCompatibleWithAnalysis), both on one grid (IsCompatible), with opposite
        // deltas. The grid is theirs, as in mvu; the super only supplies the planes.
        VectorInfo info[2];
        VSNode *nodes[2] = {d->mvfw, d->mvbw};
        for (int i = 0; i < 2; ++i) {
            const VectorInfo &v = info[i] = ReadVectorInfo(nodes[i], d->prefix, vsapi);
            const SuperLayout analysed = ImportSuperLayout(nodes[i], d->prefix, vsapi);
            if (i == 0 && (!SameGeometry(analysed, L) || v.width != L.aw || v.height != L.ah || v.realWidth != L.width || v.realHeight != L.height ||
                           v.hpad != L.pad || v.vpad != L.padY || v.pel != L.pel || (v.chroma && (v.xRatio != L.format.xr || v.yRatio != L.format.yr))))
                throw std::runtime_error("wrong source or super clip frame size");
            if (v.blkX != analysed.blk || v.blkY != analysed.blk || v.overlapX != analysed.overlap || v.overlapY != analysed.overlap || v.nbx != analysed.nbx ||
                v.nby != analysed.nby)
                throw std::runtime_error("the vectors' grid isn't their super's; they must come from mvgpu.Analyse");
        }
        const VectorInfo &fw = info[0], &bw = info[1];
        if (fw.width != bw.width || fw.height != bw.height || fw.realWidth != bw.realWidth || fw.realHeight != bw.realHeight || fw.hpad != bw.hpad ||
            fw.vpad != bw.vpad || fw.pel != bw.pel || fw.bits != bw.bits || fw.chroma != bw.chroma || fw.xRatio != bw.xRatio || fw.yRatio != bw.yRatio ||
            fw.blkX != bw.blkX || fw.blkY != bw.blkY || fw.overlapX != bw.overlapX || fw.overlapY != bw.overlapY || fw.nbx != bw.nbx || fw.nby != bw.nby ||
            bw.delta != -fw.delta || fw.delta > 0 || bw.delta < 0)
            throw std::runtime_error("mvfw and mvbw must be compatible with each other and have opposite sign delta");
        d->off = -fw.delta;
        d->blk = fw.blkX;
        d->overlap = fw.overlapX;
        d->step = d->blk - d->overlap;
        d->nbx = fw.nbx;
        d->nby = fw.nby;

        // mvu's ScaleThSCD and GetThSCDScaleFactor, for mvfw's block size, chroma and bit depth
        constexpr int maxSAD = 8 * 8 * 255;
        if (thscd1 < 0 || thscd1 > maxSAD)
            throw std::runtime_error("thscd1 must be between 0 and " + std::to_string(maxSAD));
        if (!std::isfinite(thscd2) || thscd2 < 0.0f || thscd2 > 100.0f)
            throw std::runtime_error("thscd2 must be a percentage between 0 and 100");
        const double scale = static_cast<double>(fw.blkX * fw.blkY) / (8.0 * 8.0) * (fw.chroma ? (1.0 + 2.0 / (fw.xRatio * fw.yRatio)) : 1.0) *
                             (((1 << std::min(16, fw.bits)) - 1) / 255.0);
        d->thscd1 = static_cast<int>(static_cast<int64_t>(thscd1 * scale + 0.5));
        const float blocksTh2 = static_cast<float>(static_cast<double>(thscd2) * fw.nbx * fw.nby / 100.0);
        d->scdLimit = static_cast<int>(std::floor(blocksTh2)); // a count above the float limit is above its floor

        if (d->fps) {
            if (d->vi.fpsNum == 0 || d->vi.fpsDen == 0)
                throw std::runtime_error("input clip must have known framerate");
            if (num < 0 || den < 0)
                throw std::runtime_error("num and den must not be negative");
            d->fa = d->vi.fpsNum;
            d->fb = d->vi.fpsDen;
            if (num != 0 && den != 0) {
                d->vi.fpsNum = num;
                d->vi.fpsDen = den;
            } else {
                d->vi.fpsNum *= 2;
            }
            vsh::reduceRational(&d->vi.fpsNum, &d->vi.fpsDen);
            vsh::muldivRational(&d->fa, &d->fb, d->vi.fpsDen, d->vi.fpsNum);
            d->vi.numFrames = static_cast<int>(d->vi.numFrames * d->fb / d->fa);
        }

        d->vc = VulkanContext::Get(core, vsapi);
        VulkanContext &vc = *d->vc;
        // Specialization constant 6: chroma's subsampling, bit 0 horizontal, bit 1 vertical
        const int chromaLog = (L.format.xr > 1 ? 1 : 0) | (L.format.yr > 1 ? 2 : 0);
        d->prep = vc.Pipeline(Kernel::FlowPrep, 0, L.pel, chromaLog);
        d->pixels = vc.Pipeline(Kernel::FlowInter, 0, L.pel, chromaLog | (L.format.bits > 8 ? 4 : 0));
        if (static_cast<VkDeviceSize>(2) * d->nbx * d->nby * 4 > vc.limits.maxStorageBufferRange)
            throw std::runtime_error("the frame is too large for the device's storage buffers");

        // The resize's taps: luma's columns and rows, then chroma's, its blocks luma's shifted by the
        // subsampling, as mvu's MaskResizer has them (4:4:4 chroma takes luma's)
        std::vector<int32_t> tables;
        for (int c = 0; c < (L.format.xr > 1 || L.format.yr > 1 ? 2 : 1); ++c) {
            const int lx = c ? d->vi.format.subSamplingW : 0, ly = c ? d->vi.format.subSamplingH : 0;
            const int stepX = (d->blk >> lx) - (d->overlap >> lx), stepY = (d->blk >> ly) - (d->overlap >> ly);
            const int w = d->vi.width >> lx, h = d->vi.height >> ly;
            if (!HorizontalFirst(d->nbx, w, stepX, d->overlap >> lx, d->nby, h, stepY, d->overlap >> ly))
                throw std::runtime_error("zimg would resize the masks vertically first, which isn't implemented");
            d->colOff[c] = static_cast<int>(tables.size());
            const std::vector<int32_t> cols = ResizeTaps(d->nbx, w, stepX, d->overlap >> lx);
            tables.insert(tables.end(), cols.begin(), cols.end());
            d->rowOff[c] = static_cast<int>(tables.size());
            const std::vector<int32_t> rows = ResizeTaps(d->nby, h, stepY, d->overlap >> ly);
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
            d->profile = std::make_unique<StageProfiler>(d->name + " (pel " + std::to_string(L.pel) + ")", std::vector<std::string>{"masks", "pixels"});
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, (d->name + ": " + e.what()).c_str());
        return;
    }

    VSFilterDependency deps[4] = {
        {d->node, rpGeneral},
        {d->super, rpGeneral},
        {d->mvbw, rpGeneral},
        {d->mvfw, rpGeneral},
    };
    vsapi->createVideoFilterEx(out, d->name.c_str(), &d->vi, flowGetFrame, filterFree<FlowData>, fmParallel, ffGPUOutput, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void flowRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("FlowInter",
                             "clip:vnode:gpu;"
                             "super:vnode:gpu;"
                             "vectors:vnode[]:gpu;"
                             "time:float:opt;"
                             "ml:float:opt;"
                             "blend:int:opt;"
                             "thscd1:int:opt;"
                             "thscd2:float:opt;"
                             "prefix:data:opt;",
                             "clip:vnode:gpu;", flowCreate, nullptr, plugin);
    vspapi->registerFunction("FlowFPS",
                             "clip:vnode:gpu;"
                             "super:vnode:gpu;"
                             "vectors:vnode[]:gpu;"
                             "num:int:opt;"
                             "den:int:opt;"
                             "extramask:int:opt;"
                             "ml:float:opt;"
                             "blend:int:opt;"
                             "thscd1:int:opt;"
                             "thscd2:float:opt;"
                             "prefix:data:opt;",
                             "clip:vnode:gpu;", flowCreate, reinterpret_cast<void *>(1), plugin);
}
