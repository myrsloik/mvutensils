#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "SuperPyramid.h"
#include "Common.h"


struct SuperData {
    VSNode *node = nullptr;
    VSNode *pelclip = nullptr;

    VSVideoInfo vi;

    int nHPad;
    int nVPad;
    int nPel;
    int nLevels;
    SharpParam sharp;
    RFilterParam rfilter;

    int nBlkSizeX;
    int nBlkSizeY;
    int nOverlapX;
    int nOverlapY;

    bool usePelClip;

    std::string prefix;

    const VSAPI *vsapi;

    SuperData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~SuperData() {
        vsapi->freeNode(node);
        vsapi->freeNode(pelclip);
    }
};

static const VSFrame *VS_CC superGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    SuperData *d = reinterpret_cast<SuperData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (d->usePelClip)
            vsapi->requestFrameFilter(n, d->pelclip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        VSFrame *dst = nullptr;
        const VSFrame *src = nullptr;
        const VSFrame *srcPel = nullptr;

        try {
            src = vsapi->getFrameFilter(n, d->node, frameCtx);
            dst = vsapi->copyFrame(src, core);

            if (d->usePelClip) {
                srcPel = vsapi->getFrameFilter(n, d->pelclip, frameCtx);
                FramePyramid pyramid(src, d->nLevels, d->nBlkSizeX, d->nBlkSizeY, d->nOverlapX, d->nOverlapY, d->nHPad, d->nVPad, d->rfilter, d->nPel, srcPel, core, vsapi);
                vsapi->freeFrame(srcPel);
                srcPel = nullptr;
                pyramid.ExportFrameData(dst, d->prefix);
            } else {
                FramePyramid pyramid(src, d->nLevels, d->nBlkSizeX, d->nBlkSizeY, d->nOverlapX, d->nOverlapY, d->nHPad, d->nVPad, d->rfilter, d->nPel, d->sharp, core, vsapi);
                pyramid.ExportFrameData(dst, d->prefix);
            }

            vsapi->freeFrame(src);
            src = nullptr;

            return dst;
        } catch (const std::exception &e) {
            vsapi->freeFrame(dst);
            vsapi->freeFrame(src);
            vsapi->freeFrame(srcPel);
            vsapi->setFilterError(("Super: " + std::string(e.what())).c_str(), frameCtx);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC superCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<SuperData> d = std::make_unique<SuperData>(vsapi);
    int err;

    try {
        GetHVPairArgument(d->nHPad, d->nVPad, "pad", 16, 16, in, vsapi);

        if (d->nHPad <= 0 || d->nVPad <= 0)
            throw std::runtime_error("pad must be positive");

        d->nPel = vsapi->mapGetIntSaturated(in, "pel", 0, &err);
        if (err)
            d->nPel = 2;

        d->nLevels = !!vsapi->mapGetIntSaturated(in, "onelevel", 0, &err) ? 1 : 0;
        if (err)
            d->nLevels = 0;

        d->sharp = static_cast<SharpParam>(vsapi->mapGetIntSaturated(in, "sharp", 0, &err));
        if (err)
            d->sharp = SharpParam::Wiener;

        d->rfilter = static_cast<RFilterParam>(vsapi->mapGetIntSaturated(in, "rfilter", 0, &err));
        if (err)
            d->rfilter = RFilterParam::Bilinear;

        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        if (prefix)
            d->prefix = prefix;
        else
            d->prefix = DEFAULT_MVUTENSILS_PREFIX;

        if ((d->nPel != 1) && (d->nPel != 2) && (d->nPel != 4))
            throw std::runtime_error("pel must be 1, 2, or 4");

        if (d->sharp != SharpParam::Bilinear && d->sharp != SharpParam::Bicubic && d->sharp != SharpParam::Wiener)
            throw std::runtime_error("sharp must be between 0 and 2");

        if (d->rfilter != RFilterParam::Simple && d->rfilter != RFilterParam::Bilinear && d->rfilter != RFilterParam::Cubic)
            throw std::runtime_error("rfilter must be between 0 and 2");

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);

        if (!vsh::isConstantVideoFormat(&d->vi) || (d->vi.format.bitsPerSample > 16 && d->vi.format.sampleType == stInteger) || (d->vi.format.bitsPerSample != 32 && d->vi.format.sampleType == stFloat) ||
            d->vi.format.subSamplingW > 1 || d->vi.format.subSamplingH > 1 || (d->vi.format.colorFamily != cfYUV && d->vi.format.colorFamily != cfGray))
            throw std::runtime_error("input clip must be GRAY, YUV420, YUV422, YUV440, or YUV444, up to 16 bits integer or 32 bit float, with constant dimensions");

        int xRatioUV = 1 << d->vi.format.subSamplingW;
        int yRatioUV = 1 << d->vi.format.subSamplingH;

        // The chroma planes are padded by pad / ratio. Rounded down, a luma vector reaching -pad would map below the
        // chroma padding in the consumers that floor-shift vectors to chroma.
        if (d->nHPad % xRatioUV || d->nVPad % yRatioUV)
            throw std::runtime_error("pad must be divisible by the chroma subsampling: the horizontal pad must be even for 4:2:0 and 4:2:2, the vertical pad for 4:2:0 and 4:4:0");

        GetHVPairArgument(d->nBlkSizeX, d->nBlkSizeY, "blksize", 8, 8, in, vsapi);
        GetHVPairArgument(d->nOverlapX, d->nOverlapY, "overlap", 0, 0, in, vsapi);

        CheckBlkSize(d->nBlkSizeX, d->nBlkSizeY, d->nOverlapX, d->nOverlapY, d->vi.format.subSamplingW, d->vi.format.subSamplingH);

        int nLevelsMax = FramePyramid::GetMaxLevelsForBlockSize(d->vi.width, d->vi.height, xRatioUV, yRatioUV, d->nBlkSizeX, d->nBlkSizeY, d->nOverlapX, d->nOverlapY, d->nHPad, d->nVPad);

        if (d->nLevels <= 0 || d->nLevels > nLevelsMax)
            d->nLevels = nLevelsMax;

        d->pelclip = vsapi->mapGetNode(in, "pelclip", 0, &err);
        const VSVideoInfo *pelvi = d->pelclip ? vsapi->getVideoInfo(d->pelclip) : nullptr;

        if (pelvi && (!vsh::isConstantVideoFormat(pelvi) || !vsh::isSameVideoFormat(&pelvi->format, &d->vi.format)))
            throw std::runtime_error("pelclip must have the same format as the input clip, and it must have constant dimensions");

        d->usePelClip = false;
        if (pelvi && (d->nPel >= 2)) {
            if ((pelvi->width != d->vi.width * d->nPel) ||
                (pelvi->height != d->vi.height * d->nPel))
                throw std::runtime_error("pelclip's dimensions must be pel times the input clip's dimensions");

            if (pelvi->numFrames != d->vi.numFrames)
                throw std::runtime_error("pelclip's length must match the input clip's length");

            d->usePelClip = true;
        }


    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("Super: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[2] = { 
        { d->node, rpStrictSpatial },
        { d->pelclip, rpStrictSpatial }
    };

    vsapi->createVideoFilter(out, "Super", &d->vi, superGetFrame, filterFree<SuperData>, fmParallel, deps, d->usePelClip ? 2 : 1, d.get(), core);
    d.release();
}

void superRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Super", 
                 "clip:vnode;"
                 "blksize:int[];"
                 "overlap:int[];"
                 "pad:int[]:opt;"
                 "onelevel:int:opt;"
                 "sharp:int:opt;"
                 "rfilter:int:opt;"
                 "pel:int:opt;"
                 "pelclip:vnode:opt;"
                 "prefix:data:opt;",
                 "clip:vnode;",
                 superCreate, nullptr, plugin);
}
