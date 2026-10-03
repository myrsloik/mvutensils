// Make a motion compensate temporal denoiser
// Copyright(c)2006 A.G.Balakhnin aka Fizick
// See legal notice in Copying.txt for more information

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .

#include <cmath>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "Common.h"
#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"
#include "MaskResize.h"
#include "FlowShared.h"

struct FlowFPSData {
    VSNode *node = nullptr;
    VSVideoInfo vi;
    const VSVideoInfo *oldvi = nullptr;

    VSNode *super = nullptr;
    VSNode *mvbw = nullptr;
    VSNode *mvfw = nullptr;

    bool extraMask;
    float ml;
    bool blend;
    int64_t thscd1;
    float thscd2;

    int64_t fa;
    int64_t fb;

    int deltaFrame;

    MaskResizer maskResizerFull;
    MaskResizer maskResizerSubSampled;

    AnalysisGeometry mvfwGeometry, mvbwGeometry;
    SuperGeometry superGeometry;

    std::string prefix;

    const VSAPI *vsapi;

    FlowFPSData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~FlowFPSData() {
        vsapi->freeNode(node);
        vsapi->freeNode(super);
        vsapi->freeNode(mvbw);
        vsapi->freeNode(mvfw);
    }
};

static void SetFpsDuration(VSFrame *frame, int64_t fpsNum, int64_t fpsDen, const VSAPI *vsapi) noexcept {
    VSMap *props = vsapi->getFramePropertiesRW(frame);
    vsapi->mapSetInt(props, "_DurationNum", fpsDen, maReplace);
    vsapi->mapSetInt(props, "_DurationDen", fpsNum, maReplace);
}

static const VSFrame *PassThroughWithFpsDuration(const VSFrame *src, int64_t fpsNum, int64_t fpsDen, VSCore *core, const VSAPI *vsapi) noexcept {
    VSFrame *dst = vsapi->copyFrame(src, core);
    vsapi->freeFrame(src);
    SetFpsDuration(dst, fpsNum, fpsDen, vsapi);
    return dst;
}

template<typename PixelType>
static const VSFrame *VS_CC flowfpsGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    const FlowFPSData *d = reinterpret_cast<FlowFPSData *>(instanceData);

    if (activationReason == arInitial) {
        int off = -d->deltaFrame;

        int nleft = (int)(n * d->fa / d->fb);
        int nright = nleft + off;

        int time256 = (int)(((double)n * d->fa / d->fb - nleft) * 256 + 0.5);
        if (off > 1)
            time256 = time256 / off;

        if (time256 == 0) {
            vsapi->requestFrameFilter(std::min(nleft, d->oldvi->numFrames - 1), d->node, frameCtx);
            return nullptr;
        } else if (time256 == 256) {
            vsapi->requestFrameFilter(std::min(nright, d->oldvi->numFrames - 1), d->node, frameCtx);
            return nullptr;
        }

        if (nleft < d->oldvi->numFrames && nright < d->oldvi->numFrames) { // for the good estimation case
            if (d->extraMask)
                vsapi->requestFrameFilter(nleft, d->mvfw, frameCtx); // requests nleft - off, nleft
            vsapi->requestFrameFilter(nright, d->mvfw, frameCtx);    // requests nleft, nleft + off
            vsapi->requestFrameFilter(nleft, d->mvbw, frameCtx);     // requests nleft, nleft + off
            if (d->extraMask)
                vsapi->requestFrameFilter(nright, d->mvbw, frameCtx); // requests nleft + off, nleft + off + off

            vsapi->requestFrameFilter(nleft, d->super, frameCtx);
            vsapi->requestFrameFilter(nright, d->super, frameCtx);
        }

        vsapi->requestFrameFilter(std::min(nleft, d->oldvi->numFrames - 1), d->node, frameCtx);

        if (d->blend)
            vsapi->requestFrameFilter(std::min(nright, d->oldvi->numFrames - 1), d->node, frameCtx);

    } else if (activationReason == arAllFramesReady) {
        VSFrame *dst = nullptr;

        try {
            int nleft = (int)(n * d->fa / d->fb);
            // intermediate product may be very large! Now I know how to multiply int64
            int time256 = (int)(((double)n * d->fa / d->fb - nleft) * 256 + 0.5);

            int off = -d->deltaFrame; // integer offset of reference frame
            // usually off must be = 1
            if (off > 1)
                time256 = time256 / off;

            int nright = nleft + off;

            if (time256 == 0) {
                return PassThroughWithFpsDuration(vsapi->getFrameFilter(std::min(nleft, d->oldvi->numFrames - 1), d->node, frameCtx), d->vi.fpsNum, d->vi.fpsDen, core, vsapi); // simply left
            } else if (time256 == 256) {
                return PassThroughWithFpsDuration(vsapi->getFrameFilter(std::min(nright, d->oldvi->numFrames - 1), d->node, frameCtx), d->vi.fpsNum, d->vi.fpsDen, core, vsapi); // simply right
            }

            bool vectorsLoadFrame = (nleft < d->oldvi->numFrames && nright < d->oldvi->numFrames);

            MotionBlockPyramid vectorsF(vectorsLoadFrame ? vsapi->getFrameFilter(nright, d->mvfw, frameCtx) : nullptr, 1, d->prefix, vsapi, d->mvfwGeometry);
            MotionBlockPyramid vectorsB(vectorsLoadFrame ? vsapi->getFrameFilter(nleft, d->mvbw, frameCtx) : nullptr, 1, d->prefix, vsapi, d->mvbwGeometry);

            if (vectorsB.IsUsable(d->thscd1, d->thscd2) && vectorsF.IsUsable(d->thscd1, d->thscd2)) {
                // If both are usable, that means both nleft and nright are less than oldvi->numFrames. Thus there is no need to check nleft and nright here.
                FramePyramid src(vsapi->getFrameFilter(nleft, d->super, frameCtx), 1, d->prefix, vsapi, d->superGeometry);
                FramePyramid ref(vsapi->getFrameFilter(nright, d->super, frameCtx), 1, d->prefix, vsapi, d->superGeometry);
                const VSFrame *dstPropSrc = vsapi->getFrameFilter(nleft, d->node, frameCtx);
                dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, dstPropSrc, core);
                vsapi->freeFrame(dstPropSrc);
                SetFpsDuration(dst, d->vi.fpsNum, d->vi.fpsDen, vsapi);

                auto SmallB = vectorsB.MakeSmallVectorMasks();
                auto SmallF = vectorsF.MakeSmallVectorMasks();

                auto MaskSmallB = vectorsB.MakeVectorOcclusionMask<uint16_t>(d->ml, 1.0f, (256 - time256), true);
                auto MaskSmallF = vectorsF.MakeVectorOcclusionMask<uint16_t>(d->ml, 1.0f, time256, true);

                auto tmp = MaskResizer::GetTmpBuffer(std::max(d->maskResizerFull.tmpSize, d->maskResizerSubSampled.tmpSize));

                auto [dstTileMaskF, dstTileMaskB, dstTileXF, dstTileYF, dstTileXB, dstTileYB] = MaskResizer::GetTileBuffers<6>();

                auto bufMaskF = MaskResizer::MakeBufferPair(MaskSmallF->mask, MaskSmallF->stride, dstTileMaskF.get());
                auto bufMaskB = MaskResizer::MakeBufferPair(MaskSmallB->mask, MaskSmallB->stride, dstTileMaskB.get());

                auto bufSmallFX = MaskResizer::MakeBufferPair(SmallF->VXSmallY, SmallF->pitchVSmallY, dstTileXF.get());
                auto bufSmallFY = MaskResizer::MakeBufferPair(SmallF->VYSmallY, SmallF->pitchVSmallY, dstTileYF.get());
                auto bufSmallBX = MaskResizer::MakeBufferPair(SmallB->VXSmallY, SmallB->pitchVSmallY, dstTileXB.get());
                auto bufSmallBY = MaskResizer::MakeBufferPair(SmallB->VYSmallY, SmallB->pitchVSmallY, dstTileYB.get());

                MotionBlockPyramid vectorsFF(d->extraMask ? vsapi->getFrameFilter(nleft, d->mvfw, frameCtx) : nullptr, 1, d->prefix, vsapi, d->mvfwGeometry);
                MotionBlockPyramid vectorsBB(d->extraMask ? vsapi->getFrameFilter(nright, d->mvbw, frameCtx) : nullptr, 1, d->prefix, vsapi, d->mvbwGeometry);

                if (d->extraMask && vectorsBB.IsUsable(d->thscd1, d->thscd2) && vectorsFF.IsUsable(d->thscd1, d->thscd2)) {
                    // get vector mask from extra frames
                    auto SmallBB = vectorsBB.MakeSmallVectorMasks();
                    auto SmallFF = vectorsFF.MakeSmallVectorMasks();

                    auto [dstTileXFF, dstTileYFF, dstTileXBB, dstTileYBB] = MaskResizer::GetTileBuffers<4>();

                    auto bufSmallFFX = MaskResizer::MakeBufferPair(SmallFF->VXSmallY, SmallFF->pitchVSmallY, dstTileXFF.get());
                    auto bufSmallFFY = MaskResizer::MakeBufferPair(SmallFF->VYSmallY, SmallFF->pitchVSmallY, dstTileYFF.get());
                    auto bufSmallBBX = MaskResizer::MakeBufferPair(SmallBB->VXSmallY, SmallBB->pitchVSmallY, dstTileXBB.get());
                    auto bufSmallBBY = MaskResizer::MakeBufferPair(SmallBB->VYSmallY, SmallBB->pitchVSmallY, dstTileYBB.get());

                    ptrdiff_t dstStrideY = vsapi->getStride(dst, 0);
                    uint8_t *dstPtrY = vsapi->getWritePtr(dst, 0);

                    for (auto &tile : d->maskResizerFull.tiles) {
                        tile.Process(tmp.get(), bufMaskF, bufMaskB, bufSmallFX, bufSmallFY, bufSmallBX, bufSmallBY, bufSmallFFX, bufSmallFFY, bufSmallBBX, bufSmallBBY);

                        FlowInterExtra<PixelType>(dstPtrY + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideY, dstStrideY,
                             ref.GetLevel(0).planes[0], src.GetLevel(0).planes[0],
                             dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                             dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                             tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, time256,
                             dstTileXBB.get(), dstTileXFF.get(), dstTileYBB.get(), dstTileYFF.get());
                    }

                    if (d->vi.format.numPlanes == 3) {
                        SmallF->AdjustSmallVectorMaskSubSampling(vectorsF.nBlkX, vectorsF.nBlkY, d->vi.format.subSamplingW, d->vi.format.subSamplingH);
                        SmallB->AdjustSmallVectorMaskSubSampling(vectorsB.nBlkX, vectorsB.nBlkY, d->vi.format.subSamplingW, d->vi.format.subSamplingH);
                        SmallFF->AdjustSmallVectorMaskSubSampling(vectorsFF.nBlkX, vectorsFF.nBlkY, d->vi.format.subSamplingW, d->vi.format.subSamplingH);
                        SmallBB->AdjustSmallVectorMaskSubSampling(vectorsBB.nBlkX, vectorsBB.nBlkY, d->vi.format.subSamplingW, d->vi.format.subSamplingH);

                        ptrdiff_t dstStrideU = vsapi->getStride(dst, 1);
                        ptrdiff_t dstStrideV = vsapi->getStride(dst, 2);
                        uint8_t *dstPtrU = vsapi->getWritePtr(dst, 1);
                        uint8_t *dstPtrV = vsapi->getWritePtr(dst, 2);

                        for (auto &tile : (d->vi.format.subSamplingH > 0 || d->vi.format.subSamplingW > 0) ? d->maskResizerSubSampled.tiles : d->maskResizerFull.tiles) {
                            tile.Process(tmp.get(), bufMaskF, bufMaskB, bufSmallFX, bufSmallFY, bufSmallBX, bufSmallBY, bufSmallFFX, bufSmallFFY, bufSmallBBX, bufSmallBBY);

                            FlowInterExtra<PixelType>(dstPtrU + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideU, dstStrideU,
                                 ref.GetLevel(0).planes[1], src.GetLevel(0).planes[1],
                                 dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                                 dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                                 tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, time256,
                                 dstTileXBB.get(), dstTileXFF.get(), dstTileYBB.get(), dstTileYFF.get());

                            FlowInterExtra<PixelType>(dstPtrV + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideV, dstStrideV,
                                 ref.GetLevel(0).planes[2], src.GetLevel(0).planes[2],
                                 dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                                 dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                                 tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, time256,
                                 dstTileXBB.get(), dstTileXFF.get(), dstTileYBB.get(), dstTileYFF.get());
                        }
                    }
                } else { // old method without extra frames
                    ptrdiff_t dstStrideY = vsapi->getStride(dst, 0);
                    uint8_t *dstPtrY = vsapi->getWritePtr(dst, 0);

                    for (auto &tile : d->maskResizerFull.tiles) {
                        tile.Process(tmp.get(), bufMaskF, bufMaskB, bufSmallFX, bufSmallFY, bufSmallBX, bufSmallBY);

                        FlowInter<PixelType>(dstPtrY + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideY, dstStrideY,
                             ref.GetLevel(0).planes[0], src.GetLevel(0).planes[0],
                             dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                             dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                             tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, time256);

                    }

                    if (d->vi.format.numPlanes == 3) {
                        SmallF->AdjustSmallVectorMaskSubSampling(vectorsF.nBlkX, vectorsF.nBlkY, d->vi.format.subSamplingW, d->vi.format.subSamplingH);
                        SmallB->AdjustSmallVectorMaskSubSampling(vectorsB.nBlkX, vectorsB.nBlkY, d->vi.format.subSamplingW, d->vi.format.subSamplingH);

                        ptrdiff_t dstStrideU = vsapi->getStride(dst, 1);
                        ptrdiff_t dstStrideV = vsapi->getStride(dst, 2);
                        uint8_t *dstPtrU = vsapi->getWritePtr(dst, 1);
                        uint8_t *dstPtrV = vsapi->getWritePtr(dst, 2);

                        for (auto &tile : (d->vi.format.subSamplingH > 0 || d->vi.format.subSamplingW > 0) ? d->maskResizerSubSampled.tiles : d->maskResizerFull.tiles) {
                            tile.Process(tmp.get(), bufMaskF, bufMaskB, bufSmallFX, bufSmallFY, bufSmallBX, bufSmallBY);

                            FlowInter<PixelType>(dstPtrU + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideU, dstStrideU,
                                    ref.GetLevel(0).planes[1], src.GetLevel(0).planes[1],
                                    dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                                    dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                                    tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, time256);

                            FlowInter<PixelType>(dstPtrV + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideV, dstStrideV,
                                    ref.GetLevel(0).planes[2], src.GetLevel(0).planes[2],
                                    dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                                    dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                                    tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, time256);
                        }
                    }
                }

                return dst;
            } else {
                if (d->blend) {
                    const VSFrame *src = vsapi->getFrameFilter(std::min(nleft, d->oldvi->numFrames - 1), d->node, frameCtx);
                    const VSFrame *ref = vsapi->getFrameFilter(std::min(nright, d->oldvi->numFrames - 1), d->node, frameCtx);
                    VSFrame *dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
                    SetFpsDuration(dst, d->vi.fpsNum, d->vi.fpsDen, vsapi);

                    for (int plane = 0; plane < d->vi.format.numPlanes; plane++)
                        Blend<PixelType>(vsapi->getWritePtr(dst, plane), vsapi->getReadPtr(src, plane), vsapi->getReadPtr(ref, plane), vsapi->getFrameHeight(dst, plane), vsapi->getFrameWidth(dst, plane), vsapi->getStride(dst, plane), time256);

                    vsapi->freeFrame(src);
                    vsapi->freeFrame(ref);

                    return dst;
                } else {
                    return PassThroughWithFpsDuration(vsapi->getFrameFilter(std::min(nleft, d->oldvi->numFrames - 1), d->node, frameCtx), d->vi.fpsNum, d->vi.fpsDen, core, vsapi);
                }
            }
        } catch (const std::exception &e) {
            vsapi->freeFrame(dst);
            vsapi->setFilterError((std::string("FlowFPS: ") + e.what()).c_str(), frameCtx);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC flowfpsCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<FlowFPSData> d = std::make_unique<FlowFPSData>(vsapi);
    int err;

    int64_t num = vsapi->mapGetInt(in, "num", 0, &err);
    if (err)
        num = 25;

    int64_t den = vsapi->mapGetInt(in, "den", 0, &err);
    if (err)
        den = 1;

    d->extraMask = !!vsapi->mapGetIntSaturated(in, "extramask", 0, &err);
    if (err)
        d->extraMask = true;

    d->ml = vsapi->mapGetFloatSaturated(in, "ml", 0, &err);
    if (err)
        d->ml = 100.0f;

    d->blend = !!vsapi->mapGetInt(in, "blend", 0, &err);
    if (err)
        d->blend = 1;

    d->thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
    if (err)
        d->thscd1 = MV_DEFAULT_SCD1;

    d->thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
    if (err)
        d->thscd2 = MV_DEFAULT_SCD2;

    const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
    if (prefix)
        d->prefix = prefix;
    else
        d->prefix = DEFAULT_MVUTENSILS_PREFIX;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->oldvi = vsapi->getVideoInfo(d->node);
    d->vi = *d->oldvi;

    try {
        if (!std::isfinite(d->ml) || d->ml <= 0.0)
            throw std::runtime_error("ml must be a finite value greater than 0");

        d->ml = 1.0f / d->ml;

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);

        FramePyramid super(d->super, d->prefix, vsapi);

        if (!super.IsCompatibleWithSource(d->oldvi))
            throw std::runtime_error("super clip is not compatible with source clip");

        if (vsapi->mapNumElements(in, "vectors") != 2)
            throw std::runtime_error("vectors must have exactly 2 elements");

        d->mvbw = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        d->mvfw = vsapi->mapGetNode(in, "vectors", 1, nullptr);

        CheckClipLength(d->super, "super", d->oldvi->numFrames, "clip", vsapi);
        CheckClipLength(d->mvbw, "vectors", d->oldvi->numFrames, "clip", vsapi);
        CheckClipLength(d->mvfw, "vectors", d->oldvi->numFrames, "clip", vsapi);

        MotionBlockPyramid vectorsFw(d->mvfw, d->prefix, vsapi);
        MotionBlockPyramid vectorsBw(d->mvbw, d->prefix, vsapi);

        vectorsFw.ScaleThSCD(d->thscd1, d->thscd2, vectorsFw.bitsPerSample);

        d->deltaFrame = vectorsFw.nDeltaFrame;

        if (!vectorsFw.IsCompatibleWithAnalysis(super))
            throw std::runtime_error("wrong source or super clip frame size");

        if (!vectorsFw.IsCompatible(vectorsBw) || (vectorsBw.nDeltaFrame != -vectorsFw.nDeltaFrame) || vectorsFw.nDeltaFrame > 0 || vectorsBw.nDeltaFrame < 0)
            throw std::runtime_error("mvfw and mvbw must be compatible with each other and have opposite sign delta");

        d->mvfwGeometry = vectorsFw.Geometry();
        d->mvbwGeometry = vectorsBw.Geometry();
        d->superGeometry = super.Geometry();

        d->maskResizerFull.Init(vectorsFw.nBlkX, vectorsFw.nBlkY, vectorsFw.nBlkSizeX, vectorsFw.nBlkSizeY, vectorsFw.nOverlapX, vectorsFw.nOverlapY,
            d->vi.width, d->vi.height);

        if (d->vi.format.subSamplingH > 0 || d->vi.format.subSamplingW > 0)
            d->maskResizerSubSampled.Init(vectorsFw.nBlkX, vectorsFw.nBlkY, vectorsFw.nBlkSizeX >> d->vi.format.subSamplingW, vectorsFw.nBlkSizeY >> d->vi.format.subSamplingH, vectorsFw.nOverlapX >> d->vi.format.subSamplingW, vectorsFw.nOverlapY >> d->vi.format.subSamplingH,
                d->vi.width >> d->vi.format.subSamplingW, d->vi.height >> d->vi.format.subSamplingH);

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

        d->vi.numFrames = (int)(d->vi.numFrames * d->fb / d->fa);
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("FlowFPS: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[4] = { 
        {d->node, rpGeneral},
        {d->super, rpGeneral},
        {d->mvbw, rpGeneral},
        {d->mvfw, rpGeneral},
    };

    vsapi->createVideoFilter(out, "FlowFPS", &d->vi, SelectOnBitsPerSample(d->vi.format.bitsPerSample, flowfpsGetFrame<uint8_t>, flowfpsGetFrame<uint16_t>, flowfpsGetFrame<float>), filterFree<FlowFPSData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void flowfpsRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("FlowFPS",
                 "clip:vnode;"
                 "super:vnode;"
                 "vectors:vnode[];"
                 "num:int:opt;"
                 "den:int:opt;"
                 "extramask:int:opt;"
                 "ml:float:opt;"
                 "blend:int:opt;"
                 "thscd1:int:opt;"
                 "thscd2:float:opt;"
                 "prefix:data:opt;",
                 "clip:vnode;",
                 flowfpsCreate, nullptr, plugin);
}
