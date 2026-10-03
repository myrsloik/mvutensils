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

#include "Common.h"
#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"
#include "MaskResize.h"
#include "FlowShared.h"

struct FlowInterData {
    VSNode *node = nullptr;
    const VSVideoInfo *vi = nullptr;

    VSNode *super = nullptr;
    VSNode *mvbw = nullptr;
    VSNode *mvfw = nullptr;

    float ml;
    bool blend;
    int64_t thscd1;
    float thscd2;

    int time256;

    int deltaFrame;

    MaskResizer maskResizerFull;
    MaskResizer maskResizerSubSampled;

    AnalysisGeometry mvfwGeometry, mvbwGeometry;
    SuperGeometry superGeometry;

    std::string prefix;

    const VSAPI *vsapi;

    FlowInterData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~FlowInterData() {
        vsapi->freeNode(node);
        vsapi->freeNode(super);
        vsapi->freeNode(mvbw);
        vsapi->freeNode(mvfw);
    }
};

template<typename PixelType>
static const VSFrame *VS_CC flowinterGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    FlowInterData *d = reinterpret_cast<FlowInterData *>(instanceData);

    if (activationReason == arInitial) {
        int off = -d->deltaFrame;

        if (n + off < d->vi->numFrames) {
            vsapi->requestFrameFilter(n, d->mvfw, frameCtx);
            vsapi->requestFrameFilter(n + off, d->mvfw, frameCtx);

            vsapi->requestFrameFilter(n, d->mvbw, frameCtx);
            vsapi->requestFrameFilter(n + off, d->mvbw, frameCtx);

            vsapi->requestFrameFilter(n, d->super, frameCtx);
            vsapi->requestFrameFilter(n + off, d->super, frameCtx);
        }

        vsapi->requestFrameFilter(n, d->node, frameCtx);
        vsapi->requestFrameFilter(std::min(n + off, d->vi->numFrames - 1), d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        VSFrame *dst = nullptr;

        try {
            int off = -d->deltaFrame;

            bool vectorsLoadFrame = (n + off < d->vi->numFrames);

            MotionBlockPyramid vectorsF(vectorsLoadFrame ? vsapi->getFrameFilter(n + off, d->mvfw, frameCtx) : nullptr, 1, d->prefix, vsapi, d->mvfwGeometry);
            MotionBlockPyramid vectorsB(vectorsLoadFrame ? vsapi->getFrameFilter(n, d->mvbw, frameCtx) : nullptr, 1, d->prefix, vsapi, d->mvbwGeometry);

            if (vectorsB.IsUsable(d->thscd1, d->thscd2) && vectorsF.IsUsable(d->thscd1, d->thscd2)) {
                FramePyramid src(vsapi->getFrameFilter(n, d->super, frameCtx), 1, d->prefix, vsapi, d->superGeometry);
                FramePyramid ref(vsapi->getFrameFilter(n + off, d->super, frameCtx), 1, d->prefix, vsapi, d->superGeometry);
                const VSFrame *dstPropSrc = vsapi->getFrameFilter(n, d->node, frameCtx);
                dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, dstPropSrc, core);
                vsapi->freeFrame(dstPropSrc);

                auto SmallB = vectorsB.MakeSmallVectorMasks();
                auto SmallF = vectorsF.MakeSmallVectorMasks();

                auto MaskSmallB = vectorsB.MakeVectorOcclusionMask<uint16_t>(d->ml, 1.0f, (256 - d->time256), true);
                auto MaskSmallF = vectorsF.MakeVectorOcclusionMask<uint16_t>(d->ml, 1.0f, d->time256, true);

                auto tmp = MaskResizer::GetTmpBuffer(std::max(d->maskResizerFull.tmpSize, d->maskResizerSubSampled.tmpSize));

                auto [dstTileMaskF, dstTileMaskB, dstTileXF, dstTileYF, dstTileXB, dstTileYB] = MaskResizer::GetTileBuffers<6>();

                auto bufMaskF = MaskResizer::MakeBufferPair(MaskSmallF->mask, MaskSmallF->stride, dstTileMaskF.get());
                auto bufMaskB = MaskResizer::MakeBufferPair(MaskSmallB->mask, MaskSmallB->stride, dstTileMaskB.get());

                auto bufSmallFX = MaskResizer::MakeBufferPair(SmallF->VXSmallY, SmallF->pitchVSmallY, dstTileXF.get());
                auto bufSmallFY = MaskResizer::MakeBufferPair(SmallF->VYSmallY, SmallF->pitchVSmallY, dstTileYF.get());
                auto bufSmallBX = MaskResizer::MakeBufferPair(SmallB->VXSmallY, SmallB->pitchVSmallY, dstTileXB.get());
                auto bufSmallBY = MaskResizer::MakeBufferPair(SmallB->VYSmallY, SmallB->pitchVSmallY, dstTileYB.get());

                MotionBlockPyramid vectorsFF(vsapi->getFrameFilter(n, d->mvfw, frameCtx), 1, d->prefix, vsapi, d->mvfwGeometry);
                MotionBlockPyramid vectorsBB(vsapi->getFrameFilter(n + off, d->mvbw, frameCtx), 1, d->prefix, vsapi, d->mvbwGeometry);

                if (vectorsBB.IsUsable(d->thscd1, d->thscd2) && vectorsFF.IsUsable(d->thscd1, d->thscd2)) {
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
                        tile.Process(tmp.get(),
                             bufMaskF, bufMaskB,
                             bufSmallFX, bufSmallFY, bufSmallBX, bufSmallBY,
                             bufSmallFFX, bufSmallFFY, bufSmallBBX, bufSmallBBY);

                        FlowInterExtra<PixelType>(dstPtrY + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideY, dstStrideY,
                             ref.GetLevel(0).planes[0], src.GetLevel(0).planes[0],
                             dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                             dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                             tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->time256,
                             dstTileXBB.get(), dstTileXFF.get(), dstTileYBB.get(), dstTileYFF.get());
                    }

                    if (d->vi->format.numPlanes == 3) {
                        SmallF->AdjustSmallVectorMaskSubSampling(vectorsF.nBlkX, vectorsF.nBlkY, d->vi->format.subSamplingW, d->vi->format.subSamplingH);
                        SmallB->AdjustSmallVectorMaskSubSampling(vectorsB.nBlkX, vectorsB.nBlkY, d->vi->format.subSamplingW, d->vi->format.subSamplingH);
                        SmallFF->AdjustSmallVectorMaskSubSampling(vectorsFF.nBlkX, vectorsFF.nBlkY, d->vi->format.subSamplingW, d->vi->format.subSamplingH);
                        SmallBB->AdjustSmallVectorMaskSubSampling(vectorsBB.nBlkX, vectorsBB.nBlkY, d->vi->format.subSamplingW, d->vi->format.subSamplingH);

                        ptrdiff_t dstStrideU = vsapi->getStride(dst, 1);
                        ptrdiff_t dstStrideV = vsapi->getStride(dst, 2);
                        uint8_t *dstPtrU = vsapi->getWritePtr(dst, 1);
                        uint8_t *dstPtrV = vsapi->getWritePtr(dst, 2);

                        for (auto &tile : (d->vi->format.subSamplingH > 0 || d->vi->format.subSamplingW > 0) ? d->maskResizerSubSampled.tiles : d->maskResizerFull.tiles) {
                            tile.Process(tmp.get(),
                                 bufMaskF, bufMaskB,
                                 bufSmallFX, bufSmallFY, bufSmallBX, bufSmallBY,
                                 bufSmallFFX, bufSmallFFY, bufSmallBBX, bufSmallBBY);

                            FlowInterExtra<PixelType>(dstPtrU + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideU, dstStrideU,
                                 ref.GetLevel(0).planes[1], src.GetLevel(0).planes[1],
                                 dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                                 dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                                 tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->time256,
                                 dstTileXBB.get(), dstTileXFF.get(), dstTileYBB.get(), dstTileYFF.get());

                            FlowInterExtra<PixelType>(dstPtrV + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideV, dstStrideV,
                                 ref.GetLevel(0).planes[2], src.GetLevel(0).planes[2],
                                 dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                                 dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                                 tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->time256,
                                 dstTileXBB.get(), dstTileXFF.get(), dstTileYBB.get(), dstTileYFF.get());
                        }
                    }
                } else { // bad extra frames, use old method without extra frames
                    ptrdiff_t dstStrideY = vsapi->getStride(dst, 0);
                    uint8_t *dstPtrY = vsapi->getWritePtr(dst, 0);

                    for (auto &tile : d->maskResizerFull.tiles) {
                        tile.Process(tmp.get(),
                             bufMaskF, bufMaskB,
                             bufSmallFX, bufSmallFY, bufSmallBX, bufSmallBY);

                        FlowInter<PixelType>(dstPtrY + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideY, dstStrideY,
                             ref.GetLevel(0).planes[0], src.GetLevel(0).planes[0],
                             dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                             dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                             tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->time256);

                    }

                    if (d->vi->format.numPlanes == 3) {
                        SmallF->AdjustSmallVectorMaskSubSampling(vectorsF.nBlkX, vectorsF.nBlkY, d->vi->format.subSamplingW, d->vi->format.subSamplingH);
                        SmallB->AdjustSmallVectorMaskSubSampling(vectorsB.nBlkX, vectorsB.nBlkY, d->vi->format.subSamplingW, d->vi->format.subSamplingH);

                        ptrdiff_t dstStrideU = vsapi->getStride(dst, 1);
                        ptrdiff_t dstStrideV = vsapi->getStride(dst, 2);
                        uint8_t *dstPtrU = vsapi->getWritePtr(dst, 1);
                        uint8_t *dstPtrV = vsapi->getWritePtr(dst, 2);

                        for (auto &tile : (d->vi->format.subSamplingH > 0 || d->vi->format.subSamplingW > 0) ? d->maskResizerSubSampled.tiles : d->maskResizerFull.tiles) {
                            tile.Process(tmp.get(),
                                 bufMaskF, bufMaskB,
                                 bufSmallFX, bufSmallFY, bufSmallBX, bufSmallBY);

                            FlowInter<PixelType>(dstPtrU + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideU, dstStrideU,
                                    ref.GetLevel(0).planes[1], src.GetLevel(0).planes[1],
                                    dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                                    dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                                    tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->time256);

                            FlowInter<PixelType>(dstPtrV + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideV, dstStrideV,
                                    ref.GetLevel(0).planes[2], src.GetLevel(0).planes[2],
                                    dstTileXB.get(), dstTileXF.get(), dstTileYB.get(), dstTileYF.get(),
                                    dstTileMaskB.get(), dstTileMaskF.get(), MaskResizer::GetTileBufferStride(),
                                    tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->time256);
                        }
                    }
                }


                return dst;
            } else {
                if (d->blend) {
                    const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
                    const VSFrame *ref = vsapi->getFrameFilter(std::min(n + off, d->vi->numFrames - 1), d->node, frameCtx);
                    VSFrame *dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core);

                    for (int plane = 0; plane < d->vi->format.numPlanes; plane++)
                        Blend<PixelType>(vsapi->getWritePtr(dst, plane), vsapi->getReadPtr(src, plane), vsapi->getReadPtr(ref, plane), vsapi->getFrameHeight(dst, plane), vsapi->getFrameWidth(dst, plane), vsapi->getStride(dst, plane), d->time256);

                    vsapi->freeFrame(src);
                    vsapi->freeFrame(ref);

                    return dst;
                } else {
                    return vsapi->getFrameFilter(n, d->node, frameCtx);
                }
            }
        } catch (const std::exception &e) {
            vsapi->setFilterError((std::string("FlowInter: ") + e.what()).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC flowinterCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<FlowInterData> d = std::make_unique<FlowInterData>(vsapi);
    int err;

    float time = vsapi->mapGetFloatSaturated(in, "time", 0, &err);
    if (err)
        time = 50.0f;

    d->ml = vsapi->mapGetFloatSaturated(in, "ml", 0, &err);
    if (err)
        d->ml = 100.0f;

    d->blend = !!vsapi->mapGetInt(in, "blend", 0, &err);
    if (err)
        d->blend = true;

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

    try {

        if (!std::isfinite(time) || time < 0.0f || time > 100.0f)
            throw std::runtime_error("time must be between 0 and 100%");

        if (!std::isfinite(d->ml) || d->ml <= 0.0f)
            throw std::runtime_error("ml must be a finite value greater than 0");

        d->ml = 1.0f / d->ml;

        d->time256 = (int)(time * 256.0f / 100.0f);

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);

        FramePyramid super(d->super, d->prefix, vsapi);

        if (!super.IsCompatibleWithSource(d->vi))
            throw std::runtime_error("super clip is not compatible with source clip");

        if (vsapi->mapNumElements(in, "vectors") != 2)
            throw std::runtime_error("vectors must have exactly 2 elements");

        d->mvbw = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        d->mvfw = vsapi->mapGetNode(in, "vectors", 1, nullptr);

        CheckClipLength(d->super, "super", d->vi->numFrames, "clip", vsapi);
        CheckClipLength(d->mvbw, "vectors", d->vi->numFrames, "clip", vsapi);
        CheckClipLength(d->mvfw, "vectors", d->vi->numFrames, "clip", vsapi);

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
            d->vi->width, d->vi->height);

        if (d->vi->format.subSamplingH > 0 || d->vi->format.subSamplingW > 0)
            d->maskResizerSubSampled.Init(vectorsFw.nBlkX, vectorsFw.nBlkY, vectorsFw.nBlkSizeX >> d->vi->format.subSamplingW, vectorsFw.nBlkSizeY >> d->vi->format.subSamplingH, vectorsFw.nOverlapX >> d->vi->format.subSamplingW, vectorsFw.nOverlapY >> d->vi->format.subSamplingH,
                d->vi->width >> d->vi->format.subSamplingW, d->vi->height >> d->vi->format.subSamplingH);

    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("FlowInter: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[4] = { 
        {d->node, rpGeneral}, 
        {d->super, rpGeneral},
        {d->mvbw, rpGeneral}, 
        {d->mvfw, rpGeneral}, 
    };

    vsapi->createVideoFilter(out, "FlowInter", d->vi, SelectOnBitsPerSample(d->vi->format.bitsPerSample, flowinterGetFrame<uint8_t>, flowinterGetFrame<uint16_t>, flowinterGetFrame<float>), filterFree<FlowInterData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void flowinterRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("FlowInter",
                 "clip:vnode;"
                 "super:vnode;"
                 "vectors:vnode[];"
                 "time:float:opt;"
                 "ml:float:opt;"
                 "blend:int:opt;"
                 "thscd1:int:opt;"
                 "thscd2:float:opt;"
                 "prefix:data:opt;",
                 "clip:vnode;",
                 flowinterCreate, nullptr, plugin);
}
