// Pixels flow motion function
// Copyright(c)2005 A.G.Balakhnin aka Fizick

// See legal notice in Copying.txt for more information

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; version 2 of the License.
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

#include <cstdint>
#include <cstring>
#include <memory>
#include <algorithm>
#include <cmath>

#include <VapourSynth4.h>

#include "Common.h"
#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"
#include "MaskResize.h"
#include "FlowShared.h" // FlowFetch dispatch (AVX-512/AVX2 vpgatherdd > scalar) + FlowFetch_scalar

struct FlowData {
    VSNode *clip = nullptr;
    VSNode *super = nullptr;
    VSNode *vectors = nullptr;
    const VSVideoInfo *vi = nullptr;
    
    int deltaFrame;
    int time256;
    int64_t thscd1;
    float thscd2;

    MaskResizer maskResizerFull;
    MaskResizer maskResizerSubSampled;

    AnalysisGeometry geometry;
    SuperGeometry superGeometry;

    std::string prefix;

    const VSAPI *vsapi;

    FlowData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~FlowData() {
        vsapi->freeNode(clip);
        vsapi->freeNode(super);
        vsapi->freeNode(vectors);
    }
};

// flowFetch (motion-compensated copy) now lives in FlowShared.h as FlowFetch (dispatch) / FlowFetch_scalar.

template<typename PixelType>
static const VSFrame *VS_CC flowGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    FlowData *d =  reinterpret_cast<FlowData *>(instanceData);

    int nref = n + d->deltaFrame;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->clip, frameCtx);
        vsapi->requestFrameFilter(n, d->vectors, frameCtx);

        if (nref >= 0 && nref < d->vi->numFrames)
            vsapi->requestFrameFilter(nref, d->super, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        VSFrame *dst = nullptr;

        try {
            MotionBlockPyramid vectors(vsapi->getFrameFilter(n, d->vectors, frameCtx), 1, d->prefix, vsapi, d->geometry);

            if (nref >= 0 && nref < d->vi->numFrames && vectors.IsUsable(d->thscd1, d->thscd2)) {
                const VSFrame *propSrc = vsapi->getFrameFilter(n, d->clip, frameCtx);
                dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, propSrc, core);
                vsapi->freeFrame(propSrc);

                const VSFrame *ref = vsapi->getFrameFilter(nref, d->super, frameCtx);
                FramePyramid refGOF(ref, 1, d->prefix, vsapi, d->superGeometry);

                auto smallMasks = vectors.MakeSmallVectorMasks();

                auto tmp = MaskResizer::GetTmpBuffer(std::max(d->maskResizerFull.tmpSize, d->maskResizerSubSampled.tmpSize));

                auto [dstTileVX, dstTileVY] = MaskResizer::GetTileBuffers<2>();

                auto bufVX = MaskResizer::MakeBufferPair(smallMasks->VXSmallY, smallMasks->pitchVSmallY, dstTileVX.get());
                auto bufVY = MaskResizer::MakeBufferPair(smallMasks->VYSmallY, smallMasks->pitchVSmallY, dstTileVY.get());

                ptrdiff_t dstStrideY = vsapi->getStride(dst, 0);
                uint8_t *dstPtrY = vsapi->getWritePtr(dst, 0);

                for (auto &tile : d->maskResizerFull.tiles) {
                    tile.Process(tmp.get(), bufVX, bufVY);

                    FlowFetch<PixelType>(dstPtrY + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideY, dstStrideY, refGOF.GetLevel(0).planes[0],
                        dstTileVX.get(), dstTileVY.get(), MaskResizer::GetTileBufferStride(),
                        tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->time256);
                }

                if (d->vi->format.numPlanes == 3) {
                    smallMasks->AdjustSmallVectorMaskSubSampling(vectors.nBlkX, vectors.nBlkY, d->vi->format.subSamplingW, d->vi->format.subSamplingH);

                    ptrdiff_t dstStrideU = vsapi->getStride(dst, 1);
                    ptrdiff_t dstStrideV = vsapi->getStride(dst, 2);
                    uint8_t *dstPtrU = vsapi->getWritePtr(dst, 1);
                    uint8_t *dstPtrV = vsapi->getWritePtr(dst, 2);

                    for (auto &tile : (d->vi->format.subSamplingH > 0 || d->vi->format.subSamplingW > 0) ? d->maskResizerSubSampled.tiles : d->maskResizerFull.tiles) {
                        tile.Process(tmp.get(), bufVX, bufVY);

                        FlowFetch<PixelType>(dstPtrU + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideU, dstStrideU, refGOF.GetLevel(0).planes[1],
                            dstTileVX.get(), dstTileVY.get(), MaskResizer::GetTileBufferStride(),
                            tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->time256);

                        FlowFetch<PixelType>(dstPtrV + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideV, dstStrideV, refGOF.GetLevel(0).planes[2],
                            dstTileVX.get(), dstTileVY.get(), MaskResizer::GetTileBufferStride(),
                            tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->time256);
                    }
                }

                return dst;

            } else {
                return vsapi->getFrameFilter(n, d->clip, frameCtx);
            }
        } catch (const std::exception &e) {
            vsapi->setFilterError(("Flow: " + std::string(e.what())).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC flowCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<FlowData> d = std::make_unique<FlowData>(vsapi);

    int err;

    double time = vsapi->mapGetFloat(in, "time", 0, &err);
    if (err)
        time = 100.0;

    d->thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
    if (err)
        d->thscd1 = MV_DEFAULT_SCD1;

    d->thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
    if (err)
        d->thscd2 = MV_DEFAULT_SCD2;

    try {

        if (!std::isfinite(time) || time < 0.0 || time > 100.0)
            throw std::runtime_error("time must be between 0 and 100%");

        d->time256 = (int)(time * 256.0 / 100.0);

        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        if (prefix)
            d->prefix = prefix;
        else
            d->prefix = DEFAULT_MVUTENSILS_PREFIX;

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);

        FramePyramid super(d->super, d->prefix, vsapi);

        d->vectors = vsapi->mapGetNode(in, "vectors", 0, nullptr);

        d->clip = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->clip);

        if (!super.IsCompatibleWithSource(d->vi))
            throw std::runtime_error("source clip isn't compatible with super clip");

        CheckClipLength(d->super, "super", d->vi->numFrames, "clip", vsapi);
        CheckClipLength(d->vectors, "vectors", d->vi->numFrames, "clip", vsapi);

        MotionBlockPyramid vectors(d->vectors, d->prefix, vsapi);

        vectors.ScaleThSCD(d->thscd1, d->thscd2, vectors.bitsPerSample);

        d->deltaFrame = vectors.nDeltaFrame;

        if (!vectors.IsCompatibleWithAnalysis(super))
            throw std::runtime_error("wrong source or super clip frame size");

        d->geometry = vectors.Geometry();
        d->superGeometry = super.Geometry();

        d->maskResizerFull.Init(vectors.nBlkX, vectors.nBlkY, vectors.nBlkSizeX, vectors.nBlkSizeY, vectors.nOverlapX, vectors.nOverlapY,
            d->vi->width, d->vi->height);

        if (d->vi->format.subSamplingH > 0 || d->vi->format.subSamplingW > 0)
            d->maskResizerSubSampled.Init(vectors.nBlkX, vectors.nBlkY, vectors.nBlkSizeX >> d->vi->format.subSamplingW, vectors.nBlkSizeY >> d->vi->format.subSamplingH, vectors.nOverlapX >> d->vi->format.subSamplingW, vectors.nOverlapY >> d->vi->format.subSamplingH,
                d->vi->width >> d->vi->format.subSamplingW, d->vi->height >> d->vi->format.subSamplingH);

    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("Flow: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[3] = {
        {d->clip, rpStrictSpatial},
        {d->super, rpGeneral},
        {d->vectors, rpStrictSpatial},
    };

    vsapi->createVideoFilter(out, "Flow", d->vi, SelectOnBitsPerSample(d->vi->format.bitsPerSample, flowGetFrame<uint8_t>, flowGetFrame<uint16_t>, flowGetFrame<float>), filterFree<FlowData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void flowRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Flow",
                 "clip:vnode;"
                 "super:vnode;"
                 "vectors:vnode;"
                 "time:float:opt;"
                 "thscd1:int:opt;"
                 "thscd2:float:opt;"
                 "prefix:data:opt;",
                 "clip:vnode;",
                 flowCreate, nullptr, plugin);
}
