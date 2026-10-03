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

#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include <VapourSynth4.h>

#include "Common.h"
#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"
#include "MaskResize.h"
#include "FlowShared.h"

struct FlowBlurData {
    VSNode *node = nullptr;
    const VSVideoInfo *vi = nullptr;
    VSNode *super = nullptr;
    VSNode *mvbw = nullptr;
    VSNode *mvfw = nullptr;

    int prec;
    int64_t thscd1;
    float thscd2;

    int blur256;

    int deltaFrame;

    MaskResizer maskResizerFull;
    MaskResizer maskResizerSubSampled;

    AnalysisGeometry mvfwGeometry, mvbwGeometry;
    SuperGeometry superGeometry;

    std::string prefix;

    const VSAPI *vsapi;

    FlowBlurData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~FlowBlurData() {
        vsapi->freeNode(node);
        vsapi->freeNode(super);
        vsapi->freeNode(mvbw);
        vsapi->freeNode(mvfw);
    }
};

template<typename PixelType>
static void FlowBlur_scalar(uint8_t * MVU_RESTRICT pdst8, ptrdiff_t dst_pitch, const PyramidPlane &pref,
                         const uint16_t * MVU_RESTRICT VXFullB, const uint16_t *MVU_RESTRICT VXFullF, const uint16_t *MVU_RESTRICT VYFullB, const uint16_t *MVU_RESTRICT VYFullF,
                         ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int blur256, int prec) noexcept {
    PixelType *pdst = (PixelType *)pdst8;

    dst_pitch /= sizeof(PixelType);
    tilePitch /= sizeof(int16_t);
    int nPelLog = ilog2(pref.nPel);
    const int64_t thresh = 256LL * prec; // max(|vx0|,|vy0|) < thresh  <=>  m == 0 (no taps; centre pixel only)

    /* very slow, but precise motion blur */
    for (int h = 0; h < height; h++) {
        int yBase = (h + dstY) << nPelLog;
        // Centre (zero-motion) row taken straight from the base sub-plane (pPlane[0]) instead of GetPointer():
        // both coords are nPel multiples so GetPointer resolves to sub-plane 0 anyway, and a plain base+offset
        // restrict pointer lets the vectorizer handle the common both-zero copy below.
        const PixelType *MVU_RESTRICT prefPtr = reinterpret_cast<const PixelType *>(
            pref.pPlane[0] + (ptrdiff_t)(h + dstY + pref.nVPadding) * pref.nPitch
                           + (ptrdiff_t)(dstX + pref.nHPadding) * (ptrdiff_t)sizeof(PixelType));
        for (int w = 0; w < width; w++) {
            int vxF0 = (static_cast<int>(VXFullF[w]) - (1 << 15)) * blur256;
            int vyF0 = (static_cast<int>(VYFullF[w]) - (1 << 15)) * blur256;
            int vxB0 = (static_cast<int>(VXFullB[w]) - (1 << 15)) * blur256;
            int vyB0 = (static_cast<int>(VYFullB[w]) - (1 << 15)) * blur256;
            int aF = std::max(abs(vxF0), abs(vyF0));
            int aB = std::max(abs(vxB0), abs(vyB0));
            // Dominant case (~60% of pixels, static regions): no forward and no backward taps, so the result
            // is just the centre pixel. Skips both /prec divisions and the 64-bit /(mF+mB+1) divide.
            if (aF < thresh && aB < thresh) {
                pdst[w] = prefPtr[w];
                continue;
            }
            int xBase = (w + dstX) << nPelLog;
            std::conditional_t<std::is_integral_v<PixelType>, int64_t, double> blurredsum = prefPtr[w];
            int mF = (aF / prec) >> 8;
            if (mF > 0) {
                vxF0 /= mF;
                vyF0 /= mF;
                int vxF = vxF0;
                int vyF = vyF0;
                for (int i = 0; i < mF; i++) {
                    blurredsum += *reinterpret_cast<const PixelType *>(pref.GetPointer<PixelType>((vxF >> 8) + xBase, (vyF >> 8) + yBase));
                    vxF += vxF0;
                    vyF += vyF0;
                }
            }
            int mB = (aB / prec) >> 8;
            if (mB > 0) {
                vxB0 /= mB;
                vyB0 /= mB;
                int vxB = vxB0;
                int vyB = vyB0;
                for (int i = 0; i < mB; i++) {
                    blurredsum += *reinterpret_cast<const PixelType *>(pref.GetPointer<PixelType>((vxB >> 8) + xBase, (vyB >> 8) + yBase));
                    vxB += vxB0;
                    vyB += vyB0;
                }
            }
            pdst[w] = static_cast<PixelType>(blurredsum / (mF + mB + 1));
        }
        pdst += dst_pitch;
        VXFullB += tilePitch;
        VYFullB += tilePitch;
        VXFullF += tilePitch;
        VYFullF += tilePitch;
    }
}

// Dispatch: per-pixel AVX-512 tap-gather when the int32 gather offsets fit, else the scalar walk.
// u8/u16 are bit-exact with the scalar across all build variants. The float gather reorders the double
// sum but matches scalar float bit-for-bit for normal pixel ranges (measured 0 error); only pathological
// magnitudes could differ sub-ULP between the avx512 build and the scalar baseline/avx2 builds.
template<typename PixelType>
static MVU_FORCE_INLINE void FlowBlur(uint8_t * MVU_RESTRICT pdst8, ptrdiff_t dst_pitch, const PyramidPlane &pref,
                         const uint16_t * MVU_RESTRICT VXFullB, const uint16_t *MVU_RESTRICT VXFullF, const uint16_t *MVU_RESTRICT VYFullB, const uint16_t *MVU_RESTRICT VYFullF,
                         ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int blur256, int prec) noexcept {
#if defined(MVTOOLS_X86)
    if (FlowGatherFits(pref) && (g_cpuinfo & MVU_CPU_AVX512_BASE)) {
        if constexpr (sizeof(PixelType) == 1)
            FlowBlur_avx512_u8(pdst8, dst_pitch, pref, VXFullB, VXFullF, VYFullB, VYFullF, tilePitch, dstX, dstY, width, height, blur256, prec);
        else if constexpr (sizeof(PixelType) == 2)
            FlowBlur_avx512_u16(pdst8, dst_pitch, pref, VXFullB, VXFullF, VYFullB, VYFullF, tilePitch, dstX, dstY, width, height, blur256, prec);
        else
            FlowBlur_avx512_f32(pdst8, dst_pitch, pref, VXFullB, VXFullF, VYFullB, VYFullF, tilePitch, dstX, dstY, width, height, blur256, prec);
        return;
    }
#endif
    FlowBlur_scalar<PixelType>(pdst8, dst_pitch, pref, VXFullB, VXFullF, VYFullB, VYFullF, tilePitch, dstX, dstY, width, height, blur256, prec);
}

template<typename PixelType>
static const VSFrame *VS_CC flowblurGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    FlowBlurData *d = reinterpret_cast<FlowBlurData *>(instanceData);

    if (activationReason == arInitial) {
        if (n + d->deltaFrame >= 0 && n - d->deltaFrame < d->vi->numFrames) {
            vsapi->requestFrameFilter(n + d->deltaFrame, d->mvbw, frameCtx);
            vsapi->requestFrameFilter(n - d->deltaFrame, d->mvfw, frameCtx);
        }

        vsapi->requestFrameFilter(n, d->super, frameCtx);
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        VSFrame *dst = nullptr;

        try {
            bool vectorsLoadFrame = (n + d->deltaFrame >= 0 && n - d->deltaFrame < d->vi->numFrames);

            MotionBlockPyramid vectorsfw(vectorsLoadFrame ? vsapi->getFrameFilter(n - d->deltaFrame, d->mvfw, frameCtx) : nullptr, 1, d->prefix, vsapi, d->mvfwGeometry);
            MotionBlockPyramid vectorsbw(vectorsLoadFrame ? vsapi->getFrameFilter(n + d->deltaFrame, d->mvbw, frameCtx) : nullptr, 1, d->prefix, vsapi, d->mvbwGeometry);

            if (vectorsfw.IsUsable(d->thscd1, d->thscd2) && vectorsbw.IsUsable(d->thscd1, d->thscd2)) {
                const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
                dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core);
                vsapi->freeFrame(src);

                const VSFrame *ref = vsapi->getFrameFilter(n, d->super, frameCtx);
                FramePyramid refGOF(ref, 1, d->prefix, vsapi, d->superGeometry);

                auto smallMasksFw = vectorsfw.MakeSmallVectorMasks();
                auto smallMasksBw = vectorsbw.MakeSmallVectorMasks();

                auto tmp = MaskResizer::GetTmpBuffer(std::max(d->maskResizerFull.tmpSize, d->maskResizerSubSampled.tmpSize));

                auto [dstTileVXFw, dstTileVYFw, dstTileVXBw, dstTileVYBw] = MaskResizer::GetTileBuffers<4>();

                auto bufVXFw = MaskResizer::MakeBufferPair(smallMasksFw->VXSmallY, smallMasksFw->pitchVSmallY, dstTileVXFw.get());
                auto bufVYFw = MaskResizer::MakeBufferPair(smallMasksFw->VYSmallY, smallMasksFw->pitchVSmallY, dstTileVYFw.get());
                auto bufVXBw = MaskResizer::MakeBufferPair(smallMasksBw->VXSmallY, smallMasksBw->pitchVSmallY, dstTileVXBw.get());
                auto bufVYBw = MaskResizer::MakeBufferPair(smallMasksBw->VYSmallY, smallMasksBw->pitchVSmallY, dstTileVYBw.get());

                ptrdiff_t dstStrideY = vsapi->getStride(dst, 0);
                uint8_t *dstPtrY = vsapi->getWritePtr(dst, 0);

                for (auto &tile : d->maskResizerFull.tiles) {
                    tile.Process(tmp.get(), bufVXFw, bufVYFw, bufVXBw, bufVYBw);

                    FlowBlur<PixelType>(dstPtrY + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideY, dstStrideY, refGOF.GetLevel(0).planes[0],
                             dstTileVXBw.get(), dstTileVXFw.get(), dstTileVYBw.get(), dstTileVYFw.get(), MaskResizer::GetTileBufferStride(),
                             tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->blur256, d->prec);
                }

                if (d->vi->format.numPlanes == 3) {
                    smallMasksFw->AdjustSmallVectorMaskSubSampling(vectorsfw.nBlkX, vectorsfw.nBlkY, d->vi->format.subSamplingW, d->vi->format.subSamplingH);
                    smallMasksBw->AdjustSmallVectorMaskSubSampling(vectorsbw.nBlkX, vectorsbw.nBlkY, d->vi->format.subSamplingW, d->vi->format.subSamplingH);

                    ptrdiff_t dstStrideU = vsapi->getStride(dst, 1);
                    ptrdiff_t dstStrideV = vsapi->getStride(dst, 2);
                    uint8_t *dstPtrU = vsapi->getWritePtr(dst, 1);
                    uint8_t *dstPtrV = vsapi->getWritePtr(dst, 2);

                    for (auto &tile : (d->vi->format.subSamplingH > 0 || d->vi->format.subSamplingW > 0) ? d->maskResizerSubSampled.tiles : d->maskResizerFull.tiles) {
                        tile.Process(tmp.get(), bufVXFw, bufVYFw, bufVXBw, bufVYBw);

                        FlowBlur<PixelType>(dstPtrU + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideU, dstStrideU, refGOF.GetLevel(0).planes[1],
                             dstTileVXBw.get(), dstTileVXFw.get(), dstTileVYBw.get(), dstTileVYFw.get(), MaskResizer::GetTileBufferStride(),
                             tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->blur256, d->prec);
                        FlowBlur<PixelType>(dstPtrV + tile.dstX * sizeof(PixelType) + tile.dstY * dstStrideV, dstStrideV, refGOF.GetLevel(0).planes[2],
                             dstTileVXBw.get(), dstTileVXFw.get(), dstTileVYBw.get(), dstTileVYFw.get(), MaskResizer::GetTileBufferStride(),
                             tile.dstX, tile.dstY, tile.dstWidth, tile.dstHeight, d->blur256, d->prec);
                    }
                }

                return dst;
            } else {
                return vsapi->getFrameFilter(n, d->node, frameCtx);
            }
        } catch (const std::exception &e) {
            vsapi->freeFrame(dst);
            vsapi->setFilterError(("FlowBlur: " + std::string(e.what())).c_str(), frameCtx);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC flowblurCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<FlowBlurData> d = std::make_unique<FlowBlurData>(vsapi);
    int err;

    float blur = vsapi->mapGetFloatSaturated(in, "blur", 0, &err);
    if (err)
        blur = 50.0f;

    d->prec = vsapi->mapGetIntSaturated(in, "prec", 0, &err);
    if (err)
        d->prec = 1;

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

        if (!std::isfinite(blur) || blur < 0.0f || blur > 200.0f)
            throw std::runtime_error("blur must be between 0 and 200");

        if (d->prec < 1)
            throw std::runtime_error("prec must be at least 1");

        d->blur256 = (int)(blur * 256.0f / 200.0f);

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);

        FramePyramid super(d->super, d->prefix, vsapi);

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        if (!super.IsCompatibleWithSource(d->vi))
            throw std::runtime_error("source clip isn't compatible with super clip");

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
        vsapi->mapSetError(out, ("FlowBlur: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[4] = { 
        {d->node, rpStrictSpatial}, 
        {d->super, rpStrictSpatial},
        {d->mvbw, rpGeneral}, 
        {d->mvfw, rpGeneral}, 
    };

    vsapi->createVideoFilter(out, "FlowBlur", d->vi, SelectOnBitsPerSample(d->vi->format.bitsPerSample, flowblurGetFrame<uint8_t>, flowblurGetFrame<uint16_t>, flowblurGetFrame<float>), filterFree<FlowBlurData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void flowblurRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("FlowBlur",
                 "clip:vnode;"
                 "super:vnode;"
                 "vectors:vnode[];"
                 "blur:float:opt;"
                 "prec:int:opt;"
                 "thscd1:int:opt;"
                 "thscd2:float:opt;"
                 "prefix:data:opt;",
                 "clip:vnode;",
                 flowblurCreate, nullptr, plugin);
}
