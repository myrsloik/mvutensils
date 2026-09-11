// Make a motion compensate temporal denoiser
// Author: Manao
// Copyright(c)2006 A.G.Balakhnin aka Fizick (YUY2, overlap, edges processing)
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

#include <memory>
#include <cmath>
#include <VapourSynth4.h>

#include "CopyCode.h"
#include "Overlap.h"
#include "Common.h"
#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"



struct CompensateData {
    VSNode *node = nullptr;
    VSNode *super = nullptr;
    VSNode *vectors = nullptr;

    const VSVideoInfo *vi = nullptr;
    const VSVideoInfo *supervi = nullptr;

    int64_t thSAD;
    bool fields;
    int time256;
    int64_t nSCD1;
    float nSCD2;
    int tff;
    int tff_exists;
    int deltaFrame;

    bool chroma;
    int xRatioUV;
    int yRatioUV;

    int dstTempPitch;
    int dstTempPitchUV;

    OverlapWindows OverWins;
    OverlapWindows OverWinsUV;

    OverlapsFunction OVERS[3];
    COPYFunction BLIT[3];

    std::string prefix;

    const VSAPI *vsapi;

    CompensateData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~CompensateData() {
        vsapi->freeNode(node);
        vsapi->freeNode(super);
        vsapi->freeNode(vectors);
    }
};


template<typename PixelType>
static const VSFrame *VS_CC compensateGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    CompensateData *d = reinterpret_cast<CompensateData *>(instanceData);
    int nref = n + d->deltaFrame;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);

        vsapi->requestFrameFilter(n, d->vectors, frameCtx);

        if (nref < n && nref >= 0)
            vsapi->requestFrameFilter(nref, d->super, frameCtx);

        vsapi->requestFrameFilter(n, d->super, frameCtx);

        if (nref >= n && nref < d->vi->numFrames)
            vsapi->requestFrameFilter(nref, d->super, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        uint8_t *pDstCur[3] = {};
        ptrdiff_t nDstPitches[3] = {};

        VSFrame *dst = nullptr;

        try {
            // Construct (and validate) the vectors inside the try: deserialization can throw on corrupt data.
            MotionBlockPyramid vectors(vsapi->getFrameFilter(n, d->vectors, frameCtx), 1, d->prefix, vsapi);

            const int ySubUV = ilog2(d->yRatioUV);
            const int xSubUV = ilog2(d->xRatioUV);
            const int nWidth[3] = { vectors.nWidth, nWidth[0] >> xSubUV, nWidth[1] };
            const int nHeight[3] = { vectors.nHeight, nHeight[0] >> ySubUV, nHeight[1] };
            const int nOverlapX[3] = { vectors.nOverlapX, nOverlapX[0] >> xSubUV, nOverlapX[1] };
            const int nOverlapY[3] = { vectors.nOverlapY, nOverlapY[0] >> ySubUV, nOverlapY[1] };
            const int nBlkSizeX[3] = { vectors.nBlkSizeX, nBlkSizeX[0] >> xSubUV, nBlkSizeX[1] };
            const int nBlkSizeY[3] = { vectors.nBlkSizeY, nBlkSizeY[0] >> ySubUV, nBlkSizeY[1] };
            const int nBlkX = vectors.nBlkX;
            const int nBlkY = vectors.nBlkY;
            const int64_t thSAD = d->thSAD;
            const int dstTempPitch[3] = { d->dstTempPitch, d->dstTempPitchUV, d->dstTempPitchUV };
            const bool chroma = d->chroma;
            const int nPel = vectors.nPel;
            const int time256 = d->time256;

            int bitsPerSample = d->supervi->format.bitsPerSample;

            int nWidth_B[3] = { nBlkX * (nBlkSizeX[0] - nOverlapX[0]) + nOverlapX[0], nWidth_B[0] >> xSubUV, nWidth_B[1] };
            int nHeight_B[3] = { nBlkY * (nBlkSizeY[0] - nOverlapY[0]) + nOverlapY[0], nHeight_B[0] >> ySubUV, nHeight_B[1] };


            int num_planes = chroma ? 3 : 1;

            const VSFrame *src = vsapi->getFrameFilter(n, d->super, frameCtx);
            FramePyramid pSrcGOF(src, 1, d->prefix, vsapi);
            const auto &pSrcPlanes = pSrcGOF.GetLevel(0).planes;

            if (nref >= 0 && nref < d->vi->numFrames && vectors.IsUsable(d->nSCD1, d->nSCD2)) {
                const VSFrame *ref = vsapi->getFrameFilter(nref, d->super, frameCtx);
                FramePyramid pRefGOF(ref, 1, d->prefix, vsapi);
                const auto &pRefPlanes = pRefGOF.GetLevel(0).planes;

                const VSFrame *realSrc = vsapi->getFrameFilter(n, d->node, frameCtx);
                dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, realSrc, core);
                vsapi->freeFrame(realSrc);

                for (int i = 0; i < d->supervi->format.numPlanes; i++) {
                    pDstCur[i] = vsapi->getWritePtr(dst, i);
                    nDstPitches[i] = vsapi->getStride(dst, i);
                }

                int fieldShift = 0;
                if (d->fields && nPel > 1 && ((nref - n) % 2 != 0)) {
                    bool src_top_field = GetTopField(src, n, d->tff_exists, d->tff, true, vsapi);
                    bool ref_top_field = GetTopField(ref, nref, d->tff_exists, d->tff, true, vsapi);
                    fieldShift = ComputeFieldShift(src_top_field, ref_top_field, nPel);
                }

                if (nOverlapX[0] == 0 && nOverlapY[0] == 0) {
                    // Uses a more restrictive copy function to the right/bottom when necessary to handle block dimension padded frames

                    size_t blitSizeRight[3] = {};
                    size_t blitSizeBottom[3] = {};

                    for (int plane = 0; plane < num_planes; plane++) {
                        blitSizeRight[plane] = (nBlkSizeX[plane] - (pSrcPlanes[plane].nWidth - vsapi->getFrameWidth(dst, plane))) * sizeof(PixelType);
                        blitSizeBottom[plane] = (nBlkSizeY[plane] - (pSrcPlanes[plane].nHeight - vsapi->getFrameHeight(dst, plane)));
                    }

                    for (int by = 0; by < nBlkY; by++) {
                        bool slowBlitY = (by == nBlkY - 1) && (blitSizeBottom[0] > 0);
                        int xx[3] = {};

                        for (int bx = 0; bx < nBlkX; bx++) {
                            bool slowBlitX = (bx == nBlkX - 1) && (blitSizeRight[0] > 0);
                            int i = by * nBlkX + bx;
                            const BlockData block = vectors.GetBlock(i);

                            int blx[3], bly[3];

                            if (block.vector.sad < thSAD) {
                                blx[0] = block.x * nPel + block.vector.x * time256 / 256;
                                bly[0] = block.y * nPel + block.vector.y * time256 / 256 + fieldShift;
                            } else {
                                blx[0] = bx * nBlkSizeX[0] * nPel;
                                bly[0] = by * nBlkSizeY[0] * nPel + fieldShift;
                            }

                            const auto &pPlanes = (block.vector.sad < thSAD) ? pRefPlanes : pSrcPlanes;

                            blx[1] = blx[2] = blx[0] >> xSubUV;
                            bly[1] = bly[2] = bly[0] >> ySubUV;

                            if (slowBlitX || slowBlitY) {
                                for (int plane = 0; plane < num_planes; plane++) {
                                    mvu_bitblt(pDstCur[plane] + xx[plane], nDstPitches[plane], pPlanes[plane].GetPointer<PixelType>(blx[plane], bly[plane]), pPlanes[plane].nPitch, (slowBlitX && blitSizeRight[plane]) ? blitSizeRight[plane] : (nBlkSizeX[plane] * sizeof(PixelType)), (slowBlitY && blitSizeBottom[plane]) ? blitSizeBottom[plane] : nBlkSizeY[plane]);
                                    xx[plane] += nBlkSizeX[plane] * sizeof(PixelType);
                                }
                            } else {
                                for (int plane = 0; plane < num_planes; plane++) {
                                    d->BLIT[plane](pDstCur[plane] + xx[plane], nDstPitches[plane], pPlanes[plane].GetPointer<PixelType>(blx[plane], bly[plane]), pPlanes[plane].nPitch);
                                    xx[plane] += nBlkSizeX[plane] * sizeof(PixelType);
                                }
                            }
                        }

                        for (int plane = 0; plane < num_planes; plane++)
                            pDstCur[plane] += nBlkSizeY[plane] * nDstPitches[plane];
                    }
                } else { // overlap
                    uint8_t *DstTemp[3] = {};
                    MvuAlignedPtr<uint8_t> DstTempBuffers[3] = { {nullptr, mvu_aligned_free}, {nullptr, mvu_aligned_free}, {nullptr, mvu_aligned_free} };

                    // Allocate buffer for only nBlkSizeY rows instead of full frame height
                    // We'll output finalized rows and reuse the buffer as a sliding window
                    for (int plane = 0; plane < num_planes; plane++) {
                        DstTempBuffers[plane] = mvu_make_aligned<uint8_t>(nBlkSizeY[plane] * dstTempPitch[plane]);
                        DstTemp[plane] = DstTempBuffers[plane].get();
                    }

                    for (int by = 0; by < nBlkY; by++) {
                        int wby = (by == 0) ? 0 : (by == nBlkY - 1) ? 6 : 3;
                        int wbx = 0;
                        int xx[3] = { 0 };

                        // Clear the non-overlapping region for this block row
                        for (int plane = 0; plane < num_planes; plane++) {
                            int clearStart = (by == 0) ? 0 : nOverlapY[plane];
                            int clearRows = (by == 0) ? nBlkSizeY[plane] : (nBlkSizeY[plane] - nOverlapY[plane]);
                            memset(DstTemp[plane] + clearStart * dstTempPitch[plane], 0, clearRows * dstTempPitch[plane]);
                        }

                        for (int bx = 0; bx < nBlkX; bx++) {
                            wbx = bx == nBlkX - 1 ? 2 : wbx;
                            const int16_t *winOver[3] = { d->OverWins.GetWindow(wby + wbx) };
                            if (chroma)
                                winOver[1] = winOver[2] = d->OverWinsUV.GetWindow(wby + wbx);

                            int i = by * nBlkX + bx;
                            const BlockData block = vectors.GetBlock(i);

                            int blx[3], bly[3];

                            if (block.vector.sad < thSAD) {
                                blx[0] = block.x * nPel + block.vector.x * time256 / 256;
                                bly[0] = block.y * nPel + block.vector.y * time256 / 256 + fieldShift;
                            } else {
                                blx[0] = bx * (nBlkSizeX[0] - nOverlapX[0]) * nPel;
                                bly[0] = by * (nBlkSizeY[0] - nOverlapY[0]) * nPel + fieldShift;
                            }

                            const auto &pPlanes = (block.vector.sad < thSAD) ? pRefPlanes : pSrcPlanes;

                            blx[1] = blx[2] = blx[0] >> xSubUV;
                            bly[1] = bly[2] = bly[0] >> ySubUV;

                            for (int plane = 0; plane < num_planes; plane++) {
                                d->OVERS[plane](DstTemp[plane] + xx[plane] * (std::is_integral_v<PixelType> ? 2 : 1), dstTempPitch[plane], pPlanes[plane].GetPointer<PixelType>(blx[plane], bly[plane]), pPlanes[plane].nPitch, winOver[plane], nBlkSizeX[plane]);
                                xx[plane] += (nBlkSizeX[plane] - nOverlapX[plane]) * sizeof(PixelType);
                            }
                            wbx = 1;
                        }

                        // Output the finalized rows (non-overlapping portion)
                        for (int plane = 0; plane < num_planes; plane++) {
                            int planeRowsToOutput = (by == nBlkY - 1) ? nBlkSizeY[plane] : (nBlkSizeY[plane] - nOverlapY[plane]);
                            int outputHeight = std::min(planeRowsToOutput, std::min(vsapi->getFrameHeight(dst, plane), nHeight_B[plane]) - by * (nBlkSizeY[plane] - nOverlapY[plane]));

                            if (outputHeight > 0) {
                                int outputWidth = std::min(vsapi->getFrameWidth(dst, plane), nWidth_B[plane]);
                                ToPixels<PixelType>(pDstCur[plane], nDstPitches[plane], DstTemp[plane], dstTempPitch[plane], outputWidth, outputHeight, bitsPerSample);
                            }

                            pDstCur[plane] += nDstPitches[plane] * (nBlkSizeY[plane] - nOverlapY[plane]);
                        }

                        // Shift the overlapping rows to the beginning of the buffer for next iteration
                        if (by < nBlkY - 1) {
                            for (int plane = 0; plane < num_planes; plane++) {
                                memmove(DstTemp[plane],
                                    DstTemp[plane] + (nBlkSizeY[plane] - nOverlapY[plane]) * dstTempPitch[plane],
                                    nOverlapY[plane] * dstTempPitch[plane]);
                            }
                        }
                    }
                }
            } else {
                assert(!dst);
                return vsapi->getFrameFilter(n, d->node, frameCtx);
            }

        } catch (const std::exception &e) {
            vsapi->setFilterError((std::string("Compensate: ") + e.what()).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        return dst;
    }

    return nullptr;
}


static void VS_CC compensateCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<CompensateData> d = std::make_unique<CompensateData>(vsapi);
    int err;

    d->thSAD = vsapi->mapGetInt(in, "thsad", 0, &err);
    if (err)
        d->thSAD = 10000;

    d->fields = !!vsapi->mapGetInt(in, "fields", 0, &err);

    double time = vsapi->mapGetFloat(in, "time", 0, &err);
    if (err)
        time = 100.0;

    d->nSCD1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
    if (err)
        d->nSCD1 = MV_DEFAULT_SCD1;

    d->nSCD2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
    if (err)
        d->nSCD2 = MV_DEFAULT_SCD2;

    d->tff = !!vsapi->mapGetInt(in, "tff", 0, &err);
    d->tff_exists = !err;

    try {
        if (!std::isfinite(time) || time < 0.0 || time > 100.0)
            throw std::runtime_error("time must be between 0.0 and 100.0");

        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        if (prefix)
            d->prefix = prefix;
        else
            d->prefix = DEFAULT_MVUTENSILS_PREFIX;

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);

        FramePyramid super(d->super, d->prefix, vsapi);

        d->vectors = vsapi->mapGetNode(in, "vectors", 0, nullptr);

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        if (!super.IsCompatibleWithSource(d->vi))
            throw std::runtime_error("source clip isn't compatible with super clip");

        d->xRatioUV = super.xRatioUV;
        d->yRatioUV = super.yRatioUV;

        MotionBlockPyramid vectors(d->vectors, d->prefix, vsapi);

        vectors.ScaleThSCD(d->nSCD1, d->nSCD2, vectors.bitsPerSample);

        d->deltaFrame = vectors.nDeltaFrame;

        if (d->fields && vectors.nPel < 2)
            throw std::runtime_error("fields option requires pel > 1");

        d->thSAD = (int64_t)(d->thSAD * vectors.GetThSCDScaleFactor(vectors.bitsPerSample) + 0.5);

        // accumulator is 1x pixel width for float (bytesPerSample 4), 2x for 8/16-bit integer.
        const int accRatio = (d->vi->format.bytesPerSample == 4) ? 1 : 2;
        d->dstTempPitch = ((vectors.nWidth + 15) / 16) * 16 * d->vi->format.bytesPerSample * accRatio;
        d->dstTempPitchUV = (((vectors.nWidth / d->xRatioUV) + 15) / 16) * 16 * d->vi->format.bytesPerSample * accRatio;

        d->supervi = vsapi->getVideoInfo(d->super);

        if (!vectors.IsCompatibleWithAnalysis(super))
            throw std::runtime_error("wrong source or super clip frame size");;

        d->chroma = (d->vi->format.colorFamily != cfGray);

        if (vectors.nOverlapX > 0 || vectors.nOverlapY > 0) {
            d->OverWins.Init(vectors.nBlkSizeX, vectors.nBlkSizeY, vectors.nOverlapX, vectors.nOverlapY);
            if (d->chroma)
                d->OverWinsUV.Init(vectors.nBlkSizeX / d->xRatioUV, vectors.nBlkSizeY / d->yRatioUV, vectors.nOverlapX / d->xRatioUV, vectors.nOverlapY / d->yRatioUV);
        }

        const unsigned bits = d->vi->format.bytesPerSample * 8;

        d->OVERS[0] = selectOverlapsFunction(vectors.nBlkSizeX, vectors.nBlkSizeY, bits);
        d->BLIT[0] = selectCopyFunction(vectors.nBlkSizeX, vectors.nBlkSizeY, bits);

        d->OVERS[1] = d->OVERS[2] = selectOverlapsFunction(vectors.nBlkSizeX / d->xRatioUV, vectors.nBlkSizeY / d->yRatioUV, bits);
        d->BLIT[1] = d->BLIT[2] = selectCopyFunction(vectors.nBlkSizeX / d->xRatioUV, vectors.nBlkSizeY / d->yRatioUV, bits);

        d->time256 = (int)(time * 256 / 100);

    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("Compensate: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[3] = { 
        {d->node, rpStrictSpatial},
        {d->super, rpGeneral},
        {d->vectors, rpNoFrameReuse},
    };

    vsapi->createVideoFilter(out, "Compensate", d->vi, SelectOnBitsPerSample(d->vi->format.bitsPerSample, compensateGetFrame<uint8_t>, compensateGetFrame<uint16_t>, compensateGetFrame<float>), filterFree<CompensateData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}


void compensateRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Compensate",
                 "clip:vnode;"
                 "super:vnode;"
                 "vectors:vnode;"
                 "thsad:int:opt;"
                 "fields:int:opt;"
                 "time:float:opt;"
                 "thscd1:int:opt;"
                 "thscd2:float:opt;"
                 "tff:int:opt;"
                 "prefix:data:opt;",
                 "clip:vnode;",
                 compensateCreate, nullptr, plugin);
}
