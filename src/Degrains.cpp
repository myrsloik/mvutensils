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

#include <stdexcept>
#include <string>
#include <memory>
#include <cmath>
#include <limits>
#include <array>
#include <utility>

#include <VapourSynth4.h>

#include "Degrains.h"
#include "FunctionTable.h"
#include "Overlap.h"
#include "Common.h"
#include "CPU.h"


static inline int64_t interpolateThSAD(int64_t thsad, int64_t thsad2, int d, int tr) noexcept {
    if (d <= 1 || tr <= 1)
        return thsad;

    constexpr double kPi = 3.14159265358979323846;
    const double x = (d - 1) * kPi / (tr - 1);
    const double lerp = (1.0 - std::cos(x)) * 0.5; // 0 at d==1, 1 at d==tr
    return static_cast<int64_t>(std::floor(thsad + lerp * static_cast<double>(thsad2 - thsad) + 0.5));
}


template<int radius>
struct DegrainData {
    VSNode *node = nullptr;
    VSNode *super = nullptr;
    VSNode *vectors[radius * 2] = {};
    int deltaFrame[radius * 2] = {};

    const VSVideoInfo *vi = nullptr;

    int64_t thSAD[radius * 2][3];
    int nLimit[3];
    float fLimit[3];
    bool needsLimit[3];
    int64_t nSCD1;
    float nSCD2;

    int userWeights[radius * 2 + 1];

    ptrdiff_t dstTempPitch;

    OverlapsFunction OVERS[3];
    DenoiseFunction DEGRAIN[3];

    bool process[3];

    int xSubUV;
    int ySubUV;

    int nWidth[3];
    int nHeight[3];
    int nOverlapX[3];
    int nOverlapY[3];
    int nBlkSizeX[3];
    int nBlkSizeY[3];
    int nWidth_B[3];
    int nHeight_B[3];

    OverlapWindows OverWins[3];

    std::string prefix;
    std::string filterName;

    const VSAPI *vsapi;

    DegrainData(const VSAPI *vsapi) : vsapi(vsapi) {}

    ~DegrainData() {
        for (int r = 0; r < radius * 2; r++) {
            if (vectors[r])
                vsapi->freeNode(vectors[r]);
        }
        if (super)
            vsapi->freeNode(super);
        if (node)
            vsapi->freeNode(node);
    }
};

template<int radius, typename PixelType>
static const VSFrame *VS_CC degrainGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    DegrainData<radius> *d = reinterpret_cast<DegrainData<radius> *>(instanceData);

    // FIXME, investigate weird vector delta frame checks
    if (activationReason == arInitial) {

        for (int r = 0; r < radius * 2; r += 2) {
            //Backward
            vsapi->requestFrameFilter(n, d->vectors[r], frameCtx);
            //Forward
            vsapi->requestFrameFilter(n, d->vectors[r + 1], frameCtx);

            // Backward
            int offB = d->deltaFrame[r];
            if (n + offB < d->vi->numFrames && n + offB >= 0)
                vsapi->requestFrameFilter(n + offB, d->super, frameCtx);

            vsapi->requestFrameFilter(n, d->super, frameCtx);

            // Forward
            int offF = d->deltaFrame[r + 1];
            if (n + offF >= 0 && n + offF < d->vi->numFrames)
                vsapi->requestFrameFilter(n + offF, d->super, frameCtx);
        }

        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);

        const int pl[] = { 0, 1, 2 };
        const VSFrame *fr[] = {
            d->process[0] ? nullptr : src,
            d->process[1] ? nullptr : src,
            d->process[2] ? nullptr : src
        };

        VSFrame *dst = vsapi->newVideoFrame2(&d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);
        vsapi->freeFrame(src);

        int bitsPerSample = d->vi->format.bitsPerSample;
        
        constexpr int bytesPerSample = sizeof(PixelType);

        uint8_t *pDst[3] = {};
        uint8_t *pDstCur[3] = {};
        const uint8_t *pSrcCur[3] = {};
        const uint8_t *pSrc[3] = {};
        ptrdiff_t nDstPitches[3] = {};
        ptrdiff_t nSrcPitches[3] = {};
        bool isUsable[radius * 2] = {};

        try {
            std::optional<MotionBlockPyramid> fgops[radius * 2];
            std::optional<FramePyramid> pRefGOF[radius * 2];

            for (int r = 0; r < radius * 2; r++) {
                const VSFrame *frame = vsapi->getFrameFilter(n, d->vectors[r], frameCtx);
                fgops[r].emplace(frame, 1, d->prefix, vsapi);
                isUsable[r] = fgops[r]->IsUsable(d->nSCD1, d->nSCD2);

                if (isUsable[r]) {
                    if (fgops[r]->nDeltaFrame != d->deltaFrame[r])
                        throw std::runtime_error("vector clip " + std::to_string(r) + " reports delta " + std::to_string(fgops[r]->nDeltaFrame) +
                            " at frame " + std::to_string(n) + " but was created with delta " + std::to_string(d->deltaFrame[r]) +
                            "; the delta must be constant for the whole clip");

                    int offset = d->deltaFrame[r];
                    if (n + offset >= 0 && n + offset < d->vi->numFrames)
                        pRefGOF[r].emplace(vsapi->getFrameFilter(n + offset, d->super, frameCtx), 1, d->prefix, vsapi);
                    else
                        isUsable[r] = false; // reference out of range was never requested at arInitial; treat as unusable
                }
            }

            int nLogPel = (fgops[0]->nPel == 4) ? 2 : (fgops[0]->nPel == 2) ? 1 : 0;

            FramePyramid pSrcFrame(vsapi->getFrameFilter(n, d->super, frameCtx), 1, d->prefix, vsapi);
            const auto &srcLevel = pSrcFrame.GetLevel(0);

            for (int i = 0; i < d->vi->format.numPlanes; i++) {
                pDst[i] = vsapi->getWritePtr(dst, i);
                nDstPitches[i] = vsapi->getStride(dst, i);
                pSrc[i] = srcLevel.planes[i].GetPointer<PixelType>(0, 0);
                nSrcPitches[i] = srcLevel.planes[i].nPitch;
            }

            const int xSubUV = d->xSubUV;
            const int ySubUV = d->ySubUV;
            const int nBlkX = fgops[0]->nBlkX;
            const int nBlkY = fgops[0]->nBlkY;
            const ptrdiff_t dstTempPitch = d->dstTempPitch;
            const int *nOverlapX = d->nOverlapX;
            const int *nOverlapY = d->nOverlapY;
            const int *nBlkSizeX = d->nBlkSizeX;
            const int *nBlkSizeY = d->nBlkSizeY;
            const int *nWidth_B = d->nWidth_B;
            const auto *thSAD = d->thSAD;

            OverlapWindows *OverWins[3] = { &d->OverWins[0], &d->OverWins[1], &d->OverWins[2] };
            MvuAlignedPtr<uint8_t> DstTempAlloc(nullptr, mvu_aligned_free);
            uint8_t *DstTemp = nullptr;
            int tmpBlockPitch = nBlkSizeX[0] * bytesPerSample;
            if (nOverlapX[0] > 0 || nOverlapY[0] > 0) {
                DstTempAlloc.reset(mvu_aligned_malloc<uint8_t>(dstTempPitch * nBlkSizeY[0], MVU_MEMORY_ALIGN));
                DstTemp = DstTempAlloc.get();
            }

            auto tmpBlockAlloc = mvu_make_aligned<uint8_t>(tmpBlockPitch * nBlkSizeY[0]);
            uint8_t *tmpBlock = tmpBlockAlloc.get();

            const FramePyramidLevel *pPlanes[radius * 2] = {};

            for (int r = 0; r < radius * 2; r++) {
                if (isUsable[r])
                    pPlanes[r] = &pRefGOF[r]->GetLevel(0);
            }

            pDstCur[0] = pDst[0];
            pDstCur[1] = pDst[1];
            pDstCur[2] = pDst[2];
            pSrcCur[0] = pSrc[0];
            pSrcCur[1] = pSrc[1];
            pSrcCur[2] = pSrc[2];
            // -----------------------------------------------------------------------------

            for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
                if (!d->process[plane])
                    continue;

                const ptrdiff_t nRefPitch = nSrcPitches[plane];

                if (nOverlapX[0] == 0 && nOverlapY[0] == 0) {
                    const int frameW = vsapi->getFrameWidth(dst, plane);
                    const int frameH = vsapi->getFrameHeight(dst, plane);

                    for (int by = 0; by < nBlkY; by++) {
                        const int dstPixY = by * nBlkSizeY[plane];
                        const int validH = std::min(nBlkSizeY[plane], frameH - dstPixY);
                        if (validH <= 0)
                            break;

                        int xx = 0;
                        for (int bx = 0; bx < nBlkX; bx++) {
                            int i = by * nBlkX + bx;

                            const uint8_t *pointers[radius * 2];
                            uint16_t WSrc, WRefs[radius * 2];

                            for (int r = 0; r < radius * 2; r++)
                                useBlock<PixelType>(pointers[r], WRefs[r], isUsable[r], fgops[r], i, pPlanes[r], pSrcCur, xx, nLogPel, plane, xSubUV, ySubUV, thSAD[r]);

                            normaliseWeights<radius>(WSrc, WRefs, d->userWeights);

                            const int dstPixX = xx / bytesPerSample;
                            const int validW = std::min(nBlkSizeX[plane], frameW - dstPixX);

                            if (validW == nBlkSizeX[plane] && validH == nBlkSizeY[plane]) {
                                // Block fits entirely — write directly
                                d->DEGRAIN[plane](pDstCur[plane] + xx, nDstPitches[plane], pSrcCur[plane] + xx, nSrcPitches[plane],
                                    pointers, nRefPitch, WSrc, WRefs);
                            } else if (validW > 0) {
                                // Edge block — write to tmpBlock, then copy only the valid region
                                d->DEGRAIN[plane](tmpBlock, tmpBlockPitch, pSrcCur[plane] + xx, nSrcPitches[plane],
                                    pointers, nRefPitch, WSrc, WRefs);
                                mvu_bitblt(pDstCur[plane] + xx, nDstPitches[plane],
                                    tmpBlock, tmpBlockPitch,
                                    validW * bytesPerSample, validH);
                            }

                            xx += nBlkSizeX[plane] * bytesPerSample;
                        }

                        pDstCur[plane] += nBlkSizeY[plane] * nDstPitches[plane];
                        pSrcCur[plane] += nBlkSizeY[plane] * nSrcPitches[plane];
                    }
                } else { // overlap - sliding window
                    const int stepY = nBlkSizeY[plane] - nOverlapY[plane];

                    // Clear the whole buffer for the first block row
                    memset(DstTemp, 0, dstTempPitch * nBlkSizeY[plane]);

                    for (int by = 0; by < nBlkY; by++) {
                        int wby = (by == 0) ? 0 : (by == nBlkY - 1) ? 6 : 3;
                        int wbx = 0;
                        int xx = 0;

                        // For subsequent rows only clear the new (non-overlap) region;
                        // the top nOverlapY rows were preserved by the memmove below
                        if (by > 0)
                            memset(DstTemp + nOverlapY[plane] * dstTempPitch, 0, dstTempPitch * stepY);

                        for (int bx = 0; bx < nBlkX; bx++) {
                            // select window
                            wbx = bx == nBlkX - 1 ? 2 : wbx;
                            const int16_t *winOver = OverWins[plane]->GetWindow(wby + wbx);

                            int i = by * nBlkX + bx;

                            const uint8_t *pointers[radius * 2];
                            uint16_t WSrc, WRefs[radius * 2];

                            for (int r = 0; r < radius * 2; r++)
                                useBlock<PixelType>(pointers[r], WRefs[r], isUsable[r], fgops[r], i, pPlanes[r], pSrcCur, xx, nLogPel, plane, xSubUV, ySubUV, thSAD[r]);

                            normaliseWeights<radius>(WSrc, WRefs, d->userWeights);

                            d->DEGRAIN[plane](tmpBlock, tmpBlockPitch, pSrcCur[plane] + xx, nSrcPitches[plane],
                                pointers, nRefPitch,
                                WSrc, WRefs);
                            // accumulator is 1x pixel width for float, 2x for 8/16-bit integer.
                            constexpr int accRatio = std::is_floating_point_v<PixelType> ? 1 : 2;
                            d->OVERS[plane](DstTemp + xx * accRatio, dstTempPitch, tmpBlock, tmpBlockPitch, winOver, nBlkSizeX[plane]);

                            xx += (nBlkSizeX[plane] - nOverlapX[plane]) * bytesPerSample;
                            wbx = 1;
                        }

                        // Last block row outputs all nBlkSizeY rows; others output only stepY
                        int rowsToOutput = (by == nBlkY - 1) ? nBlkSizeY[plane] : stepY;
                        int outputHeight = std::min(rowsToOutput, vsapi->getFrameHeight(dst, plane) - by * stepY);
                        int outputWidth = std::min(nWidth_B[plane], vsapi->getFrameWidth(dst, plane));

                        if (outputHeight > 0)
                            ToPixels<PixelType>(pDstCur[plane], nDstPitches[plane], DstTemp, dstTempPitch,
                                outputWidth, outputHeight, bitsPerSample);

                        pDstCur[plane] += nDstPitches[plane] * rowsToOutput;
                        pSrcCur[plane] += stepY * nSrcPitches[plane];

                        // Slide: preserve the overlap rows at the top for the next block row
                        if (by < nBlkY - 1)
                            memmove(DstTemp, DstTemp + stepY * dstTempPitch, nOverlapY[plane] * dstTempPitch);
                    }
                }

                if (d->needsLimit[plane]) {
                    if constexpr (std::is_integral_v<PixelType>)
                        LimitChanges_C<PixelType>(pDst[plane], nDstPitches[plane],
                            pSrc[plane], nSrcPitches[plane],
                            vsapi->getFrameWidth(dst, plane), vsapi->getFrameHeight(dst, plane), d->nLimit[plane]);
                    else
                        LimitChanges_C<PixelType>(pDst[plane], nDstPitches[plane],
                            pSrc[plane], nSrcPitches[plane],
                            vsapi->getFrameWidth(dst, plane), vsapi->getFrameHeight(dst, plane), d->fLimit[plane]);
                }
            }

            return dst;

        } catch (const std::exception &e) {
            vsapi->freeFrame(dst);
            vsapi->setFilterError((d->filterName + ": " + e.what()).c_str(), frameCtx);
            return nullptr;
        }
    }

    return nullptr;
}


#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8

#define DEGRAIN(radius, width, height) \
    { KEY(width, height, 8), Degrain_C8<radius, width, height> }, \
    { KEY(width, height, 16), Degrain_C16<radius, width, height> }, \
    { KEY(width, height, 32), Degrain_F32<radius, width, height> },

#define DEGRAIN_LEVEL(radius)\
    {\
        DEGRAIN(radius, 2, 2)\
        DEGRAIN(radius, 2, 4)\
        DEGRAIN(radius, 4, 2)\
        DEGRAIN(radius, 4, 4)\
        DEGRAIN(radius, 4, 8)\
        DEGRAIN(radius, 8, 1)\
        DEGRAIN(radius, 8, 2)\
        DEGRAIN(radius, 8, 4)\
        DEGRAIN(radius, 8, 8)\
        DEGRAIN(radius, 8, 16)\
        DEGRAIN(radius, 16, 1)\
        DEGRAIN(radius, 16, 2)\
        DEGRAIN(radius, 16, 4)\
        DEGRAIN(radius, 16, 8)\
        DEGRAIN(radius, 16, 16)\
        DEGRAIN(radius, 16, 32)\
        DEGRAIN(radius, 32, 8)\
        DEGRAIN(radius, 32, 16)\
        DEGRAIN(radius, 32, 32)\
        DEGRAIN(radius, 32, 64)\
        DEGRAIN(radius, 64, 16)\
        DEGRAIN(radius, 64, 32)\
        DEGRAIN(radius, 64, 64)\
        DEGRAIN(radius, 64, 128)\
        DEGRAIN(radius, 128, 32)\
        DEGRAIN(radius, 128, 64)\
        DEGRAIN(radius, 128, 128)\
    }

// One table per radius, generated for 1..kMaxDegrainRadius so the cap lives in a single constant.
// consteval, not constexpr: these only ever build the table below, and a stray runtime call would
// materialise the whole array on the stack instead of using the one baked into the image.
template <int Radius>
static consteval auto makeDegrainTable() {
    return std::to_array<FunctionTableEntry<DenoiseFunction>>(DEGRAIN_LEVEL(Radius));
}

template <int... Is>
static consteval auto makeDegrainTables(std::integer_sequence<int, Is...>) {
    return std::array{ makeDegrainTable<Is + 1>()... };
}

static constexpr auto degrain_functions = makeDegrainTables(std::make_integer_sequence<int, kMaxDegrainRadius>{});

static DenoiseFunction selectDegrainFunction(unsigned radius, unsigned width, unsigned height, unsigned bits) {
    if (radius < 1 || radius > kMaxDegrainRadius)
        throw std::out_of_range("degrain radius out of range");

    return findFunctionOrThrow(degrain_functions[radius - 1], KEY(width, height, bits));
}

#undef DEGRAIN
#undef DEGRAIN_LEVEL

#undef KEY


template <int radius>
static void selectFunctions(DegrainData<radius> &d, const MotionBlockPyramid &vectors) {
    const unsigned xRatioUV = vectors.xRatioUV;
    const unsigned yRatioUV = vectors.yRatioUV;
    const unsigned nBlkSizeX = vectors.nBlkSizeX;
    const unsigned nBlkSizeY = vectors.nBlkSizeY;
    const unsigned bits = d.vi->format.bytesPerSample * 8;

    d.OVERS[0] = selectOverlapsFunction(nBlkSizeX, nBlkSizeY, bits);
    d.DEGRAIN[0] = selectDegrainFunction(radius, nBlkSizeX, nBlkSizeY, bits);

    d.OVERS[1] = d.OVERS[2] = selectOverlapsFunction(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bits);
    d.DEGRAIN[1] = d.DEGRAIN[2] = selectDegrainFunction(radius, nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bits);
}

static inline void getProcessPlanesArg(const VSMap *in, bool process[3], const VSAPI *vsapi) {
    int m = vsapi->mapNumElements(in, "planes");

    for (int i = 0; i < 3; i++)
        process[i] = (m <= 0);

    for (int i = 0; i < m; i++) {
        int64_t o = vsapi->mapGetInt(in, "planes", i, nullptr);

        if (o < 0 || o >= 3)
            throw std::runtime_error("plane index out of range");


        if (process[o])
            throw std::runtime_error("plane specified twice");

        process[o] = true;
    }
}

template <int radius>
static void VS_CC degrainCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<DegrainData<radius>> d(new DegrainData<radius>(vsapi));

    d->filterName = "Degrain" + std::to_string(radius);

    int err;

    d->nSCD1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
    if (err)
        d->nSCD1 = MV_DEFAULT_SCD1;

    d->nSCD2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
    if (err)
        d->nSCD2 = MV_DEFAULT_SCD2;

    d->super = vsapi->mapGetNode(in, "super", 0, nullptr);

    const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
    if (prefix)
        d->prefix = prefix;
    else
        d->prefix = DEFAULT_MVUTENSILS_PREFIX;

    try {
        getProcessPlanesArg(in, d->process, vsapi);

        // Optional per-reference weights, given in temporal order
        // [bw_radius, ..., bw_1, centre, fw_1, ..., fw_radius] and reordered into
        // userWeights to match the [centre, bw_1, fw_1, bw_2, fw_2, ...] layout
        // normaliseWeights expects. Absent means all weights are 1 (unweighted).
        int weightCount = vsapi->mapNumElements(in, "weights");
        if (weightCount == -1) {
            for (int r = 0; r <= radius * 2; r++)
                d->userWeights[r] = 1;
        } else if (weightCount != radius * 2 + 1) {
            throw std::runtime_error("weights, if given, must have exactly " + std::to_string(radius * 2 + 1) + " elements");
        } else {
            d->userWeights[0] = vsapi->mapGetIntSaturated(in, "weights", radius, nullptr);
            for (int r = 0; r < radius; r++) {
                d->userWeights[r * 2 + 1] = vsapi->mapGetIntSaturated(in, "weights", radius - (r + 1), nullptr);
                d->userWeights[r * 2 + 2] = vsapi->mapGetIntSaturated(in, "weights", radius + (r + 1), nullptr);
            }
            // Largest weight that keeps normaliseWeights' int WSum accumulation
            // (256 * weight summed over 2*radius+1 terms, plus 1) within INT_MAX.
            // Only the ratios matter, so this ceiling is far beyond any real use.
            constexpr int maxWeight = (std::numeric_limits<int>::max() - 1) / (256 * (radius * 2 + 1));
            for (int r = 0; r <= radius * 2; r++)
                if (d->userWeights[r] < 0 || d->userWeights[r] > maxWeight)
                    throw std::runtime_error("weights must be between 0 and " + std::to_string(maxWeight));
        }

        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        FramePyramid super(d->super, d->prefix, vsapi);

        int numVectors = vsapi->mapNumElements(in, "vectors");
        if (numVectors != radius * 2)
            throw std::runtime_error("the number of vector clips must be exactly " + std::to_string(radius * 2));

        std::optional<MotionBlockPyramid> vectors[radius * 2];

        for (int r = 0; r < radius * 2; r++) {
            d->vectors[r] = vsapi->mapGetNode(in, "vectors", r, nullptr);

            vectors[r].emplace(d->vectors[r], d->prefix, vsapi);

            if (r > 0 && !vectors[r]->IsCompatible(*vectors[r - 1]))
                throw std::runtime_error("The motion vectors passed are not compatible with each other");

            if (!vectors[r]->IsCompatibleWithAnalysis(super))
                throw std::runtime_error("The motion vectors passed are not compatible with the super clip");

            d->deltaFrame[r] = vectors[r]->nDeltaFrame;

            if (r % 2 == 1) {
                if (d->deltaFrame[r] != -d->deltaFrame[r - 1])
                    throw std::runtime_error("forward and backward vector clips must be symmetric in their delta frame");
                if (r >= 2) {
                    if (abs(d->deltaFrame[r - 2]) >= abs(d->deltaFrame[r]))
                        throw std::runtime_error("vector clips must have increasing number of delta frames");
                }
            }
        }

        if (!super.IsCompatibleWithSource(d->vi))
            throw std::runtime_error("super clip is not compatible with the source clip");

        int64_t thsadRaw[3], thsad2Raw[3];
        GetHVPairArgument(thsadRaw[0], thsadRaw[1], "thsad", 400, 400, in, vsapi);
        thsadRaw[2] = thsadRaw[1];
        GetHVPairArgument(thsad2Raw[0], thsad2Raw[1], "thsad2", thsadRaw[0], thsadRaw[1], in, vsapi);
        thsad2Raw[2] = thsad2Raw[1];

        vectors[0]->ScaleThSCD(d->nSCD1, d->nSCD2, vectors[0]->bitsPerSample);
        double thscdScale = vectors[0]->GetThSCDScaleFactor(vectors[0]->bitsPerSample);

        int64_t thsadScaled[3], thsad2Scaled[3];
        for (int p = 0; p < 3; p++) {
            thsadScaled[p] = static_cast<int64_t>(thsadRaw[p] * thscdScale + .5);
            thsad2Scaled[p] = static_cast<int64_t>(thsad2Raw[p] * thscdScale + .5);
        }

        for (int p = 0; p < 2; p++) {
            if (thsadScaled[p] >= std::numeric_limits<int>::max() || thsad2Scaled[p] >= std::numeric_limits<int>::max()) {
                int64_t maximum = static_cast<int64_t>((std::numeric_limits<int>::max() - 1) / thscdScale);
                std::string which = (p == 1) ? "thsad/thsad2 chroma values" : "thsad/thsad2 luma values";
                throw std::runtime_error("with this block size and video format, the " + which + " must not exceed " + std::to_string(maximum) + " or some calculations would overflow");
            }
        }

        for (int r = 0; r < radius * 2; r++)
            for (int p = 0; p < 3; p++)
                d->thSAD[r][p] = interpolateThSAD(thsadScaled[p], thsad2Scaled[p], r / 2 + 1, radius);

        GetHVPairArgument(d->fLimit[0], d->fLimit[1], "limit", std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), in, vsapi);
        d->fLimit[2] = d->fLimit[1];
        for (int i = 0; i < 3; i++) {
            d->needsLimit[i] = std::isfinite(d->fLimit[i]);

            if (d->needsLimit[i]) {
                if (d->fLimit[i] <= 0.0f)
                    throw std::runtime_error("limit must be non-negative");
                if (d->vi->format.sampleType == stInteger) {
                    int pixelMax = (1 << d->vi->format.bitsPerSample) - 1;
                    if (d->fLimit[i] >= (float)pixelMax)
                        d->needsLimit[i] = false;
                    else
                        d->nLimit[i] = static_cast<int>(d->fLimit[i] + 0.5f);
                }
            }
        }

        // accumulator is 1x pixel width for float (bytesPerSample 4), 2x for 8/16-bit integer.
        d->dstTempPitch = ((vectors[0]->nWidth + 15) / 16) * 16 * d->vi->format.bytesPerSample * (d->vi->format.bytesPerSample == 4 ? 1 : 2);

        d->xSubUV = d->vi->format.subSamplingW;
        d->ySubUV = d->vi->format.subSamplingH;

        d->nWidth[0] = vectors[0]->nWidth;
        d->nWidth[1] = d->nWidth[2] = d->nWidth[0] >> d->xSubUV;

        d->nHeight[0] = vectors[0]->nHeight;
        d->nHeight[1] = d->nHeight[2] = d->nHeight[0] >> d->ySubUV;

        d->nOverlapX[0] = vectors[0]->nOverlapX;
        d->nOverlapX[1] = d->nOverlapX[2] = d->nOverlapX[0] >> d->xSubUV;

        d->nOverlapY[0] = vectors[0]->nOverlapY;
        d->nOverlapY[1] = d->nOverlapY[2] = d->nOverlapY[0] >> d->ySubUV;

        d->nBlkSizeX[0] = vectors[0]->nBlkSizeX;
        d->nBlkSizeX[1] = d->nBlkSizeX[2] = d->nBlkSizeX[0] >> d->xSubUV;

        d->nBlkSizeY[0] = vectors[0]->nBlkSizeY;
        d->nBlkSizeY[1] = d->nBlkSizeY[2] = d->nBlkSizeY[0] >> d->ySubUV;

        d->nWidth_B[0] = vectors[0]->nBlkX * (d->nBlkSizeX[0] - d->nOverlapX[0]) + d->nOverlapX[0];
        d->nWidth_B[1] = d->nWidth_B[2] = d->nWidth_B[0] >> d->xSubUV;

        d->nHeight_B[0] = vectors[0]->nBlkY * (d->nBlkSizeY[0] - d->nOverlapY[0]) + d->nOverlapY[0];
        d->nHeight_B[1] = d->nHeight_B[2] = d->nHeight_B[0] >> d->ySubUV;

        if (d->nOverlapX[0] || d->nOverlapY[0]) {
            d->OverWins[0].Init(d->nBlkSizeX[0], d->nBlkSizeY[0], d->nOverlapX[0], d->nOverlapY[0]);

            if (d->vi->format.colorFamily != cfGray) {
                d->OverWins[1].Init(d->nBlkSizeX[1], d->nBlkSizeY[1], d->nOverlapX[1], d->nOverlapY[1]);
                d->OverWins[2].Init(d->nBlkSizeX[2], d->nBlkSizeY[2], d->nOverlapX[2], d->nOverlapY[2]);
            }
        }

        selectFunctions<radius>(*d, *vectors[0]);

        const int numDeps = 2 + radius * 2; // input clip, super, and corresponding backward and forward vectors.
        std::vector<VSFilterDependency> deps;
        deps.reserve(numDeps);
        deps.push_back({ d->node, rpStrictSpatial });
        deps.push_back({ d->super, rpGeneral });
        for (int r = 0; r < radius * 2; r++)
            deps.push_back({ d->vectors[r], rpStrictSpatial });

        assert(numDeps == deps.size());

        vsapi->createVideoFilter(out, d->filterName.c_str(), d->vi, SelectOnBitsPerSample(d->vi->format.bitsPerSample, degrainGetFrame<radius, uint8_t>, degrainGetFrame<radius, uint16_t>, degrainGetFrame<radius, float>), filterFree<DegrainData<radius>>, fmParallel, deps.data(), numDeps, d.get(), core);
        d.release();

    } catch (const std::exception &e) {
        vsapi->mapSetError(out, (d->filterName + ": " + e.what()).c_str());
    }
}

static void VS_CC degrainNCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    int numElems = vsapi->mapNumElements(in, "vectors");
    if (numElems % 2 != 0) {
        vsapi->mapSetError(out, "Degrain: number of vectors must be even");
        return;
    }

    numElems /= 2;

    if (numElems < 1 || numElems > kMaxDegrainRadius) {
        vsapi->mapSetError(out, ("Degrain: number of vector pairs must be between 1 and " + std::to_string(kMaxDegrainRadius)).c_str());
        return;
    }

    std::string functionName = "Degrain" + std::to_string(numElems);
    VSPlugin *thisPlugin = vsapi->getPluginByID("com.vapoursynth.mvutensils", core);

    VSMap *ret = vsapi->invoke(thisPlugin, functionName.c_str(), in);
    if (vsapi->mapGetError(ret)) {
        vsapi->mapSetError(out, ("Degrain: " + std::string(vsapi->mapGetError(ret))).c_str());
    } else {
        vsapi->mapConsumeNode(out, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maAppend);
    }
    vsapi->freeMap(ret);
}

constexpr const char *degrain_args =
    "clip:vnode;"
    "super:vnode;"
    "vectors:vnode[];"
    "thsad:int[]:opt;"
    "thsad2:int[]:opt;"
    "planes:int[]:opt;"
    "limit:float[]:opt;"
    "thscd1:int:opt;"
    "thscd2:float:opt;"
    "weights:int[]:opt;"
    "prefix:data:opt;";

// Registers Degrain1 .. DegrainN for the whole 1..kMaxDegrainRadius range.
template <int... Is>
static void registerDegrainRadii(VSPlugin *plugin, const VSPLUGINAPI *vspapi, std::integer_sequence<int, Is...>) noexcept {
    (vspapi->registerFunction(("Degrain" + std::to_string(Is + 1)).c_str(),
                 degrain_args,
                 "clip:vnode;",
                 degrainCreate<Is + 1>, nullptr, plugin), ...);
}

void degrainsRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    registerDegrainRadii(plugin, vspapi, std::make_integer_sequence<int, kMaxDegrainRadius>{});
    vspapi->registerFunction("Degrain",
                 degrain_args,
                 "clip:vnode;",
                 degrainNCreate, nullptr, plugin);
}
