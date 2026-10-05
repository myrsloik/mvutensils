#include "SuperPyramid.h"
#include "Common.h"

#include <VSHelper4.h>
#include <cassert>
#include <memory>
#include <type_traits>

// Extra scanline(s) of slack allocated at the very end of the level-0 pel plane so the 32-bit SIMD
// gathers (FlowInter / flowFetch) can over-read up to 3 bytes past the last pixel without leaving the
// allocation. It is invisible to the layout: added to the frame height when the plane is created
// (CopyAndPadPlane) and subtracted back out wherever the height is read to reconstruct nPaddedHeight
// (FromExternalPlane / FromExternalPelPlanes). Only level 0 carries it (only level 0 is gathered).
static constexpr int kLevel0GatherGuardLines = 1;

// Reads an "h" / "h,v" list argument. Only the per-element getter differs by T; an if-constexpr lambda
// picks it (mapGetInt for int64_t, mapGetFloatSaturated for float, mapGetIntSaturated for int).
template <typename T>
void GetHVPairArgument(T &h, T &v, const char *name, std::type_identity_t<T> defaultH, std::type_identity_t<T> defaultV, const VSMap *in, const VSAPI *vsapi) {
    static_assert(std::is_same_v<T, int> || std::is_same_v<T, int64_t> || std::is_same_v<T, float>,
                  "GetHVPairArgument supports int, int64_t and float");
    int err;
    int numElems = vsapi->mapNumElements(in, name);
    if (numElems > 2)
        throw std::runtime_error(std::string("Too many values passed to ") + name);

    auto getElem = [&](int idx) -> T {
        if constexpr (std::is_same_v<T, int64_t>)
            return vsapi->mapGetInt(in, name, idx, &err);
        else if constexpr (std::is_same_v<T, float>)
            return vsapi->mapGetFloatSaturated(in, name, idx, &err);
        else
            return vsapi->mapGetIntSaturated(in, name, idx, &err);
    };

    h = getElem(0);
    if (err)
        h = defaultH;

    v = getElem(1);
    if (err)
        v = (numElems == 1) ? h : defaultV;
}

template void GetHVPairArgument<int>(int &, int &, const char *, int, int, const VSMap *, const VSAPI *);
template void GetHVPairArgument<int64_t>(int64_t &, int64_t &, const char *, int64_t, int64_t, const VSMap *, const VSAPI *);
template void GetHVPairArgument<float>(float &, float &, const char *, float, float, const VSMap *, const VSAPI *);

void CheckBlkSize(int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int subSamplingW, int subSamplingH, bool useSatd) {
    if (useSatd && nBlkSizeX == 16 && nBlkSizeY == 2)
        throw std::runtime_error("satd cannot work with 16x2 blocks");

    if ((nBlkSizeX != 4 || nBlkSizeY != 4) &&
        (nBlkSizeX != 8 || nBlkSizeY != 4) &&
        (nBlkSizeX != 8 || nBlkSizeY != 8) &&
        (nBlkSizeX != 16 || nBlkSizeY != 2) &&
        (nBlkSizeX != 16 || nBlkSizeY != 8) &&
        (nBlkSizeX != 16 || nBlkSizeY != 16) &&
        (nBlkSizeX != 32 || nBlkSizeY != 16) &&
        (nBlkSizeX != 32 || nBlkSizeY != 32) &&
        (nBlkSizeX != 64 || nBlkSizeY != 32) &&
        (nBlkSizeX != 64 || nBlkSizeY != 64) &&
        (nBlkSizeX != 128 || nBlkSizeY != 64) &&
        (nBlkSizeX != 128 || nBlkSizeY != 128))
        throw std::runtime_error("the block size must be 4x4, 8x4, 8x8, 16x2, 16x8, 16x16, 32x16, 32x32, 64x32, 64x64, 128x64 or 128x128");

    if (nOverlapX < 0 || nOverlapX > nBlkSizeX / 2 || nOverlapY < 0 || nOverlapY > nBlkSizeY / 2)
        throw std::runtime_error("overlap must be between 0 and half of blksize");

    if (nOverlapX % (1 << subSamplingW) || nOverlapY % (1 << subSamplingH))
        throw std::runtime_error("the specified overlap is incompatible with the super clip's subsampling");
}

template<typename PixelType>
void PyramidPlane::CopyAndPadPlane(const VSFrame *src, int plane, int hPad, int vPad, int nBlkSizePadX, int nBlkSizePadY, int nPel, VSCore *core, const VSAPI *vsapi) noexcept {
    const VSVideoFormat *format = vsapi->getVideoFrameFormat(src);
    VSVideoFormat dstFormat = {};
    vsapi->queryVideoFormat(&dstFormat, cfGray, format->sampleType, format->bitsPerSample, 0, 0, core);
    
    this->nPel = nPel;
    nRealWidth = vsapi->getFrameWidth(src, plane);
    nRealHeight = vsapi->getFrameHeight(src, plane);
    nWidth = nRealWidth + nBlkSizePadX;
    nHeight = nRealHeight + nBlkSizePadY;
    nPaddedWidth = nWidth + 2 * hPad;
    nPaddedHeight = nHeight + 2 * vPad;
    nHPadding = hPad;
    nVPadding = vPad;
    nHPaddingPel = nHPadding;
    nVPaddingPel = nVPadding;

    VSFrame *dst = vsapi->newVideoFrame(&dstFormat, nPaddedWidth, nPaddedHeight * nPel * nPel + kLevel0GatherGuardLines, nullptr, core);
    storage = dst;
    nPitch = vsapi->getStride(dst, 0);
    subPelPlaneOffset = nPitch * nPaddedHeight;
    nOffsetPadding = nPitch * nVPadding + nHPadding * sizeof(PixelType);

    const PixelType *srcP  = reinterpret_cast<const PixelType *>(vsapi->getReadPtr(src, plane));
    PixelType *dstP = reinterpret_cast<PixelType *>(vsapi->getWritePtr(dst, 0));
    ptrdiff_t srcPitch = vsapi->getStride(src, plane) / sizeof(PixelType);
    ptrdiff_t dstPitch = nPitch / sizeof(PixelType);

    // Copy frame data and pad sides by extending the edges
    dstP += dstPitch * vPad;

    for (int h = 0; h < nRealHeight; h++) {
        PixelType padValueLeft = srcP[0];
        for (int w = 0; w < hPad; w++)
            dstP[w] = padValueLeft;
        memcpy(dstP + hPad, srcP, nRealWidth * sizeof(PixelType));
        PixelType padValueRight = srcP[nRealWidth - 1];
        for (int w = nRealWidth + hPad; w < nPaddedWidth; w++)
            dstP[w] = padValueRight;
        srcP += srcPitch;
        dstP += dstPitch;
    }

    // Top and bottom padding by copying the first and last actual image line that's already been extended horizontally
    const PixelType *dstPLastLine = dstP - dstPitch;
    for (int h = 0; h < vPad + nBlkSizePadY; h++) {
        memcpy(dstP, dstPLastLine, nPitch);
        dstP += dstPitch;
    }

    dstP = reinterpret_cast<PixelType *>(vsapi->getWritePtr(dst, 0));
    const PixelType *dstPFirstLine = dstP + dstPitch * vPad;
    for (int h = 0; h < vPad; h++) {
        memcpy(dstP, dstPFirstLine, nPitch);
        dstP += dstPitch;
    }

    for (int p = 0; p < nPel * nPel; p++)
        pPlane[p] = vsapi->getWritePtr(dst, 0) + p * subPelPlaneOffset;
}

template <typename PixelType>
static void RB2F_C(uint8_t *MVU_RESTRICT pDst8, const uint8_t *MVU_RESTRICT pSrc8, ptrdiff_t nDstPitch,
    ptrdiff_t nSrcPitch, int nWidth, int nHeight) noexcept {
    PixelType *pDst = (PixelType *)pDst8;
    PixelType *pSrc = (PixelType *)pSrc8;

    nDstPitch /= sizeof(PixelType);
    nSrcPitch /= sizeof(PixelType);

    for (int y = 0; y < nHeight; y++) {
        for (int x = 0; x < nWidth; x++)
            pDst[x] = ShiftDivide<2, 2>(pSrc[x * 2] + pSrc[x * 2 + 1]
                + pSrc[x * 2 + nSrcPitch + 1] + pSrc[x * 2 + nSrcPitch]);

        pDst += nDstPitch;
        pSrc += nSrcPitch * 2;
    }
}

// separable BilinearFiltered with 1/8, 3/8, 3/8, 1/8 filter for smoothing and anti-aliasing
// interleaves vertical and horizontal downscaling row by row using tempBuffer as scratch
// tempBuffer must point to at least nWidth * 8 * sizeof(PixelType) bytes
template <typename PixelType>
static void RB2BilinearFiltered(uint8_t *pDst, const uint8_t * MVU_RESTRICT pSrc, ptrdiff_t nDstPitch,
    ptrdiff_t nSrcPitch, int nWidth, int nHeight, uint8_t * MVU_RESTRICT tempBuffer) noexcept {

    PixelType *dst = (PixelType *)pDst;
    const PixelType *src = (const PixelType *)pSrc;
    PixelType *tmp = (PixelType *)tempBuffer; // nWidth*2 entries used per row

    nDstPitch /= sizeof(PixelType);
    nSrcPitch /= sizeof(PixelType);

    const int srcWidth2 = nWidth * 2;

    for (int y = 0; y < nHeight; y++) {
        const PixelType *row = src + y * 2 * nSrcPitch;

        // Vertical filter: produce srcWidth2 intermediate samples into tmp
        if (y == 0 || y == nHeight - 1) {
            // Edge rows: simple average of the two source rows
            for (int x = 0; x < srcWidth2; x++)
                tmp[x] = ShiftDivide<1, 1>(row[x] + row[x + nSrcPitch]);
        } else {
            // Middle rows: 4-tap filter (1/8, 3/8, 3/8, 1/8) across rows 2y-1..2y+2
            for (int x = 0; x < srcWidth2; x++)
                tmp[x] = ShiftDivide<3, 4>(row[x - nSrcPitch] + (row[x] + row[x + nSrcPitch]) * 3 + row[x + nSrcPitch * 2]);
        }

        // Horizontal filter: reduce tmp from srcWidth2 to nWidth, write to dst row y
        PixelType *dstRow = dst + y * nDstPitch;

        dstRow[0] = ShiftDivide<1, 1>(tmp[0] + tmp[1]);

        for (int x = 1; x < nWidth - 1; x++)
            dstRow[x] = ShiftDivide<3, 4>(tmp[x * 2 - 1] + (tmp[x * 2] + tmp[x * 2 + 1]) * 3 + tmp[x * 2 + 2]);

        dstRow[nWidth - 1] = ShiftDivide<1, 1>(tmp[(nWidth - 1) * 2] + tmp[(nWidth - 1) * 2 + 1]);
    }
}

// separable filtered cubic with 1/32, 5/32, 10/32, 10/32, 5/32, 1/32 filter for smoothing and anti-aliasing
// interleaves vertical and horizontal downscaling row by row using tempBuffer as scratch
// tempBuffer must point to at least nWidth * 8 * sizeof(PixelType) bytes
template <typename PixelType>
static void RB2Cubic(uint8_t *pDst, const uint8_t *pSrc, ptrdiff_t nDstPitch,
    ptrdiff_t nSrcPitch, int nWidth, int nHeight, uint8_t *MVU_RESTRICT tempBuffer) noexcept {

    PixelType *dst = (PixelType *)pDst;
    const PixelType *src = (const PixelType *)pSrc;
    PixelType *tmp = (PixelType *)tempBuffer; // nWidth*2 entries used per row

    nDstPitch /= sizeof(PixelType);
    nSrcPitch /= sizeof(PixelType);

    using TempStorage = std::conditional_t<std::is_integral_v<PixelType>, int, float>;

    const int srcWidth2 = nWidth * 2;

    for (int y = 0; y < nHeight; y++) {
        const PixelType *row = src + y * 2 * nSrcPitch;

        // Vertical filter: produce srcWidth2 intermediate samples into tmp
        if (y == 0 || y == nHeight - 1) {
            // Edge rows: simple average of the two source rows
            for (int x = 0; x < srcWidth2; x++)
                tmp[x] = ShiftDivide<1, 1>(row[x] + row[x + nSrcPitch]);
        } else {
            // Middle rows: 6-tap filter (1/32, 5/32, 10/32, 10/32, 5/32, 1/32) across rows 2y-2..2y+3
            for (int x = 0; x < srcWidth2; x++) {
                TempStorage m0 = row[x - nSrcPitch * 2];
                TempStorage m1 = row[x - nSrcPitch];
                TempStorage m2 = row[x];
                TempStorage m3 = row[x + nSrcPitch];
                TempStorage m4 = row[x + nSrcPitch * 2];
                TempStorage m5 = row[x + nSrcPitch * 3];

                m2 = (m2 + m3) * 10;
                m1 = (m1 + m4) * 5;
                m0 += m5 + m2 + m1;
                tmp[x] = ShiftDivide<5, 16>(m0);
            }
        }

        // Horizontal filter: reduce tmp from srcWidth2 to nWidth, write to dst row y
        PixelType *dstRow = dst + y * nDstPitch;

        dstRow[0] = AveragePixels(tmp[0], tmp[1]);

        for (int x = 1; x < nWidth - 1; x++) {
            TempStorage m0 = tmp[x * 2 - 2];
            TempStorage m1 = tmp[x * 2 - 1];
            TempStorage m2 = tmp[x * 2];
            TempStorage m3 = tmp[x * 2 + 1];
            TempStorage m4 = tmp[x * 2 + 2];
            TempStorage m5 = tmp[x * 2 + 3];

            m2 = (m2 + m3) * 10;
            m1 = (m1 + m4) * 5;
            m0 += m5 + m2 + m1;
            dstRow[x] = ShiftDivide<5, 16>(m0);
        }

        dstRow[nWidth - 1] = AveragePixels(tmp[(nWidth - 1) * 2], tmp[(nWidth - 1) * 2 + 1]);
    }
}

static int PlaneDimensionLuma(int numPixels, int ratioUV, int pad) noexcept {
      return (pad >= ratioUV) ? ((numPixels / ratioUV + 1) / 2) * ratioUV : ((numPixels / ratioUV) / 2) * ratioUV;
}

// Level 0 size: the frame dimension grown to the smallest whole block grid covering it
int BlockAlignedDimension(int size, int blkSize, int overlap) noexcept {
    int step = blkSize - overlap;
    int size_B = step * ((size - overlap) / step) + overlap;
    return (size_B < size) ? size_B + step : size;
}

template<typename PixelType>
void PyramidPlane::ReducePlane(const PyramidPlane &src, int xRatioUV, int yRatioUV, RFilterParam rFilter, uint8_t *tempBuffer, VSCore *core, const VSAPI *vsapi) noexcept {
    nVPadding = src.nVPadding;
    nHPadding = src.nHPadding;

    nHPaddingPel = nHPadding;
    nVPaddingPel = nVPadding;

    nWidth = PlaneDimensionLuma(src.nWidth, xRatioUV, nHPadding);
    nHeight = PlaneDimensionLuma(src.nHeight, yRatioUV, nVPadding);

    nRealWidth = nWidth;
    nRealHeight = nHeight;

    nPaddedWidth = nWidth + 2 * nHPadding;
    nPaddedHeight = nHeight + 2 * nVPadding;

    VSFrame *dst = vsapi->newVideoFrame(vsapi->getVideoFrameFormat(src.storage), nPaddedWidth, nPaddedHeight, nullptr, core);
    storage = dst;
    uint8_t *dstP = vsapi->getWritePtr(dst, 0);
    pPlane[0] = dstP;
    nPitch = vsapi->getStride(dst, 0);
    nOffsetPadding = nPitch * nVPadding + nHPadding * sizeof(PixelType);

    if (rFilter == RFilterParam::Simple) {
        RB2F_C<PixelType>(dstP + nOffsetPadding, src.pPlane[0] + src.nOffsetPadding, nPitch, src.nPitch, nWidth, nHeight);
    } else if (rFilter == RFilterParam::Bilinear) {
        RB2BilinearFiltered<PixelType>(dstP + nOffsetPadding, src.pPlane[0] + src.nOffsetPadding, nPitch, src.nPitch, nWidth, nHeight, tempBuffer);
    } else if (rFilter == RFilterParam::Cubic) {
        RB2Cubic<PixelType>(dstP + nOffsetPadding, src.pPlane[0] + src.nOffsetPadding, nPitch, src.nPitch, nWidth, nHeight, tempBuffer);
    }

    PadPlaneData<PixelType>(0);
}


template <typename PixelType>
static void VerticalBilinear(uint8_t *MVU_RESTRICT pDst8, const uint8_t *MVU_RESTRICT pSrc8,
    ptrdiff_t nPitch, int nWidth, int nHeight, [[maybe_unused]] int bitsPerSample) noexcept {
    PixelType *pDst = (PixelType *)pDst8;
    PixelType *pSrc = (PixelType *)pSrc8;

    nPitch /= sizeof(PixelType);

    for (int j = 0; j < nHeight - 1; j++) {
        for (int i = 0; i < nWidth; i++)
            pDst[i] = AveragePixels(pSrc[i], pSrc[i + nPitch]);
        pDst += nPitch;
        pSrc += nPitch;
    }
    /* last row */
    for (int i = 0; i < nWidth; i++)
        pDst[i] = pSrc[i];
}


template <typename PixelType>
static void HorizontalBilinear(uint8_t *MVU_RESTRICT pDst8, const uint8_t *MVU_RESTRICT pSrc8,
    ptrdiff_t nPitch, int nWidth, int nHeight, [[maybe_unused]] int bitsPerSample) noexcept {
    PixelType *pDst = (PixelType *)pDst8;
    PixelType *pSrc = (PixelType *)pSrc8;

    nPitch /= sizeof(PixelType);

    for (int j = 0; j < nHeight; j++) {
        for (int i = 0; i < nWidth - 1; i++)
            pDst[i] = AveragePixels(pSrc[i], pSrc[i + 1]);

        pDst[nWidth - 1] = pSrc[nWidth - 1];
        pDst += nPitch;
        pSrc += nPitch;
    }
}


template <typename PixelType>
static void DiagonalBilinear(uint8_t *MVU_RESTRICT pDst8, const uint8_t *MVU_RESTRICT pSrc8,
    ptrdiff_t nPitch, int nWidth, int nHeight, [[maybe_unused]] int bitsPerSample) noexcept {
    PixelType *pDst = (PixelType *)pDst8;
    PixelType *pSrc = (PixelType *)pSrc8;

    nPitch /= sizeof(PixelType);

    for (int j = 0; j < nHeight - 1; j++) {
        for (int i = 0; i < nWidth - 1; i++)
            pDst[i] = AveragePixels(pSrc[i], pSrc[i + 1], pSrc[i + nPitch], pSrc[i + nPitch + 1]);

        pDst[nWidth - 1] = AveragePixels(pSrc[nWidth - 1], pSrc[nWidth + nPitch - 1]);
        pDst += nPitch;
        pSrc += nPitch;
    }
    for (int i = 0; i < nWidth - 1; i++)
        pDst[i] = AveragePixels(pSrc[i], pSrc[i + 1]);
    pDst[nWidth - 1] = pSrc[nWidth - 1];
}

// so called Wiener interpolation. (sharp, similar to Lanczos ?)
// invarint simplified, 6 taps. Weights: (1, -5, 20, 20, -5, 1)/32
template <typename PixelType>
static void VerticalWiener(uint8_t *MVU_RESTRICT pDst8, const uint8_t *MVU_RESTRICT pSrc8,
    ptrdiff_t nPitch, int nWidth, int nHeight, int bitsPerSample) noexcept {
    PixelType *pDst = (PixelType *)pDst8;
    PixelType *pSrc = (PixelType *)pSrc8;

    nPitch /= sizeof(PixelType);

    using TempStorage = std::conditional_t<std::is_integral_v<PixelType>, int, float>;

    int pixelMax = PixelMaxValue<PixelType>(bitsPerSample);

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < nWidth; i++)
            pDst[i] = AveragePixels(pSrc[i], pSrc[i + nPitch]);
        pDst += nPitch;
        pSrc += nPitch;
    }
    for (int j = 2; j < nHeight - 4; j++) {
        for (int i = 0; i < nWidth; i++) {
            TempStorage m0 = pSrc[i - nPitch * 2];
            TempStorage m1 = pSrc[i - nPitch];
            TempStorage m2 = pSrc[i];
            TempStorage m3 = pSrc[i + nPitch];
            TempStorage m4 = pSrc[i + nPitch * 2];
            TempStorage m5 = pSrc[i + nPitch * 3];

            m2 = (m2 + m3) * 4;

            m2 -= m1 + m4;
            m2 *= 5;

            m0 += m5 + m2;
            m0 = ShiftDivide<5, 16>(m0);

            pDst[i] = ClampIntToRange(m0, pixelMax);
        }
        pDst += nPitch;
        pSrc += nPitch;
    }
    for (intptr_t j = nHeight - 4; j < nHeight - 1; j++) {
        for (intptr_t i = 0; i < nWidth; i++) {
            pDst[i] = AveragePixels(pSrc[i], pSrc[i + nPitch]);
        }

        pDst += nPitch;
        pSrc += nPitch;
    }
    /* last row */
    for (int i = 0; i < nWidth; i++)
        pDst[i] = pSrc[i];
}


template <typename PixelType>
static void HorizontalWiener(uint8_t *MVU_RESTRICT pDst8, const uint8_t *MVU_RESTRICT pSrc8,
    ptrdiff_t nPitch, int nWidth, int nHeight, int bitsPerSample) noexcept {
    PixelType *pDst = (PixelType *)pDst8;
    PixelType *pSrc = (PixelType *)pSrc8;

    nPitch /= sizeof(PixelType);

    using TempStorage = std::conditional_t<std::is_integral_v<PixelType>, int, float>;

    int pixelMax = PixelMaxValue<PixelType>(bitsPerSample);

    for (int j = 0; j < nHeight; j++) {
        pDst[0] = AveragePixels(pSrc[0], pSrc[1]);
        pDst[1] = AveragePixels(pSrc[1], pSrc[2]);

        for (int i = 2; i < nWidth - 4; i++) {
            TempStorage m0 = pSrc[i - 2];
            TempStorage m1 = pSrc[i - 1];
            TempStorage m2 = pSrc[i];
            TempStorage m3 = pSrc[i + 1];
            TempStorage m4 = pSrc[i + 2];
            TempStorage m5 = pSrc[i + 3];

            m2 = (m2 + m3) * 4;

            m2 -= m1 + m4;
            m2 *= 5;

            m0 += m5 + m2;
            m0 = ShiftDivide<5, 16>(m0);

            pDst[i] = ClampIntToRange(m0, pixelMax);
        }

        for (intptr_t i = nWidth - 4; i < nWidth - 1; i++)
            pDst[i] = AveragePixels(pSrc[i], pSrc[i + 1]);

        pDst[nWidth - 1] = pSrc[nWidth - 1];
        pDst += nPitch;
        pSrc += nPitch;
    }
}


// bicubic (Catmull-Rom 4 taps interpolation)
template <typename PixelType>
static void VerticalBicubic(uint8_t *MVU_RESTRICT pDst8, const uint8_t *MVU_RESTRICT pSrc8,
    ptrdiff_t nPitch, int nWidth, int nHeight, int bitsPerSample) noexcept {
    PixelType *pDst = (PixelType *)pDst8;
    PixelType *pSrc = (PixelType *)pSrc8;

    nPitch /= sizeof(PixelType);

    int pixelMax = PixelMaxValue<PixelType>(bitsPerSample);

    for (int j = 0; j < 1; j++) {
        for (int i = 0; i < nWidth; i++)
            pDst[i] = AveragePixels(pSrc[i], pSrc[i + nPitch]);
        pDst += nPitch;
        pSrc += nPitch;
    }
    for (int j = 1; j < nHeight - 3; j++) {
        for (int i = 0; i < nWidth; i++) {
            pDst[i] = ClampIntToRange(ShiftDivide<4, 8>(-pSrc[i - nPitch] - pSrc[i + nPitch * 2] + (pSrc[i] + pSrc[i + nPitch]) * 9), pixelMax);
        }
        pDst += nPitch;
        pSrc += nPitch;
    }
    for (intptr_t j = nHeight - 3; j < nHeight - 1; j++) {
        for (int i = 0; i < nWidth; i++) {
            pDst[i] = AveragePixels(pSrc[i], pSrc[i + nPitch]);
        }

        pDst += nPitch;
        pSrc += nPitch;
    }
    /* last row */
    for (int i = 0; i < nWidth; i++)
        pDst[i] = pSrc[i];
}


template <typename PixelType>
static void HorizontalBicubic(uint8_t *MVU_RESTRICT pDst8, const uint8_t *MVU_RESTRICT pSrc8,
    ptrdiff_t nPitch, int nWidth, int nHeight, int bitsPerSample) noexcept {
    PixelType *pDst = (PixelType *)pDst8;
    PixelType *pSrc = (PixelType *)pSrc8;

    nPitch /= sizeof(PixelType);

    int pixelMax = PixelMaxValue<PixelType>(bitsPerSample);

    for (int j = 0; j < nHeight; j++) {
        pDst[0] = AveragePixels(pSrc[0], pSrc[1]);
        for (int i = 1; i < nWidth - 3; i++) {
            pDst[i] = ClampIntToRange(ShiftDivide<4, 8>(-(pSrc[i - 1] + pSrc[i + 2]) + (pSrc[i] + pSrc[i + 1]) * 9), pixelMax);
        }
        for (intptr_t i = nWidth - 3; i < nWidth - 1; i++)
            pDst[i] = AveragePixels(pSrc[i], pSrc[i + 1]);

        pDst[nWidth - 1] = pSrc[nWidth - 1];
        pDst += nPitch;
        pSrc += nPitch;
    }
}

template <typename PixelType>
static void GeneratePelQuarters(uint8_t *MVU_RESTRICT const P[16], ptrdiff_t nPitch, int nWidth, int nHeight) noexcept {
    const ptrdiff_t pp = nPitch / static_cast<ptrdiff_t>(sizeof(PixelType));
    auto row = [&](int i, int y) { return reinterpret_cast<PixelType *>(P[i] + static_cast<size_t>(y) * nPitch); };

    // The samples 3/4 of a pixel right (down) take the next column's (row's) full-pel samples, which the
    // last column (row) repeats, as the padding would go on. No vector the search allows reaches them;
    // they are filled so that no sample of the planes is left undefined
    const int last = nWidth - 1;
    for (int y = 0; y < nHeight; y++) {
        const PixelType *p0 = row(0, y), *p2 = row(2, y), *p8 = row(8, y), *p10 = row(10, y);
        PixelType *d1 = row(1, y), *d9 = row(9, y), *d4 = row(4, y), *d6 = row(6, y);
        PixelType *d5 = row(5, y), *d3 = row(3, y), *d11 = row(11, y), *d7 = row(7, y);
        PixelType *d12 = row(12, y), *d14 = row(14, y), *d13 = row(13, y), *d15 = row(15, y);

        for (int x = 0; x < nWidth; x++) {
            d1[x] = AveragePixels(p0[x], p2[x]);
            d9[x] = AveragePixels(p8[x], p10[x]);
            d4[x] = AveragePixels(p0[x], p8[x]);
            d6[x] = AveragePixels(p2[x], p10[x]);
        }
        for (int x = 0; x < nWidth; x++)
            d5[x] = AveragePixels(d4[x], d6[x]);
        for (int x = 0; x < last; x++) {
            d3[x] = AveragePixels(p0[x + 1], p2[x]);
            d11[x] = AveragePixels(p8[x + 1], p10[x]);
            d7[x] = AveragePixels(d4[x + 1], d6[x]);
        }
        d3[last] = AveragePixels(p0[last], p2[last]);
        d11[last] = AveragePixels(p8[last], p10[last]);
        d7[last] = AveragePixels(d4[last], d6[last]);
        const PixelType *p0n = y < nHeight - 1 ? p0 + pp : p0, *p2n = y < nHeight - 1 ? p2 + pp : p2;
        for (int x = 0; x < nWidth; x++) {
            d12[x] = AveragePixels(p0n[x], p8[x]);
            d14[x] = AveragePixels(p2n[x], p10[x]);
        }
        for (int x = 0; x < nWidth; x++)
            d13[x] = AveragePixels(d12[x], d14[x]);
        for (int x = 0; x < last; x++)
            d15[x] = AveragePixels(d12[x + 1], d14[x]);
        d15[last] = AveragePixels(d12[last], d14[last]);
    }
}

template<typename PixelType>
void PyramidPlane::GeneratePelPlanes(SharpParam sharp, const VSAPI *vsapi) noexcept {
    using RefineFunction = void (*)(uint8_t *pDst, const uint8_t *pSrc, ptrdiff_t nPitch, int nWidth, int nHeight, int bitsPerSample);

    RefineFunction refine[3];

    if (sharp == SharpParam::Bilinear) {
        refine[0] = HorizontalBilinear<PixelType>;
        refine[1] = VerticalBilinear<PixelType>;
        refine[2] = DiagonalBilinear<PixelType>;
    } else if (sharp == SharpParam::Bicubic) {
        refine[0] = refine[2] = HorizontalBicubic<PixelType>;
        refine[1] = VerticalBicubic<PixelType>;
    } else { // Wiener
        refine[0] = refine[2] = HorizontalWiener<PixelType>;
        refine[1] = VerticalWiener<PixelType>;
    }

    const uint8_t *src[3] = {};
    uint8_t *dst[3] = {};

    const VSVideoFormat *format = vsapi->getVideoFrameFormat(storage);
    if (nPel == 2) {
        dst[0] = const_cast<uint8_t *>(pPlane[1]);
        dst[1] = const_cast<uint8_t *>(pPlane[2]);
        dst[2] = const_cast<uint8_t *>(pPlane[3]);
        src[0] = src[1] = pPlane[0];
        if (sharp == SharpParam::Bilinear)
            src[2] = pPlane[0];
        else
            src[2] = pPlane[2];

        for (int i = 0; i < 3; i++)
            refine[i](dst[i], src[i], nPitch, nPaddedWidth, nPaddedHeight, format->bitsPerSample);
    } else if (nPel == 4) {
        uint8_t *pPlaneW[16] = {};

        // This cast is allowed since we always assign pPlane[i] with a write pointer when creating the frame in the constructor
        for (int i = 0; i < 16; i++)
            pPlaneW[i] = const_cast<uint8_t *>(pPlane[i]);

        dst[0] = pPlaneW[2];
        dst[1] = pPlaneW[8];
        dst[2] = pPlaneW[10];
        src[0] = src[1] = pPlane[0];
        if (sharp == SharpParam::Bilinear)
            src[2] = pPlane[0];
        else
            src[2] = pPlane[8];

        for (int i = 0; i < 3; i++)
            refine[i](dst[i], src[i], nPitch, nPaddedWidth, nPaddedHeight, format->bitsPerSample);

        GeneratePelQuarters<PixelType>(pPlaneW, nPitch, nPaddedWidth, nPaddedHeight);
    }

    nVPaddingPel = nVPadding * nPel;
    nHPaddingPel = nHPadding * nPel;
}


template<typename PixelType>
void PyramidPlane::SetExtPel2(const VSFrame *pelFrame, int plane, const VSAPI *vsapi) {
    const PixelType *pSrc2x = reinterpret_cast<const PixelType *>(vsapi->getReadPtr(pelFrame, plane));
    ptrdiff_t nSrc2xPitch = vsapi->getStride(pelFrame, plane);

    PixelType *pp[4] = {};
    // This cast is allowed since we always assign pPlane[i] with a write pointer when creating the frame in the constructor
    for (int i = 1; i < 4; i++)
        pp[i] = (PixelType *)pPlane[i];
    nSrc2xPitch /= sizeof(PixelType);
    ptrdiff_t nPitchTmp = nPitch / sizeof(PixelType);

    ptrdiff_t offset = nPitchTmp * nVPadding + nHPadding;
    pp[1] += offset;
    pp[2] += offset;
    pp[3] += offset;

    for (int h = 0; h < nRealHeight; h++) {
        for (int w = 0; w < nRealWidth; w++) {
            pp[1][w] = pSrc2x[(w << 1) + 1];
            pp[2][w] = pSrc2x[(w << 1) + nSrc2xPitch];
            pp[3][w] = pSrc2x[(w << 1) + nSrc2xPitch + 1];
        }
        pp[1] += nPitchTmp;
        pp[2] += nPitchTmp;
        pp[3] += nPitchTmp;
        pSrc2x += nSrc2xPitch * 2;
    }

    for (int i = 1; i < 4; i++)
        PadPlaneData<PixelType>(i);
}


template<typename PixelType>
void PyramidPlane::SetExtPel4(const VSFrame *pelFrame, int plane, const VSAPI *vsapi) {
    const PixelType *pSrc2x = reinterpret_cast<const PixelType *>(vsapi->getReadPtr(pelFrame, plane));
    ptrdiff_t nSrc2xPitch = vsapi->getStride(pelFrame, plane);

    PixelType *pp[16] = {};
    // This cast is allowed since we always assign pPlane[i] with a write pointer when creating the frame in the constructor
    for (int i = 1; i < 16; i++)
        pp[i] = (PixelType *)pPlane[i];

    nSrc2xPitch /= sizeof(PixelType);
    ptrdiff_t nPitchTmp = nPitch / sizeof(PixelType);

    ptrdiff_t offset = nPitchTmp * nVPadding + nHPadding;
    for (int i = 1; i < 16; i++)
        pp[i] += offset;

    for (int h = 0; h < nRealHeight; h++) {
        for (int w = 0; w < nRealWidth; w++) {
            pp[1][w] = pSrc2x[(w << 2) + 1];
            pp[2][w] = pSrc2x[(w << 2) + 2];
            pp[3][w] = pSrc2x[(w << 2) + 3];
            pp[4][w] = pSrc2x[(w << 2) + nSrc2xPitch];
            pp[5][w] = pSrc2x[(w << 2) + nSrc2xPitch + 1];
            pp[6][w] = pSrc2x[(w << 2) + nSrc2xPitch + 2];
            pp[7][w] = pSrc2x[(w << 2) + nSrc2xPitch + 3];
            pp[8][w] = pSrc2x[(w << 2) + nSrc2xPitch * 2];
            pp[9][w] = pSrc2x[(w << 2) + nSrc2xPitch * 2 + 1];
            pp[10][w] = pSrc2x[(w << 2) + nSrc2xPitch * 2 + 2];
            pp[11][w] = pSrc2x[(w << 2) + nSrc2xPitch * 2 + 3];
            pp[12][w] = pSrc2x[(w << 2) + nSrc2xPitch * 3];
            pp[13][w] = pSrc2x[(w << 2) + nSrc2xPitch * 3 + 1];
            pp[14][w] = pSrc2x[(w << 2) + nSrc2xPitch * 3 + 2];
            pp[15][w] = pSrc2x[(w << 2) + nSrc2xPitch * 3 + 3];
        }
        for (int i = 1; i < 16; i++)
            pp[i] += nPitchTmp;
        pSrc2x += nSrc2xPitch * 4;
    }

    for (int i = 1; i < 16; i++)
        PadPlaneData<PixelType>(i);
}


template<typename PixelType>
void PyramidPlane::SetExternalPelPlanes(const VSFrame *pelFrame, int plane, const VSAPI *vsapi) {
    if (nPel == 2) {
        SetExtPel2<PixelType>(pelFrame, plane, vsapi);
    } else if (nPel == 4) {
        SetExtPel4<PixelType>(pelFrame, plane, vsapi);
    }

    nVPaddingPel = nVPadding * nPel;
    nHPaddingPel = nHPadding * nPel;
}

template<typename PixelType>
void PyramidPlane::PadPlaneData(int plane) noexcept {
    PixelType *dstP = (PixelType *)(pPlane[plane]);
    ptrdiff_t nUsedPich = nPitch / sizeof(PixelType);

    // Pad sides by extending the edges
    dstP += nUsedPich * nVPadding;

    for (int h = 0; h < nRealHeight; h++) {
        PixelType padValueLeft = dstP[nHPadding];
        for (int w = 0; w < nHPadding; w++)
            dstP[w] = padValueLeft;
        PixelType padValueRight = dstP[nHPadding + nRealWidth - 1];
        for (int w = nRealWidth + nHPadding; w < nPaddedWidth; w++)
            dstP[w] = padValueRight;
        dstP += nUsedPich;
    }

    // Top and bottom padding by copying the first and last actual image line that's already been extended horizontally
    const PixelType *dstPLastLine = dstP - nUsedPich;
    for (int h = 0; h < nPaddedHeight - nVPadding - nRealHeight; h++) {
        memcpy(dstP, dstPLastLine, nPitch);
        dstP += nUsedPich;
    }

    dstP = (PixelType *)(pPlane[plane]);
    const PixelType *dstPFirstLine = dstP + nUsedPich * nVPadding;
    for (int h = 0; h < nVPadding; h++) {
        memcpy(dstP, dstPFirstLine, nPitch);
        dstP += nUsedPich;
    }
}

void PyramidPlane::FromExternalPlane(const VSFrame *planeFrame, int hPad, int vPad, int tailGuardLines, const VSAPI *vsapi) noexcept {
    const VSVideoFormat *format = vsapi->getVideoFrameFormat(planeFrame);
    storage = planeFrame;
    pPlane[0] = vsapi->getReadPtr(planeFrame, 0);
    nPitch = vsapi->getStride(planeFrame, 0);
    nHPadding = hPad;
    nVPadding = vPad;
    nHPaddingPel = nHPadding;
    nVPaddingPel = nVPadding;
    nOffsetPadding = nPitch * nVPadding + nHPadding * format->bytesPerSample;

    nPaddedWidth = vsapi->getFrameWidth(planeFrame, 0);
    // tailGuardLines is the gather slack (kLevel0GatherGuardLines for the gathered level-0 plane, 0 otherwise).
    nPaddedHeight = vsapi->getFrameHeight(planeFrame, 0) - tailGuardLines;

    nWidth = nPaddedWidth - 2 * nHPadding;
    nHeight = nPaddedHeight - 2 * nVPadding;

    subPelPlaneOffset = nPitch * nPaddedHeight;
}

void PyramidPlane::FromExternalPelPlanes(const VSFrame *pelFrame, int pel, int hPad, int vPad, const VSAPI *vsapi) {
    assert(pel == 2 || pel == 4);
    nPel = pel;
    const VSVideoFormat *format = vsapi->getVideoFrameFormat(pelFrame);
    nPitch = vsapi->getStride(pelFrame, 0);
    nHPadding = hPad;
    nVPadding = vPad;
    nHPaddingPel = nHPadding * pel;
    nVPaddingPel = nVPadding * pel;

    nOffsetPadding = nPitch * nVPadding + nHPadding * format->bytesPerSample;

    nPaddedWidth = vsapi->getFrameWidth(pelFrame, 0);
    // Subtract the gather guard line(s) before splitting the stacked sub-planes (see kLevel0GatherGuardLines).
    nPaddedHeight = (vsapi->getFrameHeight(pelFrame, 0) - kLevel0GatherGuardLines) / (pel * pel);

    nWidth = nPaddedWidth - 2 * nHPadding;
    nHeight = nPaddedHeight - 2 * nVPadding;

    subPelPlaneOffset = nPitch * nPaddedHeight;
    
    storage = pelFrame;
    for (int i = 0; i < pel * pel; i++)
        pPlane[i] = vsapi->getReadPtr(pelFrame, 0) + i * subPelPlaneOffset;
}

void FramePyramid::SharedInit(const VSFrame *srcFrame, int levels, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int hPad, int vPad, RFilterParam rFilter, int pel, VSCore *core, const VSAPI *vsapi) {
    this->vsapi = vsapi;

    if (!srcFrame)
        throw SuperPyramidError("Invalid source frame");
    if (levels < 1)
        throw SuperPyramidError("Must have at least one level");
    if (hPad <= 0)
        throw SuperPyramidError("Horizontal padding must be positive");
    if (vPad <= 0)
        throw SuperPyramidError("Vertical padding must be positive");

    this->nBlkSizeX = nBlkSizeX;
    this->nBlkSizeY = nBlkSizeY;
    this->nOverlapX = nOverlapX;
    this->nOverlapY = nOverlapY;
    nPel = pel;

    pyramidLevels.resize(levels);
    nLevels = levels;

    const VSVideoFormat *srcFormat = vsapi->getVideoFrameFormat(srcFrame);
    chroma = (srcFormat->colorFamily != cfGray);
    bitsPerSample = srcFormat->bitsPerSample;

    assert(srcFormat->subSamplingW <= 1 && srcFormat->subSamplingH <= 1);
    if (chroma) {
        xRatioUV = 1 << (srcFormat->subSamplingW);
        yRatioUV = 1 << (srcFormat->subSamplingH);
    }

    nHPad[0] = hPad;
    nHPad[1] = hPad / xRatioUV;
    nHPad[2] = hPad / xRatioUV;

    nVPad[0] = vPad;
    nVPad[1] = vPad / yRatioUV;
    nVPad[2] = vPad / yRatioUV;

    for (int plane = 0; plane < srcFormat->numPlanes; plane++) {
        nRealWidth[plane] = vsapi->getFrameWidth(srcFrame, plane);
        nRealHeight[plane] = vsapi->getFrameHeight(srcFrame, plane);

        nWidth[plane] = nRealWidth[plane];
        nHeight[plane] = nRealHeight[plane];
    }

    // Pad level 0 out to a whole block grid
    nWidth[0] = BlockAlignedDimension(nRealWidth[0], nBlkSizeX, nOverlapX);
    nHeight[0] = BlockAlignedDimension(nRealHeight[0], nBlkSizeY, nOverlapY);
    nWidth[1] = nWidth[2] = nWidth[0] / xRatioUV;
    nHeight[1] = nHeight[2] = nHeight[0] / yRatioUV;

    for (int plane = 0; plane < srcFormat->numPlanes; plane++) {
        nBlkSizePadX[plane] = nWidth[plane] - nRealWidth[plane];
        nBlkSizePadY[plane] = nHeight[plane] - nRealHeight[plane];
    }

    size_t tempBufferSize = static_cast<size_t>(nWidth[0]) * srcFormat->bytesPerSample * 8;
    MvuAlignedPtr tempBuffer = mvu_make_aligned<uint8_t>(tempBufferSize);

    if (srcFormat->bytesPerSample == 1) {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++) {
            pyramidLevels[0].planes[plane].CopyAndPadPlane<uint8_t>(srcFrame, plane, nHPad[plane], nVPad[plane], nWidth[plane] - nRealWidth[plane], nHeight[plane] - nRealHeight[plane], nPel, core, vsapi);
            for (int i = 1; i < levels; i++)
                pyramidLevels[i].planes[plane].ReducePlane<uint8_t>(pyramidLevels[i - 1].planes[plane], xRatioUV, yRatioUV, rFilter, tempBuffer.get(), core, vsapi);
        }
    } else if (srcFormat->bytesPerSample == 2) {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++) {
            pyramidLevels[0].planes[plane].CopyAndPadPlane<uint16_t>(srcFrame, plane, nHPad[plane], nVPad[plane], nWidth[plane] - nRealWidth[plane], nHeight[plane] - nRealHeight[plane], nPel, core, vsapi);
            for (int i = 1; i < levels; i++)
                pyramidLevels[i].planes[plane].ReducePlane<uint16_t>(pyramidLevels[i - 1].planes[plane], xRatioUV, yRatioUV, rFilter, tempBuffer.get(), core, vsapi);
        }
    } else {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++) {
            pyramidLevels[0].planes[plane].CopyAndPadPlane<float>(srcFrame, plane, nHPad[plane], nVPad[plane], nWidth[plane] - nRealWidth[plane], nHeight[plane] - nRealHeight[plane], nPel, core, vsapi);
            for (int i = 1; i < levels; i++)
                pyramidLevels[i].planes[plane].ReducePlane<float>(pyramidLevels[i - 1].planes[plane], xRatioUV, yRatioUV, rFilter, tempBuffer.get(), core, vsapi);
        }
    }
}

FramePyramid::FramePyramid(const VSFrame *srcFrame, int levels, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int hPad, int vPad, RFilterParam rFilter, int pel, SharpParam sharp, VSCore *core, const VSAPI *vsapi) {
    SharedInit(srcFrame, levels, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, hPad, vPad, rFilter, pel, core, vsapi);
    if (pel > 1)
        GeneratePelPlanes(sharp, vsapi);
}

FramePyramid::FramePyramid(const VSFrame *srcFrame, int levels, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int hPad, int vPad, RFilterParam rFilter, int pel, const VSFrame *pelFrame, VSCore *core, const VSAPI *vsapi) {
    SharedInit(srcFrame, levels, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, hPad, vPad, rFilter, pel, core, vsapi);
    if (pel > 1)
        SetExternalPelPlanes(srcFrame, pelFrame, vsapi);
}

void FramePyramid::LoadFrameData(const VSFrame *srcFrame, int maxLevel, const std::string &prefix) {
    if (!srcFrame)
        throw SuperPyramidError("Invalid source frame");

    serializedData = srcFrame; // take ownership; the single try below frees it on any throw before construction completes.

    try {
        const VSMap *props = vsapi->getFramePropertiesRO(srcFrame);
        int err;
        xRatioUV = vsapi->mapGetIntSaturated(props, (prefix + "SuperXRatioUV").c_str(), 0, &err);
        yRatioUV = vsapi->mapGetIntSaturated(props, (prefix + "SuperYRatioUV").c_str(), 0, &err);
        nWidth[0] = vsapi->mapGetIntSaturated(props, (prefix + "SuperWidth").c_str(), 0, &err);
        nHeight[0] = vsapi->mapGetIntSaturated(props, (prefix + "SuperHeight").c_str(), 0, &err);
        nRealWidth[0] = vsapi->mapGetIntSaturated(props, (prefix + "SuperRealWidth").c_str(), 0, &err);
        nRealHeight[0] = vsapi->mapGetIntSaturated(props, (prefix + "SuperRealHeight").c_str(), 0, &err);
        nHPad[0] = vsapi->mapGetIntSaturated(props, (prefix + "SuperHPad").c_str(), 0, &err);
        nVPad[0] = vsapi->mapGetIntSaturated(props, (prefix + "SuperVPad").c_str(), 0, &err);
        bitsPerSample = vsapi->mapGetIntSaturated(props, (prefix + "SuperBitsPerSample").c_str(), 0, &err);
        nBlkSizeX = vsapi->mapGetIntSaturated(props, (prefix + "SuperBlkSizeX").c_str(), 0, &err);
        nBlkSizeY = vsapi->mapGetIntSaturated(props, (prefix + "SuperBlkSizeY").c_str(), 0, &err);
        nOverlapX = vsapi->mapGetIntSaturated(props, (prefix + "SuperOverlapX").c_str(), 0, &err);
        nOverlapY = vsapi->mapGetIntSaturated(props, (prefix + "SuperOverlapY").c_str(), 0, &err);

        nBlkSizePadX[0] = nWidth[0] - nRealWidth[0];
        nBlkSizePadY[0] = nHeight[0] - nRealHeight[0];

        nPel = vsapi->mapGetIntSaturated(props, (prefix + "SuperPel").c_str(), 0, &err);
        nLevels = vsapi->mapGetIntSaturated(props, (prefix + "SuperLevels").c_str(), 0, &err);
        chroma = !!vsapi->mapGetInt(props, (prefix + "SuperChroma").c_str(), 0, &err);

        if (xRatioUV < 1 || yRatioUV < 1 || xRatioUV > 2 || yRatioUV > 2 || nRealWidth[0] > nWidth[0] || nRealHeight[0] > nHeight[0]
            || nVPad[0] < 0 || nHPad[0] < 0 || nRealHeight[0] < 1 || nRealWidth[0] < 1 || nLevels < 1 || (nPel != 1 && nPel != 2 && nPel != 4)
            || bitsPerSample < 8 || (bitsPerSample > 16 && bitsPerSample != 32) || (chroma && (nHPad[0] % xRatioUV || nVPad[0] % yRatioUV)))
            throw SuperPyramidError("Invalid super frame metadata");

        if (chroma) {
            nWidth[1] = nWidth[0] / xRatioUV;
            nWidth[2] = nWidth[0] / xRatioUV;
            nHeight[1] = nHeight[0] / yRatioUV;
            nHeight[2] = nHeight[0] / yRatioUV;
            nRealWidth[1] = nRealWidth[0] / xRatioUV;
            nRealWidth[2] = nRealWidth[0] / xRatioUV;
            nRealHeight[1] = nRealHeight[0] / yRatioUV;
            nRealHeight[2] = nRealHeight[0] / yRatioUV;
            nHPad[1] = nHPad[0] / xRatioUV;
            nHPad[2] = nHPad[0] / xRatioUV;
            nVPad[1] = nVPad[0] / yRatioUV;
            nVPad[2] = nVPad[0] / yRatioUV;
            nBlkSizePadX[1] = nWidth[1] - nRealWidth[1];
            nBlkSizePadX[2] = nWidth[2] - nRealWidth[2];
            nBlkSizePadY[1] = nHeight[1] - nRealHeight[1];
            nBlkSizePadY[2] = nHeight[2] - nRealHeight[2];
        }

        bitsPerSample = vsapi->getVideoFrameFormat(srcFrame)->bitsPerSample;

        int loadLevels = (maxLevel < 0) ? nLevels : std::min(maxLevel, nLevels);

        pyramidLevels.resize(loadLevels);

        for (int level = 0; level < loadLevels; level++) {
            std::string propStr = prefix + "SuperLevel" + std::to_string(level);
            for (int plane = 0; plane < (chroma ? 3 : 1); plane++) {
                const VSFrame *frame = vsapi->mapGetFrame(props, propStr.c_str(), plane, &err);
                if (!frame)
                    throw SuperPyramidError("Plane data missing in super frame metadata");
                if (level == 0 && nPel > 1)
                    pyramidLevels[level].planes[plane].FromExternalPelPlanes(frame, nPel, nHPad[plane], nVPad[plane], vsapi);
                else
                    // Only level 0 carries the gather guard line(s); deeper levels (ReducePlane) do not.
                    pyramidLevels[level].planes[plane].FromExternalPlane(frame, nHPad[plane], nVPad[plane], level == 0 ? kLevel0GatherGuardLines : 0, vsapi);
            }
        }

        // Validate that every loaded plane's padded dimensions match the declared metadata.
        // Padding is constant across all levels; only content shrinks via PlaneDimensionLuma,
        // which is the same reduction ReducePlane applies when building the pyramid.
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++) {
            int expectedWidth = nWidth[plane];
            int expectedHeight = nHeight[plane];
            for (int level = 0; level < loadLevels; level++) {
                const PyramidPlane &p = pyramidLevels[level].planes[plane];
                const int expectedPaddedWidth = expectedWidth + 2 * nHPad[plane];
                const int expectedPaddedHeight = expectedHeight + 2 * nVPad[plane];
                if (p.nPaddedWidth != expectedPaddedWidth || p.nPaddedHeight != expectedPaddedHeight)
                    throw SuperPyramidError(
                        "Level " + std::to_string(level) + " plane " + std::to_string(plane) +
                        " dimensions mismatch: expected " +
                        std::to_string(expectedPaddedWidth) + "x" + std::to_string(expectedPaddedHeight) +
                        " but got " +
                        std::to_string(p.nPaddedWidth) + "x" + std::to_string(p.nPaddedHeight));
                expectedWidth = PlaneDimensionLuma(expectedWidth, xRatioUV, nHPad[plane]);
                expectedHeight = PlaneDimensionLuma(expectedHeight, yRatioUV, nVPad[plane]);
            }
        }

        // Propagate real dimensions to level 0 where they're relevant
        if (!pyramidLevels.empty()) {
            for (int plane = 0; plane < (chroma ? 3 : 1); plane++) {
                pyramidLevels[0].planes[plane].nRealHeight = nRealHeight[plane];
                pyramidLevels[0].planes[plane].nRealWidth = nRealWidth[plane];
            }
        }

    } catch (...) {
        FreeFrames();
        throw;
    }
}

FramePyramid::FramePyramid(const VSFrame *srcFrame, int maxLevel, const std::string &prefix, const VSAPI *vsapi)
: vsapi(vsapi) {
    LoadFrameData(srcFrame, maxLevel, prefix);
}

// The delegated constructor has completed, so a throw here runs the destructor, which frees the frames
FramePyramid::FramePyramid(const VSFrame *srcFrame, int maxLevel, const std::string &prefix, const VSAPI *vsapi, const SuperGeometry &expected)
: FramePyramid(srcFrame, maxLevel, prefix, vsapi) {
    if (Geometry() != expected)
        throw SuperPyramidError("a super clip frame was made with different Super arguments than its first frame");
}

FramePyramid::FramePyramid(VSNode *node, const std::string &prefix, const VSAPI *vsapi)
    : vsapi(vsapi) {

    char errorMsg[ERROR_SIZE] = {};
    const VSFrame *srcFrame = vsapi->getFrame(0, node, errorMsg, ERROR_SIZE);
    if (!srcFrame)
        throw std::runtime_error("Failed to retrieve first frame from super clip. Error message: " + std::string(errorMsg));

    LoadFrameData(srcFrame, 0, prefix);
}

void FramePyramid::FreeFrames() noexcept {
    for (auto &level : pyramidLevels) {
        for (int i = 0; i < 3; i++)
            vsapi->freeFrame(level.planes[i].storage);
    }

    vsapi->freeFrame(serializedData);
}

FramePyramid::~FramePyramid() {
    FreeFrames();
}

void FramePyramid::GeneratePelPlanes(SharpParam sharp, const VSAPI *vsapi) {
    if (nPel != 2 && nPel != 4)
        throw SuperPyramidError("Pel value must be 2 or 4");
    if (bitsPerSample == 8) {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++)
            pyramidLevels[0].planes[plane].GeneratePelPlanes<uint8_t>(sharp, vsapi);
    } else if (bitsPerSample == 32) {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++)
            pyramidLevels[0].planes[plane].GeneratePelPlanes<float>(sharp, vsapi);
    } else {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++)
            pyramidLevels[0].planes[plane].GeneratePelPlanes<uint16_t>(sharp, vsapi);
    }
}

void FramePyramid::SetExternalPelPlanes(const VSFrame *srcFrame, const VSFrame *pelFrame, const VSAPI *vsapi) {
    if (nPel != 2 && nPel != 4)
        throw SuperPyramidError("Pel value must be 2 or 4");

    assert(pyramidLevels[0].planes[0].storage);

    const VSVideoFormat *pelFormat = vsapi->getVideoFrameFormat(pelFrame);
    const VSVideoFormat *srcFormat = vsapi->getVideoFrameFormat(srcFrame);

    if (!vsh::isSameVideoFormat(pelFormat, srcFormat))
        throw SuperPyramidError("Pel frame format does not match source frame format");

    if (vsapi->getFrameWidth(pelFrame, 0) != pyramidLevels[0].planes[0].nRealWidth * nPel ||
        vsapi->getFrameHeight(pelFrame, 0) != pyramidLevels[0].planes[0].nRealHeight * nPel)
        throw SuperPyramidError("Pel frame dimensions are not a suitable multiple of the source frame dimensions");

    if (bitsPerSample == 8) {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++)
            pyramidLevels[0].planes[plane].SetExternalPelPlanes<uint8_t>(pelFrame, plane, vsapi);
    } else if (bitsPerSample == 32) {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++)
            pyramidLevels[0].planes[plane].SetExternalPelPlanes<float>(pelFrame, plane, vsapi);
    } else {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++)
            pyramidLevels[0].planes[plane].SetExternalPelPlanes<uint16_t>(pelFrame, plane, vsapi);
    }
}

void FramePyramid::ExportFrameData(VSFrame *dst, const std::string &prefix) const noexcept {
    VSMap *props = vsapi->getFramePropertiesRW(dst);

    for (size_t level = 0; level < pyramidLevels.size(); level++) {
        for (int plane = 0; plane < (chroma ? 3 : 1); plane++) {
            assert(pyramidLevels[level].planes[plane].storage);
            vsapi->mapSetFrame(props, (prefix + "SuperLevel" + std::to_string(level)).c_str(), pyramidLevels[level].planes[plane].storage, maAppend);
        }
    }

    vsapi->mapSetInt(props, (prefix + "SuperWidth").c_str(), nWidth[0], maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperHeight").c_str(), nHeight[0], maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperRealWidth").c_str(), nRealWidth[0], maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperRealHeight").c_str(), nRealHeight[0], maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperHPad").c_str(), nHPad[0], maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperVPad").c_str(), nVPad[0], maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperPel").c_str(), nPel, maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperLevels").c_str(), pyramidLevels.size(), maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperChroma").c_str(), chroma, maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperXRatioUV").c_str(), xRatioUV, maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperYRatioUV").c_str(), yRatioUV, maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperBitsPerSample").c_str(), bitsPerSample, maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperBlkSizeX").c_str(), nBlkSizeX, maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperBlkSizeY").c_str(), nBlkSizeY, maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperOverlapX").c_str(), nOverlapX, maReplace);
    vsapi->mapSetInt(props, (prefix + "SuperOverlapY").c_str(), nOverlapY, maReplace);
}

const FramePyramidLevel &FramePyramid::GetLevel(int level) const noexcept {
    assert(level >= 0 && level < static_cast<int>(pyramidLevels.size()));
    return pyramidLevels[level];
}

bool FramePyramid::IsCompatible(const FramePyramid &other) const noexcept {
    SuperGeometry otherGeometry = other.Geometry();
    otherGeometry.levels = nLevels; // a onelevel super (Degrain's centersuper) goes with a full one
    return Geometry() == otherGeometry;
}

SuperGeometry FramePyramid::Geometry() const noexcept {
    return { nWidth[0], nHeight[0], nRealWidth[0], nRealHeight[0], nHPad[0], nVPad[0], nBlkSizePadX[0], nBlkSizePadY[0],
        nPel, xRatioUV, yRatioUV, bitsPerSample, nLevels, chroma };
}

bool FramePyramid::IsCompatibleWithSource(const VSVideoInfo *vi) const noexcept {
    if (!vsh::isConstantVideoFormat(vi))
        return false;
    if (vi->format.numPlanes != (chroma ? 3 : 1))
        return false;
    if (nRealWidth[0] != vi->width || nRealHeight[0] != vi->height || (xRatioUV != 1 << vi->format.subSamplingW)
        || yRatioUV != (1 << vi->format.subSamplingH) || bitsPerSample != vi->format.bitsPerSample
        || (bitsPerSample == 32 && vi->format.sampleType != stFloat) || (bitsPerSample <= 16 && vi->format.sampleType != stInteger))
        return false;
    return true;
}


int FramePyramid::GetMaxLevelsForBlockSize(int width, int height, int xRatioUV, int yRatioUV, int blkSizeX, int blkSizeY, int overlapX, int overlapY, int padX, int padY) noexcept {
    // Number of levels, level 0 included, that can still hold a whole block; any smaller level is useless.
    // Counted on the planes SharedInit actually builds, starting from the block aligned level 0. Always at least 1:
    // level 0 is kept even when the frame is too small for a block, which Analyse then rejects.
    // Terminates since halving only stalls at ratioUV (<= 2) while every block is at least 4 wide.
    width = BlockAlignedDimension(width, blkSizeX, overlapX);
    height = BlockAlignedDimension(height, blkSizeY, overlapY);

    int nLevelsMax = 0;
    while (width >= blkSizeX && height >= blkSizeY) {
        nLevelsMax++;
        width = PlaneDimensionLuma(width, xRatioUV, padX);
        height = PlaneDimensionLuma(height, yRatioUV, padY);
    }

    return std::max(1, nLevelsMax);
}
