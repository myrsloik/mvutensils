#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <algorithm>

#include "Common.h"
#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"

// Largest temporal radius Degrain supports. Everything radius-dependent derives from this constant --
// the per-radius dispatch tables, the Degrain1..N registrations and the argument validation -- so
// raising the cap only means changing this value and paying for the extra template instantiations.
// The kernels themselves are radius-agnostic: normaliseWeights renormalises to sum(WSrc + WRefs) == 256
// regardless of radius, which is what keeps Degrain_C8's uint16 pixel accumulator in range.
inline constexpr int kMaxDegrainRadius = 25;

// normaliseWeights accumulates 256 * userWeight over 2 * radius + 1 terms (plus 1) in an int; the
// per-weight ceiling enforced in degrainCreate is derived from this and must stay >= 1.
static_assert(kMaxDegrainRadius >= 1 &&
    (std::numeric_limits<int>::max() - 1) / (256LL * (kMaxDegrainRadius * 2 + 1)) >= 1,
    "kMaxDegrainRadius too large: the normaliseWeights int accumulation would overflow");


using DenoiseFunction = void (*)(uint8_t *pDst, ptrdiff_t nDstPitch, const uint8_t *pSrc, ptrdiff_t nSrcPitch, const uint8_t **_pRefs, uint16_t WSrc, const uint16_t *WRefs) noexcept;


// XXX Both Degrain_C8/Degrain_C16 move the pointers passed in pRefs8. This is okay
// because they are not used after the function is done with them.

template <int radius, int blockWidth, int blockHeight>
static void Degrain_C8(uint8_t * MVU_RESTRICT pDst8, ptrdiff_t nDstPitch, const uint8_t * MVU_RESTRICT pSrc8, ptrdiff_t nSrcPitch, const uint8_t ** MVU_RESTRICT pRefs8, uint16_t WSrc, const uint16_t * MVU_RESTRICT WRefs) noexcept {
    const uint16_t wsrc = WSrc;
    uint16_t wref[radius * 2];
    for (int r = 0; r < radius * 2; r++)
        wref[r] = WRefs[r];

    for (int y = 0; y < blockHeight; y++) {
        const uint8_t * MVU_RESTRICT pSrc = pSrc8;
        uint8_t * MVU_RESTRICT pDst = pDst8;

        for (int x = 0; x < blockWidth; x++) {
            uint16_t sum = (uint16_t)(128u + (uint16_t)((uint16_t)pSrc[x] * wsrc));
            for (int r = 0; r < radius * 2; r++) {
                const uint8_t * MVU_RESTRICT pRef = pRefs8[r];
                sum = (uint16_t)(sum + (uint16_t)((uint16_t)pRef[x] * wref[r]));
            }
            pDst[x] = (uint8_t)(sum >> 8);
        }

        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
        for (int r = 0; r < radius * 2; r++)
            pRefs8[r] += nSrcPitch;
    }
}

template <int radius, int blockWidth, int blockHeight>
static void Degrain_C16(uint8_t * MVU_RESTRICT pDst8, ptrdiff_t nDstPitch, const uint8_t * MVU_RESTRICT pSrc8, ptrdiff_t nSrcPitch, const uint8_t ** MVU_RESTRICT pRefs8, uint16_t WSrc, const uint16_t * MVU_RESTRICT WRefs) noexcept {
    const int wsrc = WSrc;
    int wref[radius * 2];
    for (int r = 0; r < radius * 2; r++)
        wref[r] = WRefs[r];

    for (int y = 0; y < blockHeight; y++) {
        for (int x = 0; x < blockWidth; x++) {
            const uint16_t *pSrc = (const uint16_t * __restrict)pSrc8;
            uint16_t *pDst = (uint16_t * __restrict)pDst8;

            int sum = 128 + pSrc[x] * wsrc;

            for (int r = 0; r < radius * 2; r++) {
                const uint16_t *pRef = (const uint16_t * __restrict)pRefs8[r];
                sum += pRef[x] * wref[r];
            }

            pDst[x] = static_cast<uint16_t>(sum >> 8);
        }

        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
        for (int r = 0; r < radius * 2; r++)
            pRefs8[r] += nSrcPitch;
    }
}

template <int radius, int blockWidth, int blockHeight>
static void Degrain_F32(uint8_t *MVU_RESTRICT pDst8, ptrdiff_t nDstPitch, const uint8_t *MVU_RESTRICT pSrc8, ptrdiff_t nSrcPitch, const uint8_t **MVU_RESTRICT pRefs8, uint16_t WSrc, const uint16_t *MVU_RESTRICT WRefs) noexcept {
    const float wsrc = WSrc;
    float wref[radius * 2];
    for (int r = 0; r < radius * 2; r++)
        wref[r] = WRefs[r];

    for (int y = 0; y < blockHeight; y++) {
        for (int x = 0; x < blockWidth; x++) {
            const float *pSrc = (const float * __restrict)pSrc8;
            float *pDst = (float * __restrict)pDst8;

            float sum = pSrc[x] * wsrc;

            for (int r = 0; r < radius * 2; r++) {
                const float *pRef = (const float * __restrict)pRefs8[r];
                sum += pRef[x] * wref[r];
            }

            pDst[x] = sum / 256.0f;
        }

        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
        for (int r = 0; r < radius * 2; r++)
            pRefs8[r] += nSrcPitch;
    }
}



// LimitChanges clamps each output pixel to [pSrc - nLimit, pSrc + nLimit]. nLimit is
// validated to [0, pixelMax]
// The idiom used here is very friendly to compiler optimizers and helps them to not
// widen the type used for calculations
template <typename PixelType>
static void LimitChanges_C(uint8_t * MVU_RESTRICT pDst8, ptrdiff_t nDstPitch, const uint8_t * MVU_RESTRICT pSrc8, ptrdiff_t nSrcPitch, int nWidth, int nHeight, PixelType nLimit) noexcept {
    if constexpr (std::is_integral_v<PixelType>) {
        const int lim = nLimit;
        const int maxValue = (PixelType)~(PixelType)0; // 255 or 65535
        for (int h = 0; h < nHeight; h++) {
            const PixelType *pSrc = (const PixelType *)pSrc8;
            PixelType *pDst = (PixelType *)pDst8;
            for (int i = 0; i < nWidth; i++) {
                const int s = pSrc[i];
                const int lo = std::max(s - lim, 0);
                const int hi = std::min(s + lim, maxValue);
                pDst[i] = (PixelType)std::clamp((int)pDst[i], lo, hi);
            }
            pDst8 += nDstPitch;
            pSrc8 += nSrcPitch;
        }
    } else {
        const float lim = nLimit;
        for (int h = 0; h < nHeight; h++) {
            const PixelType *pSrc = (const PixelType *)pSrc8;
            PixelType *pDst = (PixelType *)pDst8;
            for (int i = 0; i < nWidth; i++)
                pDst[i] = std::clamp(pDst[i], pSrc[i] - lim, pSrc[i] + lim);
            pDst8 += nDstPitch;
            pSrc8 += nSrcPitch;
        }
    }
}

static inline uint16_t DegrainWeight(int64_t thSAD, int64_t blockSAD) noexcept {
    if (blockSAD >= thSAD)
        return 0;

    const double r = (double)blockSAD / (double)thSAD; // r in [0, 1)
    return static_cast<uint16_t>(256.0 * (1.0 - r * r) / (1.0 + r * r)); // in [0, 256]
}

template<typename PixelType>
static inline void useBlock(const uint8_t *&p, uint16_t &WRef, int isUsable, const std::optional<MotionBlockPyramid> &blocks, int i, const FramePyramidLevel *pPlane, const uint8_t **pSrcCur, int xx, int nLogPel, int plane, int xSubUV, int ySubUV, const int64_t *thSAD) noexcept {
    // Resolves only the block pointer and its weight; the row stride is the centre's, shared by every reference
    // (see DenoiseFunction), so the kernel steps them all with nSrcPitch.
    if (isUsable) {
        const BlockData block = blocks->GetBlock(i);
        int blx = (block.x << nLogPel) + block.vector.x;
        int bly = (block.y << nLogPel) + block.vector.y;
        p = pPlane->planes[plane].GetPointer<PixelType>(plane ? blx >> xSubUV : blx, plane ? bly >> ySubUV : bly);
        int64_t blockSAD = block.vector.sad;
        WRef = DegrainWeight(thSAD[plane], blockSAD);
    } else {
        p = pSrcCur[plane] + xx;
        WRef = 0;
    }
}


// userWeights biases each reference's contribution before normalisation:
// userWeights[0] scales the source/centre weight, userWeights[r + 1] scales
// WRefs[r]. All-ones reproduces the unweighted result exactly.
template <int radius>
static inline void normaliseWeights(uint16_t &WSrc, uint16_t *WRefs, const int *userWeights) noexcept {
    int wsrc = 256;
    int weighted[radius * 2];
    int WSum = wsrc * userWeights[0] + 1;
    for (int r = 0; r < radius * 2; r++) {
        weighted[r] = WRefs[r] * userWeights[r + 1];
        WSum += weighted[r];
    }

    double scale = 256.0 / WSum;

    for (int r = 0; r < radius * 2; r++) {
        int w = static_cast<int>(weighted[r] * scale);
        WRefs[r] = static_cast<uint16_t>(w);
        wsrc -= w;
    }

    WSrc = static_cast<uint16_t>(wsrc < 0 ? 0 : wsrc);
}
