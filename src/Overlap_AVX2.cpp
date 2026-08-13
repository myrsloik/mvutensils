#include "FunctionTable.h"
#include "Overlap.h"

#if defined(MVTOOLS_X86)

#include <immintrin.h>

template <int blockWidth, int blockHeight>
static void overlaps_avx2(uint8_t *pDst8, ptrdiff_t nDstPitch, const uint8_t *pSrc, ptrdiff_t nSrcPitch, const int16_t *pWin, ptrdiff_t nWinPitch) {
    static_assert(blockWidth >= 16, "");

    /* pWin from 0 to 2048 */
    for (unsigned y = 0; y < (unsigned)blockHeight; y++) {
        for (unsigned x = 0; x < (unsigned)blockWidth; x += 16) {
            uint16_t *pDst = (uint16_t *)pDst8;

            __m256i src = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(pSrc + x)));
            __m256i win = _mm256_loadu_si256((const __m256i *)(pWin + x));
            __m256i dst = _mm256_loadu_si256((const __m256i *)(pDst + x));

            // (src * win) >> 6 == high 16 bits of (src<<8)*(win<<2) = src*win*2^10 (>>16 nets >>6).
            src = _mm256_slli_epi16(src, 8);
            win = _mm256_slli_epi16(win, 2);
            dst = _mm256_adds_epu16(dst, _mm256_mulhi_epu16(src, win));
            _mm256_storeu_si256((__m256i *)(pDst + x), dst);
        }

        pDst8 += nDstPitch;
        pSrc += nSrcPitch;
        pWin += nWinPitch;
    }
}

template <int blockWidth, int blockHeight>
static void overlaps_avx2_u16(uint8_t *pDst8, ptrdiff_t nDstPitch, const uint8_t *pSrc8, ptrdiff_t nSrcPitch, const int16_t *pWin, ptrdiff_t nWinPitch) {
    static_assert(blockWidth >= 8, "");

    for (unsigned y = 0; y < (unsigned)blockHeight; y++) {
        uint32_t *pDst = (uint32_t *)pDst8;
        const uint16_t *pSrc = (const uint16_t *)pSrc8;
        for (unsigned x = 0; x < (unsigned)blockWidth; x += 8) {
            __m256i src = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)&pSrc[x]));
            __m256i win = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)&pWin[x]));
            __m256i p = _mm256_srli_epi32(_mm256_mullo_epi32(src, win), 6);
            _mm256_storeu_si256((__m256i *)&pDst[x], _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)&pDst[x]), p));
        }
        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
        pWin += nWinPitch;
    }
}

#endif


#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8

#if defined(MVTOOLS_X86)
#define OVERS_AVX2(width, height) \
    { KEY(width, height, 8), overlaps_avx2<width, height> },
#define OVERS_AVX2_16(width, height) \
    { KEY(width, height, 16), overlaps_avx2_u16<width, height> },
#else
#define OVERS_AVX2(width, height)
#define OVERS_AVX2_16(width, height)
#endif

static constexpr auto overlaps_functions = std::to_array<FunctionTableEntry<OverlapsFunction>>({
    // 8-bit: width 8 intentionally omitted -- the SSE2 intrinsic is faster there (see overlaps_avx2 note).
    // 16-bit: width 8 IS included -- a ymm holds 8 uint32 = one full width-8 row, so no row stitching.
    OVERS_AVX2(16, 1)
    OVERS_AVX2(16, 2)
    OVERS_AVX2(16, 4)
    OVERS_AVX2(16, 8)
    OVERS_AVX2(16, 16)
    OVERS_AVX2(16, 32)
    OVERS_AVX2(32, 8)
    OVERS_AVX2(32, 16)
    OVERS_AVX2(32, 32)
    OVERS_AVX2(32, 64)
    OVERS_AVX2(64, 16)
    OVERS_AVX2(64, 32)
    OVERS_AVX2(64, 64)
    OVERS_AVX2(64, 128)
    OVERS_AVX2(128, 32)
    OVERS_AVX2(128, 64)
    OVERS_AVX2(128, 128)
    OVERS_AVX2_16(8, 1)
    OVERS_AVX2_16(8, 2)
    OVERS_AVX2_16(8, 4)
    OVERS_AVX2_16(8, 8)
    OVERS_AVX2_16(8, 16)
    OVERS_AVX2_16(16, 1)
    OVERS_AVX2_16(16, 2)
    OVERS_AVX2_16(16, 4)
    OVERS_AVX2_16(16, 8)
    OVERS_AVX2_16(16, 16)
    OVERS_AVX2_16(16, 32)
    OVERS_AVX2_16(32, 8)
    OVERS_AVX2_16(32, 16)
    OVERS_AVX2_16(32, 32)
    OVERS_AVX2_16(32, 64)
    OVERS_AVX2_16(64, 16)
    OVERS_AVX2_16(64, 32)
    OVERS_AVX2_16(64, 64)
    OVERS_AVX2_16(64, 128)
    OVERS_AVX2_16(128, 32)
    OVERS_AVX2_16(128, 64)
    OVERS_AVX2_16(128, 128)
});


void selectOverlapsFunctionAVX2(unsigned width, unsigned height, unsigned bits, OverlapsFunction &overs) {
    if (OverlapsFunction fn = findFunction(overlaps_functions, KEY(width, height, bits)))
        overs = fn;
}
