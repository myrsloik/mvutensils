#include <cstdlib>
#include <stdexcept>

#include "CPU.h"
#include "FunctionTable.h"
#include "SADFunctions.h"
#include "Common.h"

#if defined(MVTOOLS_X86)

#include <emmintrin.h>

#define zeroes _mm_setzero_si128()

// This version used for width >= 16.
template <unsigned width, unsigned height>
struct SADWrapperU8 {
    static_assert(width >= 16, "");

    static unsigned int sad_u8_sse2(const uint8_t *pSrc, [[maybe_unused]] intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        __m128i sum = zeroes;

        for (unsigned y = 0; y < height; y++) {
            for (unsigned x = 0; x < width; x += 16) {
                __m128i m2 = _mm_loadu_si128((const __m128i *)&pSrc[x]);
                __m128i m3 = _mm_loadu_si128((const __m128i *)&pRef[x]);

                __m128i diff = _mm_sad_epu8(m2, m3);

                sum = _mm_add_epi64(sum, diff);
            }

            pSrc += /*nSrcPitch*/ width;
            pRef += nRefPitch;
        }

        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));

        return (unsigned)_mm_cvtsi128_si32(sum);
    }

};


template <unsigned height>
struct SADWrapperU8<2, height> {
    static_assert(height % 2 == 0, "");

    static unsigned int sad_u8_sse2(const uint8_t *pSrc, [[maybe_unused]] intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        __m128i sum = zeroes;

        // 2-wide: src is contiguous, so two rows are 4 adjacent bytes. Load each ref row as a
        // 2-byte value and interleave with unpacklo (same idea as SADWrapperU16<2, h>), then one
        // _mm_sad_epu8 per row pair (the unused upper bytes are zero, contributing nothing).
        for (unsigned y = 0; y < height; y += 2) {
            __m128i m2 = _mm_cvtsi32_si128(*(const int *)(pSrc + y * 2));
            __m128i r0 = _mm_cvtsi32_si128(*(const uint16_t *)(pRef + y * nRefPitch));
            __m128i r1 = _mm_cvtsi32_si128(*(const uint16_t *)(pRef + (y + 1) * nRefPitch));

            sum = _mm_add_epi64(sum, _mm_sad_epu8(m2, _mm_unpacklo_epi16(r0, r1)));
        }

        return (unsigned)_mm_cvtsi128_si32(sum);
    }
};


template <unsigned height>
struct SADWrapperU8<4, height> {

    static unsigned int sad_u8_sse2(const uint8_t *pSrc, [[maybe_unused]] intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        __m128i sum = zeroes;

        for (unsigned y = 0; y < height; y++) {
            __m128i m2 = _mm_cvtsi32_si128(*(const int *)pSrc);
            __m128i m3 = _mm_cvtsi32_si128(*(const int *)pRef);

            __m128i diff = _mm_sad_epu8(m2, m3);

            sum = _mm_add_epi64(sum, diff);

            pSrc += /*nSrcPitch*/ 4;
            pRef += nRefPitch;
        }

        return (unsigned)_mm_cvtsi128_si32(sum);
    }

};


template <unsigned height>
struct SADWrapperU8<8, height> {
    static_assert(height == 1 || height % 2 == 0, "");

    static unsigned int sad_u8_sse2(const uint8_t *pSrc, [[maybe_unused]] intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        __m128i sum = zeroes;

        if (height == 1) {
            __m128i m2 = _mm_loadl_epi64((const __m128i *)pSrc);
            __m128i m3 = _mm_loadl_epi64((const __m128i *)pRef);
            sum = _mm_sad_epu8(m2, m3);
            return (unsigned)_mm_cvtsi128_si32(sum);
        }

        for (int y = 0; (unsigned)y < height; y += 2) {
            __m128i m2 = _mm_loadu_si128((const __m128i *)(pSrc + y * 8));
            __m128i m3 = _mm_loadl_epi64((const __m128i *)(pRef + y * nRefPitch));
            m3 = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(m3), (const double *)(pRef + (y + 1) * nRefPitch)));

            __m128i diff = _mm_sad_epu8(m2, m3);

            sum = _mm_add_epi64(sum, diff);
        }

        sum = _mm_add_epi64(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(0, 0, 3, 2)));
        return (unsigned)_mm_cvtsi128_si32(sum);
    }
};


static MVU_FORCE_INLINE __m128i abs_diff_epu16(const __m128i &a, const __m128i &b) {
    return _mm_or_si128(_mm_subs_epu16(a, b),
                        _mm_subs_epu16(b, a));
}


static MVU_FORCE_INLINE __m128i hsum_epi32(const __m128i &a) {
    __m128i sum = a;
    __m128i m0 = _mm_srli_si128(sum, 8);
    sum = _mm_add_epi32(sum, m0);
    m0 = _mm_srli_epi64(sum, 32);
    sum = _mm_add_epi32(sum, m0);

    return sum;
}


// This version used for width >= 8.
template <unsigned width, unsigned height>
struct SADWrapperU16 {

    // 16-bit SAD via a bias + pmaddwd widen. abs_diff_epu16 is unsigned [0,65535]; xor 0x8000 maps it to
    // signed [-32768,32767] (== v-32768), and pmaddwd(.,1) fuses the 16->32 widen + adjacent-pair sum into a
    // single op, so the compile-time constant 32768*width*height is added back at the end. Replaces the
    // unpacklo/unpackhi + two adds (~1.1-1.2x for width<=64, up to ~2x at width=128 where that widen turns
    // port-bound). Full vectors only (width % 8 == 0), so every lane is a valid pixel; int32 can't overflow up
    // to 128x128 (max |partial| ~1.3e8/lane, final SAD < 1.1e9). Bench: bench_degrain/sad16_madd_bench.cpp.
    static unsigned int sad_u16_sse2(const uint8_t *pSrc8, intptr_t nSrcPitch, const uint8_t *pRef8, intptr_t nRefPitch) noexcept {
        const __m128i ones = _mm_set1_epi16(1), bias = _mm_set1_epi16((short)0x8000);
        __m128i sum = zeroes;

        for (unsigned y = 0; y < height; y++) {
            for (unsigned x = 0; x < width; x += 8) {
                const uint16_t *pSrc = (const uint16_t *)pSrc8;
                const uint16_t *pRef = (const uint16_t *)pRef8;

                __m128i m2 = _mm_loadu_si128((const __m128i *)&pSrc[x]);
                __m128i m3 = _mm_loadu_si128((const __m128i *)&pRef[x]);

                __m128i diff = abs_diff_epu16(m2, m3);

                sum = _mm_add_epi32(sum, _mm_madd_epi16(_mm_xor_si128(diff, bias), ones));
            }

            pSrc8 += nSrcPitch;
            pRef8 += nRefPitch;
        }

        return (unsigned)_mm_cvtsi128_si32(hsum_epi32(sum)) + 32768u * width * height;
    }

};


template <>
struct SADWrapperU16<2, 2> {

    static unsigned int sad_u16_sse2(const uint8_t *pSrc, intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        __m128i m2 = _mm_cvtsi32_si128(*(const int *)(pSrc));
        __m128i m4 = _mm_cvtsi32_si128(*(const int *)(pSrc + nSrcPitch));
        m2 = _mm_unpacklo_epi16(m2, m4);
        __m128i m3 = _mm_cvtsi32_si128(*(const int *)(pRef));
        __m128i m5 = _mm_cvtsi32_si128(*(const int *)(pRef + nRefPitch));
        m3 = _mm_unpacklo_epi16(m3, m5);

        m2 = abs_diff_epu16(m2, m3);

        m2 = _mm_unpacklo_epi16(m2, zeroes);

        return (unsigned)_mm_cvtsi128_si32(hsum_epi32(m2));
    }

};


template <>
struct SADWrapperU16<2, 4> {

    static unsigned int sad_u16_sse2(const uint8_t *pSrc, intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        __m128i m2 = _mm_cvtsi32_si128(*(const int *)(pSrc));
        __m128i m4 = _mm_cvtsi32_si128(*(const int *)(pSrc + nSrcPitch));
        m2 = _mm_unpacklo_epi16(m2, m4);
        __m128i m3 = _mm_cvtsi32_si128(*(const int *)(pRef));
        __m128i m5 = _mm_cvtsi32_si128(*(const int *)(pRef + nRefPitch));
        m3 = _mm_unpacklo_epi16(m3, m5);

        __m128i m6 = _mm_cvtsi32_si128(*(const int *)(pSrc + nSrcPitch * 2));
        __m128i m8 = _mm_cvtsi32_si128(*(const int *)(pSrc + nSrcPitch * 3));
        m6 = _mm_unpacklo_epi16(m6, m8);
        __m128i m7 = _mm_cvtsi32_si128(*(const int *)(pRef + nRefPitch * 2));
        __m128i m9 = _mm_cvtsi32_si128(*(const int *)(pRef + nRefPitch * 3));
        m7 = _mm_unpacklo_epi16(m7, m9);

        m2 = _mm_unpacklo_epi16(m2, m6);
        m3 = _mm_unpacklo_epi16(m3, m7);

        m2 = abs_diff_epu16(m2, m3);

        m2 = _mm_add_epi32(_mm_unpacklo_epi16(m2, zeroes),
                           _mm_unpackhi_epi16(m2, zeroes));

        return (unsigned)_mm_cvtsi128_si32(hsum_epi32(m2));
    }

};


template <unsigned height>
struct SADWrapperU16<4, height> {

    static unsigned int sad_u16_sse2(const uint8_t *pSrc, intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        __m128i sum = zeroes;

        for (unsigned y = 0; y < height; y++) {
            __m128i m2 = _mm_loadl_epi64((const __m128i *)pSrc);
            __m128i m3 = _mm_loadl_epi64((const __m128i *)pRef);

            __m128i diff = abs_diff_epu16(m2, m3);

            sum = _mm_add_epi32(sum, _mm_unpacklo_epi16(diff, zeroes));

            pSrc += nSrcPitch;
            pRef += nRefPitch;
        }

        return (unsigned)_mm_cvtsi128_si32(hsum_epi32(sum));
    }

};


#undef zeroes
#endif


template <unsigned width, unsigned height, typename PixelType>
unsigned int sad_c(const uint8_t *pSrc8, intptr_t nSrcPitch, const uint8_t *pRef8, intptr_t nRefPitch) noexcept {
    unsigned int sum = 0;
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            const PixelType *pSrc = (const PixelType *)pSrc8;
            const PixelType *pRef = (const PixelType *)pRef8;
            sum += (unsigned)std::abs(pSrc[x] - pRef[x]);
        }
        pSrc8 += nSrcPitch;
        pRef8 += nRefPitch;
    }
    return sum;
}


#define KEY(width, height, bits) ((unsigned)(width) << 24 | (height) << 16 | (bits) << 8)

// SSE2 is baseline on x86-64 and an intrinsic wrapper exists for every block size, so on x86 the
// SSE2 kernel takes the scalar version's place directly in the map (no instruction-set key, no
// separate map). On non-x86 the same macro emits the scalar kernel.
#if defined(MVTOOLS_X86)
#define SAD(width, height) \
    { KEY(width, height, 8), SADWrapperU8<width, height>::sad_u8_sse2 }, \
    { KEY(width, height, 16), SADWrapperU16<width, height>::sad_u16_sse2 },
#else
#define SAD(width, height) \
    { KEY(width, height, 8), sad_c<width, height, uint8_t> }, \
    { KEY(width, height, 16), sad_c<width, height, uint16_t> },
#endif

static constexpr auto sad_functions = std::to_array<FunctionTableEntry<SADFunction>>({
    SAD(2, 2)
    SAD(2, 4)
    SAD(4, 2)
    SAD(4, 4)
    SAD(4, 8)
    SAD(8, 1)
    SAD(8, 2)
    SAD(8, 4)
    SAD(8, 8)
    SAD(8, 16)
    SAD(16, 1)
    SAD(16, 2)
    SAD(16, 4)
    SAD(16, 8)
    SAD(16, 16)
    SAD(16, 32)
    SAD(32, 8)
    SAD(32, 16)
    SAD(32, 32)
    SAD(32, 64)
    SAD(64, 16)
    SAD(64, 32)
    SAD(64, 64)
    SAD(64, 128)
    SAD(128, 32)
    SAD(128, 64)
    SAD(128, 128)
});

SADFunction selectSADFunction(unsigned width, unsigned height, unsigned bits) {
    // 32-bit float pixels: separate dispatch (auto-vec C baseline, overridden by the hand AVX2/AVX-512
    // kernels). The result is scaled to the 16-bit integer pixel range inside the kernel.
    if (bits == 32) {
        SADFunction sad = nullptr;
        selectSADFunctionFloat(width, height, sad);
#if defined(MVTOOLS_X86)
        if (g_cpuinfo & MVU_CPU_AVX2)
            selectSADFunctionAVX2(width, height, bits, sad);
        if (g_cpuinfo & MVU_CPU_AVX512_BASE)
            selectSADFunctionAVX512(width, height, bits, sad);
#endif
        return sad;
    }

    SADFunction sad = findFunctionOrThrow(sad_functions, KEY(width, height, bits));

#if defined(MVTOOLS_X86)
    if (g_cpuinfo & MVU_CPU_AVX2)
        selectSADFunctionAVX2(width, height, bits, sad);
    if (g_cpuinfo & MVU_CPU_AVX512_BASE)
        selectSADFunctionAVX512(width, height, bits, sad);
    if (g_cpuinfo & MVU_CPU_AVX512VNNI)
        selectSADFunctionAVX512VNNI(width, height, bits, sad); // 16-bit only, beats the base kernel at W 8-64
#endif

    return sad;
}

#undef SAD



#define HADAMARD4(d0, d1, d2, d3, s0, s1, s2, s3) \
    {                                             \
        SumType2 t0 = s0 + s1;                    \
        SumType2 t1 = s0 - s1;                    \
        SumType2 t2 = s2 + s3;                    \
        SumType2 t3 = s2 - s3;                    \
        d0 = t0 + t2;                             \
        d2 = t0 - t2;                             \
        d1 = t1 + t3;                             \
        d3 = t1 - t3;                             \
    }


// in: a pseudo-simd number of the form x+(y<<16)
// return: abs(x)+(abs(y)<<16)
template <typename SumType, typename SumType2>
static MVU_FORCE_INLINE SumType2 abs2(SumType2 a) {
    int bitsPerSum = 8 * sizeof(SumType);

    SumType2 s = ((a >> (bitsPerSum - 1)) & (((SumType2)1 << bitsPerSum) + 1)) * ((SumType)-1);
    return (a + s) ^ s;
}


template <typename PixelType, typename SumType, typename SumType2>
static MVU_FORCE_INLINE unsigned int Real_Satd_4x4_C(const uint8_t *pSrc8, intptr_t nSrcPitch, const uint8_t *pRef8, intptr_t nRefPitch) {
    int bitsPerSum = 8 * sizeof(SumType);

    SumType2 tmp[4][2];
    SumType2 a0, a1, a2, a3, b0, b1;
    SumType2 sum = 0;

    for (int i = 0; i < 4; i++) {
        const PixelType *pSrc = (const PixelType *)pSrc8;
        const PixelType *pRef = (const PixelType *)pRef8;

        a0 = pSrc[0] - pRef[0];
        a1 = pSrc[1] - pRef[1];
        b0 = (a0 + a1) + ((a0 - a1) << bitsPerSum);
        a2 = pSrc[2] - pRef[2];
        a3 = pSrc[3] - pRef[3];
        b1 = (a2 + a3) + ((a2 - a3) << bitsPerSum);
        tmp[i][0] = b0 + b1;
        tmp[i][1] = b0 - b1;

        pSrc8 += nSrcPitch;
        pRef8 += nRefPitch;
    }

    for (int i = 0; i < 2; i++) {
        HADAMARD4(a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i]);
        a0 = abs2<SumType, SumType2>(a0) + abs2<SumType, SumType2>(a1) + abs2<SumType, SumType2>(a2) + abs2<SumType, SumType2>(a3);
        sum += ((SumType)a0) + (a0 >> bitsPerSum);
    }

    return (unsigned int)(sum >> 1);
}


template <typename PixelType>
static MVU_FORCE_INLINE unsigned int Satd_4x4_C(const uint8_t *pSrc, intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) {
    if (sizeof(PixelType) == 1)
        return Real_Satd_4x4_C<PixelType, uint16_t, uint32_t>(pSrc, nSrcPitch, pRef, nRefPitch);
    else
        return Real_Satd_4x4_C<PixelType, uint32_t, uint64_t>(pSrc, nSrcPitch, pRef, nRefPitch);
}


template <typename PixelType, typename SumType, typename SumType2>
static MVU_FORCE_INLINE unsigned int Real_Satd_8x4_C(const uint8_t *pSrc8, intptr_t nSrcPitch, const uint8_t *pRef8, intptr_t nRefPitch) {
    int bitsPerSum = 8 * sizeof(SumType);

    SumType2 tmp[4][4];
    SumType2 a0, a1, a2, a3;
    SumType2 sum = 0;

    for (int i = 0; i < 4; i++) {
        const PixelType *pSrc = (const PixelType *)pSrc8;
        const PixelType *pRef = (const PixelType *)pRef8;

        a0 = (pSrc[0] - pRef[0]) + ((SumType2)(pSrc[4] - pRef[4]) << bitsPerSum);
        a1 = (pSrc[1] - pRef[1]) + ((SumType2)(pSrc[5] - pRef[5]) << bitsPerSum);
        a2 = (pSrc[2] - pRef[2]) + ((SumType2)(pSrc[6] - pRef[6]) << bitsPerSum);
        a3 = (pSrc[3] - pRef[3]) + ((SumType2)(pSrc[7] - pRef[7]) << bitsPerSum);
        HADAMARD4(tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], a0, a1, a2, a3);

        pSrc8 += nSrcPitch;
        pRef8 += nRefPitch;
    }
    for (int i = 0; i < 4; i++) {
        HADAMARD4(a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i]);
        sum += abs2<SumType, SumType2>(a0) + abs2<SumType, SumType2>(a1) + abs2<SumType, SumType2>(a2) + abs2<SumType, SumType2>(a3);
    }

    return (unsigned int)((((SumType)sum) + (sum >> bitsPerSum)) >> 1);
}

template <typename PixelType>
static MVU_FORCE_INLINE unsigned int Satd_8x4_C(const uint8_t *pSrc, intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) {
    if (sizeof(PixelType) == 1)
        return Real_Satd_8x4_C<PixelType, uint16_t, uint32_t>(pSrc, nSrcPitch, pRef, nRefPitch);
    else
        return Real_Satd_8x4_C<PixelType, uint32_t, uint64_t>(pSrc, nSrcPitch, pRef, nRefPitch);
}


// Doesn't handle 16x2 blocks.
template <int nBlkWidth, int nBlkHeight, typename PixelType>
static unsigned int Satd_C(const uint8_t *pSrc, intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
    if (nBlkWidth == 4 && nBlkHeight == 4)
        return Satd_4x4_C<PixelType>(pSrc, nSrcPitch, pRef, nRefPitch);
    else {
        const int bytesPerSample = sizeof(PixelType);
        const int partition_width = 8;
        const int partition_height = 4;

        uint64_t sum = 0;

        for (int y = 0; y < nBlkHeight; y += partition_height) {
            for (int x = 0; x < nBlkWidth; x += partition_width)
                sum += Satd_8x4_C<PixelType>(pSrc + x * bytesPerSample, nSrcPitch,
                                             pRef + x * bytesPerSample, nRefPitch);

            pSrc += nSrcPitch * partition_height;
            pRef += nRefPitch * partition_height;
        }

        return static_cast<unsigned int>(sum > 0xFFFFFFFFu ? 0xFFFFFFFFu : sum);
    }
}

#if defined(MVTOOLS_X86)
// Intrinsic SATD (Hadamard), replacing the x264 asm. SATD tiles into 4x4 units; each unit's
// score is (sum of |coeff| of the separable 4x4 Hadamard of src-ref) >> 1. That abs-sum is
// always even (sum of all coeffs == 16*DC), so the block SATD = (total abs-sum over all 4x4
// sub-blocks) >> 1, matching the scalar Satd_C bit-for-bit. Core: diff -> vertical Hadamard
// -> 4x4 transpose -> Hadamard -> abs -> sum; abs-sum is order/sign invariant so any valid
// 4-point butterfly works. 8-bit packs two 4x4 in the 128-bit lanes (8x4 core); 16-bit uses
// int32 lanes (4x4 core) with int64 accumulation.
static MVU_FORCE_INLINE __m128i satd_abs_epi16(__m128i x) noexcept {
    __m128i s = _mm_srai_epi16(x, 15);
    return _mm_sub_epi16(_mm_xor_si128(x, s), s);
}
static MVU_FORCE_INLINE __m128i satd_abs_epi32(__m128i x) noexcept {
    __m128i s = _mm_srai_epi32(x, 31);
    return _mm_sub_epi32(_mm_xor_si128(x, s), s);
}
static MVU_FORCE_INLINE unsigned satd_hsum_epi32(__m128i a) noexcept {
    a = _mm_add_epi32(a, _mm_srli_si128(a, 8));
    a = _mm_add_epi32(a, _mm_srli_si128(a, 4));
    return (unsigned)_mm_cvtsi128_si32(a);
}
static MVU_FORCE_INLINE uint64_t satd_hsum2_epi64(__m128i a) noexcept {
    return (uint64_t)_mm_cvtsi128_si64(a) + (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(a, a));
}
static MVU_FORCE_INLINE __m128i satd_4x4_u8(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    const __m128i z = _mm_setzero_si128();
    __m128i d0 = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int *)(s + 0 * sp)), z), _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int *)(r + 0 * rp)), z));
    __m128i d1 = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int *)(s + 1 * sp)), z), _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int *)(r + 1 * rp)), z));
    __m128i d2 = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int *)(s + 2 * sp)), z), _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int *)(r + 2 * rp)), z));
    __m128i d3 = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int *)(s + 3 * sp)), z), _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int *)(r + 3 * rp)), z));
    __m128i t0 = _mm_add_epi16(d0, d1), t1 = _mm_sub_epi16(d0, d1), t2 = _mm_add_epi16(d2, d3), t3 = _mm_sub_epi16(d2, d3);
    __m128i h0 = _mm_add_epi16(t0, t2), h1 = _mm_add_epi16(t1, t3), h2 = _mm_sub_epi16(t0, t2), h3 = _mm_sub_epi16(t1, t3);
    __m128i p0 = _mm_unpacklo_epi16(h0, h1), p1 = _mm_unpacklo_epi16(h2, h3);
    __m128i lo = _mm_unpacklo_epi32(p0, p1), hi = _mm_unpackhi_epi32(p0, p1);
    __m128i T0 = _mm_move_epi64(lo), T1 = _mm_srli_si128(lo, 8), T2 = _mm_move_epi64(hi), T3 = _mm_srli_si128(hi, 8);
    __m128i u0 = _mm_add_epi16(T0, T1), u1 = _mm_sub_epi16(T0, T1), u2 = _mm_add_epi16(T2, T3), u3 = _mm_sub_epi16(T2, T3);
    // |x+y|+|x-y| == 2*max(|x|,|y|): fold last butterfly + abs + satd >>1 into one max (pmaxsw is SSE2).
    __m128i m0 = _mm_max_epi16(satd_abs_epi16(u0), satd_abs_epi16(u2));
    __m128i m1 = _mm_max_epi16(satd_abs_epi16(u1), satd_abs_epi16(u3));
    return _mm_madd_epi16(_mm_add_epi16(m0, m1), _mm_set1_epi16(1)); // already halved; caller must NOT >>1
}
static MVU_FORCE_INLINE __m128i satd_8x4_u8(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    const __m128i z = _mm_setzero_si128();
    __m128i d0 = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 0 * sp)), z), _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(r + 0 * rp)), z));
    __m128i d1 = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 1 * sp)), z), _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(r + 1 * rp)), z));
    __m128i d2 = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 2 * sp)), z), _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(r + 2 * rp)), z));
    __m128i d3 = _mm_sub_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 3 * sp)), z), _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(r + 3 * rp)), z));
    __m128i t0 = _mm_add_epi16(d0, d1), t1 = _mm_sub_epi16(d0, d1), t2 = _mm_add_epi16(d2, d3), t3 = _mm_sub_epi16(d2, d3);
    __m128i h0 = _mm_add_epi16(t0, t2), h1 = _mm_add_epi16(t1, t3), h2 = _mm_sub_epi16(t0, t2), h3 = _mm_sub_epi16(t1, t3);
    __m128i lo01 = _mm_unpacklo_epi16(h0, h1), hi01 = _mm_unpackhi_epi16(h0, h1);
    __m128i lo23 = _mm_unpacklo_epi16(h2, h3), hi23 = _mm_unpackhi_epi16(h2, h3);
    __m128i loA = _mm_unpacklo_epi32(lo01, lo23), hiA = _mm_unpackhi_epi32(lo01, lo23);
    __m128i loB = _mm_unpacklo_epi32(hi01, hi23), hiB = _mm_unpackhi_epi32(hi01, hi23);
    __m128i r0 = _mm_unpacklo_epi64(loA, loB), r1 = _mm_unpackhi_epi64(loA, loB);
    __m128i r2 = _mm_unpacklo_epi64(hiA, hiB), r3 = _mm_unpackhi_epi64(hiA, hiB);
    __m128i u0 = _mm_add_epi16(r0, r1), u1 = _mm_sub_epi16(r0, r1), u2 = _mm_add_epi16(r2, r3), u3 = _mm_sub_epi16(r2, r3);
    // |x+y|+|x-y| == 2*max(|x|,|y|): fold last butterfly + abs + satd >>1 into one max (pmaxsw is SSE2).
    __m128i m0 = _mm_max_epi16(satd_abs_epi16(u0), satd_abs_epi16(u2));
    __m128i m1 = _mm_max_epi16(satd_abs_epi16(u1), satd_abs_epi16(u3));
    return _mm_madd_epi16(_mm_add_epi16(m0, m1), _mm_set1_epi16(1)); // already halved; caller must NOT >>1
}
static MVU_FORCE_INLINE __m128i satd_4x4_u16(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    const __m128i z = _mm_setzero_si128();
    __m128i d0 = _mm_sub_epi32(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)(s + 0 * sp)), z), _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)(r + 0 * rp)), z));
    __m128i d1 = _mm_sub_epi32(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)(s + 1 * sp)), z), _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)(r + 1 * rp)), z));
    __m128i d2 = _mm_sub_epi32(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)(s + 2 * sp)), z), _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)(r + 2 * rp)), z));
    __m128i d3 = _mm_sub_epi32(_mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)(s + 3 * sp)), z), _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i *)(r + 3 * rp)), z));
    __m128i t0 = _mm_add_epi32(d0, d1), t1 = _mm_sub_epi32(d0, d1), t2 = _mm_add_epi32(d2, d3), t3 = _mm_sub_epi32(d2, d3);
    __m128i h0 = _mm_add_epi32(t0, t2), h1 = _mm_add_epi32(t1, t3), h2 = _mm_sub_epi32(t0, t2), h3 = _mm_sub_epi32(t1, t3);
    __m128i p0 = _mm_unpacklo_epi32(h0, h1), p1 = _mm_unpacklo_epi32(h2, h3);
    __m128i p2 = _mm_unpackhi_epi32(h0, h1), p3 = _mm_unpackhi_epi32(h2, h3);
    __m128i T0 = _mm_unpacklo_epi64(p0, p1), T1 = _mm_unpackhi_epi64(p0, p1), T2 = _mm_unpacklo_epi64(p2, p3), T3 = _mm_unpackhi_epi64(p2, p3);
    __m128i u0 = _mm_add_epi32(T0, T1), u1 = _mm_sub_epi32(T0, T1), u2 = _mm_add_epi32(T2, T3), u3 = _mm_sub_epi32(T2, T3);
    __m128i v0 = _mm_add_epi32(u0, u2), v1 = _mm_add_epi32(u1, u3), v2 = _mm_sub_epi32(u0, u2), v3 = _mm_sub_epi32(u1, u3);
    return _mm_add_epi32(_mm_add_epi32(satd_abs_epi32(v0), satd_abs_epi32(v1)), _mm_add_epi32(satd_abs_epi32(v2), satd_abs_epi32(v3)));
}
template <unsigned W, unsigned H>
static unsigned int satd_u8_sse2(const uint8_t *src, intptr_t sp, const uint8_t *ref, intptr_t rp) noexcept {
    __m128i acc = _mm_setzero_si128();
    if (W == 4) {
        for (unsigned y = 0; y < H; y += 4)
            acc = _mm_add_epi32(acc, satd_4x4_u8(src + y * sp, sp, ref + y * rp, rp));
    } else {
        for (unsigned y = 0; y < H; y += 4)
            for (unsigned x = 0; x < W; x += 8)
                acc = _mm_add_epi32(acc, satd_8x4_u8(src + y * sp + x, sp, ref + y * rp + x, rp));
    }
    uint64_t sum = satd_hsum_epi32(acc); // cores already halved (amax)
    return (unsigned)(sum > 0xFFFFFFFFu ? 0xFFFFFFFFu : sum);
}
template <unsigned W, unsigned H>
static unsigned int satd_u16_sse2(const uint8_t *src, intptr_t sp, const uint8_t *ref, intptr_t rp) noexcept {
    const __m128i z = _mm_setzero_si128();
    __m128i acc = _mm_setzero_si128(); // 2 int64
    for (unsigned y = 0; y < H; y += 4)
        for (unsigned x = 0; x < W; x += 4) {
            __m128i a = satd_4x4_u16(src + y * sp + (intptr_t)x * 2, sp, ref + y * rp + (intptr_t)x * 2, rp);
            acc = _mm_add_epi64(acc, _mm_add_epi64(_mm_unpacklo_epi32(a, z), _mm_unpackhi_epi32(a, z)));
        }
    uint64_t sum = satd_hsum2_epi64(acc) >> 1;
    return (unsigned)(sum > 0xFFFFFFFFu ? 0xFFFFFFFFu : sum);
}
#endif

// As with SAD: on x86 the SSE2 kernel replaces the scalar version directly in the map.
#if defined(MVTOOLS_X86)
#define SATD(width, height) \
    { KEY(width, height, 8), satd_u8_sse2<width, height> }, \
    { KEY(width, height, 16), satd_u16_sse2<width, height> },
#else
#define SATD(width, height) \
    { KEY(width, height, 8), Satd_C<width, height, uint8_t> }, \
    { KEY(width, height, 16), Satd_C<width, height, uint16_t> },
#endif

static constexpr auto satd_functions = std::to_array<FunctionTableEntry<SADFunction>>({
    SATD(4, 4)
    SATD(8, 4)
    SATD(8, 8)
    SATD(16, 8)
    SATD(16, 16)
    SATD(32, 16)
    SATD(32, 32)
    SATD(64, 32)
    SATD(64, 64)
    SATD(128, 64)
    SATD(128, 128)
});

SADFunction selectSATDFunction(unsigned width, unsigned height, unsigned bits) {
    // 32-bit float pixels: pure scalar SATD (auto-vectorized at the baseline ISA), no per-ISA overrides.
    // The result is 16-bit-scaled internally.
    if (bits == 32) {
        SADFunction satd = nullptr;
        selectSATDFunctionFloat(width, height, satd);
        return satd;
    }

    SADFunction satd = findFunctionOrThrow(satd_functions, KEY(width, height, bits));

#if defined(MVTOOLS_X86)
    if (g_cpuinfo & MVU_CPU_AVX2)
        selectSATDFunctionAVX2(width, height, bits, satd);
    if (g_cpuinfo & MVU_CPU_AVX512_BASE)
        selectSATDFunctionAVX512(width, height, bits, satd);
#endif

    return satd;
}

#undef SATD

#undef KEY
