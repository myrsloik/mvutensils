#include <cstdint>

#include "FunctionTable.h"
#include "SADFunctions.h"
#include "SADFunctions_Float.h"
#include "Common.h"

#if defined(MVTOOLS_X86)

#include <immintrin.h>

// 32-bit float SAD (hand intrinsics, scaled to the 16-bit pixel range). Width-capped to YMM/XMM here.
template <unsigned W, unsigned H>
static unsigned int sad_f32_avx2(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    return scale_f32_sad(sad_f32_hand_raw<W, H>(s, sp, r, rp));
}


// This version used for width >= 32.
template <unsigned width, unsigned height>
struct SADWrapperU8_AVX2 {
    static_assert(width >= 32 && height % 2 == 0, "");

    static unsigned int sad_u8_avx2(const uint8_t *pSrc, [[maybe_unused]] intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        // Two accumulators (even/odd rows) -> two independent _mm256_add_epi64 chains. vpsadbw is the
        // bottleneck for most sizes (so this is neutral there), but for the largest blocks the single
        // accumulator's loop-carried add chain dominates; splitting it is ~22% faster at 128x128 and
        // neutral elsewhere. (This is the x264 sad-a.asm trick of accumulating into multiple registers.)
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();

        for (unsigned y = 0; y < height; y += 2) {
            for (unsigned x = 0; x < width; x += 32) {
                sum0 = _mm256_add_epi64(sum0, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)&pSrc[x]),
                                                              _mm256_loadu_si256((const __m256i *)&pRef[x])));
                sum1 = _mm256_add_epi64(sum1, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)&pSrc[width + x]),
                                                              _mm256_loadu_si256((const __m256i *)&pRef[nRefPitch + x])));
            }

            pSrc += /*nSrcPitch*/ 2 * width;
            pRef += 2 * nRefPitch;
        }

        __m256i sum = _mm256_add_epi64(sum0, sum1);
        sum = _mm256_add_epi64(sum, _mm256_permute4x64_epi64(sum, _MM_SHUFFLE(0, 0, 3, 2)));
        sum = _mm256_add_epi64(sum, _mm256_shuffle_epi32(sum, _MM_SHUFFLE(0, 0, 3, 2)));
        return (unsigned)_mm_cvtsi128_si32(_mm256_castsi256_si128(sum));
    }

};

template <unsigned height>
struct SADWrapperU8_AVX2<16, height> {
    static_assert(height >= 2, "");

    static unsigned int sad_u8_avx2(const uint8_t *pSrc, [[maybe_unused]] intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        __m256i sum = _mm256_setzero_si256();

        for (int y = 0; (unsigned)y < height; y += 2) {
            __m256i m2 = _mm256_loadu_si256((const __m256i *)(pSrc + y * 16));
            __m256i m3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(pRef + y * nRefPitch)));
            m3 = _mm256_inserti128_si256(m3, _mm_loadu_si128((const __m128i *)(pRef + (y + 1) * nRefPitch)), 1);

            __m256i diff = _mm256_sad_epu8(m2, m3);
            sum = _mm256_add_epi64(sum, diff);
        }

        sum = _mm256_add_epi64(sum, _mm256_permute4x64_epi64(sum, _MM_SHUFFLE(0, 0, 3, 2)));
        sum = _mm256_add_epi64(sum, _mm256_shuffle_epi32(sum, _MM_SHUFFLE(0, 0, 3, 2)));
        return (unsigned)_mm_cvtsi128_si32(_mm256_castsi256_si128(sum));
    }
};


// 16-bit SAD. There is no psadbw for 16-bit pixels, so this does abs_diff_epu16 then a bias + pmaddwd
// widen: abs_diff is unsigned [0,65535], xor 0x8000 maps it to signed [-32768,32767] (== v-32768), and
// pmaddwd(.,1) fuses the 16->32 widen + adjacent-pair sum into one op (the compile-time constant
// 32768*width*height is added back). ~1.1-1.2x over the unpacklo/unpackhi + two adds it replaces (widened
// to ymm, 16 uint16/row). width >= 16 fills one ymm per row; width-8 16-bit stays on the SSE2 path.
// int32 can't overflow up to 128x128. Bench: bench_degrain/sad16_madd_bench.cpp.
template <unsigned width, unsigned height>
struct SADWrapperU16_AVX2 {
    static_assert(width >= 16, "");

    static unsigned int sad_u16_avx2(const uint8_t *pSrc8, intptr_t nSrcPitch, const uint8_t *pRef8, intptr_t nRefPitch) noexcept {
        const __m256i ones = _mm256_set1_epi16(1), bias = _mm256_set1_epi16((short)0x8000);
        __m256i sum = _mm256_setzero_si256();

        for (unsigned y = 0; y < height; y++) {
            const uint16_t *pSrc = (const uint16_t *)pSrc8;
            const uint16_t *pRef = (const uint16_t *)pRef8;

            for (unsigned x = 0; x < width; x += 16) {
                __m256i m2 = _mm256_loadu_si256((const __m256i *)&pSrc[x]);
                __m256i m3 = _mm256_loadu_si256((const __m256i *)&pRef[x]);
                __m256i diff = _mm256_or_si256(_mm256_subs_epu16(m2, m3), _mm256_subs_epu16(m3, m2));

                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_xor_si256(diff, bias), ones));
            }

            pSrc8 += nSrcPitch;
            pRef8 += nRefPitch;
        }

        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
        s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
        return (unsigned)_mm_cvtsi128_si32(s) + 32768u * width * height;
    }
};


#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8

// This AVX2 map is private to this TU and only queried via selectSADFunctionAVX2, which overwrites
// the caller's function pointer when a kernel exists for the requested size.
#define SAD_U8_AVX2(width, height) \
    { KEY(width, height, 8), SADWrapperU8_AVX2<width, height>::sad_u8_avx2 },
#define SAD_U16_AVX2(width, height) \
    { KEY(width, height, 16), SADWrapperU16_AVX2<width, height>::sad_u16_avx2 },
#define SAD_F32_AVX2(width, height) \
    { KEY(width, height, 32), sad_f32_avx2<width, height> },

static constexpr auto sad_functions = std::to_array<FunctionTableEntry<SADFunction>>({
    SAD_U8_AVX2(16, 2)
    SAD_U8_AVX2(16, 4)
    SAD_U8_AVX2(16, 8)
    SAD_U8_AVX2(16, 16)
    SAD_U8_AVX2(16, 32)
    SAD_U8_AVX2(32, 8)
    SAD_U8_AVX2(32, 16)
    SAD_U8_AVX2(32, 32)
    SAD_U8_AVX2(32, 64)
    SAD_U8_AVX2(64, 16)
    SAD_U8_AVX2(64, 32)
    SAD_U8_AVX2(64, 64)
    SAD_U8_AVX2(64, 128)
    SAD_U8_AVX2(128, 32)
    SAD_U8_AVX2(128, 64)
    SAD_U8_AVX2(128, 128)
    // 16-bit: width >= 16 only (width-8 16-bit stays on the SSE2 path)
    SAD_U16_AVX2(16, 1)
    SAD_U16_AVX2(16, 2)
    SAD_U16_AVX2(16, 4)
    SAD_U16_AVX2(16, 8)
    SAD_U16_AVX2(16, 16)
    SAD_U16_AVX2(16, 32)
    SAD_U16_AVX2(32, 8)
    SAD_U16_AVX2(32, 16)
    SAD_U16_AVX2(32, 32)
    SAD_U16_AVX2(32, 64)
    SAD_U16_AVX2(64, 16)
    SAD_U16_AVX2(64, 32)
    SAD_U16_AVX2(64, 64)
    SAD_U16_AVX2(64, 128)
    SAD_U16_AVX2(128, 32)
    SAD_U16_AVX2(128, 64)
    SAD_U16_AVX2(128, 128)
    // 32-bit float: width >= 8 only (YMM). W < 8 uses the auto-vec C baseline -- no SSE2 float kernel.
    SAD_F32_AVX2(8, 1) SAD_F32_AVX2(8, 2) SAD_F32_AVX2(8, 4) SAD_F32_AVX2(8, 8) SAD_F32_AVX2(8, 16)
    SAD_F32_AVX2(16, 1) SAD_F32_AVX2(16, 2) SAD_F32_AVX2(16, 4) SAD_F32_AVX2(16, 8) SAD_F32_AVX2(16, 16) SAD_F32_AVX2(16, 32)
    SAD_F32_AVX2(32, 8) SAD_F32_AVX2(32, 16) SAD_F32_AVX2(32, 32) SAD_F32_AVX2(32, 64)
    SAD_F32_AVX2(64, 16) SAD_F32_AVX2(64, 32) SAD_F32_AVX2(64, 64) SAD_F32_AVX2(64, 128)
    SAD_F32_AVX2(128, 32) SAD_F32_AVX2(128, 64) SAD_F32_AVX2(128, 128)
});

void selectSADFunctionAVX2(unsigned width, unsigned height, unsigned bits, SADFunction &sad) {
    if (SADFunction fn = findFunction(sad_functions, KEY(width, height, bits)))
        sad = fn;
}


// ===================== Intrinsic SATD (Hadamard), AVX2 =====================
// 8-bit width >= 16 uses the x264 pmaddubsw/hmul technique: the horizontal stage-1 butterfly is
// fused into the load so there is no 4x4 transpose (see satd_16x4_u8). 8x8 packs two 8x4 in the
// ymm lanes (satd_pair8x4_body_u8); the 16-bit core does an 8x4 tile (two 4x4). The final Hadamard
// stage uses the amax identity |x+y|+|x-y| == 2*max(|x|,|y|), which also absorbs the satd >>1.
// Only sizes where AVX2 beats the SSE2 intrinsic are registered (8-bit width >= 16 plus 8x8,
// 16-bit width >= 8); selectSATDFunction falls back to the SSE2 intrinsic for the rest.

static MVU_FORCE_INLINE unsigned satd_hsum_epi32(__m256i a) noexcept {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
    return (unsigned)_mm_cvtsi128_si32(s);
}
static MVU_FORCE_INLINE uint64_t satd_hsum4_epi64(__m256i a) noexcept {
    __m128i s = _mm_add_epi64(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    return (uint64_t)_mm_cvtsi128_si64(s) + (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(s, s));
}
// Shared 8-bit body: 4 diff-rows, each ymm 128-lane an independent 8x4 (two 4x4). -> 8 int32.
static MVU_FORCE_INLINE __m256i satd_pair8x4_body_u8(__m256i d0, __m256i d1, __m256i d2, __m256i d3) noexcept {
    __m256i t0 = _mm256_add_epi16(d0, d1), t1 = _mm256_sub_epi16(d0, d1), t2 = _mm256_add_epi16(d2, d3), t3 = _mm256_sub_epi16(d2, d3);
    __m256i h0 = _mm256_add_epi16(t0, t2), h1 = _mm256_add_epi16(t1, t3), h2 = _mm256_sub_epi16(t0, t2), h3 = _mm256_sub_epi16(t1, t3);
    __m256i lo01 = _mm256_unpacklo_epi16(h0, h1), hi01 = _mm256_unpackhi_epi16(h0, h1);
    __m256i lo23 = _mm256_unpacklo_epi16(h2, h3), hi23 = _mm256_unpackhi_epi16(h2, h3);
    __m256i loA = _mm256_unpacklo_epi32(lo01, lo23), hiA = _mm256_unpackhi_epi32(lo01, lo23);
    __m256i loB = _mm256_unpacklo_epi32(hi01, hi23), hiB = _mm256_unpackhi_epi32(hi01, hi23);
    __m256i r0 = _mm256_unpacklo_epi64(loA, loB), r1 = _mm256_unpackhi_epi64(loA, loB);
    __m256i r2 = _mm256_unpacklo_epi64(hiA, hiB), r3 = _mm256_unpackhi_epi64(hiA, hiB);
    __m256i u0 = _mm256_add_epi16(r0, r1), u1 = _mm256_sub_epi16(r0, r1), u2 = _mm256_add_epi16(r2, r3), u3 = _mm256_sub_epi16(r2, r3);
    // |x+y| + |x-y| == 2*max(|x|,|y|): folds the last butterfly, the abs, and the satd >>1 into one max.
    __m256i m0 = _mm256_max_epi16(_mm256_abs_epi16(u0), _mm256_abs_epi16(u2));
    __m256i m1 = _mm256_max_epi16(_mm256_abs_epi16(u1), _mm256_abs_epi16(u3));
    return _mm256_madd_epi16(_mm256_add_epi16(m0, m1), _mm256_set1_epi16(1)); // already halved; caller must NOT >>1
}
// 16x4 via pmaddubsw/hmul (x264 technique): the horizontal stage-1 butterfly is fused into the
// load, so there is NO 4x4 transpose. Each row is broadcast into both lanes; one vpmaddubsw with
// hmul (lane0 = +1x16 -> pair sums, lane1 = +1,-1 x8 -> pair diffs) yields [s0..s7 | d0..d7] of
// the row's diffs. Vertical Hadamard across the 4 rows, then the horizontal stage-2 collapses to an
// adjacent-pair amax (|x+y|+|x-y| == 2*max(|x|,|y|)) -- a vpshufb swap + vpmaxsw, no transpose.
static MVU_FORCE_INLINE __m256i satd_16x4_u8(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    const __m256i hmul = _mm256_setr_epi8(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1);
    auto row = [&](int y) {
        __m256i sv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(s + y * sp)));
        __m256i rv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(r + y * rp)));
        return _mm256_sub_epi16(_mm256_maddubs_epi16(sv, hmul), _mm256_maddubs_epi16(rv, hmul));
    };
    __m256i R0 = row(0), R1 = row(1), R2 = row(2), R3 = row(3);
    __m256i t0 = _mm256_add_epi16(R0, R1), t1 = _mm256_sub_epi16(R0, R1), t2 = _mm256_add_epi16(R2, R3), t3 = _mm256_sub_epi16(R2, R3);
    __m256i V0 = _mm256_add_epi16(t0, t2), V1 = _mm256_add_epi16(t1, t3), V2 = _mm256_sub_epi16(t0, t2), V3 = _mm256_sub_epi16(t1, t3);
    const __m256i sw = _mm256_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13, 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
    auto pm = [&](__m256i v) { __m256i a = _mm256_abs_epi16(v); return _mm256_max_epi16(a, _mm256_shuffle_epi8(a, sw)); };
    __m256i m = _mm256_add_epi16(_mm256_add_epi16(pm(V0), pm(V1)), _mm256_add_epi16(pm(V2), pm(V3)));
    // m holds each adjacent-pair max twice; madd with [1,0,...] keeps one copy -> 8 int32 (already halved).
    const __m256i evenmask = _mm256_setr_epi16(1,0,1,0,1,0,1,0, 1,0,1,0,1,0,1,0);
    return _mm256_madd_epi16(m, evenmask);
}
// 8x8: lane0 = top 8x4 (rows 0-3), lane1 = bottom 8x4 (rows 4-7) -- two 8-byte loads/row.
static MVU_FORCE_INLINE __m256i satd_8x8_u8(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    auto dr = [&](int y) {
        __m128i st = _mm_loadl_epi64((const __m128i *)(s + y * sp)), sb = _mm_loadl_epi64((const __m128i *)(s + (y + 4) * sp));
        __m128i rt = _mm_loadl_epi64((const __m128i *)(r + y * rp)), rb = _mm_loadl_epi64((const __m128i *)(r + (y + 4) * rp));
        return _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_unpacklo_epi64(st, sb)), _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(rt, rb)));
    };
    return satd_pair8x4_body_u8(dr(0), dr(1), dr(2), dr(3));
}
static MVU_FORCE_INLINE __m256i satd_8x4_u16(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    auto dr = [&](int y) {
        __m256i a = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(s + y * sp)));
        __m256i b = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(r + y * rp)));
        return _mm256_sub_epi32(a, b);
    };
    __m256i d0 = dr(0), d1 = dr(1), d2 = dr(2), d3 = dr(3);
    __m256i t0 = _mm256_add_epi32(d0, d1), t1 = _mm256_sub_epi32(d0, d1), t2 = _mm256_add_epi32(d2, d3), t3 = _mm256_sub_epi32(d2, d3);
    __m256i h0 = _mm256_add_epi32(t0, t2), h1 = _mm256_add_epi32(t1, t3), h2 = _mm256_sub_epi32(t0, t2), h3 = _mm256_sub_epi32(t1, t3);
    __m256i p0 = _mm256_unpacklo_epi32(h0, h1), p1 = _mm256_unpacklo_epi32(h2, h3);
    __m256i p2 = _mm256_unpackhi_epi32(h0, h1), p3 = _mm256_unpackhi_epi32(h2, h3);
    __m256i T0 = _mm256_unpacklo_epi64(p0, p1), T1 = _mm256_unpackhi_epi64(p0, p1), T2 = _mm256_unpacklo_epi64(p2, p3), T3 = _mm256_unpackhi_epi64(p2, p3);
    __m256i u0 = _mm256_add_epi32(T0, T1), u1 = _mm256_sub_epi32(T0, T1), u2 = _mm256_add_epi32(T2, T3), u3 = _mm256_sub_epi32(T2, T3);
    // |x+y| + |x-y| == 2*max(|x|,|y|): folds the last butterfly, the abs, and the satd >>1 into one max.
    __m256i m0 = _mm256_max_epi32(_mm256_abs_epi32(u0), _mm256_abs_epi32(u2));
    __m256i m1 = _mm256_max_epi32(_mm256_abs_epi32(u1), _mm256_abs_epi32(u3));
    return _mm256_add_epi32(m0, m1); // already halved; caller must NOT >>1
}
template <unsigned W, unsigned H>
static unsigned int satd_u8_avx2(const uint8_t *src, intptr_t sp, const uint8_t *ref, intptr_t rp) noexcept {
    if constexpr (W == 8 && H == 8) { // 8x8: the two stacked 8x4 in the two ymm lanes
        uint64_t sum = satd_hsum_epi32(satd_8x8_u8(src, sp, ref, rp)); // body already halved (amax)
        return (unsigned)(sum > 0xFFFFFFFFu ? 0xFFFFFFFFu : sum);
    } else {
        // The generic path loads 16 bytes/row in 16-wide tiles; only register sizes it can handle.
        static_assert(W % 16 == 0 && H % 4 == 0, "AVX2 SATD generic path needs width %16, height %4 (else 8x8)");
        __m256i acc = _mm256_setzero_si256();
        for (unsigned y = 0; y < H; y += 4)
            for (unsigned x = 0; x < W; x += 16)
                acc = _mm256_add_epi32(acc, satd_16x4_u8(src + y * sp + x, sp, ref + y * rp + x, rp));
        uint64_t sum = satd_hsum_epi32(acc); // bodies already halved (amax)
        return (unsigned)(sum > 0xFFFFFFFFu ? 0xFFFFFFFFu : sum);
    }
}
template <unsigned W, unsigned H>
static unsigned int satd_u16_avx2(const uint8_t *src, intptr_t sp, const uint8_t *ref, intptr_t rp) noexcept {
    // Loads 8 u16 (16 bytes)/row in 8-wide tiles; only register sizes it can handle.
    static_assert(W % 8 == 0 && H % 4 == 0, "AVX2 SATD16 path needs width %8, height %4");
    const __m256i z = _mm256_setzero_si256();
    __m256i acc = _mm256_setzero_si256(); // 4 int64
    for (unsigned y = 0; y < H; y += 4)
        for (unsigned x = 0; x < W; x += 8) {
            __m256i a = satd_8x4_u16(src + y * sp + (intptr_t)x * 2, sp, ref + y * rp + (intptr_t)x * 2, rp);
            acc = _mm256_add_epi64(acc, _mm256_add_epi64(_mm256_unpacklo_epi32(a, z), _mm256_unpackhi_epi32(a, z)));
        }
    uint64_t sum = satd_hsum4_epi64(acc); // core already halved (amax)
    return (unsigned)(sum > 0xFFFFFFFFu ? 0xFFFFFFFFu : sum);
}

#define SATD_U8_AVX2(width, height) { KEY(width, height, 8), satd_u8_avx2<width, height> },
#define SATD_U16_AVX2(width, height) { KEY(width, height, 16), satd_u16_avx2<width, height> },
static constexpr auto satd_functions = std::to_array<FunctionTableEntry<SADFunction>>({
    SATD_U8_AVX2(8, 8)
    SATD_U8_AVX2(16, 8) SATD_U8_AVX2(16, 16)
    SATD_U8_AVX2(32, 16) SATD_U8_AVX2(32, 32) SATD_U8_AVX2(64, 32)
    SATD_U8_AVX2(64, 64) SATD_U8_AVX2(128, 64) SATD_U8_AVX2(128, 128)
    SATD_U16_AVX2(8, 4) SATD_U16_AVX2(8, 8) SATD_U16_AVX2(16, 8) SATD_U16_AVX2(16, 16)
    SATD_U16_AVX2(32, 16) SATD_U16_AVX2(32, 32) SATD_U16_AVX2(64, 32)
    SATD_U16_AVX2(64, 64) SATD_U16_AVX2(128, 64) SATD_U16_AVX2(128, 128)
});

void selectSATDFunctionAVX2(unsigned width, unsigned height, unsigned bits, SADFunction &satd) {
    if (SADFunction fn = findFunction(satd_functions, KEY(width, height, bits)))
        satd = fn;
}

#endif // MVTOOLS_X86
