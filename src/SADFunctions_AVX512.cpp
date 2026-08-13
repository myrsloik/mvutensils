#include <cstdint>

#include "FunctionTable.h"
#include "SADFunctions.h"
#include "SADFunctions_Float.h"
#include "Common.h"

#if defined(MVTOOLS_X86)

#include <immintrin.h>

// AVX-512 (x86-64-v4 / zen4) SAD + SATD. Only the block sizes where 512-bit beats the AVX2 (EVEX-256)
// kernel are registered (see bench_degrain/sad_avx512_bench.cpp, satd_avx512_bench.cpp):
//   SAD  8-bit : width >= 64 (width-32 was parity, the 2-row stitch doesn't beat AVX2's 2 accumulators)
//   SAD  16-bit: width >= 16 (width-16 stitches 2 rows per zmm; needs even height, so 16x1 stays AVX2)
//   SATD 8-bit : width >= 32   SATD 16-bit: width >= 16
// Everything else falls back to the AVX2 / SSE2 path via selectSAD/SATDFunction.

// ===================== SAD =====================
template <unsigned W, unsigned H>
struct SADWrapperU8_AVX512 {
    static_assert(W >= 64 && W % 64 == 0, "");
    static unsigned int sad_u8_avx512(const uint8_t *pSrc, [[maybe_unused]] intptr_t nSrcPitch, const uint8_t *pRef, intptr_t nRefPitch) noexcept {
        __m512i s0 = _mm512_setzero_si512(), s1 = _mm512_setzero_si512();
        for (unsigned y = 0; y < H; y++) {
            s0 = _mm512_add_epi64(s0, _mm512_sad_epu8(_mm512_loadu_si512(pSrc), _mm512_loadu_si512(pRef)));
            for (unsigned x = 64; x < W; x += 64)
                s1 = _mm512_add_epi64(s1, _mm512_sad_epu8(_mm512_loadu_si512(pSrc + x), _mm512_loadu_si512(pRef + x)));
            pSrc += W;
            pRef += nRefPitch;
        }
        return (unsigned)_mm512_reduce_add_epi64(_mm512_add_epi64(s0, s1));
    }
};


template <unsigned W, unsigned H>
struct SADWrapperU16_AVX512 {
    static_assert(W >= 16, "");
    // Deliberately keeps the unpacklo/unpackhi + two-adds widen, NOT the bias + vpmaddwd trick the SSE2/AVX2
    // SADWrapperU16 kernels use. The madd form is faster in isolation (~1.3-1.5x at width 16-64, see
    // bench_degrain/sad16_madd_bench.cpp) but a 512-bit vpmaddwd regressed the whole motion search end-to-end
    // (~-10% at 32x32 on a real bbb benchmark, on top of the 128x128 throughput regression): the surrounding
    // search pays a frequency/port cost for the heavy 512-bit multiply that the small SAD speedup doesn't cover.
    // The 256-bit AVX2 vpmaddwd has no such effect (it wins end-to-end). This path only runs on AVX-512 parts
    // WITHOUT VNNI (VNNI otherwise takes width<=64), a rare audience not worth an unstable widen -- so leave it.
    static unsigned int sad_u16_avx512(const uint8_t *pSrc8, intptr_t nSrcPitch, const uint8_t *pRef8, intptr_t nRefPitch) noexcept {
        const __m512i z = _mm512_setzero_si512();
        __m512i acc = z;
        if constexpr (W >= 32) {
            for (unsigned y = 0; y < H; y++) {
                const uint16_t *pSrc = (const uint16_t *)pSrc8, *pRef = (const uint16_t *)pRef8;
                for (unsigned x = 0; x < W; x += 32) {
                    __m512i a = _mm512_loadu_si512(pSrc + x), b = _mm512_loadu_si512(pRef + x);
                    __m512i d = _mm512_or_si512(_mm512_subs_epu16(a, b), _mm512_subs_epu16(b, a));
                    acc = _mm512_add_epi32(acc, _mm512_unpacklo_epi16(d, z));
                    acc = _mm512_add_epi32(acc, _mm512_unpackhi_epi16(d, z));
                }
                pSrc8 += nSrcPitch;
                pRef8 += nRefPitch;
            }
        } else { // W == 16: two rows per zmm (registered only for even heights)
            for (unsigned y = 0; y < H; y += 2) {
                __m512i a = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(pSrc8))), _mm256_loadu_si256((const __m256i *)(pSrc8 + nSrcPitch)), 1);
                __m512i b = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(pRef8))), _mm256_loadu_si256((const __m256i *)(pRef8 + nRefPitch)), 1);
                __m512i d = _mm512_or_si512(_mm512_subs_epu16(a, b), _mm512_subs_epu16(b, a));
                acc = _mm512_add_epi32(acc, _mm512_unpacklo_epi16(d, z));
                acc = _mm512_add_epi32(acc, _mm512_unpackhi_epi16(d, z));
                pSrc8 += 2 * nSrcPitch;
                pRef8 += 2 * nRefPitch;
            }
        }
        return (unsigned)_mm512_reduce_add_epi32(acc);
    }
};


// 16-bit SAD via AVX-512 VNNI (vpdpwssd). Full-16-bit correct via a bias trick: the abs-diff
// or(subs_epu16(a,b),subs_epu16(b,a)) is UNSIGNED [0,65535] which overflows the signed word vpdpwssd
// wants, so XOR 0x8000 maps it to signed [-32768,32767] (== ad-32768); vpdpwssd(.,+1) then fuses
// widen+pair-sum+accumulate into int32, and the compile-time constant 32768*W*H is added back at the
// end. The or+xor pair is written as one explicit vpternlog (imm 0x56 = (A|B)^C) instead of relying on
// the compiler to spot the fusion. ~1.25-1.5x over the plain kernel for widths 8..64 (see
// bench_degrain/sad16_vnni*_bench.cpp).
// 4 row-interleaved accumulators hide vpdpwssd's ~5c latency. Requires H % 4 == 0 (all registered).
// clang-cl's /arch:AVX512 is only x86-64-v4 (no VNNI), so the kernel opts into VNNI with a function
// target attribute; MSVC's /arch:AVX512 is a broad umbrella that already exposes the VNNI intrinsics
// (and rejects __attribute__), so the attribute is clang-only. Runtime-gated on MVU_CPU_AVX512VNNI.
// clang and GCC both compile this TU at x86-64-v4 (-march / clang-cl /arch:AVX512), which lacks VNNI,
// so the kernel opts VNNI in with a function target attribute. Both compilers ADD these to the v4
// baseline (which already has avx512f/bw/vl), so only the *added* features are listed -- listing
// avx512f explicitly trips a clang feature-resolution bug on the 256-bit _mm256_dpwssd path. MSVC
// cl.exe rejects __attribute__ but its /arch:AVX512 umbrella already exposes VNNI, so it's attribute-free.
// The TU is x86-64-v4 (clang-cl /arch:AVX512, GCC -march=x86-64-v4) which lacks VNNI, so opt it in per
// function. clang-cl defines __clang__ (not __GNUC__); GCC defines __GNUC__ (not __clang__). The lists
// differ by compiler: on clang, listing avx512f explicitly trips a bug that then rejects the 256-bit
// _mm256_dpwssd, so only the added feature + its 256/128 companions are listed (clang keeps avx512f
// from the baseline). GCC needs the full set spelled out (its attribute can reset the feature set).
// MSVC cl.exe (neither macro) needs nothing -- its /arch:AVX512 umbrella already exposes VNNI.
#if defined(__clang__)
#define MVU_TARGET_VNNI __attribute__((target("avx512vnni,avx512bw,avx512vl")))
#elif defined(__GNUC__)
#define MVU_TARGET_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
#else
#define MVU_TARGET_VNNI
#endif
template <unsigned W, unsigned H>
MVU_TARGET_VNNI
static unsigned int sad_u16_avx512_vnni(const uint8_t *pSrc8, intptr_t nSrcPitch, const uint8_t *pRef8, intptr_t nRefPitch) noexcept {
    static_assert(W >= 8 && H % 4 == 0, "VNNI SAD: width >= 8, processes 4 rows per iteration");
    const uint32_t correction = 32768u * W * H;
    if constexpr (W >= 32) { // 512-bit, 32 words/vector, 4 row-interleaved accumulators
        const __m512i ones = _mm512_set1_epi16(1), bias = _mm512_set1_epi16((short)0x8000);
        __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
        for (unsigned y = 0; y < H; y += 4) {
            const uint16_t *s0 = (const uint16_t *)(pSrc8), *s1 = (const uint16_t *)(pSrc8 + nSrcPitch),
                           *s2 = (const uint16_t *)(pSrc8 + 2 * nSrcPitch), *s3 = (const uint16_t *)(pSrc8 + 3 * nSrcPitch);
            const uint16_t *r0 = (const uint16_t *)(pRef8), *r1 = (const uint16_t *)(pRef8 + nRefPitch),
                           *r2 = (const uint16_t *)(pRef8 + 2 * nRefPitch), *r3 = (const uint16_t *)(pRef8 + 3 * nRefPitch);
            for (unsigned x = 0; x < W; x += 32) {
                __m512i A0 = _mm512_loadu_si512(s0 + x), B0 = _mm512_loadu_si512(r0 + x);
                a0 = _mm512_dpwssd_epi32(a0, _mm512_ternarylogic_epi64(_mm512_subs_epu16(A0, B0), _mm512_subs_epu16(B0, A0), bias, 0x56), ones);
                __m512i A1 = _mm512_loadu_si512(s1 + x), B1 = _mm512_loadu_si512(r1 + x);
                a1 = _mm512_dpwssd_epi32(a1, _mm512_ternarylogic_epi64(_mm512_subs_epu16(A1, B1), _mm512_subs_epu16(B1, A1), bias, 0x56), ones);
                __m512i A2 = _mm512_loadu_si512(s2 + x), B2 = _mm512_loadu_si512(r2 + x);
                a2 = _mm512_dpwssd_epi32(a2, _mm512_ternarylogic_epi64(_mm512_subs_epu16(A2, B2), _mm512_subs_epu16(B2, A2), bias, 0x56), ones);
                __m512i A3 = _mm512_loadu_si512(s3 + x), B3 = _mm512_loadu_si512(r3 + x);
                a3 = _mm512_dpwssd_epi32(a3, _mm512_ternarylogic_epi64(_mm512_subs_epu16(A3, B3), _mm512_subs_epu16(B3, A3), bias, 0x56), ones);
            }
            pSrc8 += 4 * nSrcPitch; pRef8 += 4 * nRefPitch;
        }
        __m512i acc = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
        return (unsigned)((uint32_t)_mm512_reduce_add_epi32(acc) + correction);
    } else if constexpr (W == 16) { // 512-bit, 2 rows per zmm, 2 accumulators
        const __m512i ones = _mm512_set1_epi16(1), bias = _mm512_set1_epi16((short)0x8000);
        __m512i a0 = _mm512_setzero_si512(), a1 = a0;
        for (unsigned y = 0; y < H; y += 4) {
            __m512i S0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(pSrc8))), _mm256_loadu_si256((const __m256i *)(pSrc8 + nSrcPitch)), 1);
            __m512i R0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(pRef8))), _mm256_loadu_si256((const __m256i *)(pRef8 + nRefPitch)), 1);
            a0 = _mm512_dpwssd_epi32(a0, _mm512_ternarylogic_epi64(_mm512_subs_epu16(S0, R0), _mm512_subs_epu16(R0, S0), bias, 0x56), ones);
            __m512i S1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(pSrc8 + 2 * nSrcPitch))), _mm256_loadu_si256((const __m256i *)(pSrc8 + 3 * nSrcPitch)), 1);
            __m512i R1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(pRef8 + 2 * nRefPitch))), _mm256_loadu_si256((const __m256i *)(pRef8 + 3 * nRefPitch)), 1);
            a1 = _mm512_dpwssd_epi32(a1, _mm512_ternarylogic_epi64(_mm512_subs_epu16(S1, R1), _mm512_subs_epu16(R1, S1), bias, 0x56), ones);
            pSrc8 += 4 * nSrcPitch; pRef8 += 4 * nRefPitch;
        }
        return (unsigned)((uint32_t)_mm512_reduce_add_epi32(_mm512_add_epi32(a0, a1)) + correction);
    } else { // W == 8 : 256-bit, 2 rows per ymm, 2 accumulators
        const __m256i ones = _mm256_set1_epi16(1), bias = _mm256_set1_epi16((short)0x8000);
        __m256i a0 = _mm256_setzero_si256(), a1 = a0;
        for (unsigned y = 0; y < H; y += 4) {
            __m256i S0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(pSrc8))), _mm_loadu_si128((const __m128i *)(pSrc8 + nSrcPitch)), 1);
            __m256i R0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(pRef8))), _mm_loadu_si128((const __m128i *)(pRef8 + nRefPitch)), 1);
            a0 = _mm256_dpwssd_epi32(a0, _mm256_ternarylogic_epi64(_mm256_subs_epu16(S0, R0), _mm256_subs_epu16(R0, S0), bias, 0x56), ones);
            __m256i S1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(pSrc8 + 2 * nSrcPitch))), _mm_loadu_si128((const __m128i *)(pSrc8 + 3 * nSrcPitch)), 1);
            __m256i R1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(pRef8 + 2 * nRefPitch))), _mm_loadu_si128((const __m128i *)(pRef8 + 3 * nRefPitch)), 1);
            a1 = _mm256_dpwssd_epi32(a1, _mm256_ternarylogic_epi64(_mm256_subs_epu16(S1, R1), _mm256_subs_epu16(R1, S1), bias, 0x56), ones);
            pSrc8 += 4 * nSrcPitch; pRef8 += 4 * nRefPitch;
        }
        __m256i acc = _mm256_add_epi32(a0, a1);
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
        return (unsigned)((uint32_t)_mm_cvtsi128_si32(s) + correction);
    }
}

// ===================== SATD (Hadamard) =====================
// 16x4 tile of 16-bit pixels = four 4x4 sub-blocks (one per zmm 128-lane); direct widening of the AVX2
// 8x4 core (the unpack-based 4x4 transpose is per-128-lane, so the ymm logic replicates to all 4 lanes).
static MVU_FORCE_INLINE __m512i satd_16x4_u16_z(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    auto dr = [&](int y) {
        __m512i a = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(s + y * sp)));
        __m512i b = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(r + y * rp)));
        return _mm512_sub_epi32(a, b);
    };
    __m512i d0 = dr(0), d1 = dr(1), d2 = dr(2), d3 = dr(3);
    __m512i t0 = _mm512_add_epi32(d0, d1), t1 = _mm512_sub_epi32(d0, d1), t2 = _mm512_add_epi32(d2, d3), t3 = _mm512_sub_epi32(d2, d3);
    __m512i h0 = _mm512_add_epi32(t0, t2), h1 = _mm512_add_epi32(t1, t3), h2 = _mm512_sub_epi32(t0, t2), h3 = _mm512_sub_epi32(t1, t3);
    __m512i p0 = _mm512_unpacklo_epi32(h0, h1), p1 = _mm512_unpacklo_epi32(h2, h3);
    __m512i p2 = _mm512_unpackhi_epi32(h0, h1), p3 = _mm512_unpackhi_epi32(h2, h3);
    __m512i T0 = _mm512_unpacklo_epi64(p0, p1), T1 = _mm512_unpackhi_epi64(p0, p1), T2 = _mm512_unpacklo_epi64(p2, p3), T3 = _mm512_unpackhi_epi64(p2, p3);
    __m512i u0 = _mm512_add_epi32(T0, T1), u1 = _mm512_sub_epi32(T0, T1), u2 = _mm512_add_epi32(T2, T3), u3 = _mm512_sub_epi32(T2, T3);
    __m512i m0 = _mm512_max_epi32(_mm512_abs_epi32(u0), _mm512_abs_epi32(u2));
    __m512i m1 = _mm512_max_epi32(_mm512_abs_epi32(u1), _mm512_abs_epi32(u3));
    return _mm512_add_epi32(m0, m1); // already halved (amax); caller must NOT >>1
}
template <unsigned W, unsigned H>
static unsigned int satd_u16_avx512(const uint8_t *src, intptr_t sp, const uint8_t *ref, intptr_t rp) noexcept {
    static_assert(W % 16 == 0 && H % 4 == 0, "");
    const __m512i z = _mm512_setzero_si512();
    __m512i acc = z; // widen to int64 (large-block overflow), matches the AVX2 path
    for (unsigned y = 0; y < H; y += 4)
        for (unsigned x = 0; x < W; x += 16) {
            __m512i a = satd_16x4_u16_z(src + y * sp + (intptr_t)x * 2, sp, ref + y * rp + (intptr_t)x * 2, rp);
            acc = _mm512_add_epi64(acc, _mm512_add_epi64(_mm512_unpacklo_epi32(a, z), _mm512_unpackhi_epi32(a, z)));
        }
    uint64_t sum = _mm512_reduce_add_epi64(acc);
    return (unsigned)(sum > 0xFFFFFFFFu ? 0xFFFFFFFFu : sum);
}

// 32x4 tile of 8-bit = two 16x4 (cols 0-15 -> zmm lanes 0,1; cols 16-31 -> lanes 2,3). The x264
// maddubsw/hmul technique with the ymm hmul/swap/evenmask patterns broadcast to both 256-halves.
static MVU_FORCE_INLINE __m512i satd_32x4_u8_z(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    const __m256i hmul256 = _mm256_setr_epi8(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1);
    const __m512i hmul = _mm512_broadcast_i64x4(hmul256);
    auto arrange = [&](const uint8_t *p) {
        __m256i m = _mm256_loadu_si256((const __m256i *)p);
        __m256i lo = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(m));
        __m256i hi = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(m, 1));
        return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    };
    auto row = [&](int y) { return _mm512_sub_epi16(_mm512_maddubs_epi16(arrange(s + y * sp), hmul), _mm512_maddubs_epi16(arrange(r + y * rp), hmul)); };
    __m512i R0 = row(0), R1 = row(1), R2 = row(2), R3 = row(3);
    __m512i t0 = _mm512_add_epi16(R0, R1), t1 = _mm512_sub_epi16(R0, R1), t2 = _mm512_add_epi16(R2, R3), t3 = _mm512_sub_epi16(R2, R3);
    __m512i V0 = _mm512_add_epi16(t0, t2), V1 = _mm512_add_epi16(t1, t3), V2 = _mm512_sub_epi16(t0, t2), V3 = _mm512_sub_epi16(t1, t3);
    const __m256i sw256 = _mm256_setr_epi8(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13, 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13);
    const __m512i sw = _mm512_broadcast_i64x4(sw256);
    auto pm = [&](__m512i v) { __m512i a = _mm512_abs_epi16(v); return _mm512_max_epi16(a, _mm512_shuffle_epi8(a, sw)); };
    __m512i m = _mm512_add_epi16(_mm512_add_epi16(pm(V0), pm(V1)), _mm512_add_epi16(pm(V2), pm(V3)));
    const __m256i ev256 = _mm256_setr_epi16(1,0,1,0,1,0,1,0, 1,0,1,0,1,0,1,0);
    const __m512i evenmask = _mm512_broadcast_i64x4(ev256);
    return _mm512_madd_epi16(m, evenmask); // 16 int32, already halved
}
template <unsigned W, unsigned H>
static unsigned int satd_u8_avx512(const uint8_t *src, intptr_t sp, const uint8_t *ref, intptr_t rp) noexcept {
    static_assert(W % 32 == 0 && H % 4 == 0, "");
    __m512i acc = _mm512_setzero_si512();
    for (unsigned y = 0; y < H; y += 4)
        for (unsigned x = 0; x < W; x += 32)
            acc = _mm512_add_epi32(acc, satd_32x4_u8_z(src + y * sp + x, sp, ref + y * rp + x, rp));
    uint64_t sum = (uint32_t)_mm512_reduce_add_epi32(acc);
    return (unsigned)(sum > 0xFFFFFFFFu ? 0xFFFFFFFFu : sum);
}

// 32-bit float SAD (hand intrinsics, scaled to 16-bit pixel range). ZMM for W>=16; W<16 falls through
// to the AVX2 (YMM) override since 512-bit doesn't help there (see bench_degrain/sad_f32_bench.cpp).
template <unsigned W, unsigned H>
static unsigned int sad_f32_avx512(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    return scale_f32_sad(sad_f32_hand_raw<W, H>(s, sp, r, rp));
}

#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8

#define SAD_U8_AVX512(width, height)  { KEY(width, height, 8), SADWrapperU8_AVX512<width, height>::sad_u8_avx512 },
#define SAD_U16_AVX512(width, height) { KEY(width, height, 16), SADWrapperU16_AVX512<width, height>::sad_u16_avx512 },
#define SAD_F32_AVX512(width, height) { KEY(width, height, 32), sad_f32_avx512<width, height> },

static constexpr auto sad_functions = std::to_array<FunctionTableEntry<SADFunction>>({
    // 8-bit: width >= 64 only
    SAD_U8_AVX512(64, 16) SAD_U8_AVX512(64, 32) SAD_U8_AVX512(64, 64) SAD_U8_AVX512(64, 128)
    SAD_U8_AVX512(128, 32) SAD_U8_AVX512(128, 64) SAD_U8_AVX512(128, 128)
    // 16-bit: width >= 16 (16x1 excluded -- odd height, the 2-row stitch needs even height)
    SAD_U16_AVX512(16, 2) SAD_U16_AVX512(16, 4) SAD_U16_AVX512(16, 8) SAD_U16_AVX512(16, 16) SAD_U16_AVX512(16, 32)
    SAD_U16_AVX512(32, 8) SAD_U16_AVX512(32, 16) SAD_U16_AVX512(32, 32) SAD_U16_AVX512(32, 64)
    SAD_U16_AVX512(64, 16) SAD_U16_AVX512(64, 32) SAD_U16_AVX512(64, 64) SAD_U16_AVX512(64, 128)
    SAD_U16_AVX512(128, 32) SAD_U16_AVX512(128, 64) SAD_U16_AVX512(128, 128)
    // 32-bit float: width >= 16 (ZMM); smaller widths use the AVX2 YMM kernel
    SAD_F32_AVX512(16, 1) SAD_F32_AVX512(16, 2) SAD_F32_AVX512(16, 4) SAD_F32_AVX512(16, 8) SAD_F32_AVX512(16, 16) SAD_F32_AVX512(16, 32)
    SAD_F32_AVX512(32, 8) SAD_F32_AVX512(32, 16) SAD_F32_AVX512(32, 32) SAD_F32_AVX512(32, 64)
    SAD_F32_AVX512(64, 16) SAD_F32_AVX512(64, 32) SAD_F32_AVX512(64, 64) SAD_F32_AVX512(64, 128)
    SAD_F32_AVX512(128, 32) SAD_F32_AVX512(128, 64) SAD_F32_AVX512(128, 128)
});

#define SATD_U8_AVX512(width, height)  { KEY(width, height, 8), satd_u8_avx512<width, height> },
#define SATD_U16_AVX512(width, height) { KEY(width, height, 16), satd_u16_avx512<width, height> },

static constexpr auto satd_functions = std::to_array<FunctionTableEntry<SADFunction>>({
    // 8-bit: width >= 32
    SATD_U8_AVX512(32, 16) SATD_U8_AVX512(32, 32) SATD_U8_AVX512(64, 32)
    SATD_U8_AVX512(64, 64) SATD_U8_AVX512(128, 64) SATD_U8_AVX512(128, 128)
    // 16-bit: width >= 16
    SATD_U16_AVX512(16, 8) SATD_U16_AVX512(16, 16) SATD_U16_AVX512(32, 16) SATD_U16_AVX512(32, 32)
    SATD_U16_AVX512(64, 32) SATD_U16_AVX512(64, 64) SATD_U16_AVX512(128, 64) SATD_U16_AVX512(128, 128)
});

void selectSADFunctionAVX512(unsigned width, unsigned height, unsigned bits, SADFunction &sad) {
    if (SADFunction fn = findFunction(sad_functions, KEY(width, height, bits)))
        sad = fn;
}

// 16-bit VNNI SAD: only the sizes where the vpdpwssd bias kernel beats the plain AVX-512 kernel
// (widths 8-64, height >= 8; 4x4/8x4 were ~parity, 128-wide regressed -- bench_degrain/sad16_vnni*).
#define SAD_U16_AVX512_VNNI(width, height) { KEY(width, height, 16), sad_u16_avx512_vnni<width, height> },
static constexpr auto sad_vnni_functions = std::to_array<FunctionTableEntry<SADFunction>>({
    SAD_U16_AVX512_VNNI(8, 8)  SAD_U16_AVX512_VNNI(8, 16)  SAD_U16_AVX512_VNNI(8, 32)
    SAD_U16_AVX512_VNNI(16, 8) SAD_U16_AVX512_VNNI(16, 16) SAD_U16_AVX512_VNNI(16, 32)
    SAD_U16_AVX512_VNNI(32, 8) SAD_U16_AVX512_VNNI(32, 16) SAD_U16_AVX512_VNNI(32, 32) SAD_U16_AVX512_VNNI(32, 64)
    SAD_U16_AVX512_VNNI(64, 16) SAD_U16_AVX512_VNNI(64, 32) SAD_U16_AVX512_VNNI(64, 64) SAD_U16_AVX512_VNNI(64, 128)
});

void selectSADFunctionAVX512VNNI(unsigned width, unsigned height, unsigned bits, SADFunction &sad) {
    if (bits != 16)
        return;
    if (SADFunction fn = findFunction(sad_vnni_functions, KEY(width, height, bits)))
        sad = fn;
}

void selectSATDFunctionAVX512(unsigned width, unsigned height, unsigned bits, SADFunction &satd) {
    if (SADFunction fn = findFunction(satd_functions, KEY(width, height, bits)))
        satd = fn;
}

#endif // MVTOOLS_X86
