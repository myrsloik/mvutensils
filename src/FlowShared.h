#pragma once

#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <limits>
#include "SuperPyramid.h"
#include "Common.h"
#include "CPU.h"

// FlowBlur divides by each moving pixel's small tap counts: precomputed divisors for 1..kFlowBlurTapDivisors. Like the
// prec divisor, the table is built in FlowBlur.cpp, which is compiled for the baseline ISA (see InvariantDivisor).
constexpr int kFlowBlurTapDivisors = 512;
const InvariantDivisor *FlowBlurTapDivisors() noexcept;

// n / d for d >= 1
static MVU_FORCE_INLINE int DivideByTaps(const InvariantDivisor *divisors, int n, int d) noexcept {
    return d <= kFlowBlurTapDivisors ? divisors[d].Divide(n) : n / d;
}

// The average of d summed pixels; an integer sum of at most kFlowBlurTapDivisors 16-bit pixels fits an int
template <typename T>
static MVU_FORCE_INLINE T DivideTapSum(const InvariantDivisor *divisors, T sum, int d) noexcept {
    if constexpr (std::is_integral_v<T>) {
        if (d <= kFlowBlurTapDivisors)
            return divisors[d].Divide(static_cast<int>(sum));
    }
    return sum / d;
}

#if defined(MVTOOLS_X86)
// AVX-512 gather kernels (FlowShared_AVX512.cpp). See that file for the subpel-gather addressing and
// the documented <= 3-byte tail over-read of the pPlane[0] allocation.
void FlowInter_avx512_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowInter_avx512_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowInter_avx512_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowInterExtra_avx512_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
    const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept;
void FlowInterExtra_avx512_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
    const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept;
void FlowInterExtra_avx512_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
    const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept;
void FlowFetch_avx512_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
    const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowFetch_avx512_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
    const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowFetch_avx512_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
    const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;

// FlowBlur per-pixel tap-gather (FlowShared_AVX512.cpp): the mF+mB motion-blur taps of one output pixel
// are dense-packed and gathered in up to four 8-wide masked vpgatherdd chunks, then horizontally summed.
// u8/u16 only (float keeps the scalar's in-order double accumulation). Bit-exact with FlowBlur_scalar.
// precDiv divides by prec.
void FlowBlur_avx512_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int blur256, int prec, const InvariantDivisor &precDiv) noexcept;
void FlowBlur_avx512_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int blur256, int prec, const InvariantDivisor &precDiv) noexcept;
// Float gather variant: tap values gathered as float32, summed in double (reordered vs the scalar walk).
// For normal pixel ranges the double sum is order-independent-exact, so it matches FlowBlur_scalar<float>
// bit-for-bit (measured 0 error); only extreme magnitudes could differ sub-ULP (build-variant-dependent).
void FlowBlur_avx512_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int blur256, int prec, const InvariantDivisor &precDiv) noexcept;

// AVX2 gather kernels (FlowShared_AVX2.cpp): 8 px/iter, same addressing/limits as the AVX-512 ones.
void FlowInter_avx2_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowInter_avx2_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowInter_avx2_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowInterExtra_avx2_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
    const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept;
void FlowInterExtra_avx2_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
    const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept;
void FlowInterExtra_avx2_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &prefB, const PyramidPlane &prefF,
    const uint16_t *VXFullB, const uint16_t *VXFullF, const uint16_t *VYFullB, const uint16_t *VYFullF,
    const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256,
    const uint16_t *VXFullBB, const uint16_t *VXFullFF, const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept;
void FlowFetch_avx2_u8(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
    const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowFetch_avx2_u16(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
    const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;
void FlowFetch_avx2_f32(uint8_t *pdst, ptrdiff_t dst_pitch, const PyramidPlane &pref,
    const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch, int dstX, int dstY, int width, int height, int time256) noexcept;

// Both gather paths (AVX2 vpgatherdd 8-wide, AVX-512 16-wide) compute signed int32 indices relative to
// the MIDDLE sub-plane (pPlane[nPel*nPel/2]), so the largest |index| is the bigger half of the
// allocation: (nPel*nPel - nPel*nPel/2) sub-planes. That must fit int32 or the index would wrap; we
// fall back to the (64-bit, unlimited) scalar above it. Centering ~doubles the ceiling vs a pPlane[0] base.
static MVU_FORCE_INLINE bool FlowGatherFits(const PyramidPlane &p) {
    const int n = p.nPel * p.nPel;
    return (int64_t)p.subPelPlaneOffset * (n - n / 2) <= std::numeric_limits<int32_t>::max();
}
#endif

// time-weihted blend src with ref frames (used for interpolation for poor motion estimation)
template <typename PixelType>
static void Blend(uint8_t *MVU_RESTRICT pdst, const uint8_t *MVU_RESTRICT psrc, const uint8_t *MVU_RESTRICT pref, int height, int width, ptrdiff_t stride, int time256) noexcept {
    for (int h = 0; h < height; h++) {
        // Hoist the row casts out of the inner loop; recomputing them per pixel
        // defeats autovectorization (the result is the same unit-stride row).
        const PixelType *psrc_ = (const PixelType *)psrc;
        const PixelType *pref_ = (const PixelType *)pref;
        PixelType *pdst_ = (PixelType *)pdst;

        for (int w = 0; w < width; w++)
            pdst_[w] = ShiftDivide<8, 0>(psrc_[w] * (256 - time256) + pref_[w] * time256);

        pdst += stride;
        psrc += stride;
        pref += stride;
    }
}

// Loop-level hoisting accessor mirroring the AVX-512 gather. Copying the plane fields into locals once
// (rather than re-reading them through PyramidPlane::GetPointer every pixel) is what lets the compiler
// keep them in registers and auto-vectorize the inner loops -- a branchless GetPointer alone does NOT,
// because it can't prove the dst store doesn't alias the PyramidPlane object so it reloads each call.
// Offsets are 64-bit so this stays correct at any resolution (unlike the int32 gather).
template <typename PixelType>
struct PlaneGather {
    const uint8_t *base;
    ptrdiff_t spo, pitch;
    int hpad, vpad, log, mask;
    explicit PlaneGather(const PyramidPlane &p) noexcept
        : base(p.pPlane[0]), spo(p.subPelPlaneOffset), pitch(p.nPitch),
          hpad(p.nHPaddingPel), vpad(p.nVPaddingPel), log(ilog2(p.nPel)), mask(p.nPel - 1) {}
    MVU_FORCE_INLINE PixelType operator()(int nx, int ny) const noexcept {
        int X = nx + hpad, Y = ny + vpad;
        ptrdiff_t off = (ptrdiff_t)((X & mask) | ((Y & mask) << log)) * spo
                      + (ptrdiff_t)(X >> log) * (ptrdiff_t)sizeof(PixelType)
                      + (ptrdiff_t)(Y >> log) * pitch;
        return *reinterpret_cast<const PixelType *>(base + off);
    }
};

template <typename PixelType>
static void FlowInter_scalar(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF,
        const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY,
        int width, int height,
        int time256) noexcept {

    PixelType *pdst = (PixelType *)pdst8;

    tilePitch /= sizeof(uint16_t);
    dst_pitch /= sizeof(PixelType);

    int nPelLog = ilog2(prefB.nPel);
    PlaneGather<PixelType> gF(prefF), gB(prefB);

    for (int h = 0; h < height; h++) {
        int yBase = (h + dstY) << nPelLog;
        const PixelType *prefF0Ptr = reinterpret_cast<const PixelType *>(prefF.GetPointer<PixelType>(dstX << nPelLog, yBase));
        const PixelType *prefB0Ptr = reinterpret_cast<const PixelType *>(prefB.GetPointer<PixelType>(dstX << nPelLog, yBase));
        for (int w = 0; w < width; w++) {
            int xBase = (w + dstX) << nPelLog;
            int vxF = ((static_cast<int>(VXFullF[w]) - (1 << 15)) * time256) >> 8;
            int vyF = ((static_cast<int>(VYFullF[w]) - (1 << 15)) * time256) >> 8;
            int vxB = ((static_cast<int>(VXFullB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            int vyB = ((static_cast<int>(VYFullB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            // Hoisted branchless loads (see PlaneGather) so this loop auto-vectorizes.
            // Integer path: exact original fixed-point blend ((x+256)>>8 rounding, final -1) via ShiftDivide.
            // Float path: same structure with /256 and no integer-rounding bias / -1 (ShiftDivide handles both).
            using Acc = std::conditional_t<std::is_integral_v<PixelType>, int64_t, float>;
            Acc dstF = gF(vxF + xBase, vyF + yBase);
            Acc dstB = gB(vxB + xBase, vyB + yBase);
            Acc dstF0 = prefF0Ptr[w];
            Acc dstB0 = prefB0Ptr[w];
            Acc occF = ShiftDivide<8, 256>(dstF * (256 - MaskF[w]) + ShiftDivide<8, 256>(MaskF[w] * (dstB * (256 - MaskB[w]) + MaskB[w] * dstF0)));
            Acc occB = ShiftDivide<8, 256>(dstB * (256 - MaskB[w]) + ShiftDivide<8, 256>(MaskB[w] * (dstF * (256 - MaskF[w]) + MaskF[w] * dstB0)));
            Acc blended = ShiftDivide<8, 0>(occF * (256 - time256) + occB * time256);
            if constexpr (std::is_integral_v<PixelType>)
                pdst[w] = (PixelType)(blended - 1);
            else
                pdst[w] = (PixelType)blended;
        }

        pdst += dst_pitch;
        VXFullB += tilePitch;
        VYFullB += tilePitch;
        VXFullF += tilePitch;
        VYFullF += tilePitch;
        MaskB += tilePitch;
        MaskF += tilePitch;
    }
}

template <typename PixelType>
static void FlowInterExtra_scalar(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF,
        const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY,
        int width, int height,
        int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF,
        const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {

    PixelType *pdst = (PixelType *)pdst8;

    dst_pitch /= sizeof(PixelType);
    tilePitch /= sizeof(int16_t);

    int nPelLog = ilog2(prefB.nPel);
    PlaneGather<PixelType> gF(prefF), gB(prefB);

    for (int h = 0; h < height; h++) {
        int yBase = (h + dstY) << nPelLog;
        for (int w = 0; w < width; w++) {
            int xBase = (w + dstX) << nPelLog;
            int vxF = ((static_cast<int>(VXFullF[w]) - (1 << 15)) * time256) >> 8;
            int vyF = ((static_cast<int>(VYFullF[w]) - (1 << 15)) * time256) >> 8;
            int vxFF = ((static_cast<int>(VXFullFF[w]) - (1 << 15)) * time256) >> 8;
            int vyFF = ((static_cast<int>(VYFullFF[w]) - (1 << 15)) * time256) >> 8;
            int vxB = ((static_cast<int>(VXFullB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            int vyB = ((static_cast<int>(VYFullB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            int vxBB = ((static_cast<int>(VXFullBB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            int vyBB = ((static_cast<int>(VYFullBB[w]) - (1 << 15)) * (256 - time256)) >> 8;
            // Hoisted branchless loads (see PlaneGather) so this loop auto-vectorizes.
            using Acc = std::conditional_t<std::is_integral_v<PixelType>, int, float>;
            Acc dstF = gF(vxF + xBase, vyF + yBase);
            Acc dstFF = gF(vxFF + xBase, vyFF + yBase);
            Acc dstB = gB(vxB + xBase, vyB + yBase);
            Acc dstBB = gB(vxBB + xBase, vyBB + yBase);

            /* use median, firsly get min max of compensations */
            Acc minfb = std::min(dstB, dstF);
            Acc maxfb = std::max(dstB, dstF);

            Acc medianBB = std::max(minfb, std::min(dstBB, maxfb));
            Acc medianFF = std::max(minfb, std::min(dstFF, maxfb));

            // Integer path matches the original exactly (ShiftDivide<8,256> == (x+256)>>8, then -1); float uses /256.
            Acc blended = ShiftDivide<8, 0>(ShiftDivide<8, 256>(medianBB * MaskF[w] + dstF * (256 - MaskF[w])) * (256 - time256) +
                ShiftDivide<8, 256>(medianFF * MaskB[w] + dstB * (256 - MaskB[w])) * time256);
            if constexpr (std::is_integral_v<PixelType>)
                pdst[w] = (PixelType)(blended - 1);
            else
                pdst[w] = (PixelType)blended;
        }
        pdst += dst_pitch;
        VXFullB += tilePitch;
        VYFullB += tilePitch;
        VXFullF += tilePitch;
        VYFullF += tilePitch;
        MaskB += tilePitch;
        MaskF += tilePitch;
        VXFullBB += tilePitch;
        VYFullBB += tilePitch;
        VXFullFF += tilePitch;
        VYFullFF += tilePitch;
    }
}

// Motion-compensated copy (fetch). One sub-pel load + store per pixel, no blend. Uses GetPointer (NOT
// PlaneGather): with ~no per-pixel arithmetic the loop never auto-vectorizes, so the branchless hoisted
// form only adds work -- GetPointer's pel-specialized path is faster here (measured, see flowfetch bench).
template <typename PixelType>
static void FlowFetch_scalar(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch, const PyramidPlane &pref,
        const uint16_t *MVU_RESTRICT VXFull, const uint16_t *MVU_RESTRICT VYFull, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256) noexcept {
    PixelType *pdst = (PixelType *)pdst8;
    dst_pitch /= sizeof(PixelType);
    tilePitch /= sizeof(int16_t);
    int nPelLog = ilog2(pref.nPel);

    for (int h = 0; h < height; h++) {
        int yBase = (h + dstY) << nPelLog;
        for (int w = 0; w < width; w++) {
            int xBase = (w + dstX) << nPelLog;
            int vx = ((static_cast<int>(VXFull[w]) - (1 << 15)) * time256 + 128) >> 8;
            int vy = ((static_cast<int>(VYFull[w]) - (1 << 15)) * time256 + 128) >> 8;
            pdst[w] = *reinterpret_cast<const PixelType *>(pref.GetPointer<PixelType>(xBase + vx, yBase + vy));
        }
        pdst += dst_pitch;
        VXFull += tilePitch;
        VYFull += tilePitch;
    }
}

// Dispatch wrappers: best available gather (AVX-512 > AVX2) when the offsets fit int32, else scalar.
template <typename PixelType>
static MVU_FORCE_INLINE void FlowInter(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF,
        const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256) noexcept {
#if defined(MVTOOLS_X86)
    if (FlowGatherFits(prefB) && FlowGatherFits(prefF)) {
        if (g_cpuinfo & MVU_CPU_AVX512_BASE) {
            if constexpr (sizeof(PixelType) == 1)
                FlowInter_avx512_u8(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
            else if constexpr (sizeof(PixelType) == 2)
                FlowInter_avx512_u16(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
            else
                FlowInter_avx512_f32(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
            return;
        }
        if (g_cpuinfo & MVU_CPU_AVX2) {
            if constexpr (sizeof(PixelType) == 1)
                FlowInter_avx2_u8(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
            else if constexpr (sizeof(PixelType) == 2)
                FlowInter_avx2_u16(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
            else
                FlowInter_avx2_f32(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
            return;
        }
    }
#endif
    FlowInter_scalar<PixelType>(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256);
}

template <typename PixelType>
static MVU_FORCE_INLINE void FlowInterExtra(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch,
        const PyramidPlane &prefB, const PyramidPlane &prefF,
        const uint16_t *VXFullB, const uint16_t *VXFullF,
        const uint16_t *VYFullB, const uint16_t *VYFullF,
        const uint16_t *MaskB, const uint16_t *MaskF, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256,
        const uint16_t *VXFullBB, const uint16_t *VXFullFF,
        const uint16_t *VYFullBB, const uint16_t *VYFullFF) noexcept {
#if defined(MVTOOLS_X86)
    if (FlowGatherFits(prefB) && FlowGatherFits(prefF)) {
        if (g_cpuinfo & MVU_CPU_AVX512_BASE) {
            if constexpr (sizeof(PixelType) == 1)
                FlowInterExtra_avx512_u8(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
            else if constexpr (sizeof(PixelType) == 2)
                FlowInterExtra_avx512_u16(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
            else
                FlowInterExtra_avx512_f32(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
            return;
        }
        if (g_cpuinfo & MVU_CPU_AVX2) {
            if constexpr (sizeof(PixelType) == 1)
                FlowInterExtra_avx2_u8(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
            else if constexpr (sizeof(PixelType) == 2)
                FlowInterExtra_avx2_u16(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
            else
                FlowInterExtra_avx2_f32(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
            return;
        }
    }
#endif
    FlowInterExtra_scalar<PixelType>(pdst8, dst_pitch, prefB, prefF, VXFullB, VXFullF, VYFullB, VYFullF, MaskB, MaskF, tilePitch, dstX, dstY, width, height, time256, VXFullBB, VXFullFF, VYFullBB, VYFullFF);
}

template <typename PixelType>
static MVU_FORCE_INLINE void FlowFetch(
        uint8_t *MVU_RESTRICT pdst8, ptrdiff_t dst_pitch, const PyramidPlane &pref,
        const uint16_t *VXFull, const uint16_t *VYFull, ptrdiff_t tilePitch,
        int dstX, int dstY, int width, int height, int time256) noexcept {
#if defined(MVTOOLS_X86)
    if (FlowGatherFits(pref)) {
        if (g_cpuinfo & MVU_CPU_AVX512_BASE) {
            if constexpr (sizeof(PixelType) == 1)
                FlowFetch_avx512_u8(pdst8, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
            else if constexpr (sizeof(PixelType) == 2)
                FlowFetch_avx512_u16(pdst8, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
            else
                FlowFetch_avx512_f32(pdst8, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
            return;
        }
        if (g_cpuinfo & MVU_CPU_AVX2) {
            if constexpr (sizeof(PixelType) == 1)
                FlowFetch_avx2_u8(pdst8, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
            else if constexpr (sizeof(PixelType) == 2)
                FlowFetch_avx2_u16(pdst8, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
            else
                FlowFetch_avx2_f32(pdst8, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
            return;
        }
    }
#endif
    FlowFetch_scalar<PixelType>(pdst8, dst_pitch, pref, VXFull, VYFull, tilePitch, dstX, dstY, width, height, time256);
}