// flow_common.glsl
//
// Shared by mvgpu.FlowInter's and mvgpu.FlowFPS's kernels (FlowInterpolate.cpp): flow_prep.comp
// counts each vector frame's blocks above thscd1 and makes the blocks' occlusion masks, and
// flow_inter.comp makes every pixel of a plane of the frame between two frames from their supers,
// the blocks' vectors and masks resized to the plane at it; and by mvgpu.Flow's and FlowBlur's
// (Flow.cpp), flow_fetch.comp and flow_blur.comp. The arithmetic is mvu's for 8-bit, 9 to 16-bit and
// float clips (FlowShared.h's FlowInter, FlowInterExtra, FlowFetch and Blend, FlowBlur.cpp's
// FlowBlur_scalar, MotionBlockPyramid.cpp's MakeVectorOcclusionMask, MakeSmallVectorMasks and
// IsSceneChange), the resize too: zimg's bilinear one in 64 x 64 tiles, as mvu's MaskResizer makes it,
// whose taps the host computes as zimg does.
//
// The kernels run in the Main layout (VulkanContext.h, FlowBinding) with push constants of their own.

#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_control_flow_attributes : require

// The vectors' units per pixel, 1, 2 or 4 (specialization constant 5), and the variant (specialization
// constant 6): chroma's subsampling in bits 0 (horizontal) and 1 (vertical), so 3 for 4:2:0 and 0
// for 4:4:4, bit 2 set for 16-bit samples and bit 3 for float ones
layout(constant_id = 5) const int kPel = 2;
layout(constant_id = 6) const int kVariant = 3;
const int kChromaLog = kVariant & 3;
const int kLogX = kChromaLog & 1, kLogY = kChromaLog >> 1;
const bool kWide = (kVariant & 4) != 0;
const bool kFloat = (kVariant & 8) != 0;
const bool kImage = kPel == 4 && kChromaLog != 0; // subsampled chroma at pel 4 is its quarter-pel image (SuperLayout.h)

// Matches FlowParams in VulkanContext.h
layout(push_constant) uniform Params {
    int nbx, nby, recStride, flags;         // the grid, vector records per row, kHave* and kBlend
    int pad, padY, padc, padcY;             // the supers' padding, luma's and chroma's
    int wp, hp, wc, hc;                     // their storage frames' strides and rows per plane
    int plane, width, height, outStride;    // the plane (0 luma, 1 chroma: U and V together), its size, the output planes' stride
    int clipStride, time256, thscd1, scdLimit; // the clip planes' stride; the time (flow_blur.comp's blur256); the scene change test
    int colOff, rowOff;                     // flow_inter.comp: the plane's resize taps in taps[]
    float occnormX, occnormY;               // flow_prep.comp: MakeVectorOcclusionMask's occnorm
    int time4096FX, time4096FY, time4096BX, time4096BY; // and its time4096 for F and B; flow_blur.comp's prec in time4096FX
    int stepY, overlapY;                    // compensate.comp: its grid's vertical step and overlap
} pc;

// flags: F and B have vectors, FF and BB have (for the extra masks), blend when F and B don't serve;
// the plane's resize goes down first (Interpolate)
const int kHaveFB = 1, kHaveExtra = 2, kBlend = 4, kVerticalFirst = 8;

// The bindings, FlowBinding in VulkanContext.h: the supers' planes of the frame before (src) and
// the frame after (ref); the vector frames, records (x, y, SAD, 0) per block, pc.recStride per row:
// F, the vectors of the frame after back toward the frame before, B from the frame before forward,
// FF of the frame before back, BB of the frame after forward; the occlusion masks, F's nbx * nby
// then B's; the scene change counts of F, B, FF, BB; the clip's plane in both frames; the output;
// the resize's taps, for each column of each plane's output, then each row (FlowInterpolate.cpp's
// ResizeTaps): the block to its left (above it) << 15 | that block's 14-bit weight, the next block
// taking the rest; and for chroma, whose U and V a lane makes together, V's output and clip planes
// (U's in the output's and the clip's bindings). The supers hold chroma's U and V interleaved, sample
// by sample (SuperLayout.h). The planes' samples are bytes, 16 bits each for 9 to 16-bit clips
// (kWide) or floats (kFloat), every plane buffer declared all three ways.
layout(std430, set = 0, binding = 0) readonly buffer SrcLuma { uint8_t srcY[]; };
layout(std430, set = 0, binding = 1) readonly buffer SrcChroma { uint8_t srcC[]; };
layout(std430, set = 0, binding = 2) readonly buffer RefLuma { uint8_t refY[]; };
layout(std430, set = 0, binding = 3) readonly buffer RefChroma { uint8_t refC[]; };
layout(std430, set = 0, binding = 0) readonly buffer SrcLuma16 { uint16_t srcY16[]; };
layout(std430, set = 0, binding = 1) readonly buffer SrcChroma16 { uint16_t srcC16[]; };
layout(std430, set = 0, binding = 2) readonly buffer RefLuma16 { uint16_t refY16[]; };
layout(std430, set = 0, binding = 3) readonly buffer RefChroma16 { uint16_t refC16[]; };
layout(std430, set = 0, binding = 0) readonly buffer SrcLumaF { float srcYF[]; };
layout(std430, set = 0, binding = 1) readonly buffer SrcChromaF { float srcCF[]; };
layout(std430, set = 0, binding = 2) readonly buffer RefLumaF { float refYF[]; };
layout(std430, set = 0, binding = 3) readonly buffer RefChromaF { float refCF[]; };
layout(std430, set = 0, binding = 4) readonly buffer VecF { ivec4 vecF[]; };
layout(std430, set = 0, binding = 5) readonly buffer VecB { ivec4 vecB[]; };
layout(std430, set = 0, binding = 6) readonly buffer VecFF { ivec4 vecFF[]; };
layout(std430, set = 0, binding = 7) readonly buffer VecBB { ivec4 vecBB[]; };
layout(std430, set = 0, binding = 8) buffer Masks { uint masks[]; };
layout(std430, set = 0, binding = 9) buffer Counts { uint counts[]; };
layout(std430, set = 0, binding = 10) readonly buffer ClipSrc { uint8_t clipSrc[]; };
layout(std430, set = 0, binding = 11) readonly buffer ClipRef { uint8_t clipRef[]; };
layout(std430, set = 0, binding = 12) writeonly buffer Out { uint8_t outPx[]; };
layout(std430, set = 0, binding = 10) readonly buffer ClipSrc16 { uint16_t clipSrc16[]; };
layout(std430, set = 0, binding = 11) readonly buffer ClipRef16 { uint16_t clipRef16[]; };
layout(std430, set = 0, binding = 12) writeonly buffer Out16 { uint16_t outPx16[]; };
layout(std430, set = 0, binding = 10) readonly buffer ClipSrcF { float clipSrcF[]; };
layout(std430, set = 0, binding = 11) readonly buffer ClipRefF { float clipRefF[]; };
layout(std430, set = 0, binding = 12) writeonly buffer OutF { float outPxF[]; };
layout(std430, set = 0, binding = 13) readonly buffer Taps { int taps[]; };
layout(std430, set = 0, binding = 14) writeonly buffer OutV { uint8_t outPxV[]; };
layout(std430, set = 0, binding = 15) readonly buffer ClipSrcV { uint8_t clipSrcV[]; };
layout(std430, set = 0, binding = 16) readonly buffer ClipRefV { uint8_t clipRefV[]; };
layout(std430, set = 0, binding = 14) writeonly buffer OutV16 { uint16_t outPxV16[]; };
layout(std430, set = 0, binding = 15) readonly buffer ClipSrcV16 { uint16_t clipSrcV16[]; };
layout(std430, set = 0, binding = 16) readonly buffer ClipRefV16 { uint16_t clipRefV16[]; };
layout(std430, set = 0, binding = 14) writeonly buffer OutVF { float outPxVF[]; };
layout(std430, set = 0, binding = 15) readonly buffer ClipSrcVF { float clipSrcVF[]; };
layout(std430, set = 0, binding = 16) readonly buffer ClipRefVF { float clipRefVF[]; };

// A sample of luma's or chroma's planes of the frame before (src) or the frame after (ref)
uint LumaPx(bool ref, uint i) {
    if (kWide)
        return ref ? uint(refY16[i]) : uint(srcY16[i]);
    return ref ? uint(refY[i]) : uint(srcY[i]);
}
uint ChromaPx(bool ref, uint i) {
    if (kWide)
        return ref ? uint(refC16[i]) : uint(srcC16[i]);
    return ref ? uint(refC[i]) : uint(srcC[i]);
}

// A sample of the clip's plane in the frame before or the frame after: luma's or U's, and V's
uint ClipPx(bool ref, uint i) {
    if (kWide)
        return ref ? uint(clipRef16[i]) : uint(clipSrc16[i]);
    return ref ? uint(clipRef[i]) : uint(clipSrc[i]);
}
uint ClipPxV(bool ref, uint i) {
    if (kWide)
        return ref ? uint(clipRefV16[i]) : uint(clipSrcV16[i]);
    return ref ? uint(clipRefV[i]) : uint(clipSrcV[i]);
}

// The output's pixel at i: luma's or U's, and V's
void Store(uint i, uint v) {
    if (kWide)
        outPx16[i] = uint16_t(v);
    else
        outPx[i] = uint8_t(v);
}
void StoreV(uint i, uint v) {
    if (kWide)
        outPxV16[i] = uint16_t(v);
    else
        outPxV[i] = uint8_t(v);
}

// The rounded average mvu.Super's quarter planes take of two samples
uint Avg(uint a, uint b) {
    return (a + b + 1u) >> 1u;
}

// Float clips: a sample of luma's or chroma's planes, and mvu.Super's float AveragePixels, (a + b) * 0.5
float LumaPxF(bool ref, uint i) {
    return ref ? refYF[i] : srcYF[i];
}
float ChromaPxF(bool ref, uint i) {
    return ref ? refCF[i] : srcCF[i];
}
float AvgF(float a, float b) {
    precise float s = a + b;
    return s * 0.5;
}

// The address of sample (Xh, Yh) of the half-pel grid of a plane with rows stride samples apart,
// its four half-pel planes rows rows each
uint HalfAddr(int Xh, int Yh, int stride, int rows) {
    return uint((Xh & 1) | ((Yh & 1) << 1)) * uint(stride * rows) + uint((Yh >> 1) * stride + (Xh >> 1));
}

// mvu's PlaneGather of luma: the sample at (X, Y), 1 / pel pixels from the plane's top left pixel, of
// the super of the frame before or the frame after (at pel 1 its full-pel plane); at pel 4 between the
// half-pel samples the rounded average of their neighbours, as mvu.Super's quarter planes hold it. The
// position is clamped to the storage, where mvu reads outside the padding.
uint Sample(bool ref, int X, int Y) {
    int stride = pc.wp, rows = pc.hp;
    X += kPel * pc.pad;
    Y += kPel * pc.padY;
    if (kPel == 1) {
        X = clamp(X, 0, stride - 1);
        Y = clamp(Y, 0, rows - 1);
        return LumaPx(ref, uint(Y * stride + X));
    }
    if (kPel == 2) {
        X = clamp(X, 0, 2 * stride - 1);
        Y = clamp(Y, 0, 2 * rows - 1);
        return LumaPx(ref, HalfAddr(X, Y, stride, rows));
    }
    X = clamp(X, 0, 4 * stride - 2);
    Y = clamp(Y, 0, 4 * rows - 2);
    int Xa = X >> 1, Xb = Xa + (X & 1), Ya = Y >> 1, Yb = Ya + (Y & 1);
    return Avg(Avg(LumaPx(ref, HalfAddr(Xa, Ya, stride, rows)), LumaPx(ref, HalfAddr(Xa, Yb, stride, rows))),
               Avg(LumaPx(ref, HalfAddr(Xb, Ya, stride, rows)), LumaPx(ref, HalfAddr(Xb, Yb, stride, rows))));
}

// And of chroma: U's and V's samples at (X, Y), which the interleaved planes hold side by side, U's
// where one plane alone would hold the sample at a at 2a; 4:4:4 at pel 4 between its half-pel samples
// as luma, subsampled chroma at pel 4 from its quarter-pel image
uvec2 SampleUV(bool ref, int X, int Y) {
    int stride = pc.wc, rows = pc.hc;
    X += kPel * pc.padc;
    Y += kPel * pc.padcY;
    uint a;
    if (kImage) {
        X = clamp(X, 0, 4 * stride - 1);
        Y = clamp(Y, 0, 4 * rows - 1);
        a = 2u * uint(Y * 4 * stride + X);
        return uvec2(ChromaPx(ref, a), ChromaPx(ref, a + 1u));
    }
    if (kPel == 1) {
        X = clamp(X, 0, stride - 1);
        Y = clamp(Y, 0, rows - 1);
        a = 2u * uint(Y * stride + X);
        return uvec2(ChromaPx(ref, a), ChromaPx(ref, a + 1u));
    }
    if (kPel == 2) {
        X = clamp(X, 0, 2 * stride - 1);
        Y = clamp(Y, 0, 2 * rows - 1);
        a = 2u * HalfAddr(X, Y, stride, rows);
        return uvec2(ChromaPx(ref, a), ChromaPx(ref, a + 1u));
    }
    X = clamp(X, 0, 4 * stride - 2);
    Y = clamp(Y, 0, 4 * rows - 2);
    int Xa = X >> 1, Xb = Xa + (X & 1), Ya = Y >> 1, Yb = Ya + (Y & 1);
    uint aa = 2u * HalfAddr(Xa, Ya, stride, rows), ab = 2u * HalfAddr(Xa, Yb, stride, rows);
    uint ba = 2u * HalfAddr(Xb, Ya, stride, rows), bb = 2u * HalfAddr(Xb, Yb, stride, rows);
    return uvec2(Avg(Avg(ChromaPx(ref, aa), ChromaPx(ref, ab)), Avg(ChromaPx(ref, ba), ChromaPx(ref, bb))),
                 Avg(Avg(ChromaPx(ref, aa + 1u), ChromaPx(ref, ab + 1u)), Avg(ChromaPx(ref, ba + 1u), ChromaPx(ref, bb + 1u))));
}

// Sample() and SampleUV(), of a float clip
float SampleF(bool ref, int X, int Y) {
    int stride = pc.wp, rows = pc.hp;
    X += kPel * pc.pad;
    Y += kPel * pc.padY;
    if (kPel == 1) {
        X = clamp(X, 0, stride - 1);
        Y = clamp(Y, 0, rows - 1);
        return LumaPxF(ref, uint(Y * stride + X));
    }
    if (kPel == 2) {
        X = clamp(X, 0, 2 * stride - 1);
        Y = clamp(Y, 0, 2 * rows - 1);
        return LumaPxF(ref, HalfAddr(X, Y, stride, rows));
    }
    X = clamp(X, 0, 4 * stride - 2);
    Y = clamp(Y, 0, 4 * rows - 2);
    int Xa = X >> 1, Xb = Xa + (X & 1), Ya = Y >> 1, Yb = Ya + (Y & 1);
    return AvgF(AvgF(LumaPxF(ref, HalfAddr(Xa, Ya, stride, rows)), LumaPxF(ref, HalfAddr(Xa, Yb, stride, rows))),
                AvgF(LumaPxF(ref, HalfAddr(Xb, Ya, stride, rows)), LumaPxF(ref, HalfAddr(Xb, Yb, stride, rows))));
}
vec2 SampleUVF(bool ref, int X, int Y) {
    int stride = pc.wc, rows = pc.hc;
    X += kPel * pc.padc;
    Y += kPel * pc.padcY;
    uint a;
    if (kImage) {
        X = clamp(X, 0, 4 * stride - 1);
        Y = clamp(Y, 0, 4 * rows - 1);
        a = 2u * uint(Y * 4 * stride + X);
        return vec2(ChromaPxF(ref, a), ChromaPxF(ref, a + 1u));
    }
    if (kPel == 1) {
        X = clamp(X, 0, stride - 1);
        Y = clamp(Y, 0, rows - 1);
        a = 2u * uint(Y * stride + X);
        return vec2(ChromaPxF(ref, a), ChromaPxF(ref, a + 1u));
    }
    if (kPel == 2) {
        X = clamp(X, 0, 2 * stride - 1);
        Y = clamp(Y, 0, 2 * rows - 1);
        a = 2u * HalfAddr(X, Y, stride, rows);
        return vec2(ChromaPxF(ref, a), ChromaPxF(ref, a + 1u));
    }
    X = clamp(X, 0, 4 * stride - 2);
    Y = clamp(Y, 0, 4 * rows - 2);
    int Xa = X >> 1, Xb = Xa + (X & 1), Ya = Y >> 1, Yb = Ya + (Y & 1);
    uint aa = 2u * HalfAddr(Xa, Ya, stride, rows), ab = 2u * HalfAddr(Xa, Yb, stride, rows);
    uint ba = 2u * HalfAddr(Xb, Ya, stride, rows), bb = 2u * HalfAddr(Xb, Yb, stride, rows);
    return vec2(AvgF(AvgF(ChromaPxF(ref, aa), ChromaPxF(ref, ab)), AvgF(ChromaPxF(ref, ba), ChromaPxF(ref, bb))),
                AvgF(AvgF(ChromaPxF(ref, aa + 1u), ChromaPxF(ref, ab + 1u)), AvgF(ChromaPxF(ref, ba + 1u), ChromaPxF(ref, bb + 1u))));
}

// The resize's blocks and weights at the plane's pixel (x, y): blocks gIa and gIb = gIa + 1 of the
// rows gJa and gJb = gJa + 1, weighted gCa and 16384 - gCa across, gDa and 16384 - gDa down
int gIa, gIb, gCa, gJa, gJb, gDa;

void Weights(int x, int y) {
    int cx = taps[pc.colOff + x], ry = taps[pc.rowOff + y];
    gIa = cx >> 15;
    gCa = cx & 0x7FFF;
    gIb = min(gIa + 1, pc.nbx - 1);
    gJa = ry >> 15;
    gDa = ry & 0x7FFF;
    gJb = min(gJa + 1, pc.nby - 1);
}

// zimg's 16-bit bilinear step, the weights adding up to 1 << 14: its result rounded and offset
// back (unpack_pixel_u16, pack_pixel_u16), which comes to this
int Lerp(int a, int b, int c) {
    return (c * a + (16384 - c) * b + 8192) >> 14;
}

// The resize of the four blocks' values at the pixel: across each row, then down, each pass rounded
// to 16 bits, as zimg resizes when it scales both ways up (resize_h_first); or down each column, then
// across, where it goes vertically first (kVerticalFirst: grids whose rows are a pixel apart in the
// plane, Flow.cpp's TilesPassOrder)
int Interpolate(int aa, int ab, int ba, int bb) {
    if ((pc.flags & kVerticalFirst) != 0)
        return Lerp(Lerp(aa, ba, gDa), Lerp(ab, bb, gDa), gCa);
    return Lerp(Lerp(aa, ab, gCa), Lerp(ba, bb, gCa), gDa);
}

// A block's vector as mvu's MakeSmallVectorMasks and AdjustSmallVectorMaskSubSampling keep it:
// biased into 16 bits, saturated, chroma's then shifted by the subsampling
ivec2 Biased(ivec4 rec) {
    ivec2 v = clamp(rec.xy + 32768, 0, 65535);
    if (pc.plane != 0)
        v = ((v - 32768) >> ivec2(kLogX, kLogY)) + 32768;
    return v;
}

// A vector frame's vector at the pixel, unbiased, in 1 / pel pixels of the plane
ivec2 VectorAt(int which) {
    ivec4 raa, rab, rba, rbb;
    int aa = gJa * pc.recStride + gIa, ab = gJa * pc.recStride + gIb, ba = gJb * pc.recStride + gIa, bb = gJb * pc.recStride + gIb;
    if (which == 0) {
        raa = vecF[aa]; rab = vecF[ab]; rba = vecF[ba]; rbb = vecF[bb];
    } else if (which == 1) {
        raa = vecB[aa]; rab = vecB[ab]; rba = vecB[ba]; rbb = vecB[bb];
    } else if (which == 2) {
        raa = vecFF[aa]; rab = vecFF[ab]; rba = vecFF[ba]; rbb = vecFF[bb];
    } else {
        raa = vecBB[aa]; rab = vecBB[ab]; rba = vecBB[ba]; rbb = vecBB[bb];
    }
    ivec2 baa = Biased(raa), bab = Biased(rab), bba = Biased(rba), bbb = Biased(rbb);
    return ivec2(Interpolate(baa.x, bab.x, bba.x, bbb.x), Interpolate(baa.y, bab.y, bba.y, bbb.y)) - 32768;
}
