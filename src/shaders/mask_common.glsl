// mask_common.glsl
//
// Shared by the kernels of mvgpu.VectorLengthMask, SADMask and OcclusionMask (Mask.cpp) and of
// mvgpu.SCDetection (SCDetection.cpp): mask_blocks.comp counts a vector frame's blocks above thscd1
// and makes every block's mask value, mask_resize.comp spreads the values over the frame as mvu's
// PlaneResizer does, with zimg's bilinear resize of the whole plane, or fills the frame with scval
// where the vectors don't serve. The values are mvu's (MotionBlockPyramid.cpp's MakeVectorLengthMask,
// MakeSADMask and MakeVectorOcclusionMask) operation for operation, but for the pow a gamma other than
// 1 (other than 1 or 2 for the vector length) takes, which is computed here (PowD), close enough to the
// C library's that its results agree but for values within some 1e-10 of a rounding boundary (on a
// device without doubles the GPU's float pow, which puts some values a step apart).
//
// The kernels run in the Main layout (VulkanContext.h, MaskBinding) with push constants of their own.

#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_control_flow_attributes : require

// The mask's samples (specialization constant 6): 0 8-bit, 1 9 to 16-bit, 2 float
layout(constant_id = 6) const int kKind = 0;

// Matches MaskParams in VulkanContext.h
layout(push_constant) uniform Params {
    int nbx, nby, recStride, kind;       // the grid, vector records per row; the mask (kLength, kSad, kOcclusion) or kCount
    int width, height, outStride, flags; // mask_resize.comp: the frame, its plane's stride; kNoVectors
    int thscd1, scdLimit, pel, maxVal;   // the scene change test; the vectors' units per pixel; the mask's peak, 1 for float
    int time4096X, time4096Y, sadShift, backward; // SADMask's and OcclusionMask's time4096; SADMask's SAD shift to 8 bits; OcclusionMask's direction
    float norm, normY, gamma, scvalF;    // the masks' norm factors (OcclusionMask's for x, then y; the vector length's 1 / pel^2 in normY); gamma; scval, float
    int scval, colOff, rowOff, countSlot; // scval, integer; mask_resize.comp: the taps of the columns and the rows in taps[]; mask_blocks.comp: the count's place in counts[]
} pc;

// pc.kind: VectorLengthMask, SADMask, OcclusionMask, or only the scene change test's count (SCDetection)
const int kLength = 0, kSad = 1, kOcclusion = 2, kCount = 3;
// pc.flags: the frame has no vectors (mask_resize.comp fills it with scval); zimg resizes the
// mask down first
const int kNoVectors = 1, kVerticalFirst = 2;

// The bindings, MaskBinding in VulkanContext.h: the vector frame, a record (x, y, SAD, 0) per block,
// pc.recStride per row; the blocks' mask values, nbx * nby, integers or float bits; the count of
// blocks above thscd1 (mask_blocks.comp's at pc.countSlot, mask_resize.comp's at 0); the output plane, bytes, 16 bits each for 9 to 16-bit masks or floats; the
// resize's taps (Mask.cpp's PlaneTaps), (left, c, f0, f1) for each column, then for each row.
layout(std430, set = 0, binding = 0) readonly buffer Vectors { ivec4 recs[]; };
layout(std430, set = 0, binding = 1) buffer Values { uint values[]; };
layout(std430, set = 0, binding = 2) buffer Counts { uint counts[]; };
layout(std430, set = 0, binding = 3) writeonly buffer Out { uint8_t outPx[]; };
layout(std430, set = 0, binding = 3) writeonly buffer Out16 { uint16_t outPx16[]; };
layout(std430, set = 0, binding = 3) writeonly buffer OutF { float outPxF[]; };
layout(std430, set = 0, binding = 4) readonly buffer Taps { ivec4 taps[]; };

// x^y for x >= 0, y > 0, as near the C library's pow and powf as the masks need: in double precision
// where the device has doubles (log2 from frexp and the atanh series of the mantissa's logarithm, 2^t
// from the Taylor series of e^(f ln 2) scaled by ldexp), good to some units in the double's last
// place, so that rounded to float (powf) or multiplied by the mask's peak and truncated (pow) it gives
// the C library's result unless that lies within some 1e-10 of a rounding boundary; else the GPU's
// float pow, a unit or two of the float's last place off.
#if MVGPU_FLOAT64
double PowD(double x, double y) {
    if (x == 0.0)
        return 0.0;
    int e;
    double m = frexp(x, e);
    if (m < 0.70710678118654752440lf) {
        m *= 2.0;
        e -= 1;
    }
    // ln m = 2 atanh(s) = s (2 + 2/3 s^2 + 2/5 s^4 + ...), |s| <= 0.1716
    const double kSeries[12] = double[](2.0lf / 23.0lf, 2.0lf / 21.0lf, 2.0lf / 19.0lf, 2.0lf / 17.0lf, 2.0lf / 15.0lf, 2.0lf / 13.0lf,
                                        2.0lf / 11.0lf, 2.0lf / 9.0lf, 2.0lf / 7.0lf, 2.0lf / 5.0lf, 2.0lf / 3.0lf, 2.0lf);
    precise double s = (m - 1.0) / (m + 1.0);
    precise double s2 = s * s;
    precise double sum = 0.0;
    [[unroll]] for (int k = 0; k < 12; ++k)
        sum = sum * s2 + kSeries[k];
    const double kLn2 = 0.69314718055994530942lf;
    precise double t = y * (double(e) + s * sum / kLn2);
    double n = floor(t + 0.5);
    precise double f = (t - n) * kLn2;
    // e^f = 1 + f (1 + f/2 (1 + f/3 (...))), |f| <= 0.347
    const double kInverse[17] = double[](1.0lf / 17.0lf, 1.0lf / 16.0lf, 1.0lf / 15.0lf, 1.0lf / 14.0lf, 1.0lf / 13.0lf, 1.0lf / 12.0lf, 1.0lf / 11.0lf,
                                         1.0lf / 10.0lf, 1.0lf / 9.0lf, 1.0lf / 8.0lf, 1.0lf / 7.0lf, 1.0lf / 6.0lf, 1.0lf / 5.0lf, 1.0lf / 4.0lf,
                                         1.0lf / 3.0lf, 1.0lf / 2.0lf, 1.0lf);
    precise double p = 1.0;
    [[unroll]] for (int k = 0; k < 17; ++k)
        p = 1.0 + p * f * kInverse[k];
    return ldexp(p, int(n));
}

float PowF(float x, float y) {
    return float(PowD(double(x), double(y)));
}
#else
float PowF(float x, float y) {
    return pow(x, y);
}
#endif
