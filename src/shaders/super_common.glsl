// super_common.glsl
//
// Shared by mvgpu.Super's kernels (Super.cpp): super.comp builds a frame's level-0 planes,
// super_qpel.comp chroma's pel 4 quarter samples and pyr_reduce.comp the coarse levels. Samples
// are the clip's, 8-bit, 9 to 16-bit or float (specialization constant 6, kKind), every buffer read
// and written through the view of that kind. Integer samples are computed as int, float ones as
// float, both in exactly mvu.Super's operations (SuperPyramid.cpp); the float ones are precise, so
// the compiler neither fuses nor reorders them, and divide only by powers of two, as products.
//
// The kernels run in the Main layout (VulkanContext.h) with push constants of their own.

#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_control_flow_attributes : require

layout(constant_id = 5) const int kPel = 2;
layout(constant_id = 6) const int kKind = 0; // 0 8-bit, 1 16-bit, 2 float: SuperFormat::Kind
const bool kFloat = kKind == 2;

// Matches SuperParams in VulkanContext.h
layout(push_constant) uniform Params {
    int w, h, aw, ah;             // the frame, and the block-aligned frame the planes cover
    int padX, padY, padcX, padcY; // the planes' padding, luma's and chroma's
    int wp, hp, wc, hc;           // the storage frames' strides, samples, and each sub-pel plane's rows
    int lumaPlanes, chromaPlanes; // sub-pel planes stored per plane
    int xr, yr;                   // chroma's subsampling
    int planes;                   // 1 for Gray, 3 for YUV
    int step;                     // super.comp's step
    int sharp;                    // 0 bilinear, 1 bicubic, 2 Wiener
    int pixelMax;                 // integer samples' largest value
    int srcStrideY, srcStrideC;   // the frame's strides, samples
    int pelStrideY, pelStrideC;   // the pelclip's strides, samples
    int level;                    // pyr_reduce.comp: the level it makes
    int rfilter;                  // pyr_reduce.comp: 0 simple, 1 bilinear, 2 cubic
    int pel;                      // the super's pel
    int quad;                     // pyr_reduce.comp: four pixels per lane, else one
} pc;

// The buffers, numbered for Get and Put: the storage frames (SuperLayout.h), luma's sub-pel
// planes and U's then V's; the frame's planes; the pelclip's planes; the pyramid
const int kLuma = 0, kChroma = 1, kRawY = 2, kRawU = 3, kRawV = 4, kPelY = 5, kPelU = 6, kPelV = 7, kPyramid = 8;

layout(std430, set = 0, binding = 0) buffer LumaB { uint8_t lumaB[]; };
layout(std430, set = 0, binding = 0) buffer LumaH { uint16_t lumaH[]; };
layout(std430, set = 0, binding = 0) buffer LumaF { float lumaF[]; };
layout(std430, set = 0, binding = 2) buffer ChromaB { uint8_t chromaB[]; };
layout(std430, set = 0, binding = 2) buffer ChromaH { uint16_t chromaH[]; };
layout(std430, set = 0, binding = 2) buffer ChromaF { float chromaF[]; };
layout(std430, set = 0, binding = 29) readonly buffer RawYB { uint8_t rawYB[]; };
layout(std430, set = 0, binding = 29) readonly buffer RawYH { uint16_t rawYH[]; };
layout(std430, set = 0, binding = 29) readonly buffer RawYF { float rawYF[]; };
layout(std430, set = 0, binding = 30) readonly buffer RawUB { uint8_t rawUB[]; };
layout(std430, set = 0, binding = 30) readonly buffer RawUH { uint16_t rawUH[]; };
layout(std430, set = 0, binding = 30) readonly buffer RawUF { float rawUF[]; };
layout(std430, set = 0, binding = 31) readonly buffer RawVB { uint8_t rawVB[]; };
layout(std430, set = 0, binding = 31) readonly buffer RawVH { uint16_t rawVH[]; };
layout(std430, set = 0, binding = 31) readonly buffer RawVF { float rawVF[]; };
layout(std430, set = 0, binding = 4) readonly buffer PelYB { uint8_t pelYB[]; };
layout(std430, set = 0, binding = 4) readonly buffer PelYH { uint16_t pelYH[]; };
layout(std430, set = 0, binding = 4) readonly buffer PelYF { float pelYF[]; };
layout(std430, set = 0, binding = 5) readonly buffer PelUB { uint8_t pelUB[]; };
layout(std430, set = 0, binding = 5) readonly buffer PelUH { uint16_t pelUH[]; };
layout(std430, set = 0, binding = 5) readonly buffer PelUF { float pelUF[]; };
layout(std430, set = 0, binding = 6) readonly buffer PelVB { uint8_t pelVB[]; };
layout(std430, set = 0, binding = 6) readonly buffer PelVH { uint16_t pelVH[]; };
layout(std430, set = 0, binding = 6) readonly buffer PelVF { float pelVF[]; };
layout(std430, set = 0, binding = 16) buffer PyramidB { uint8_t pyrB[]; };
layout(std430, set = 0, binding = 16) buffer PyramidH { uint16_t pyrH[]; };
layout(std430, set = 0, binding = 16) buffer PyramidF { float pyrF[]; };

// The storage frames four samples at a time, group g holding samples 4g .. 4g + 3: 8-bit as a
// word, 16-bit as two, float as a vec4. Their rows and planes start on such groups.
layout(std430, set = 0, binding = 0) buffer LumaW { uint lumaW[]; };
layout(std430, set = 0, binding = 0) buffer LumaD { uvec2 lumaD[]; };
layout(std430, set = 0, binding = 0) buffer LumaQ { vec4 lumaQ[]; };
layout(std430, set = 0, binding = 2) buffer ChromaW { uint chromaW[]; };
layout(std430, set = 0, binding = 2) buffer ChromaD { uvec2 chromaD[]; };
layout(std430, set = 0, binding = 2) buffer ChromaQ { vec4 chromaQ[]; };

// (Branches, not ?:, which may compile to a select that makes both loads.)
ivec4 Get4I(int buf, uint g) {
    if (kKind == 0) {
        uint w;
        if (buf == kLuma)
            w = lumaW[g];
        else
            w = chromaW[g];
        return ivec4(w & 0xFFu, (w >> 8u) & 0xFFu, (w >> 16u) & 0xFFu, w >> 24u);
    }
    uvec2 d;
    if (buf == kLuma)
        d = lumaD[g];
    else
        d = chromaD[g];
    return ivec4(d.x & 0xFFFFu, d.x >> 16u, d.y & 0xFFFFu, d.y >> 16u);
}
vec4 Get4F(int buf, uint g) {
    if (buf == kLuma)
        return lumaQ[g];
    return chromaQ[g];
}
void Put4I(int buf, uint g, ivec4 v) {
    if (kKind == 0) {
        uint w = uint(v.x) | (uint(v.y) << 8u) | (uint(v.z) << 16u) | (uint(v.w) << 24u);
        if (buf == kLuma)
            lumaW[g] = w;
        else
            chromaW[g] = w;
    } else {
        uvec2 d = uvec2(uint(v.x) | (uint(v.y) << 16u), uint(v.z) | (uint(v.w) << 16u));
        if (buf == kLuma)
            lumaD[g] = d;
        else
            chromaD[g] = d;
    }
}
void Put4F(int buf, uint g, vec4 v) {
    if (buf == kLuma)
        lumaQ[g] = v;
    else
        chromaQ[g] = v;
}

// Sample i of buffer buf, integer kinds
int GetI(int buf, uint i) {
    if (kKind == 0) {
        switch (buf) {
        case kLuma: return int(lumaB[i]);
        case kChroma: return int(chromaB[i]);
        case kRawY: return int(rawYB[i]);
        case kRawU: return int(rawUB[i]);
        case kRawV: return int(rawVB[i]);
        case kPelY: return int(pelYB[i]);
        case kPelU: return int(pelUB[i]);
        case kPelV: return int(pelVB[i]);
        default: return int(pyrB[i]);
        }
    }
    switch (buf) {
    case kLuma: return int(lumaH[i]);
    case kChroma: return int(chromaH[i]);
    case kRawY: return int(rawYH[i]);
    case kRawU: return int(rawUH[i]);
    case kRawV: return int(rawVH[i]);
    case kPelY: return int(pelYH[i]);
    case kPelU: return int(pelUH[i]);
    case kPelV: return int(pelVH[i]);
    default: return int(pyrH[i]);
    }
}

// And float
float GetF(int buf, uint i) {
    switch (buf) {
    case kLuma: return lumaF[i];
    case kChroma: return chromaF[i];
    case kRawY: return rawYF[i];
    case kRawU: return rawUF[i];
    case kRawV: return rawVF[i];
    case kPelY: return pelYF[i];
    case kPelU: return pelUF[i];
    case kPelV: return pelVF[i];
    default: return pyrF[i];
    }
}

// Stores into the storage frames and the pyramid
void PutI(int buf, uint i, int v) {
    if (kKind == 0) {
        if (buf == kLuma)
            lumaB[i] = uint8_t(v);
        else if (buf == kChroma)
            chromaB[i] = uint8_t(v);
        else
            pyrB[i] = uint8_t(v);
    } else {
        if (buf == kLuma)
            lumaH[i] = uint16_t(v);
        else if (buf == kChroma)
            chromaH[i] = uint16_t(v);
        else
            pyrH[i] = uint16_t(v);
    }
}

void PutF(int buf, uint i, float v) {
    if (buf == kLuma)
        lumaF[i] = v;
    else if (buf == kChroma)
        chromaF[i] = v;
    else
        pyrF[i] = v;
}

// mvu's AveragePixels, two and four samples
int AvgI(int a, int b) {
    return (a + b + 1) >> 1;
}
float AvgF(float a, float b) {
    precise float s = a + b;
    return s * 0.5;
}
int Avg4I(int a, int b, int c, int d) {
    return (a + b + c + d + 2) >> 2;
}
float Avg4F(float a, float b, float c, float d) {
    precise float s = ((a + b) + c) + d;
    return s * 0.25;
}

// How the horizontal filter makes sample x of a row W wide, and the vertical one row y of H
// (HorizontalBilinear/Bicubic/Wiener, VerticalBilinear/Bicubic/Wiener): 0 the sample copied (the
// last), 1 the average of it and the next (near the edges; for bilinear everywhere), 2 the filter;
// where the ranges overlap on a narrow plane the later one wins, as there
int Mode(int x, int W) {
    if (x >= W - 1)
        return 0;
    if (pc.sharp == 0)
        return 1;
    if (pc.sharp == 1)
        return x >= W - 3 || x < 1 ? 1 : 2;
    return x >= W - 4 || x < 2 ? 1 : 2;
}

// The Wiener filter's tap (1, -5, 20, 20, -5, 1) / 32 across m0 .. m5 (VerticalWiener, HorizontalWiener)
int WienerI(int m0, int m1, int m2, int m3, int m4, int m5) {
    int m = ((m2 + m3) * 4 - (m1 + m4)) * 5 + m0 + m5;
    return clamp((m + 16) >> 5, 0, pc.pixelMax);
}
float WienerF(float m0, float m1, float m2, float m3, float m4, float m5) {
    precise float t = (m2 + m3) * 4.0;
    t = t - (m1 + m4);
    t = t * 5.0;
    precise float s = m0 + (m5 + t);
    return s * (1.0 / 32.0);
}

// The bicubic (Catmull-Rom) tap (-1, 9, 9, -1) / 16 across a .. d (VerticalBicubic: -a - d, the
// same as HorizontalBicubic's -(a + d) in float)
int BicubicI(int a, int b, int c, int d) {
    return clamp((-a - d + (b + c) * 9 + 8) >> 4, 0, pc.pixelMax);
}
float BicubicF(float a, float b, float c, float d) {
    precise float s = -(a + d) + (b + c) * 9.0;
    return s * (1.0 / 16.0);
}
