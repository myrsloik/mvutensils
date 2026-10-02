// degrain_common.glsl
//
// Shared by the Degrain kernels (Degrain.cpp): degrain_count.comp counts each reference's blocks
// above thscd1 for the scene change test, degrain_weights.comp turns the vectors' SADs into each
// block's weights for luma and chroma, and degrain.comp blends every pixel from its blocks' motion
// compensated references and their overlap windows. The arithmetic is mvu.Degrain's for 8-bit
// clips, its double precision weights reproduced exactly in integers, so the result is
// mvu.Degrain's bit for bit given the same super and vectors.
//
// The bindings are one descriptor set per frame (Degrain.cpp), not pushed: every reference has its
// super's planes and its vectors, 2 * radius of each, up to kMaxRefs, more than a push descriptor
// set is guaranteed to hold. A kernel indexes them with a loop counter, the same in every lane.

#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require

const int kMaxRefs = 50; // 2 * the largest radius, VulkanContext.h's kMaxDegrainRefs

// The centre frame's super planes (SuperLayout.h): luma's four half-pel planes, the full-pel one
// first, and U's planes (at pel 4 its quarter-pel image), then V's
layout(std430, set = 0, binding = 0) readonly buffer CurLuma { uint8_t curY[]; };
layout(std430, set = 0, binding = 1) readonly buffer CurChroma { uint8_t curC[]; };
// Per block, structure of arrays, nbx * nby entries each: [0] the centre's weight, [1 + r]
// reference r's, luma in the low 16 bits and chroma in the high ones (degrain_weights.comp)
layout(std430, set = 0, binding = 2) buffer Meta { uint meta[]; };
// The tables Degrain.cpp uploads: the overlap windows of luma's and chroma's block size, each
// reference's thsad for luma and for chroma, and the user weights
layout(std430, set = 0, binding = 3) readonly buffer Tables { int tables[]; };
// Per reference, its blocks with a SAD above thscd1 (degrain_count.comp)
layout(std430, set = 0, binding = 4) buffer Counts { uint counts[]; };
// The output frame's planes; a plane that isn't processed is a dummy, never written
layout(std430, set = 0, binding = 5) writeonly buffer Out { uint8_t px[]; } outPlane[3];
// Each reference's super planes, laid out as the centre's, and its vector frame: a record
// (x, y, SAD, 0) per block, rows pc.recStride records apart (SuperLayout.h's ExportAnalysis). A
// reference without vectors, or out of the clip, is a dummy, never read.
layout(std430, set = 0, binding = 6) readonly buffer RefLuma { uint8_t px[]; } refY[kMaxRefs];
layout(std430, set = 0, binding = 7) readonly buffer RefChroma { uint8_t px[]; } refC[kMaxRefs];
layout(std430, set = 0, binding = 8) readonly buffer RefVec { ivec4 rec[]; } refVec[kMaxRefs];

// Matches DegrainParams in VulkanContext.h
layout(push_constant) uniform Params {
    int nbx, nby, step, overlap; // the luma grid: blocks kBlk wide, step apart, overlapping by overlap
    int pad, padc, wp, hp;       // the super's luma padding, its luma storage stride and rows
    int wc, hc;                  // and its chroma storage stride and rows
    int recStride;               // vector records per row
    int refs;                    // 2 * radius
    uint usable0, usable1;       // references with vectors and in the clip, bits 0-31 and 32-49
    int thscd1;                  // scene change: SAD above which a block counts as changed, scaled
    int scdLimit;                // and the changed blocks a reference may have
    int plane;                   // degrain.comp: the plane, 0 luma, 1 U, 2 V
    int width, height;           // its size
    int outStride;               // its stride in the output frame
    int limit;                   // the largest change of a pixel, -1 for none
    int winOff;                  // tables: the plane's 9 overlap windows
    int thOff;                   // each reference's thsad, luma's refs, then chroma's
    int uwOff;                   // the user weights: the centre's, then each reference's
    int nb;                      // blocks, nbx * nby
    int reserved0, reserved1, reserved2;
} pc;

// The grid's luma block size, 8, 16 or 32 (specialization constant 3), the vectors' units per pixel,
// 2 or 4 (specialization constant 5), as the other kernels have them, and chroma's subsampling
// (specialization constant 6): bit 0 horizontal, bit 1 vertical, so 3 for 4:2:0 and 0 for 4:4:4
layout(constant_id = 3) const int kBlk = 8;
layout(constant_id = 5) const int kPel = 2;
layout(constant_id = 6) const int kChromaLog = 3;
const int kLogX = kChromaLog & 1, kLogY = kChromaLog >> 1;
const bool kImage = kPel == 4 && kChromaLog != 0; // subsampled chroma at pel 4 is its quarter-pel image (SuperLayout.h)
const int kChromaPlanes = kImage ? 16 : 4;        // U's half-pel planes (or image), then V's as many plane sizes on

bool Usable(int r) {
    return (((r < 32 ? pc.usable0 >> uint(r) : pc.usable1 >> uint(r - 32)) & 1u) != 0u);
}

// The record of block (bx, by) in reference r's vector frame
ivec4 Record(int r, int bx, int by) {
    return refVec[r].rec[by * pc.recStride + bx];
}
