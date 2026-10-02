// flow_common.glsl
//
// Shared by mvgpu.FlowInter's and mvgpu.FlowFPS's kernels (FlowInterpolate.cpp): flow_prep.comp
// counts each vector frame's blocks above thscd1 and makes the blocks' occlusion masks, and
// flow_inter.comp makes every pixel of a plane of the frame between two frames from their supers,
// the blocks' vectors and masks resized to the plane at it. The arithmetic is mvu's for 8-bit clips
// (FlowShared.h's FlowInter, FlowInterExtra and Blend, MotionBlockPyramid.cpp's
// MakeVectorOcclusionMask, MakeSmallVectorMasks and IsSceneChange), the resize too: zimg's bilinear
// one in 64 x 64 tiles, as mvu's MaskResizer makes it, whose taps the host computes as zimg does.
//
// The kernels run in the Main layout (VulkanContext.h, FlowBinding) with push constants of their own.

#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : require

// The vectors' units per pixel, 2 or 4 (specialization constant 5), and chroma's subsampling
// (specialization constant 6): bit 0 horizontal, bit 1 vertical, so 3 for 4:2:0 and 0 for 4:4:4
layout(constant_id = 5) const int kPel = 2;
layout(constant_id = 6) const int kChromaLog = 3;
const int kLogX = kChromaLog & 1, kLogY = kChromaLog >> 1;
const bool kImage = kPel == 4 && kChromaLog != 0; // subsampled chroma at pel 4 is its quarter-pel image (SuperLayout.h)
const int kChromaPlanes = kImage ? 16 : 4;        // U's half-pel planes (or image), then V's as many plane sizes on

// Matches FlowParams in VulkanContext.h
layout(push_constant) uniform Params {
    int nbx, nby, recStride, flags;         // the grid, vector records per row, kHave* and kBlend
    int pad, padY, padc, padcY;             // the supers' padding, luma's and chroma's
    int wp, hp, wc, hc;                     // their storage frames' strides and rows per plane
    int plane, width, height, outStride;    // flow_inter.comp: the plane, its size, the output plane's stride
    int clipStride, time256, thscd1, scdLimit; // the clip planes' stride; the time; the scene change test
    int colOff, rowOff;                     // flow_inter.comp: the plane's resize taps in taps[]
    float occnormX, occnormY;               // flow_prep.comp: MakeVectorOcclusionMask's occnorm
    int time4096FX, time4096FY, time4096BX, time4096BY; // and its time4096 for F and B
} pc;

// flags: F and B have vectors, FF and BB have (for the extra masks), blend when F and B don't serve
const int kHaveFB = 1, kHaveExtra = 2, kBlend = 4;

// The bindings, FlowBinding in VulkanContext.h: the supers' planes of the frame before (src) and
// the frame after (ref); the vector frames, records (x, y, SAD, 0) per block, pc.recStride per row:
// F, the vectors of the frame after back toward the frame before, B from the frame before forward,
// FF of the frame before back, BB of the frame after forward; the occlusion masks, F's nbx * nby
// then B's; the scene change counts of F, B, FF, BB; the clip's plane in both frames; the output;
// the resize's taps, for each column of each plane's output, then each row (FlowInterpolate.cpp's
// ResizeTaps): the block to its left (above it) << 15 | that block's 14-bit weight, the next block
// taking the rest
layout(std430, set = 0, binding = 0) readonly buffer SrcLuma { uint8_t srcY[]; };
layout(std430, set = 0, binding = 1) readonly buffer SrcChroma { uint8_t srcC[]; };
layout(std430, set = 0, binding = 2) readonly buffer RefLuma { uint8_t refY[]; };
layout(std430, set = 0, binding = 3) readonly buffer RefChroma { uint8_t refC[]; };
layout(std430, set = 0, binding = 4) readonly buffer VecF { ivec4 vecF[]; };
layout(std430, set = 0, binding = 5) readonly buffer VecB { ivec4 vecB[]; };
layout(std430, set = 0, binding = 6) readonly buffer VecFF { ivec4 vecFF[]; };
layout(std430, set = 0, binding = 7) readonly buffer VecBB { ivec4 vecBB[]; };
layout(std430, set = 0, binding = 8) buffer Masks { uint masks[]; };
layout(std430, set = 0, binding = 9) buffer Counts { uint counts[]; };
layout(std430, set = 0, binding = 10) readonly buffer ClipSrc { uint8_t clipSrc[]; };
layout(std430, set = 0, binding = 11) readonly buffer ClipRef { uint8_t clipRef[]; };
layout(std430, set = 0, binding = 12) writeonly buffer Out { uint8_t outPx[]; };
layout(std430, set = 0, binding = 13) readonly buffer Taps { int taps[]; };
