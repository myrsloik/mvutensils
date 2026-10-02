// pyr_common.glsl
//
// Shared by the coarse search and seed building kernels (pyr_*.comp, median.comp, seed_*.comp):
// the frame pyramids, the coarse fields being searched, the per-level table, and the stored
// fields the seeds chain and invert. Everything follows the CPU reference's Pyramid and
// BuildSeeds (probe/seedrefine.cpp) exactly, so the seeds are the CPU's bit for bit.
//
// The coarse searches run in batches: the fields (n, d) of one frame n for several d, in the same
// dispatches, field z of the batch in gl_WorkGroupID.z (Analyse.cpp's coarse node). These kernels
// have a pipeline layout of their own (VulkanContext.h), with the reference frames' pyramids an
// array of bindings.

#include "refine_common.glsl"

// A frame's pyramid, the one mvgpu.Super attaches (SuperLayout.h): for every level L >= 1 its Y
// plane, then U, then V, each inside a border of repeated edge pixels (borderY, borderC wide), so
// a read anywhere a block may reach (Level.pad) sees what SmallPlane::At's clamping would give,
// and the words LevelSadOf reads around it stay inside. offY, offU and offV point at each plane's
// first pixel inside the border, its rows strideY or strideC apart, whole words. pyr_reduce.comp
// builds it, border included, writing through binding 16; the other kernels read the current
// frame's (16) and each field's reference frame's (17, element z) as words.
const int kMaxBatch = 16; // fields per batch: VulkanContext.h's kMaxBatch
#ifdef PYRAMID_BUILD
layout(std430, set = 0, binding = 16) buffer Pyramid { uint8_t pyr[]; };
#else
layout(std430, set = 0, binding = 16) readonly buffer CurPyramid { uint curPyrW[]; };
layout(std430, set = 0, binding = 17) readonly buffer RefPyramid { uint w[]; } refPyr[kMaxBatch];
#endif
// The coarse fields of the fields being searched, levels finest to top one after another, field
// z's at z * levels[0].fieldTotal; vectors are full pels of their level
layout(std430, set = 0, binding = 18) buffer LevelVec { ivec2 levelVec[]; };
layout(std430, set = 0, binding = 19) buffer LevelSad { int levelSad[]; };

// Per level: luma and chroma size, plane offsets in a frame's pyramid, blocks, the offset of its
// coarse field, its padding (pad >> L, at least 1), the offset of its lambda table, the planes'
// strides and borders. Entry 0 describes the frame itself, the samples of one frame's pyramid and
// the blocks of one field's coarse fields. Matches LevelEntry in SuperLayout.h.
struct Level {
    int w, h, wc, hc;
    int offY, offU, offV, nbx;
    int nby, fieldOff, pad, lambdaOff;
    int frameSamples, fieldTotal, strideY, strideC;
    int borderY, borderC, reserved0, reserved1;
};
layout(std430, set = 0, binding = 20) readonly buffer Levels { Level levels[]; };
// lambda by (worst neighbour SAD) >> 1 for each level, mvlambda * 2^level relaxed by lsad,
// computed on the CPU in double precision as the CPU reference does (Analyse.cpp)
layout(std430, set = 0, binding = 21) readonly buffer LevelLambda { int64_t levelLambda[]; };
// Medians (median.comp), kMedianSlots per field searched: slot L seeds level L, slot kGlobalSlot
// is the full-size field's global
layout(std430, set = 0, binding = 22) buffer Medians { ivec2 medians[]; };
// Fields refined earlier, the vector frames other mvgpu.Analyse nodes produced: (n, +-1) and
// (n +- 1, d -+ 1), which chain to (n, d), and (n + d, -d), which inverts to it. A record
// (x, y, SAD, 0) per block of the full-size grid, vectors in half-pels, rows pc.recStride records
// apart (RecIndex).
layout(std430, set = 0, binding = 23) readonly buffer StepRec { ivec4 stepRec[]; };
layout(std430, set = 0, binding = 24) readonly buffer RestRec { ivec4 restRec[]; };
layout(std430, set = 0, binding = 25) readonly buffer InvRec { ivec4 invRec[]; };
// Per full-size block, the block of the inverted field that lands nearest to it: SAD << 32 | index,
// the lowest winning; all ones where none lands (seed_build.comp resets what it reads)
layout(std430, set = 0, binding = 27) buffer InvKey { uint64_t invKey[]; };
// The coarse searches' results, a row per field: its global median, then the finest level's
// vectors; the field being refined starts at pc.coarseBase (seed_build.comp)
layout(std430, set = 0, binding = 28) readonly buffer Coarse { ivec2 coarse[]; };

// Where block b of the full-size grid has its record in a stored field
int RecIndex(int b) {
    return (b / pc.nbx) * pc.recStride + b % pc.nbx;
}

const int kMedianSlots = 32, kGlobalSlot = 31;

// A block and its four neighbours, in the CPU's order
const ivec2 kAround5[5] = ivec2[5](ivec2(0, 0), ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));

int gLX, gLY;       // the lane's block position at the level
int gField;         // the lane's field in the batch, z: its reference frame's pyramid,
int gFieldBase;     // its coarse fields in levelVec and levelSad,
int gMedianBase;    // and its medians

// Sets up the lane's field, field z of the batch (dynamically uniform: a workgroup works on one)
void SetupField(int z) {
    gField = z;
    gFieldBase = z * levels[0].fieldTotal;
    gMedianBase = z * kMedianSlots;
}

// The byte at index i of the pyramid being built, or of the current frame's
uint PyrByte(int i) {
#ifdef PYRAMID_BUILD
    return uint(pyr[i]);
#else
    return (curPyrW[i >> 2] >> ((i & 3) << 3)) & 255u;
#endif
}

// Keeps the reference block within the level's padding, as Pyramid::Bound
ivec2 LevelBound(Level lv, ivec2 v) {
    return ivec2(clamp(v.x, -(gLX + lv.pad), lv.w + lv.pad - 8 - gLX), clamp(v.y, -(gLY + lv.pad), lv.h + lv.pad - 8 - gLY));
}

#if !defined(PYRAMID_BUILD) && !defined(NO_BLOCK_CACHE)
// Sets up the lane's level block and copies its current pixels to shared memory in the form
// PackedStep takes, as LoadBlock does, with their sum in gSumCur. Every lane of the workgroup must
// call it, after SetupField, and then barrier().
void LoadLevelBlock(int group, int sub, int lanesPerBlock, Level lv, int bx, int by, bool live) {
    gGroup = group;
    gLX = bx * 8;
    gLY = by * 8;
    uint partial = 0u;
    if (live) {
        for (int k = sub; k < 24; k += lanesPerBlock) {
            uint b[4];
            for (int i = 0; i < 4; ++i) {
                int at;
                if (k < 16)
                    at = lv.offY + (gLY + k / 2) * lv.strideY + gLX + (k % 2) * 4 + i;
                else
                    at = (k < 20 ? lv.offU : lv.offV) + (gLY / 2 + (k - 16) % 4) * lv.strideC + gLX / 2 + i;
                b[i] = PyrByte(at);
                partial += b[i];
            }
            sCur[group * kLevelStride + 2 * k] = int(0x00FF00FFu | (b[0] << 8u) | (b[2] << 24u));
            sCur[group * kLevelStride + 2 * k + 1] = int(0x00FF00FFu | (b[1] << 8u) | (b[3] << 24u));
        }
    }
    gSumCur = lanesPerBlock == 8 ? subgroupClusteredAdd(partial, 8u) : subgroupAdd(partial);
}

// The lane's SAD of its level block for full-pel vector v: luma 8x8 plus U and V 4x4, the chroma
// vector rounded toward zero (Sim::CoarseSad). The border stands in for the CPU's clamping, so
// rows are read as words like LaneSad's, with the same packed arithmetic.
int LevelSadOf(Level lv, ivec2 v) {
    uint a = uint(lv.offY + (gLY + v.y) * lv.strideY + gLX + v.x);
    uint w = a >> 2u, shift = (a & 3u) << 3u, stride = uint(lv.strideY) >> 2u;
    int cur = gGroup * kLevelStride;
    uint sa = 0u, sm = 0u;
    for (int j0 = 0; j0 < pc.blockRows; j0 += kRowChunk) {
        [[unroll]] for (int jj = 0; jj < kRowChunk; ++jj) {
            int j = j0 + jj;
            uint wj = w + uint(j) * stride;
            uint w0 = refPyr[gField].w[wj], w1 = refPyr[gField].w[wj + 1u], w2 = refPyr[gField].w[wj + 2u];
            PackedStep(Bytes4(w0, w1, shift), cur + 4 * j, sa, sm);
            PackedStep(Bytes4(w1, w2, shift), cur + 4 * j + 2, sa, sm);
        }
    }
    int cvx = ChromaComponent(v.x), cvy = ChromaComponent(v.y);
    int rowC = (gLY / 2 + cvy) * lv.strideC + gLX / 2 + cvx;
    uint au = uint(lv.offU + rowC), av = uint(lv.offV + rowC);
    uint wu = au >> 2u, shiftU = (au & 3u) << 3u, wv = av >> 2u, shiftV = (av & 3u) << 3u, strideC = uint(lv.strideC) >> 2u;
    for (int j0 = 0; j0 < pc.blockRows / 2; j0 += kRowChunk) {
        [[unroll]] for (int jj = 0; jj < kRowChunk; ++jj) {
            int j = j0 + jj;
            uint u = wu + uint(j) * strideC, t = wv + uint(j) * strideC;
            PackedStep(Bytes4(refPyr[gField].w[u], refPyr[gField].w[u + 1u], shiftU), cur + 32 + 2 * j, sa, sm);
            PackedStep(Bytes4(refPyr[gField].w[t], refPyr[gField].w[t + 1u], shiftV), cur + 40 + 2 * j, sa, sm);
        }
    }
    return int(sa + gSumCur - 2u * sm);
}
#endif

// Predictor and relaxed lambda of level block b from its four neighbours, as Pyramid::Passes: a
// neighbour outside the grid repeats the block's own vector and doesn't count for the SAD
void LevelContext(Level lv, int b, out ivec2 p, out int64_t lambda) {
    int fo = gFieldBase + lv.fieldOff;
    int bx = b % lv.nbx, by = b / lv.nbx;
    int nbs[4] = int[4](bx > 0 ? b - 1 : -1, bx + 1 < lv.nbx ? b + 1 : -1, by > 0 ? b - lv.nbx : -1, by + 1 < lv.nby ? b + lv.nbx : -1);
    int xs[4], ys[4];
    int worst = 0;
    for (int i = 0; i < 4; ++i) {
        int k = nbs[i] >= 0 ? nbs[i] : b;
        ivec2 v = levelVec[fo + k];
        xs[i] = v.x;
        ys[i] = v.y;
        if (nbs[i] >= 0)
            worst = max(worst, levelSad[fo + k]);
    }
    Sort4(xs);
    Sort4(ys);
    p = ivec2((xs[1] + xs[2]) / 2, (ys[1] + ys[2]) / 2);
    lambda = levelLambda[lv.lambdaOff + (worst >> 1)];
}

// lround(a / b) for b > 0, halves away from zero, exactly
int RoundDiv(int a, int b) {
    return a >= 0 ? (2 * a + b) / (2 * b) : -((-2 * a + b) / (2 * b));
}
