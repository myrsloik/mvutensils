// refine_common.glsl
//
// Shared by the refinement kernels refine_*.comp (Analyse.cpp), and through pyr_common.glsl by
// the coarse search and seed building kernels. In the seed, pass, half-pel and quarter-pel kernels
// a workgroup of 64 lanes works on 8 blocks, 8 lanes per block, each lane measuring one candidate
// vector of its block, or on 4 blocks of large blocks, 16 lanes per block, two measuring each
// candidate (kSplit); in the fallback a workgroup of one subgroup works on one block and spreads
// the positions it searches over its lanes. A block's group picks its best candidate with one
// clustered minimum (CandidateKey). Planes, SAD, clamp, predictor and cost follow the CPU reference
// (test/reference/reference.cpp) exactly, so the result is bit-identical to it, whatever the
// subgroup size. The SAD reads whole words and works on a word's samples at a time, 4 bytes or 2
// 16-bit samples (LaneSad).
//
// VulkanContext.cpp expands the #include lines before compiling, since the core's compiler has
// no include handler.

#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_EXT_shader_subgroup_extended_types_int64 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_integer_dot_product : require

// The subgroup size the pipelines are created with, 32 or 64 (VulkanContext.cpp sets it). Kernels that
// give each block 8 lanes run workgroups of 64 lanes, one or two subgroups; kernels that spread
// one block over a subgroup run workgroups of kSubgroup lanes, specialization constant 1 sizing
// them. The search order lives in the keys, so the subgroup size never changes a result.
layout(constant_id = 0) const int kSubgroup = 64;

// The lane's index in its workgroup, counted through its subgroup, so that the 8 lanes of a block
// are 8 consecutive lanes of one subgroup
uint WorkgroupLane() {
    return gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
}

// The super's level-0 planes of the current frame n and the reference frame n + d (the storage
// frames mvgpu.Super attaches, SuperLayout.h): four planes of hp rows for luma (full, x + 1/2,
// y + 1/2, both), and for U and V four each of hc rows (U at planes 0-3, V at 4-7), but for
// subsampled chroma at pel 4, an image each of the quarter-pel grid, 4 hc rows of 4 wc samples. The
// samples are bytes, 16 bits each at 9 to 16 bits (kWide) or floats (kFloatS, one to a word, which
// LaneSadF reads as they are), the current frame's planes declared all three ways (CurLuma,
// CurChroma, CurF). wp and wc are the storage frames' strides in samples, whole words, so every row
// starts on a word. LaneSad reads the reference frame's planes as 32-bit words (RefYWord, RefCWord), up to
// two past a row's last pixel, so the storage frames carry a spare row at the end.
layout(std430, set = 0, binding = 0) readonly buffer CurLuma8 { uint8_t curY[]; };
layout(std430, set = 0, binding = 0) readonly buffer CurLuma16 { uint16_t curY16[]; };
layout(std430, set = 0, binding = 0) readonly buffer CurLuma32 { uint curY32[]; };
layout(std430, set = 0, binding = 1) readonly buffer RefLuma { uint refY[]; };
layout(std430, set = 0, binding = 2) readonly buffer CurChroma8 { uint8_t curC[]; };
layout(std430, set = 0, binding = 2) readonly buffer CurChroma16 { uint16_t curC16[]; };
layout(std430, set = 0, binding = 2) readonly buffer CurChroma32 { uint curC32[]; };
layout(std430, set = 0, binding = 3) readonly buffer RefChroma { uint refC[]; };
// Up to maxSeeds seed vectors per block (half-pels), their count, and their SADs once measured;
// seed_build.comp writes the seeds, the other kernels only read them. A SAD is unsigned, as mvu's
// is: the largest blocks' reach 2^32 (128x128 with 4:4:4 chroma at 16 bits, or with satd).
#ifdef SEED_BUILD
#define SEED_ACCESS
#else
#define SEED_ACCESS readonly
#endif
layout(std430, set = 0, binding = 4) SEED_ACCESS buffer Seeds { ivec2 seeds[]; };
layout(std430, set = 0, binding = 5) SEED_ACCESS buffer SeedCount { int seedCount[]; };
layout(std430, set = 0, binding = 6) buffer SeedSad { uint seedSad[]; };
// The field being refined, and a second copy for the kernels that must not update it in place
layout(std430, set = 0, binding = 7) buffer VecA { ivec2 vecA[]; };
layout(std430, set = 0, binding = 8) buffer SadA { uint sadA[]; };
layout(std430, set = 0, binding = 9) buffer VecB { ivec2 vecB[]; };
layout(std430, set = 0, binding = 10) buffer SadB { uint sadB[]; };
// lambda for each value of (worst neighbour SAD) >> 1, computed on the CPU exactly as the CPU
// reference does in double precision (Analyse.cpp)
layout(std430, set = 0, binding = 11) readonly buffer Lambda { int64_t lambdaOf[]; };
// Per block, the step at which its vector last changed and the step at which a pass last
// evaluated it (init is step 0, then every pass and the fallback count one step). A pass skips a
// block whose last evaluation left it unchanged and whose neighbours haven't changed since: its
// inputs are those of that evaluation, so it would only reproduce it.
layout(std430, set = 0, binding = 12) buffer LastChange { int lastChange[]; };
layout(std430, set = 0, binding = 13) buffer LastEval { int lastEval[]; };
// Statistics per step: blocks evaluated at [2 * step], blocks changed at [2 * step + 1]
layout(std430, set = 0, binding = 14) buffer Counters { uint counters[]; };
// Binding 15 is the fallback's block list (refine_flag.comp, refine_fallback.comp); 16 to 28
// belong to the coarse search and seed building (pyr_common.glsl), 26 is the output field
// (refine_halfpel.comp, refine_qpel.comp)

// Matches the Params struct in VulkanContext.h
layout(push_constant) uniform Params {
    int nbx, nby, step, stepY;  // the grid: nbx x nby blocks, step and stepY apart
    int pad, padY, padc, padcY; // the super's padding, luma's horizontal and vertical, then chroma's
    int wp, hp, wc, hc;
    int fallbackRadius, fallbackStep, badSad, maxSeeds;
    int stamp;       // this dispatch's step number
    int level;       // pyramid: the level the dispatch works on
    int flags;       // the field: 1 chained vectors available, 2 inverted vectors available, 4 no coarse search, 8 no
                     // global vector (globalmv=False); the coarse levels: 8 no global vector
    int topRadius;   // pyramid: the exhaustive search radius at the top level
    int medianScale; // median: the factor the vectors are scaled by
    int medianSlot;  // median: where the result goes
    int finest;      // pyramid: the level that seeds the full-size grid
    int blockRows;   // LaneSad: the block's luma rows, given here so the compiler keeps its loop
    int colour;      // the checkerboard passes: the colour of the blocks evaluated
    int recStride;   // fields stored as records (x, y, SAD, 0) per block: records per row
    int coarseBase;  // seed building: where the field's coarse result starts in Coarse, in vectors
    int aw, ah;      // the super's block-aligned frame (the grid's own but on another grid than the super's)
    int penalties;   // mvu's pzero, pglobal and pnew, 9 bits each from bit 0 (PZero, PGlobal, PNew)
    int moveRadius;  // the full-size passes: the positions within this many pixels of the winner (pelsearch)
} pc;

// The target grid's block size (specialization constants 3 and 7), mvu's 8x4, 8x8, 16x2, 16x8,
// 16x16, 32x16 or 32x32: luma kBlkX x kBlkY, U and V kBlkCX x kBlkCY. The pyramid's levels always
// use 8x8 blocks (pyr_common.glsl).
layout(constant_id = 3) const int kBlkX = 8;
layout(constant_id = 7) const int kBlkY = 8;

// The search's variant (specialization constant 6): bit 0 set when the SAD is luma's alone
// (Analyse's chroma=False), at every level; bit 1 set when chroma is subsampled horizontally and
// bit 9 when it is vertically (both at 4:2:0, bit 1 alone at 4:2:2, bit 9 alone at 4:4:0, neither at
// 4:4:4 or without chroma); bits 2 and 3, only for the kernels that measure 8 candidates per block
// (init, the passes, the half- and quarter-pel steps), log2 of the lanes measuring each candidate;
// bits 4 to 7 the bit depth less 8 (8 for floats); bit 8 set for float samples; bit 10 set when luma's
// SAD is its SATD (Analyse's satd=True), only in the full-size grid's kernels (TileSatd)
layout(constant_id = 6) const int kVariant = 0;
const bool kChroma = (kVariant & 1) == 0;
const bool kSatd = (kVariant & 1024) != 0;
const int kLogCX = (kVariant >> 1) & 1, kLogCY = (kVariant >> 9) & 1; // chroma's subsampling, as shifts
const int kBlkCX = kBlkX >> kLogCX, kBlkCY = kBlkY >> kLogCY;

// Lanes per candidate in those kernels, 1, 2 or 4. With 2 a block's group has 16 lanes: the second
// 8 measure the same 8 candidates as the first, over the other half of the block's rows (every other
// chunk of rows, LaneSad), and the lanes of a candidate, lane and lane ^ 8 (and with 4 lane ^ 16
// too), add their shares up (SadOf), so that each 8 of the group hold every candidate's whole SAD
// and pick the same best one. A workgroup then works on 8 / kSplit blocks instead of 8, which
// shrinks its block cache (sCur), so more workgroups fit on a compute unit to hide the SADs' memory
// reads: Analyse.cpp gives the blocks whose current pixels take 768 bytes or more 2 lanes per
// candidate and those taking 3 KB or more 4 (RefineSplit), which made 8-bit 4:4:4 32x32 fields
// 40% faster and 4:2:0 32x32 ones 14-21%.
const int kSplit = 1 << ((kVariant >> 2) & 3);
const int kGroupLanes = 8 * kSplit;  // a block's lanes
const int kGroupBlocks = 8 / kSplit; // a workgroup's blocks

// The samples: bytes at 8 bits, 16 bits each at 9 to 16 (kWide), 4 or 2 to a 32-bit word. The SADs
// of 16-bit samples count 2^kDepthShift times as much, so the lambda tables take the worst
// neighbour SAD in steps that much larger (Context). Float samples (kFloatS) take mvu's float SAD
// (LaneSadF), which mvu scales to the 16-bit range, so that everything after it is as at 16 bits.
const int kDepthShift = (kVariant >> 4) & 15;
const bool kWide = kDepthShift != 0;
const bool kFloatS = (kVariant & 256) != 0;
const int kSPW = kWide ? 2 : 4; // samples per word
const int kLogSPW = kWide ? 1 : 2;
// The full-size grid's lambda table (Analyse.cpp's LambdaEntries) takes the worst neighbour SAD / 2
// in steps of 2^kDepthShift, and for blocks of more than 32x32 pixels in steps 2^kAreaShift times
// that, so that it stays the size of 32x32 blocks' (Context)
const int kArea = kBlkX * kBlkY;
const int kAreaShift = kArea > 8192 ? 4 : (kArea > 4096 ? 3 : (kArea > 2048 ? 2 : (kArea > 1024 ? 1 : 0)));

// Sample i of the current frame's luma or chroma planes (integer samples: floats take CurF)
uint CurLuma(int i) {
    return kWide ? uint(curY16[i]) : uint(curY[i]);
}
uint CurChroma(uint i) {
    return kWide ? uint(curC16[i]) : uint(curC[i]);
}
// Word w of the reference frame's luma or chroma planes read as words
uint RefYWord(uint w) {
    return refY[w];
}
uint RefCWord(uint w) {
    return refC[w];
}
// The word holding sample a of a plane read as words, and the shift of the sample within it
uint WordOf(uint a) {
    return a >> uint(kLogSPW);
}
uint ShiftOf(uint a) {
    return (a & uint(kSPW - 1)) << (kWide ? 4u : 3u);
}

// The vectors' units per pixel, the super's pel, 1, 2 or 4 (specialization constant 5). At pel 1
// the super holds each plane's full-pel samples alone and the search stays on them (no sub-pel step
// follows the passes). At pel 2 and 4 it holds luma's four half-pel planes, and chroma's four but
// for chroma subsampled either way at pel 4, which has its quarter-pel image instead, as much as
// sixteen planes (kImage). The seeds, the passes and the half-pel step keep the luma vectors on the
// half-pel grid (ClampHalf), the fallback on the full-pel grid (ClampFull), and 4:4:4 chroma, which
// takes the luma vector itself, with them; only the quarter-pel step (refine_qpel.comp) reads
// between the grid's samples, and computes them there, exactly as mvu.Super's quarter planes hold
// them.
layout(constant_id = 5) const int kPel = 2;
const bool kImage = (kPel == 4 ? kLogCX + kLogCY : 0) != 0; // a select, as the core's compiler needs (see kCurWords)
const int kChromaPlanes = kImage ? 16 : kPel == 1 ? 1 : 4; // U's planes (or image), then V's as many plane sizes on
const int kRowWords = kBlkX / kSPW;                    // words of kSPW pixels in a luma row,
const int kRowWordsC = (kBlkCX + kSPW - 1) / kSPW;     // and in a chroma row (or part of one: kChromaMask)
const int kLumaWords = kBlkY * kRowWords;
const int kChromaWords = kBlkCY * kRowWordsC;          // per chroma plane
const int kBlockWords = kLumaWords + (kChroma ? 2 * kChromaWords : 0);
// A chroma row narrower than a word, 4x4 blocks' 2 pixels of horizontally subsampled 8-bit chroma, is
// read as a whole word with the bytes past it masked off: they count as 0 on both sides of the SAD
const uint kChromaMask = kBlkCX < kSPW ? 0xFFFFu : 0xFFFFFFFFu;
// The pixels of a block's SAD, luma's twice for a SATD, which reaches at most twice the SAD: times the
// largest sample, the largest SAD a block can have, which stays below 2^32 for every block mvu takes
const int kBlockPixels = (kSatd ? 2 : 1) * kBlkX * kBlkY + (kChroma ? 2 * kBlkCX * kBlkCY : 0);

// Where V's samples start, past U's
uint ChromaV() {
    return uint(kChromaPlanes * pc.wc * pc.hc);
}
// U's address of full-pel chroma pixel (x, y) of the padded plane: in the full-pel plane, or at
// (4x, 4y) of the image
uint ChromaPixel(int x, int y) {
    return kImage ? uint(16 * y * pc.wc + 4 * x) : uint(y * pc.wc + x);
}
// Samples from a chroma pixel to the one below it
int ChromaRow() {
    return kImage ? 16 * pc.wc : pc.wc;
}

// The current pixels of the workgroup's blocks, kBlockWords words of kSPW pixels per block (see
// LoadBlock), or kLevelWords for the pyramid's 8x8 blocks (LoadLevelBlock, in kernels that define
// LEVEL_BLOCKS), kCurWords entries each. 16-bit samples are kept as they come, two to a word. Bytes
// are kept expanded, two words of 16-bit halves holding c * 256 + 255 each, the form PackedStep
// compares; or packed, the 4 pixels themselves, expanded where PackedStep reads them, which costs a
// few instructions per word and halves the cache. Only caches that would take 12 KB or more
// expanded are packed, the 32x32 blocks' with chroma, 4 per workgroup (kSplit): expanded, 8 of them
// at 4:2:0 took 24 KB, which held so few workgroups on a compute unit that init, the passes and the
// half-pel step ran 31-48% slower than packed, and 8 4:4:4 16x16 blocks' 12 KB ran 14-28% slower
// than packed (before kSplit); in 8 KB or less, or with one block per workgroup, the expanded form
// ran 2-27% faster (RX 6900 XT). Sized to what the kernel uses,
// BLOCKS_PER_WORKGROUP blocks (kGroupBlocks unless the kernel says 1), since a larger array would
// hold fewer workgroups on a compute unit. Kernels that measure no SADs define NO_BLOCK_CACHE and
// go without it, and without the functions using it: the core compiles without an optimizer, so an
// unused shared array would stay and count against the kernel's shared memory. A level block's
// chroma is 8 >> kLogCX pixels wide and 8 >> kLogCY tall.
const int kLevelRowWords = 8 / kSPW;
const int kLevelLumaWords = 8 * kLevelRowWords;
const int kLevelRowWordsC = (8 >> kLogCX) / kSPW;
const int kLevelChromaWords = (8 >> kLogCY) * kLevelRowWordsC;
const int kLevelWords = kLevelLumaWords + (kChroma ? 2 * kLevelChromaWords : 0);
#ifndef BLOCKS_PER_WORKGROUP
#define BLOCKS_PER_WORKGROUP kGroupBlocks
#endif
#if defined(LEVEL_BLOCKS)
const int kCurWords = kWide ? 1 : 2;
#else
// (no || here: the core's compiler makes invalid SPIR-V of one between specialization constants)
const int kCurWords = kWide ? 1 : (BLOCKS_PER_WORKGROUP * 2 * kBlockWords * 4 >= 12288 ? 1 : 2);
#endif
// Blocks whose cache would take more than 16 KB (only blocks wider than 32 pixels) go without one:
// PackedStep reads their current pixels from the super every time (CurWord), where the cache has
// them read once per workgroup
#if defined(LEVEL_BLOCKS)
const bool kNoCache = false;
#else
// (float samples, which take their own SAD, go without one too: LaneSadF; a select, as kCurWords)
const bool kNoCache = (kFloatS ? 1 : (BLOCKS_PER_WORKGROUP * kBlockWords * 4 > 16384 ? 1 : 0)) != 0;
#endif
#ifndef NO_BLOCK_CACHE
#ifdef LEVEL_BLOCKS
shared int sCur[BLOCKS_PER_WORKGROUP * kCurWords * kLevelWords];
#else
shared int sCur[kNoCache ? 1 : BLOCKS_PER_WORKGROUP * kCurWords * kBlockWords];
#endif

// Word k of the cache (counted over the workgroup's blocks), pixels b0 .. b3 (b0 and b1 of 16-bit
// samples), in the kernel's form
void StoreWord(int k, uint b0, uint b1, uint b2, uint b3) {
    if (kNoCache)
        return;
    if (kWide) {
        sCur[k] = int(b0 | (b1 << 16u));
    } else if (kCurWords == 2) {
        sCur[2 * k] = int(0x00FF00FFu | (b0 << 8u) | (b2 << 24u));
        sCur[2 * k + 1] = int(0x00FF00FFu | (b1 << 8u) | (b3 << 24u));
    } else {
        sCur[k] = int(b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));
    }
}
#endif

int gGroup;    // the lane's block within the workgroup
int gX, gY;    // that block's position in the frame
uint gSumCur;  // the sum of the block's current pixels (LaneSad)
int gHalf = 0; // which of its candidate's kSplit lanes the lane is: its share of the rows

#ifndef NO_BLOCK_CACHE
// Sample i of word k of the lane's block, as LoadBlock caches it: the luma rows, kRowWords words each,
// then U's rows, then V's; 0 past a chroma row narrower than a word (kChromaMask)
uint CurSample(int k, int i) {
    if (k < kLumaWords)
        return CurLuma((gY + pc.padY + k / kRowWords) * pc.wp + gX + pc.pad + (k % kRowWords) * kSPW + i);
    int kc = k - kLumaWords;
    uint plane = kc < kChromaWords ? 0u : ChromaV();
    kc %= kChromaWords;
    int x = (kc % kRowWordsC) * kSPW + i;
    if (x >= kBlkCX)
        return 0u;
    return CurChroma(plane + ChromaPixel((gX >> kLogCX) + pc.padc + x, (gY >> kLogCY) + pc.padcY + kc / kRowWordsC));
}

// Word k of the lane's block, its pixels packed as they come: what PackedStep reads without a cache
uint CurWord(int k) {
    if (kWide)
        return CurSample(k, 0) | (CurSample(k, 1) << 16u);
    return CurSample(k, 0) | (CurSample(k, 1) << 8u) | (CurSample(k, 2) << 16u) | (CurSample(k, 3) << 24u);
}

// Word k of the current pixels in the cache's packed form: the cache's, or without one the super's.
// (The level kernels never go without, and don't reference the frame's planes: their layout lacks
// those bindings.)
uint CurPacked(int k) {
#ifdef LEVEL_BLOCKS
    return uint(sCur[k]);
#else
    return kNoCache ? CurWord(k) : uint(sCur[k]);
#endif
}

// Sets up the lane's block; the lanes of each group copy its current pixels to shared memory, as
// kBlockWords words of kSPW pixels: the luma rows, kRowWords words each, then U's rows, then V's
// (without chroma, the luma rows alone). The pixels' sum goes to every lane of the group, but for
// luma's with kSatd, whose SATD doesn't take it (SadOf). Every lane of the workgroup must call it and
// then barrier().
void LoadBlock(int group, int sub, int lanesPerBlock, int bx, int by, bool live) {
    gGroup = group;
    gX = bx * pc.step;
    gY = by * pc.stepY;
    uint partial = 0u;
    if (live && !kFloatS) {
        for (int k = sub; k < kBlockWords; k += lanesPerBlock) {
            uint b[4] = uint[4](0u, 0u, 0u, 0u);
            bool counts = kSatd ? k >= kLumaWords : true;
            for (int i = 0; i < kSPW; ++i) {
                b[i] = CurSample(k, i);
                partial += counts ? b[i] : 0u;
            }
            StoreWord(group * kBlockWords + k, b[0], b[1], b[2], b[3]);
        }
    }
    gSumCur = lanesPerBlock == 8    ? subgroupClusteredAdd(partial, 8u)
              : lanesPerBlock == 16 ? subgroupClusteredAdd(partial, 16u)
              : lanesPerBlock == 32 ? subgroupClusteredAdd(partial, 32u)
                                    : subgroupAdd(partial);
}
#endif

// The range ValidateVectors accepts for the block: within the super's block-aligned frame, aw x ah,
// and its padding. MVUtensils extends the super's grid by a column or row of blocks when they fall
// short of the frame, and its planes to match; the grid analysed, the super's or another, lies inside
// it. ClampHalf keeps to the half-pel grid inside that range, ClampFull to the full-pel grid; at pel
// 2 all three are the same.
ivec2 ClampTo(ivec2 v, int top) {
    return ivec2(clamp(v.x, -kPel * (gX + pc.pad), kPel * (pc.aw - kBlkX + pc.pad - gX) - top),
                 clamp(v.y, -kPel * (gY + pc.padY), kPel * (pc.ah - kBlkY + pc.padY - gY) - top));
}
ivec2 ClampVec(ivec2 v) {
    return ClampTo(v, 1);
}
ivec2 ClampHalf(ivec2 v) {
    return ClampTo(v, kPel == 4 ? 2 : 1);
}
ivec2 ClampFull(ivec2 v) {
    return ClampTo(v, kPel == 4 ? 4 : 1);
}

// Analyse's chroma vector: the luma vector divided by the subsampling, toward zero, each component
// by its own
ivec2 ChromaVector(ivec2 v) {
    return ivec2(kLogCX == 0 ? v.x : (v.x + (v.x < 0 ? 1 : 0)) >> 1, kLogCY == 0 ? v.y : (v.y + (v.y < 0 ? 1 : 0)) >> 1);
}

// The word's worth of a plane's samples starting `shift` bits into word lo, taking the rest from
// the next word hi: a funnel shift, defined for every shift of 0, 8, 16 or 24
uint Bytes4(uint lo, uint hi, uint shift) {
    // (A 64-bit shift of the pair, truncated, compiles to one v_lshrrev_b64 instead of three
    // instructions, but measured no faster: the SAD loops aren't bound by these.)
    return (lo >> shift) | ((hi << 1u) << (31u - shift));
}

#ifndef NO_BLOCK_CACHE
// Each sample (byte or 16-bit half) the rounded average of a's and b's, (a + b + 1) >> 1, as
// mvu.Super averages
uint Avg(uint a, uint b) {
    return (a | b) - (((a ^ b) >> 1u) & (kWide ? 0x7FFF7FFFu : 0x7F7F7F7Fu));
}

// The plane address of sample (Xh, Yh) of the half-pel grid: its plane (Xh & 1) | ((Yh & 1) << 1),
// planeBytes apart, then its row and column
uint HalfAddr(int Xh, int Yh, uint planeBytes, int stride) {
    return uint((Xh & 1) | ((Yh & 1) << 1)) * planeBytes + uint((Yh >> 1) * stride + (Xh >> 1));
}

// U's address of chroma position (Xc, Yc), 1 / pel pixels of the padded plane: at that place of the
// quarter-pel image (subsampled chroma at pel 4), else in one of the four half-pel planes, the
// position on the half-pel grid (4:4:4 at pel 4 is read only there here; refine_qpel.comp computes
// the samples between), or at pel 1 in the full-pel plane; V's is ChromaV() on
uint ChromaAddr(int Xc, int Yc) {
    if (kPel == 1)
        return uint(Yc * pc.wc + Xc);
    if (kImage)
        return uint(Yc * 4 * pc.wc + Xc);
    if (kPel == 4)
        return HalfAddr(Xc >> 1, Yc >> 1, uint(pc.wc * pc.hc), pc.wc);
    return HalfAddr(Xc, Yc, uint(pc.wc * pc.hc), pc.wc);
}

// A word's worth of chroma pixels of a row of the reference frame's quarter-pel image from sample
// a on: they sit 4 samples apart, each in a word of its own (16-bit ones every other word) at the
// same place
uint StridedWord(uint a) {
    uint w = WordOf(a), s = ShiftOf(a);
    if (kWide)
        return ((RefCWord(w) >> s) & 0xFFFFu) | ((RefCWord(w + 2u) >> s) << 16u);
    return ((RefCWord(w) >> s) & 0xFFu) | (((RefCWord(w + 1u) >> s) & 0xFFu) << 8u) | (((RefCWord(w + 2u) >> s) & 0xFFu) << 16u) | ((RefCWord(w + 3u) >> s) << 24u);
}

// The runs of words a SAD reads: the words' worth of samples of a row starting at sample a, aligned
// with funnel shifts, kRowWords of them for luma and kRowWordsC for chroma (gathered from the image
// if chroma has one)
void RunY(uint a, out uint r[kRowWords]) {
    uint w = WordOf(a), s = ShiftOf(a), lo = RefYWord(w);
    [[unroll]] for (int i = 0; i < kRowWords; ++i) {
        uint hi = RefYWord(w + uint(i) + 1u);
        r[i] = Bytes4(lo, hi, s);
        lo = hi;
    }
}
void RunC(uint a, out uint r[kRowWordsC]) {
    if (kImage) {
        [[unroll]] for (int i = 0; i < kRowWordsC; ++i)
            r[i] = StridedWord(a + uint(4 * kSPW * i)) & kChromaMask;
        return;
    }
    uint w = WordOf(a), s = ShiftOf(a), lo = RefCWord(w);
    [[unroll]] for (int i = 0; i < kRowWordsC; ++i) {
        uint hi = RefCWord(w + uint(i) + 1u);
        r[i] = Bytes4(lo, hi, s) & kChromaMask;
        lo = hi;
    }
}

// The SAD as sum(a) + sum(c) - 2 * sum(min(a, c)) over the reference pixels a and the current
// pixels c, exact in integers. One word r of 4 reference pixels against word k of the cache
// (LoadBlock): r's pixel sum into sa with one 4 x 8-bit dot product, and the sum of the 4 minimums
// into sm. The minimums come two at a time from 16-bit minimums: with a reference pixel in the high
// byte of a 16-bit half (anything below it) and the current pixel as c * 256 + 255, the high byte
// of the minimum is min(a, c) whatever the low byte holds; a dot product weighting only the high
// bytes adds them up. 16-bit samples, two to a word, add their absolute differences to sa directly.
void PackedStep(uint r, int k, inout uint sa, inout uint sm) {
    if (kWide) {
        u16vec2 a = unpack16(r), c = unpack16(CurPacked(k));
        u16vec2 d = max(a, c) - min(a, c);
        sa += uint(d.x) + uint(d.y);
        return;
    }
    uint ce, co;
    if (kCurWords == 2) {
        ce = uint(sCur[2 * k]);
        co = uint(sCur[2 * k + 1]);
    } else {
        uint c = CurPacked(k);
        ce = ((c << 8u) & 0xFF00FF00u) | 0x00FF00FFu;
        co = (c & 0xFF00FF00u) | 0x00FF00FFu;
    }
    sa = dotPacked4x8AccSatEXT(r, 0x01010101u, sa);
    uint even = pack32(min(unpack16(r << 8u), unpack16(ce)));
    uint odd = pack32(min(unpack16(r), unpack16(co)));
    sm = dotPacked4x8AccSatEXT(even, 0x01000100u, sm);
    sm = dotPacked4x8AccSatEXT(odd, 0x01000100u, sm);
}

// The SAD from the sums PackedStep made over the lane's share of the rows, added up over the
// candidate's kSplit lanes, which must all call it (with kSatd sa holds luma's SATD besides, and
// gSumCur only chroma's sum). Unsigned and modulo 2^32 all the way, so a lane's share may wrap below
// zero (8-bit samples): the whole SAD is below 2^32.
uint SadOf(uint sa, uint sm) {
    uint s = kWide ? sa : sa - 2u * sm;
    if (kSplit >= 2)
        s += subgroupShuffleXor(s, 8u);
    if (kSplit >= 4)
        s += subgroupShuffleXor(s, 16u);
    return kWide ? s : s + gSumCur;
}

// mvu's SATD (SADFunctions.cpp's Satd_C), luma's metric with kSatd: the block in 4x4 tiles, each the sum
// of the absolute values of the 4x4 Hadamard transform of its differences, halved, which is exact, the
// sum always being even. TileSatd transforms each row of differences (current minus reference), then
// takes the first step of the columns' transform; the second step's absolute values come in pairs,
// |a + b| + |a - b| = 2 max(|a|, |b|), so the halved sum is the sum of those maxima. A tile row is a
// word of 8-bit samples or two of 16-bit ones (kTileWords), and the reference's words of a band of 4
// rows are read kSatdChunk at a time.
const int kTileWords = 4 / kSPW;
const int kSatdChunk = kRowWords < 4 ? kRowWords : 4;

ivec4 Hadamard4(ivec4 d) {
    int s0 = d.x + d.y, s1 = d.x - d.y, s2 = d.z + d.w, s3 = d.z - d.w;
    return ivec4(s0 + s2, s1 + s3, s0 - s2, s1 - s3);
}

uint TileSatd(ivec4 d0, ivec4 d1, ivec4 d2, ivec4 d3) {
    ivec4 h0 = Hadamard4(d0), h1 = Hadamard4(d1), h2 = Hadamard4(d2), h3 = Hadamard4(d3);
    ivec4 p0 = h0 + h1, p1 = h0 - h1, p2 = h2 + h3, p3 = h2 - h3;
    ivec4 m = max(abs(p0), abs(p2)) + max(abs(p1), abs(p3));
    return uint(m.x + m.y + m.z + m.w);
}

// Word k of the current pixels as they come, kSPW to a word, whatever the cache's form
uint CurRaw(int k) {
    if (kWide || kCurWords == 1)
        return CurPacked(k);
    uint ce = uint(sCur[2 * k]), co = uint(sCur[2 * k + 1]);
    return ((ce >> 8u) & 0x00FF00FFu) | (co & 0xFF00FF00u);
}

// A tile row's differences, current minus reference: the reference's samples in word r0 (and r1 for
// 16-bit samples), the current ones from word k of the cache on
ivec4 TileRow(uint r0, uint r1, int k) {
    if (kWide)
        return ivec4(ivec2(unpack16(CurRaw(k))) - ivec2(unpack16(r0)), ivec2(unpack16(CurRaw(k + 1))) - ivec2(unpack16(r1)));
    return ivec4(unpack8(CurRaw(k))) - ivec4(unpack8(r0));
}

// Words t0 .. t0 + kSatdChunk - 1 of the run of a reference luma row whose first word is w, the
// block's pixels `shift` bits into it (RunY's, aligned with funnel shifts)
void RunChunk(uint w, uint shift, int t0, out uint r[kSatdChunk]) {
    uint lo = RefYWord(w + uint(t0));
    [[unroll]] for (int g = 0; g < kSatdChunk; ++g) {
        uint hi = RefYWord(w + uint(t0 + g) + 1u);
        r[g] = Bytes4(lo, hi, shift);
        lo = hi;
    }
}

// The SATD of the tiles of words t0 .. t0 + kSatdChunk - 1 of luma rows j0 .. j0 + 3, the reference's
// words in q, the current pixels from word cur of the cache on
uint ChunkSatd(uint q[4][kSatdChunk], int j0, int t0, int cur) {
    uint satd = 0u;
    [[unroll]] for (int g = 0; g < kSatdChunk; g += kTileWords) {
        int k = cur + j0 * kRowWords + t0 + g, h = g + kTileWords - 1;
        satd += TileSatd(TileRow(q[0][g], q[0][h], k), TileRow(q[1][g], q[1][h], k + kRowWords), TileRow(q[2][g], q[2][h], k + 2 * kRowWords),
                         TileRow(q[3][g], q[3][h], k + 3 * kRowWords));
    }
    return satd;
}

// The SATD of luma rows j0 .. j0 + 3 against the reference's rows from word w on, stride words apart,
// the block's pixels `shift` bits into their first words; the current pixels from word cur on
uint BandSatd(uint w, uint shift, uint stride, int j0, int cur) {
    uint satd = 0u;
    for (int t0 = 0; t0 < kRowWords; t0 += kSatdChunk) {
        uint q[4][kSatdChunk];
        [[unroll]] for (int i = 0; i < 4; ++i)
            RunChunk(w + uint(j0 + i) * stride, shift, t0, q[i]);
        satd += ChunkSatd(q, j0, t0, cur);
    }
    return satd;
}
#endif

// The rows the SADs read at once. Their loops run to pc.blockRows, a count the compiler can't see,
// so it keeps them and holds only a chunk of rows in registers; with all 12 rows of an 8x8 block
// in flight it needs twice the registers and runs slower. LaneSad's chunks hold about the same
// number of words whatever the block size, but no more rows than the block has, and at least one.
const int kRowChunk = 4;
const int kRowsPerChunk = kBlkX < 32 ? 32 / kBlkX : 1;
const int kLumaChunk = kRowsPerChunk < kBlkY ? kRowsPerChunk : kBlkY;
const int kChromaChunk = kRowsPerChunk < kBlkCY ? kRowsPerChunk : kBlkCY;

#ifndef NO_BLOCK_CACHE
// The address of luma position (X, Y), 1 / pel pixels of the padded plane, on the half-pel grid at
// pel 2 and 4 (in the half-pel plane holding it), at pel 1 in the full-pel plane
uint LumaAddr(int X, int Y) {
    if (kPel == 1)
        return uint(Y * pc.wp + X);
    if (kPel == 4) {
        X >>= 1;
        Y >>= 1;
    }
    return uint(((X & 1) | ((Y & 1) << 1)) * pc.wp * pc.hp + (Y >> 1) * pc.wp + (X >> 1));
}

// MVUtensils' SAD of the lane's block for vector v: luma kBlkX x kBlkY plus U and V kBlkCX x kBlkCY, or
// luma alone without chroma; with kSplit 2 the lane measures every other chunk of rows and SadOf
// adds the other lane's. With kSatd luma's SATD in place of its SAD, a band of 4 rows at a time, which
// kSplit's lanes share out alike.
// Each reference row comes as whole words, aligned with funnel shifts. The rows start on whole
// words (the storage frames' strides are), so every row of the block sits at the same offset in its
// words. At pel 4 the luma vector is on the half-pel grid (ClampHalf, ClampFull), so is 4:4:4
// chroma's, and subsampled chroma's anywhere, read from its quarter-pel image, its pixels gathered
// from a word each. Float samples take LaneSadF.
#ifndef LEVEL_BLOCKS
uint LaneSadF(ivec2 v, bool quarter);
#endif
uint LaneSad(ivec2 v) {
#ifndef LEVEL_BLOCKS
    if (kFloatS)
        return LaneSadF(v, false);
#endif
    uint a = LumaAddr(kPel * (gX + pc.pad) + v.x, kPel * (gY + pc.padY) + v.y);
    uint w = WordOf(a), shift = ShiftOf(a), stride = uint(pc.wp) >> uint(kLogSPW);
    int cur = kNoCache ? 0 : gGroup * kBlockWords;
    uint sa = 0u, sm = 0u;
    if (kSatd) {
        for (int j0 = 4 * gHalf; j0 < pc.blockRows; j0 += 4 * kSplit)
            sa += BandSatd(w, shift, stride, j0, cur);
    } else {
        for (int j0 = gHalf * kLumaChunk; j0 < pc.blockRows; j0 += kSplit * kLumaChunk) {
            [[unroll]] for (int jj = 0; jj < kLumaChunk; ++jj) {
                int j = j0 + jj;
                uint wj = w + uint(j) * stride;
                uint lo = RefYWord(wj);
                [[unroll]] for (int i = 0; i < kRowWords; ++i) {
                    uint hi = RefYWord(wj + uint(i) + 1u);
                    PackedStep(Bytes4(lo, hi, shift), cur + j * kRowWords + i, sa, sm);
                    lo = hi;
                }
            }
        }
    }
    if (!kChroma)
        return SadOf(sa, sm);
    ivec2 vc = ChromaVector(v);
    int Xc = kPel * ((gX >> kLogCX) + pc.padc) + vc.x, Yc = kPel * ((gY >> kLogCY) + pc.padcY) + vc.y;
    int curU = cur + kLumaWords, curV = curU + kChromaWords;
    uint ac = ChromaAddr(Xc, Yc);
    if (kImage) {
        uint av = ac + ChromaV(), rowC = uint(ChromaRow());
        for (int j0 = gHalf * kChromaChunk; j0 < pc.blockRows >> kLogCY; j0 += kSplit * kChromaChunk) {
            [[unroll]] for (int jj = 0; jj < kChromaChunk; ++jj) {
                int j = j0 + jj;
                uint u = ac + uint(j) * rowC, t = av + uint(j) * rowC;
                [[unroll]] for (int i = 0; i < kRowWordsC; ++i) {
                    PackedStep(StridedWord(u + uint(4 * kSPW * i)) & kChromaMask, curU + j * kRowWordsC + i, sa, sm);
                    PackedStep(StridedWord(t + uint(4 * kSPW * i)) & kChromaMask, curV + j * kRowWordsC + i, sa, sm);
                }
            }
        }
        return SadOf(sa, sm);
    }
    uint wu = WordOf(ac), wv = WordOf(ac + ChromaV()), shiftC = ShiftOf(ac), strideC = uint(pc.wc) >> uint(kLogSPW);
    for (int j0 = gHalf * kChromaChunk; j0 < pc.blockRows >> kLogCY; j0 += kSplit * kChromaChunk) {
        [[unroll]] for (int jj = 0; jj < kChromaChunk; ++jj) {
            int j = j0 + jj;
            uint u = wu + uint(j) * strideC, t = wv + uint(j) * strideC;
            uint loU = RefCWord(u), loV = RefCWord(t);
            [[unroll]] for (int i = 0; i < kRowWordsC; ++i) {
                uint hiU = RefCWord(u + uint(i) + 1u), hiV = RefCWord(t + uint(i) + 1u);
                PackedStep(Bytes4(loU, hiU, shiftC) & kChromaMask, curU + j * kRowWordsC + i, sa, sm);
                PackedStep(Bytes4(loV, hiV, shiftC) & kChromaMask, curV + j * kRowWordsC + i, sa, sm);
                loU = hiU;
                loV = hiV;
            }
        }
    }
    return SadOf(sa, sm);
}
#endif

#if !defined(NO_BLOCK_CACHE) && !defined(LEVEL_BLOCKS)
// Float samples (kFloatS) take their own SAD: mvu's of float blocks, its float operations in its order
// (sadf_common.glsl), the samples read as they are, one lane measuring the whole block (kSplit 1), the
// current pixels read from the super (no cache). gSadV is the vector measured, gSadQuarter whether it may
// lie between the half-pel grid's samples (refine_qpel.comp), which then come from them as mvu.Super's
// float quarter planes hold them, the rounded average (a + b) * 0.5 of two vertical averages.
ivec2 gSadV;
bool gSadQuarter = false;

float AvgF(float a, float b) {
    precise float s = (a + b) * 0.5;
    return s;
}

// The current frame's sample (x, y) of the block's plane p
float CurF(int p, int x, int y) {
    if (p == 0)
        return uintBitsToFloat(curY32[(gY + pc.padY + y) * pc.wp + gX + pc.pad + x]);
    uint base = p == 2 ? ChromaV() : 0u;
    return uintBitsToFloat(curC32[base + ChromaPixel((gX >> kLogCX) + pc.padc + x, (gY >> kLogCY) + pc.padcY + y)]);
}

// The reference's sample of plane p at (X, Y), 1 / pel pixels of the padded plane
float RefF(int p, int X, int Y) {
    if (p == 0) {
        if (gSadQuarter && ((X | Y) & 1) != 0) {
            int Xa = X >> 1, Xb = Xa + (X & 1), Ya = Y >> 1, Yb = Ya + (Y & 1);
            uint plane = uint(pc.wp * pc.hp);
            return AvgF(AvgF(uintBitsToFloat(refY[HalfAddr(Xa, Ya, plane, pc.wp)]), uintBitsToFloat(refY[HalfAddr(Xa, Yb, plane, pc.wp)])),
                        AvgF(uintBitsToFloat(refY[HalfAddr(Xb, Ya, plane, pc.wp)]), uintBitsToFloat(refY[HalfAddr(Xb, Yb, plane, pc.wp)])));
        }
        return uintBitsToFloat(refY[LumaAddr(X, Y)]);
    }
    uint base = p == 2 ? ChromaV() : 0u;
    if (gSadQuarter && !kImage && ((X | Y) & 1) != 0) {
        int Xa = X >> 1, Xb = Xa + (X & 1), Ya = Y >> 1, Yb = Ya + (Y & 1);
        uint plane = uint(pc.wc * pc.hc);
        return AvgF(AvgF(uintBitsToFloat(refC[base + HalfAddr(Xa, Ya, plane, pc.wc)]), uintBitsToFloat(refC[base + HalfAddr(Xa, Yb, plane, pc.wc)])),
                    AvgF(uintBitsToFloat(refC[base + HalfAddr(Xb, Ya, plane, pc.wc)]), uintBitsToFloat(refC[base + HalfAddr(Xb, Yb, plane, pc.wc)])));
    }
    return uintBitsToFloat(refC[base + ChromaAddr(X, Y)]);
}

float DiffF(int p, int x, int y) {
    float ref;
    if (p == 0) {
        ref = RefF(0, kPel * (gX + pc.pad + x) + gSadV.x, kPel * (gY + pc.padY + y) + gSadV.y);
    } else {
        ivec2 vc = ChromaVector(gSadV);
        ref = RefF(p, kPel * ((gX >> kLogCX) + pc.padc + x) + vc.x, kPel * ((gY >> kLogCY) + pc.padcY + y) + vc.y);
    }
    precise float d = CurF(p, x, y) - ref;
    return d;
}

#include "sadf_common.glsl"

// The block's float SAD at vector v (with kSatd luma's SATD), each plane's scaled alone and added as mvu
// adds them, the total saturated to the records' 32 bits (only float samples far outside their range reach
// them)
uint LaneSadF(ivec2 v, bool quarter) {
    gSadV = v;
    gSadQuarter = quarter;
    uint64_t s = uint64_t(kSatd ? SatdF(kBlkX, kBlkY) : SadF(0, kBlkX, kBlkY));
    if (kChroma)
        s += uint64_t(SadF(1, kBlkCX, kBlkCY)) + uint64_t(SadF(2, kBlkCX, kBlkCY));
    return uint(min(s, 0xFFFFFFFFul));
}
#endif

// The value of lane `sub` of this lane's group; call in uniform control flow
int FromLane(int value, int sub, int lanesPerBlock) {
    return subgroupShuffle(value, (gl_SubgroupInvocationID & ~uint(lanesPerBlock - 1)) + uint(sub));
}

void Sort4(inout int a[4]) {
    int t;
    if (a[0] > a[1]) { t = a[0]; a[0] = a[1]; a[1] = t; }
    if (a[2] > a[3]) { t = a[2]; a[2] = a[3]; a[3] = t; }
    if (a[0] > a[2]) { t = a[0]; a[0] = a[2]; a[2] = t; }
    if (a[1] > a[3]) { t = a[1]; a[1] = a[3]; a[3] = t; }
    if (a[1] > a[2]) { t = a[1]; a[1] = a[2]; a[2] = t; }
}

// Predictor and relaxed lambda of block b from its four neighbours in field A; a neighbour outside
// the grid repeats the block's own vector and doesn't count for the neighbourhood SAD. The worst SAD
// is capped at the largest a block's pixels can make (kBlockPixels times the largest sample), the
// table's last entry's: only float SADs, of samples outside their range, pass it.
void Context(int b, out ivec2 p, out int64_t lambda) {
    int bx = b % pc.nbx, by = b / pc.nbx;
    int nbs[4] = int[4](bx > 0 ? b - 1 : b, bx + 1 < pc.nbx ? b + 1 : b, by > 0 ? b - pc.nbx : b, by + 1 < pc.nby ? b + pc.nbx : b);
    int xs[4], ys[4];
    uint worst = 0u;
    for (int i = 0; i < 4; ++i) {
        ivec2 v = vecA[nbs[i]];
        xs[i] = v.x;
        ys[i] = v.y;
        if (nbs[i] != b)
            worst = max(worst, sadA[nbs[i]]);
    }
    Sort4(xs);
    Sort4(ys);
    p = ivec2((xs[1] + xs[2]) / 2, (ys[1] + ys[2]) / 2);
    uint cap = uint(kBlockPixels) * ((1u << uint(8 + kDepthShift)) - 1u);
    lambda = lambdaOf[min(worst, cap) >> uint(1 + kDepthShift + kAreaShift)];
}

int64_t Cost(uint sad, ivec2 v, ivec2 p, int64_t lambda) {
    int64_t dx = int64_t(v.x - p.x), dy = int64_t(v.y - p.y);
    return int64_t(sad) + ((lambda * (dx * dx + dy * dy)) >> 8);
}

// mvu's penalties, in 256ths of a candidate's SAD, which its cost takes on besides: pzero for the zero
// seed, pglobal for the field's median (the global vector), pnew for the positions a search steps to
// (the passes' positions around the winner, the top level's window, the fallback's, the sub-pel steps'),
// none for the predictors (the coarse vectors, the chained and inverted ones, the neighbours', the
// block's own)
int PZero() {
    return pc.penalties & 511;
}
int PGlobal() {
    return (pc.penalties >> 9) & 511;
}
int PNew() {
    return (pc.penalties >> 18) & 511;
}
int64_t Penalty(uint sad, int pen) {
    return (int64_t(pen) * int64_t(sad)) >> 8;
}
// The penalty of seed k of a full-size block: seed 0 is zero, seed 1 the field's median where the field
// has one (seed_build.comp)
int SeedPenalty(int k) {
    return k == 0 ? PZero() : (k == 1 && (pc.flags & 12) == 0 ? PGlobal() : 0);
}
// The SAD whose cost with penalty pen is c, SAD + Penalty(SAD, pen): that grows by at least one with
// each unit of SAD, so exactly one SAD has it, floor(256 c / (256 + pen)) or the next
uint SadOfPenalized(int64_t c, int pen) {
    int64_t s = (c << 8) / int64_t(256 + pen);
    if (s + ((int64_t(pen) * s) >> 8) < c)
        ++s;
    return uint(s);
}

// The 8 positions one step around a vector, in the CPU's order: rows top to bottom, then left to
// right, the centre left out
ivec2 Around(int k, int step) {
    int i = k < 4 ? k : k + 1;
    return ivec2((i % 3 - 1) * step, (i / 3 - 1) * step);
}

// A block's group of 8 lanes picks its best candidate with one clustered minimum of these keys
// (with kSplit 2 each 8 of its 16 lanes does, alike): the cost, then the candidate's place in the
// CPU's order, then the lane holding it, counted in its 8. The CPU takes
// a candidate only when it is strictly cheaper than the best so far, so it ends with the earliest
// of the cheapest candidates, which is the least key; the best so far itself wins ties, so the
// minimum replaces it only when strictly cheaper. Costs stay far below 2^48.
const uint64_t kNoKey = ~0ul;

uint64_t CandidateKey(int64_t cost, int order) {
    return (uint64_t(cost) << 16) | (uint64_t(order) << 3) | uint64_t(gl_SubgroupInvocationID & 7u);
}

int64_t KeyCost(uint64_t key) {
    return int64_t(key >> 16);
}

int KeyLane(uint64_t key) {
    return int(key & 7ul);
}
