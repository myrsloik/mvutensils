// refine_common.glsl
//
// Shared by the refinement kernels refine_*.comp (Analyse.cpp), and through pyr_common.glsl by
// the coarse search and seed building kernels. A lane computes whole SADs: in the
// seed, pass and half-pel kernels a workgroup of 64 lanes works on 8 blocks, 8 lanes per block,
// each lane measuring one candidate vector of its block; in the fallback a workgroup of one
// subgroup works on one block and spreads the positions it searches over its lanes. A block's
// group picks its best candidate with one clustered minimum (CandidateKey). Planes, SAD, clamp,
// predictor and cost follow the CPU reference (test/reference/reference.cpp) exactly, so the result
// is bit-identical to it, whatever the subgroup size. The SAD reads whole words and works on 4 pixels
// at a time (LaneSad).
//
// VulkanContext.cpp expands the #include lines before compiling, since the core's compiler has
// no include handler.

#extension GL_EXT_shader_8bit_storage : require
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
// y + 1/2, both), and for U and V four each of hc rows at pel 2 (U at planes 0-3, V at 4-7), at pel
// 4 an image each of the quarter-pel grid, 4 hc rows of 4 wc samples. wp and wc are the storage
// frames' strides, whole words, so every row starts on a word. LaneSad reads the reference frame's
// planes as 32-bit words, up to two past a row's last pixel, so the storage frames carry a spare row
// at the end.
layout(std430, set = 0, binding = 0) readonly buffer CurLuma { uint8_t curY[]; };
layout(std430, set = 0, binding = 1) readonly buffer RefLuma { uint refY[]; };
layout(std430, set = 0, binding = 2) readonly buffer CurChroma { uint8_t curC[]; };
layout(std430, set = 0, binding = 3) readonly buffer RefChroma { uint refC[]; };
// Up to maxSeeds seed vectors per block (half-pels), their count, and their SADs once measured;
// seed_build.comp writes the seeds, the other kernels only read them
#ifdef SEED_BUILD
#define SEED_ACCESS
#else
#define SEED_ACCESS readonly
#endif
layout(std430, set = 0, binding = 4) SEED_ACCESS buffer Seeds { ivec2 seeds[]; };
layout(std430, set = 0, binding = 5) SEED_ACCESS buffer SeedCount { int seedCount[]; };
layout(std430, set = 0, binding = 6) buffer SeedSad { int seedSad[]; };
// The field being refined, and a second copy for the kernels that must not update it in place
layout(std430, set = 0, binding = 7) buffer VecA { ivec2 vecA[]; };
layout(std430, set = 0, binding = 8) buffer SadA { int sadA[]; };
layout(std430, set = 0, binding = 9) buffer VecB { ivec2 vecB[]; };
layout(std430, set = 0, binding = 10) buffer SadB { int sadB[]; };
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
    int w, h, nbx, nby;
    int step, pad, padc, colour;
    int wp, hp, wc, hc;
    int fallbackRadius, fallbackStep, badSad, maxSeeds;
    int stamp;       // this dispatch's step number
    int level;       // pyramid: the level the dispatch works on; super.comp: its step
    int flags;       // seed building: 1 chained vectors available, 2 inverted vectors available
    int topRadius;   // pyramid: the exhaustive search radius at the top level
    int medianScale; // median: the factor the vectors are scaled by
    int medianSlot;  // median: where the result goes
    int finest;      // pyramid: the level that seeds the full-size grid
    int blockRows;   // LaneSad: the block's luma rows, given here so the compiler keeps its loop
    int srcStrideY;  // super.comp: the source frame's luma and chroma strides, bytes
    int srcStrideC;
    int recStride;   // fields stored as records (x, y, SAD, 0) per block: records per row
    int coarseBase;  // seed building: where the field's coarse result starts in Coarse, in vectors
} pc;

// The target grid's block size, 8, 16 or 32 (specialization constant 3): luma kBlk x kBlk, U and V
// kBlk / 2 square. The pyramid's levels always use 8x8 blocks (pyr_common.glsl).
layout(constant_id = 3) const int kBlk = 8;

// The search's variant (specialization constant 6): bit 0 set when the SAD is luma's alone
// (Analyse's chroma=False), at every level
layout(constant_id = 6) const int kVariant = 0;
const bool kChroma = (kVariant & 1) == 0;

// The vectors' units per pixel, the super's pel, 2 or 4 (specialization constant 5). The super
// holds luma's four half-pel planes either way, and chroma's four at pel 2 or its quarter-pel image
// at pel 4, which takes as much as sixteen planes. The seeds, the passes and the half-pel step keep
// the luma vectors on the half-pel grid (ClampHalf), the fallback on the full-pel grid (ClampFull);
// only the quarter-pel step (refine_qpel.comp) reads luma between the grid's samples, and computes
// them there, exactly as mvu.Super's quarter planes hold them.
layout(constant_id = 5) const int kPel = 2;
const int kChromaPlanes = kPel == 4 ? 16 : 4; // U's planes (or image), then V's as many plane sizes on
const int kRowWords = kBlk / 4;                        // words of 4 pixels in a luma row,
const int kRowWordsC = kBlk / 8;                       // and in a chroma row
const int kLumaWords = kBlk * kRowWords;
const int kChromaWords = kBlk / 2 * kRowWordsC;        // per chroma plane
const int kBlockWords = kLumaWords + (kChroma ? 2 * kChromaWords : 0);

// Where V's samples start, past U's
uint ChromaV() {
    return uint(kChromaPlanes * pc.wc * pc.hc);
}
// U's address of full-pel chroma pixel (x, y) of the padded plane: in the full-pel plane at pel 2,
// at (4x, 4y) of the image at pel 4
uint ChromaPixel(int x, int y) {
    return kPel == 4 ? uint(16 * y * pc.wc + 4 * x) : uint(y * pc.wc + x);
}
// Samples from a chroma pixel to the one below it
int ChromaRow() {
    return kPel == 4 ? 16 * pc.wc : pc.wc;
}

// The current pixels of the workgroup's blocks, kBlockWords words of 4 pixels per block (see
// LoadBlock), or kLevelWords for the pyramid's 8x8 blocks (LoadLevelBlock, in kernels that define
// LEVEL_BLOCKS), kCurWords entries each: expanded, two words of 16-bit halves holding c * 256 + 255
// each, the form PackedStep compares; or packed, the 4 pixels themselves, expanded where
// PackedStep reads them, which costs a few instructions per word and halves the cache. Only the
// 32x32 blocks' kernels that cache 8 blocks pack it: expanded, their 24 KB held so few workgroups on
// a compute unit that init, the passes and the half-pel step ran 31-48% slower; at 8x8 and 16x16,
// or with one block per workgroup, the expanded form ran 2-27% faster (RX 6900 XT). Sized to what
// the kernel uses, BLOCKS_PER_WORKGROUP blocks (8 unless the kernel says 1), since a larger array
// would hold fewer workgroups on a compute unit. Kernels that measure no SADs define
// NO_BLOCK_CACHE and go without it, and without the functions using it: the core compiles without
// an optimizer, so an unused shared array would stay and count against the kernel's shared memory.
const int kLevelWords = 24;
#ifndef BLOCKS_PER_WORKGROUP
#define BLOCKS_PER_WORKGROUP 8
#endif
#if defined(LEVEL_BLOCKS) || BLOCKS_PER_WORKGROUP == 1
const int kCurWords = 2;
#else
const int kCurWords = kBlk >= 32 ? 1 : 2;
#endif
#ifndef NO_BLOCK_CACHE
#ifdef LEVEL_BLOCKS
shared int sCur[BLOCKS_PER_WORKGROUP * kCurWords * kLevelWords];
#else
shared int sCur[BLOCKS_PER_WORKGROUP * kCurWords * kBlockWords];
#endif

// Word k of the cache (counted over the workgroup's blocks), pixels b0 .. b3, in the kernel's form
void StoreWord(int k, uint b0, uint b1, uint b2, uint b3) {
    if (kCurWords == 2) {
        sCur[2 * k] = int(0x00FF00FFu | (b0 << 8u) | (b2 << 24u));
        sCur[2 * k + 1] = int(0x00FF00FFu | (b1 << 8u) | (b3 << 24u));
    } else {
        sCur[k] = int(b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));
    }
}
#endif

int gGroup;   // the lane's block within the workgroup
int gX, gY;   // that block's position in the frame
uint gSumCur; // the sum of the block's current pixels (LaneSad)

#ifndef NO_BLOCK_CACHE
// Sets up the lane's block; the lanes of each group copy its current pixels to shared memory, as
// kBlockWords words of 4 pixels: the luma rows, kRowWords words each, then U's rows, then V's
// (without chroma, the luma rows alone). The pixels' sum goes to every lane of the group. Every
// lane of the workgroup must call it and then barrier().
void LoadBlock(int group, int sub, int lanesPerBlock, int bx, int by, bool live) {
    gGroup = group;
    gX = bx * pc.step;
    gY = by * pc.step;
    uint partial = 0u;
    if (live) {
        for (int k = sub; k < kBlockWords; k += lanesPerBlock) {
            uint b[4];
            for (int i = 0; i < 4; ++i) {
                if (k < kLumaWords) {
                    b[i] = uint(curY[(gY + pc.pad + k / kRowWords) * pc.wp + gX + pc.pad + (k % kRowWords) * 4 + i]);
                } else {
                    int kc = k - kLumaWords;
                    uint plane = kc < kChromaWords ? 0u : ChromaV();
                    kc %= kChromaWords;
                    b[i] = uint(curC[plane + ChromaPixel(gX / 2 + pc.padc + (kc % kRowWordsC) * 4 + i, gY / 2 + pc.padc + kc / kRowWordsC)]);
                }
                partial += b[i];
            }
            StoreWord(group * kBlockWords + k, b[0], b[1], b[2], b[3]);
        }
    }
    gSumCur = lanesPerBlock == 8 ? subgroupClusteredAdd(partial, 8u) : subgroupAdd(partial);
}
#endif

// The range ValidateVectors accepts for the block: within the block-aligned frame the grid covers,
// (nbx - 1) * step + kBlk wide, and its padding. MVUtensils extends the grid by a column or row of
// blocks when they fall short of the frame, and the super's planes to match. ClampHalf keeps to the
// half-pel grid inside that range, ClampFull to the full-pel grid; at pel 2 all three are the same.
ivec2 ClampTo(ivec2 v, int top) {
    return ivec2(clamp(v.x, -kPel * (gX + pc.pad), kPel * ((pc.nbx - 1) * pc.step + pc.pad - gX) - top),
                 clamp(v.y, -kPel * (gY + pc.pad), kPel * ((pc.nby - 1) * pc.step + pc.pad - gY) - top));
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

// Analyse rounds the chroma vector toward zero
int ChromaComponent(int v) {
    return (v + (v < 0 ? 1 : 0)) >> 1;
}

// The 4 bytes of a plane starting at the byte `shift` / 8 into word lo, taking the rest from the
// next word hi: a funnel shift, defined for every shift of 0, 8, 16 or 24
uint Bytes4(uint lo, uint hi, uint shift) {
    // (A 64-bit shift of the pair, truncated, compiles to one v_lshrrev_b64 instead of three
    // instructions, but measured no faster: the SAD loops aren't bound by these.)
    return (lo >> shift) | ((hi << 1u) << (31u - shift));
}

#ifndef NO_BLOCK_CACHE
// Each byte the rounded average of a's and b's, (a + b + 1) >> 1, as mvu.Super averages
uint Avg(uint a, uint b) {
    return (a | b) - (((a ^ b) >> 1u) & 0x7F7F7F7Fu);
}

// The plane address of sample (Xh, Yh) of the half-pel grid: its plane (Xh & 1) | ((Yh & 1) << 1),
// planeBytes apart, then its row and column
uint HalfAddr(int Xh, int Yh, uint planeBytes, int stride) {
    return uint((Xh & 1) | ((Yh & 1) << 1)) * planeBytes + uint((Yh >> 1) * stride + (Xh >> 1));
}

// U's address of chroma position (Xc, Yc), 1 / pel pixels of the padded plane: in one of the four
// half-pel planes at pel 2, at that place of the quarter-pel image at pel 4; V's is ChromaV() on
uint ChromaAddr(int Xc, int Yc) {
    if (kPel == 4)
        return uint(Yc * 4 * pc.wc + Xc);
    return HalfAddr(Xc, Yc, uint(pc.wc * pc.hc), pc.wc);
}

// At pel 4, 4 chroma pixels of a row of the reference frame's image from byte a on, as a word: they
// sit 4 samples apart, each in a word of its own at the same byte
uint Strided4(uint a) {
    uint w = a >> 2u, s = (a & 3u) << 3u;
    return ((refC[w] >> s) & 0xFFu) | (((refC[w + 1u] >> s) & 0xFFu) << 8u) | (((refC[w + 2u] >> s) & 0xFFu) << 16u) | ((refC[w + 3u] >> s) << 24u);
}

// The runs of words a SAD reads: the 4-byte groups of a row starting at byte a, aligned with funnel
// shifts, kRowWords of them for luma and kRowWordsC for chroma (at pel 4 gathered from the image)
void RunY(uint a, out uint r[kRowWords]) {
    uint w = a >> 2u, s = (a & 3u) << 3u, lo = refY[w];
    [[unroll]] for (int i = 0; i < kRowWords; ++i) {
        uint hi = refY[w + uint(i) + 1u];
        r[i] = Bytes4(lo, hi, s);
        lo = hi;
    }
}
void RunC(uint a, out uint r[kRowWordsC]) {
    if (kPel == 4) {
        [[unroll]] for (int i = 0; i < kRowWordsC; ++i)
            r[i] = Strided4(a + uint(16 * i));
        return;
    }
    uint w = a >> 2u, s = (a & 3u) << 3u, lo = refC[w];
    [[unroll]] for (int i = 0; i < kRowWordsC; ++i) {
        uint hi = refC[w + uint(i) + 1u];
        r[i] = Bytes4(lo, hi, s);
        lo = hi;
    }
}

// The SAD as sum(a) + sum(c) - 2 * sum(min(a, c)) over the reference pixels a and the current
// pixels c, exact in integers. One word r of 4 reference pixels against word k of the cache
// (LoadBlock): r's pixel sum into sa with one 4 x 8-bit dot product, and the sum of the 4 minimums
// into sm. The minimums come two at a time from 16-bit minimums: with a reference pixel in the high
// byte of a 16-bit half (anything below it) and the current pixel as c * 256 + 255, the high byte
// of the minimum is min(a, c) whatever the low byte holds; a dot product weighting only the high
// bytes adds them up.
void PackedStep(uint r, int k, inout uint sa, inout uint sm) {
    uint ce, co;
    if (kCurWords == 2) {
        ce = uint(sCur[2 * k]);
        co = uint(sCur[2 * k + 1]);
    } else {
        uint c = uint(sCur[k]);
        ce = ((c << 8u) & 0xFF00FF00u) | 0x00FF00FFu;
        co = (c & 0xFF00FF00u) | 0x00FF00FFu;
    }
    sa = dotPacked4x8AccSatEXT(r, 0x01010101u, sa);
    uint even = pack32(min(unpack16(r << 8u), unpack16(ce)));
    uint odd = pack32(min(unpack16(r), unpack16(co)));
    sm = dotPacked4x8AccSatEXT(even, 0x01000100u, sm);
    sm = dotPacked4x8AccSatEXT(odd, 0x01000100u, sm);
}
#endif

// The rows the SADs read at once. Their loops run to pc.blockRows, a count the compiler can't see,
// so it keeps them and holds only a chunk of rows in registers; with all 12 rows of an 8x8 block
// in flight it needs twice the registers and runs slower. LaneSad's chunks hold about the same
// number of words whatever the block size.
const int kRowChunk = 4;
const int kLumaChunk = 32 / kBlk, kChromaChunk = 32 / kBlk;

#ifndef NO_BLOCK_CACHE
// MVUtensils' SAD of the lane's block for vector v: luma kBlk x kBlk plus U and V at half that, or
// luma alone without chroma.
// Each reference row comes as whole words, aligned with funnel shifts. The rows start on whole
// words (the storage frames' strides are), so every row of the block sits at the same offset in its
// words. At pel 4 the luma vector is on the half-pel grid (ClampHalf, ClampFull), and the chroma
// one anywhere, read from chroma's quarter-pel image, its pixels gathered from a word each.
int LaneSad(ivec2 v) {
    int X = kPel * (gX + pc.pad) + v.x, Y = kPel * (gY + pc.pad) + v.y;
    if (kPel == 4) {
        X >>= 1;
        Y >>= 1;
    }
    int idx = (X & 1) | ((Y & 1) << 1);
    uint a = uint(idx * pc.wp * pc.hp + (Y >> 1) * pc.wp + (X >> 1));
    uint w = a >> 2u, shift = (a & 3u) << 3u, stride = uint(pc.wp) >> 2u;
    int cur = gGroup * kBlockWords;
    uint sa = 0u, sm = 0u;
    for (int j0 = 0; j0 < pc.blockRows; j0 += kLumaChunk) {
        [[unroll]] for (int jj = 0; jj < kLumaChunk; ++jj) {
            int j = j0 + jj;
            uint wj = w + uint(j) * stride;
            uint lo = refY[wj];
            [[unroll]] for (int i = 0; i < kRowWords; ++i) {
                uint hi = refY[wj + uint(i) + 1u];
                PackedStep(Bytes4(lo, hi, shift), cur + j * kRowWords + i, sa, sm);
                lo = hi;
            }
        }
    }
    if (!kChroma)
        return int(sa + gSumCur - 2u * sm);
    int Xc = kPel * (gX / 2 + pc.padc) + ChromaComponent(v.x), Yc = kPel * (gY / 2 + pc.padc) + ChromaComponent(v.y);
    int curU = cur + kLumaWords, curV = curU + kChromaWords;
    uint ac = ChromaAddr(Xc, Yc);
    if (kPel == 4) {
        uint av = ac + ChromaV(), rowC = uint(ChromaRow());
        for (int j0 = 0; j0 < pc.blockRows / 2; j0 += kChromaChunk) {
            [[unroll]] for (int jj = 0; jj < kChromaChunk; ++jj) {
                int j = j0 + jj;
                uint u = ac + uint(j) * rowC, t = av + uint(j) * rowC;
                [[unroll]] for (int i = 0; i < kRowWordsC; ++i) {
                    PackedStep(Strided4(u + uint(16 * i)), curU + j * kRowWordsC + i, sa, sm);
                    PackedStep(Strided4(t + uint(16 * i)), curV + j * kRowWordsC + i, sa, sm);
                }
            }
        }
        return int(sa + gSumCur - 2u * sm);
    }
    uint wu = ac >> 2u, wv = (ac + ChromaV()) >> 2u, shiftC = (ac & 3u) << 3u, strideC = uint(pc.wc) >> 2u;
    for (int j0 = 0; j0 < pc.blockRows / 2; j0 += kChromaChunk) {
        [[unroll]] for (int jj = 0; jj < kChromaChunk; ++jj) {
            int j = j0 + jj;
            uint u = wu + uint(j) * strideC, t = wv + uint(j) * strideC;
            uint loU = refC[u], loV = refC[t];
            [[unroll]] for (int i = 0; i < kRowWordsC; ++i) {
                uint hiU = refC[u + uint(i) + 1u], hiV = refC[t + uint(i) + 1u];
                PackedStep(Bytes4(loU, hiU, shiftC), curU + j * kRowWordsC + i, sa, sm);
                PackedStep(Bytes4(loV, hiV, shiftC), curV + j * kRowWordsC + i, sa, sm);
                loU = hiU;
                loV = hiV;
            }
        }
    }
    return int(sa + gSumCur - 2u * sm);
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
// the grid repeats the block's own vector and doesn't count for the neighbourhood SAD
void Context(int b, out ivec2 p, out int64_t lambda) {
    int bx = b % pc.nbx, by = b / pc.nbx;
    int nbs[4] = int[4](bx > 0 ? b - 1 : b, bx + 1 < pc.nbx ? b + 1 : b, by > 0 ? b - pc.nbx : b, by + 1 < pc.nby ? b + pc.nbx : b);
    int xs[4], ys[4];
    int worst = 0;
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
    lambda = lambdaOf[worst >> 1];
}

int64_t Cost(int sad, ivec2 v, ivec2 p, int64_t lambda) {
    int64_t dx = int64_t(v.x - p.x), dy = int64_t(v.y - p.y);
    return int64_t(sad) + ((lambda * (dx * dx + dy * dy)) >> 8);
}

// The 8 positions one step around a vector, in the CPU's order: rows top to bottom, then left to
// right, the centre left out
ivec2 Around(int k, int step) {
    int i = k < 4 ? k : k + 1;
    return ivec2((i % 3 - 1) * step, (i / 3 - 1) * step);
}

// A block's group of 8 lanes picks its best candidate with one clustered minimum of these keys:
// the cost, then the candidate's place in the CPU's order, then the lane holding it. The CPU takes
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
