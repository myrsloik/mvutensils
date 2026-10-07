#pragma once

// How mvgpu.Super lays out a frame on the GPU, and the frame properties that describe it to the
// filters that read it.
//
// mvu.Super stores one pyramid level per plane, laid out for the CPU. mvgpu.Super keeps what the
// kernels read, laid out for them: each frame of the super clip is one GPU frame of Gray samples,
// the clip's (8-bit, 9 to 16-bit or float), whose plane holds three parts one after another
// (Regions gives where):
//
// - luma: the padded sub-pel planes one after another, hp rows each, then a spare row, wp samples
//   wide, in the frame's own rows (which start whole words apart): the full-pel plane alone at pel
//   1; at pel 2 and 4 it and the three half-pel planes (x + 1/2, y + 1/2, both).
// - chroma (not for Gray): U and V interleaved, sample by sample, U's sample of a place then V's,
//   in rows of 2 wc samples (wc pixels of each) at a stride of their own, then a spare row: the
//   full-pel plane of hc rows at pel 1, and the three half-pel planes after it at pel 2, and at pel
//   4 when chroma isn't subsampled (4:4:4, kept as luma is); at pel 4 when it is, one image of the
//   planes' quarter-pel grid, quarter sample (fx, fy) of padded pixel (x, y) at (4x + fx, 4y + fy),
//   4 hc rows of 4 wc places, U's and V's samples side by side, each four of the region's rows (as
//   much as sixteen planes would take).
// - pyramid (not with onelevel): the coarse levels 1 .. topLevel the search starts from, luma's
//   plane and chroma's (U and V interleaved, as level 0's) of every level inside a border of
//   repeated edge pixels, at the offsets the level table gives; flat.
//
// The frame is the storage and nothing else: it shares no planes with the clip's frames and holds
// no frames in its properties, so the core's frame cache sees all of it and keeps nothing else
// alive. (The frames of a vector clip, likewise, are the vectors' records alone: ExportAnalysis.)
//
// The planes cover the block-aligned frame the analysis grid covers, as mvu.Super's do, so the
// grid is a property of the super clip: blksize and overlap go to Super. The spare rows let the
// SADs read whole words past a plane's last pixel. At pel 4 luma keeps its four half-pel planes:
// mvu.Super makes every quarter sample the rounded average of half-pel ones, whatever sharp, so the
// kernels compute them where they read luma between the half-pel samples, which measured as fast
// as reading materialized planes (probe/qpel_bench.cpp). Unsubsampled chroma takes the luma vector
// itself, so it lands where luma does and is kept the same way. Subsampled chroma, at half the
// resolution, lands between its half-pel samples wherever a luma vector has a half pel, so its
// quarter samples are materialized: a 4:2:0 super is twice pel 2's size instead of four times, a
// 4:4:4 one the same size as pel 2's. The quarter samples form one image rather
// than sixteen planes: neighbouring blocks' vectors mostly differ in sub-pel phase, and in planes a
// block's candidates and its neighbours' read up to nine of the sixteen, where in the image they read
// one region. Measured on the test clips (RX 6900 XT, 16x16/8, radius 3), the pel 4 chain's GPU
// time fell 7-19% on moving content and 1-4% on static; the half-pel step halved, while the fallback
// and init, whose candidates are scattered at one phase, read the image's spread samples 10-16%
// slower. Luma's half-pel samples stay planes: the passes read one phase at a time, which planes keep
// packed (probe/layout_bench.cpp measured luma as an image 5-19% slower). A pelclip's quarter samples
// are no such averages, so mvgpu.Super refuses a pelclip at pel 4 (a divergence from mvu.Super).
//
// Chroma is interleaved because whatever reads it takes U and V at the same places: the search's SAD
// reads one row of 2 kBlkCX samples where it read a row of U and one of V (float SADs take U's and V's
// sums side by side, in one pass), and the filters that compensate motion make a pixel's U and V in
// the same lane, which shares its vectors, weights and masks between them and finds both samples in
// the same words. Measured against the planar layout before it on the test clips (RX 6900 XT, pel 2,
// 16x16 blocks, overlap 8; 16-bit 4:2:0 at 4K unless said): the four fields' refinement took 9% less
// GPU time (17% less with float samples, at 1080p), Degrain's pixels 14-15% less (13% at 8 bits, 10%
// with floats), FlowInter's 22% and Compensate's 20%, and the chains ran 8-12% faster (0-4% with
// 32x32 blocks, 3% at 4:4:4). Prototypes of U and V in one lane from planar planes gained a third to
// two thirds as much, and lost at 8 bits: a lane reading U's and V's rows from planes far apart touches
// twice the cache lines.

#include <cstdint>
#include <string>
#include <vector>

#include <VapourSynth4.h>

// A coarse pyramid level as the kernels read it, matching pyr_common.glsl's Level. Offsets,
// strides and sizes count samples; offC is U's sample of chroma's first pixel inside the border, its
// V's the next (the plane interleaved, as level 0's), strideC the samples of its rows, wc and hc and
// borderC count pixels.
struct LevelEntry {
    int32_t w, h, wc, hc;
    int32_t offY, offC, spare, nbx;
    int32_t nby, fieldOff, pad, lambdaOff; // pad: the horizontal padding, padY the vertical
    int32_t frameSamples, fieldTotal;      // entry 0 only
    int32_t strideY, strideC;
    int32_t borderY, borderC, padY, reserved;
};
static_assert(sizeof(LevelEntry) == 80, "pyr_common.glsl's Level is 20 ints");

// Where a super frame keeps the parts of its storage: byte offsets into its plane's buffer and sizes
// of luma's storage, chroma's (none for Gray) and the coarse levels' (none without them), and the
// strides of luma's rows and chroma's (2 wc samples each, U and V interleaved: KernelWc). Offsets are
// multiples of 256 bytes, which every device takes as a storage buffer descriptor's offset.
struct SuperRegions {
    int64_t luma = 0, lumaBytes = 0, chroma = 0, chromaBytes = 0, pyramid = 0, pyramidBytes = 0;
    int64_t lumaStride = 0, chromaStride = 0;

    // The kernels' wc, for samples of bytes bytes each: chroma's rows' stride in pixels, two samples a
    // pixel (0 without chroma)
    int32_t KernelWc(int64_t bytes) const { return static_cast<int32_t>(chromaStride / (2 * bytes)); }
};

// The samples of a super: the clip's
struct SuperFormat {
    int bits = 8;        // 8 .. 16, or 32 for float
    bool chroma = true;  // YUV; false for Gray
    int xr = 2, yr = 2;  // chroma's subsampling, 1 or 2 each; 1 without chroma, as mvu has it

    bool Float() const { return bits == 32; }
    int Bytes() const { return bits == 8 ? 1 : bits <= 16 ? 2 : 4; }
    // The kernels' sample kind (specialization constant 6): 0 8-bit, 1 16-bit, 2 float
    int Kind() const { return bits == 8 ? 0 : bits <= 16 ? 1 : 2; }
    bool operator==(const SuperFormat &o) const { return bits == o.bits && chroma == o.chroma && xr == o.xr && yr == o.yr; }

    // A clip's, or throws when mvu.Super doesn't take it
    static SuperFormat Of(const VSVideoFormat &f);
};

struct SuperLayout {
    int width = 0, height = 0; // the frame
    SuperFormat format;
    // The analysis grid's blocks: blk x blkY, step and stepY apart, overlapping by overlap and
    // overlapY; nbx x nby of them, extended by a column or row while they fall short of the frame,
    // as mvu.Analyse does, covering the block-aligned frame aw x ah, (nbx - 1) * step + blk wide
    int blk = 0, blkY = 0, overlap = 0, overlapY = 0, step = 0, stepY = 0;
    int nbx = 0, nby = 0, aw = 0, ah = 0;
    int pad = 0, padY = 0, padc = 0, padcY = 0; // padding around the planes: luma's horizontal and vertical, chroma's
    int wp = 0, hp = 0, wc = 0, hc = 0;         // the padded planes: width rounded up to whole words, and rows
    int pel = 2;                                // the vectors' units per pixel, 1, 2 or 4
    int lumaPlanes = 4;                         // luma's sub-pel planes stored: 1 at pel 1, else 4
    int topLevel = 0;                           // coarse levels 1 .. topLevel; 0 when there is no pyramid
    std::vector<LevelEntry> levels;             // the kernels' level table, entry L for level L, 0 the frame itself
    int pyramidSamples = 0;                     // a frame's coarse levels
    int fieldTotal = 0;                         // blocks of one field's coarse levels, finest .. top
    int maxCoarseVector = 0;                    // the largest vector component the coarse levels can hold

    static constexpr int kFinest = 1;    // the coarse level whose field seeds the full-size grid
    static constexpr int kTopWidth = 96; // a frame is halved while the next level stays at least this wide
    // Entries of a coarse level's lambda table: (largest SAD of an 8x8 block with 4:4:4 chroma) >>
    // (1 + bits - 8), plus one, at any bit depth (the most at 16 bits)
    static constexpr int kLambdaEntries = (3 * 64 * 65535 >> 9) + 1;

    // Frames of width x height samples of format; pyramid adds the coarse levels (as many as the
    // frame is wide enough for, maybe none)
    static SuperLayout Make(const SuperFormat &format, int width, int height, int blkX, int blkY, int overlapX, int overlapY, int padX, int padY, int pel,
                            bool pyramid);

    // The same super analysed on another grid, as mvu.Analyse and mvu.Recalculate take one: blkX x blkY
    // blocks overlapping by overlapX x overlapY, as many as cover the frame (extended as mvu extends
    // them), within the super's block-aligned frame aw x ah, which stays as it is with the planes and
    // the padding; throws mvu's error when they don't fit in it
    SuperLayout WithGrid(int blkX, int blkY, int overlapX, int overlapY) const;

    // Chroma kept as its quarter-pel image: subsampled chroma at pel 4
    bool ChromaImage() const { return pel == 4 && format.chroma && (format.xr > 1 || format.yr > 1); }
    // Plane sizes chroma takes, hc rows of 2 wc samples each (U and V interleaved): the full-pel
    // plane, the four half-pel planes, or the image
    int ChromaPlanes() const { return ChromaImage() ? 16 : pel == 1 ? 1 : 4; }
    // The storage frames' sizes
    int LumaRows() const { return lumaPlanes * hp + 1; }
    int ChromaRows() const { return ChromaPlanes() * hc + 1; } // of 2 wc samples, U and V interleaved
    int PyramidWidth() const { return wp; }
    int PyramidRows() const { return (pyramidSamples + wp - 1) / wp + 1; }
    // The super clip's frames: Gray of the format's samples, FrameWidth() x FrameRows(), the storage's
    // parts where Regions puts them for the frame's stride (the rows past luma's hold the others at
    // any stride the core gives the frame, the least being FrameWidth() samples)
    int FrameWidth() const { return wp; }
    int FrameRows() const;
    SuperRegions Regions(int64_t stride) const;

    // Which filter is to read the super: the search (Analyse, AnalyseMany), or the filters that
    // compensate motion with its vectors (Degrain, FlowInter, FlowFPS)
    enum class Use { Search, Compensation };
    // What they implement so far, either of them: Gray or YUV of any subsampling, of 8 to 16-bit or
    // float samples, at any pel, any of mvu's block sizes, overlaps and paddings. Empty when the layout
    // is one of those, else what it lacks.
    std::string Unsupported(Use use) const;

    bool operator==(const SuperLayout &o) const;
};


// Sets the layout's description on a super frame, and for the tools where its parts are in it
// (SuperChromaOffset, SuperChromaStride, SuperPyramidOffset, in bytes)
void ExportSuper(VSFrame *dst, const SuperLayout &layout, const SuperRegions &regions, const std::string &prefix, const VSAPI *vsapi);

// The layout a super clip's frames were made with, read from its first frame; throws when the
// clip doesn't come from mvgpu.Super with this prefix
SuperLayout ImportSuperLayout(VSNode *node, const std::string &prefix, const VSAPI *vsapi);
// The same of the super a vector clip's vectors were analysed on, from the description its frames
// carry (ExportAnalysis)
SuperLayout ImportAnalysedLayout(VSNode *vectors, const std::string &prefix, const VSAPI *vsapi);

// Where a frame of such a clip keeps its parts; false when it isn't a frame of this layout
bool GetSuperRegions(const VSFrame *frame, const SuperLayout &layout, SuperRegions &out, const VSAPI *vsapi);
// The same for a frame a filter loads, checked as mvu checks every super frame it loads: made with the
// same Super arguments as the clip's first frame, whose layout this is (a spliced clip can mix
// supers), and this layout's storage. Empty, or what is wrong.
std::string CheckSuperFrame(const VSFrame *frame, const SuperLayout &layout, const std::string &prefix, SuperRegions &out, const VSAPI *vsapi);

// The same level-0 storage, whatever the levels above it and the grid: the planes the kernels read
bool SameStorage(const SuperLayout &a, const SuperLayout &b);
// The same, whatever the samples' bit depth: what mvu requires of the super vectors were analysed
// on (IsCompatibleWithAnalysis)
bool SameGeometry(const SuperLayout &a, const SuperLayout &b);

// A vector clip's analysis description, as mvu's filters read it from its first frame
struct VectorInfo {
    int width = 0, height = 0, realWidth = 0, realHeight = 0, hpad = 0, vpad = 0, pel = 0;
    int blkX = 0, blkY = 0, overlapX = 0, overlapY = 0, nbx = 0, nby = 0;
    int delta = 0, bits = 0, chroma = 0, xRatio = 0, yRatio = 0;
};

// Read from the clip's first frame; throws when a property is missing, AnalysisHasVectors included,
// saying so when the clip holds mvu's vectors instead (mvgpu.FromMVU converts them)
VectorInfo ReadVectorInfo(VSNode *node, const std::string &prefix, const VSAPI *vsapi);

// mvu's test of a vector frame that carries vectors against its clip's first frame
// (MotionBlockPyramid::Geometry): the whole description, the subsampling only where the SADs count
// chroma
bool SameAnalysis(const VectorInfo &a, const VectorInfo &b);

// The analysis description mvu.Analyse attaches, under the same names, on a frame of a vector clip:
// the vectors' records themselves, Gray32, a record (x, y, SAD, 0) per block, vectors in 1 / pel
// pixels, a row of records per row of blocks, 4 * nbx x nby. The frame also carries the description
// of the super it was analysed on (allocate it with the super frame as its property source).
// chroma: whether the SADs count chroma; hasVectors: whether the records hold vectors (they don't
// where the reference frame is outside the clip, as mvu's frames have none there): HasVectors;
// levels: the pyramid levels the search took (Analyse's levels), the super's all by default.
void ExportAnalysis(VSFrame *dst, const SuperLayout &layout, int delta, bool chroma, bool hasVectors, const std::string &prefix, const VSAPI *vsapi,
                    int levels = 0);

// A frame of a vector clip as the vectors' records, a new reference, or null when it has none
const VSFrame *GetAnalysisVectors(const VSFrame *frame, const std::string &prefix, const VSAPI *vsapi);
// The same for a frame a filter loads, checked as mvu checks every vector frame it loads: one with
// vectors must carry the description of its clip's first frame (expected, SameAnalysis), as a spliced
// clip can mix analyses; when it doesn't, error says so and the result is null
const VSFrame *GetAnalysisVectors(const VSFrame *frame, const VectorInfo &expected, const std::string &prefix, std::string &error, const VSAPI *vsapi);

// mvu's argument helpers: an "h" or "h,v" list argument (absent -> the defaults, one value -> v = h),
// and the block size and overlap rules
void GetPairArgument(int &h, int &v, const char *name, int defaultH, int defaultV, const VSMap *in, const VSAPI *vsapi);
void CheckBlockSize(int blkX, int blkY, int overlapX, int overlapY, int subSamplingW, int subSamplingH, bool satd = false);
