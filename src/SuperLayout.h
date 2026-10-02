#pragma once

// How mvgpu.Super lays out a frame on the GPU, and the frame properties that carry it to Analyse
// and Degrain.
//
// mvu.Super stores one pyramid level per plane, laid out for the CPU. mvgpu.Super keeps what the
// kernels read, laid out for them, in up to three GPU frames attached to each output frame, their
// samples the clip's (8-bit, 9 to 16-bit or float):
//
// - luma: the padded sub-pel planes one after another, hp rows each, then a spare row, wp samples
//   wide, its rows whole words apart: the full-pel plane alone at pel 1; at pel 2 and 4 it and the
//   three half-pel planes (x + 1/2, y + 1/2, both).
// - chroma (not for Gray): U's samples, then V's, then a spare row, wc wide: the full-pel plane of
//   hc rows at pel 1, and the three half-pel planes after it at pel 2, and at pel 4 when chroma
//   isn't subsampled (4:4:4, kept as luma is); at pel 4 when it is, one image of the plane's
//   quarter-pel grid, quarter sample (fx, fy) of padded pixel (x, y) at (4x + fx, 4y + fy), 4 hc
//   rows of 4 wc samples, each four of the frame's rows (as much as sixteen planes would take).
// - pyramid (not with onelevel): the coarse levels 1 .. topLevel the search starts from, every
//   plane of every level inside a border of repeated edge pixels, at the offsets the level table
//   gives; one flat buffer.
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

#include <cstdint>
#include <string>
#include <vector>

#include <VapourSynth4.h>

// A coarse pyramid level as the kernels read it, matching pyr_common.glsl's Level. Offsets,
// strides and sizes count samples.
struct LevelEntry {
    int32_t w, h, wc, hc;
    int32_t offY, offU, offV, nbx;
    int32_t nby, fieldOff, pad, lambdaOff;
    int32_t frameSamples, fieldTotal; // entry 0 only
    int32_t strideY, strideC;
    int32_t borderY, borderC, reserved[2];
};
static_assert(sizeof(LevelEntry) == 80, "pyr_common.glsl's Level is 20 ints");

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

    // Chroma kept as its quarter-pel image: subsampled chroma at pel 4
    bool ChromaImage() const { return pel == 4 && format.chroma && (format.xr > 1 || format.yr > 1); }
    // Plane sizes each chroma plane takes: its full-pel plane, its four half-pel planes, or its image
    int ChromaPlanes() const { return ChromaImage() ? 16 : pel == 1 ? 1 : 4; }
    // The storage frames' sizes
    int LumaRows() const { return lumaPlanes * hp + 1; }
    int ChromaRows() const { return 2 * ChromaPlanes() * hc + 1; }
    int PyramidWidth() const { return wp; }
    int PyramidRows() const { return (pyramidSamples + wp - 1) / wp + 1; }

    // Which filter is to read the super: the search (Analyse, AnalyseMany), or the filters that
    // compensate motion with its vectors (Degrain, FlowInter, FlowFPS)
    enum class Use { Search, Compensation };
    // What they implement so far, either of them: Gray, 4:2:0 or 4:4:4 of 8 to 16-bit or float samples
    // at any pel, square blocks of 8, 16 or 32 with the same overlap and padding either way, the
    // padding even with subsampled chroma. Empty when the layout is one of those, else what it lacks.
    std::string Unsupported(Use use) const;

    bool operator==(const SuperLayout &o) const;
};

// The GPU frames mvgpu.Super attaches to a frame, each holding its own reference
struct SuperFrames {
    const VSFrame *luma = nullptr;
    const VSFrame *chroma = nullptr;  // null for Gray
    const VSFrame *pyramid = nullptr; // null without coarse levels

    void Free(const VSAPI *vsapi);
};

// Attaches the frames and the layout's description; the properties take their own references
void ExportSuper(VSFrame *dst, const SuperLayout &layout, const SuperFrames &frames, const std::string &prefix, const VSAPI *vsapi);

// The layout a super clip's frames were made with, read from its first frame; throws when the
// clip doesn't come from mvgpu.Super with this prefix
SuperLayout ImportSuperLayout(VSNode *node, const std::string &prefix, const VSAPI *vsapi);

// The frames attached to a frame of such a clip, new references; false when they're missing
bool GetSuperFrames(const VSFrame *frame, const SuperLayout &layout, const std::string &prefix, SuperFrames &out, const VSAPI *vsapi);

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

// Read from the clip's first frame; throws when a property is missing
VectorInfo ReadVectorInfo(VSNode *node, const std::string &prefix, const VSAPI *vsapi);

// The analysis description mvu.Analyse attaches, under the same names, plus the vector frame when
// there is one: a 32-bit record (x, y, SAD, 0) per block, vectors in 1 / pel pixels, a row of records
// per row of blocks. chroma: whether the SADs count chroma.
void ExportAnalysis(VSFrame *dst, const SuperLayout &layout, int delta, bool chroma, const VSFrame *vectors, const std::string &prefix, const VSAPI *vsapi);

// The vector frame attached to an analysis frame, a new reference, or null
const VSFrame *GetAnalysisVectors(const VSFrame *frame, const std::string &prefix, const VSAPI *vsapi);

// mvu's argument helpers: an "h" or "h,v" list argument (absent -> the defaults, one value -> v = h),
// and the block size and overlap rules
void GetPairArgument(int &h, int &v, const char *name, int defaultH, int defaultV, const VSMap *in, const VSAPI *vsapi);
void CheckBlockSize(int blkX, int blkY, int overlapX, int overlapY, int subSamplingW, int subSamplingH);
