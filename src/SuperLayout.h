#pragma once

// How mvgpu.Super lays out a frame on the GPU, and the frame properties that carry it to Analyse.
//
// mvu.Super stores one pyramid level per plane, laid out for the CPU. mvgpu.Super keeps what the
// kernels read, laid out for them, in three GPU frames attached to each output frame:
//
// - luma: the padded full-pel plane and the three half-pel planes (x + 1/2, y + 1/2, both), one
//   after another, hp rows each, then a spare row; wp bytes wide, its rows whole words apart.
// - chroma: U's four planes, then V's, hc rows each, then a spare row; wc wide.
// - pyramid: the coarse levels 1 .. topLevel the search starts from, every plane of every level
//   inside a border of repeated edge pixels, at the offsets the level table gives; one flat
//   buffer, read as words.
//
// The planes cover the block-aligned frame the analysis grid covers, as mvu.Super's do, so the
// grid is a property of the super clip: blksize and overlap go to Super. The spare rows let the
// SADs read whole words past a plane's last pixel.

#include <cstdint>
#include <string>
#include <vector>

#include <VapourSynth4.h>

// A coarse pyramid level as the kernels read it, matching pyr_common.glsl's Level
struct LevelEntry {
    int32_t w, h, wc, hc;
    int32_t offY, offU, offV, nbx;
    int32_t nby, fieldOff, pad, lambdaOff;
    int32_t frameBytes, fieldTotal; // entry 0 only
    int32_t strideY, strideC;
    int32_t borderY, borderC, reserved[2];
};
static_assert(sizeof(LevelEntry) == 80, "pyr_common.glsl's Level is 20 ints");

struct SuperLayout {
    int width = 0, height = 0;          // the frame
    int blk = 0, overlap = 0, step = 0; // the analysis grid's blocks: blk x blk, step = blk - overlap apart
    int nbx = 0, nby = 0;               // the grid, extended by a column or row while it falls short of the frame, as mvu.Analyse does
    int aw = 0, ah = 0;                 // the block-aligned frame it covers, (nbx - 1) * step + blk
    int pad = 0, padc = 0;              // padding around the planes, luma and chroma
    int wp = 0, hp = 0, wc = 0, hc = 0; // the padded planes: width rounded up to whole words, and rows
    int topLevel = 0;                   // coarse levels 1 .. topLevel; 0 when there is no pyramid
    std::vector<LevelEntry> levels;     // the kernels' level table, entry L for level L, 0 the frame itself
    int pyramidBytes = 0;               // a frame's coarse levels
    int fieldTotal = 0;                 // blocks of one field's coarse levels, finest .. top

    static constexpr int kFinest = 1;    // the coarse level whose field seeds the full-size grid
    static constexpr int kTopWidth = 96; // a frame is halved while the next level stays at least this wide
    // Entries of a coarse level's lambda table: (largest SAD of an 8x8 block with chroma) >> 1, plus one
    static constexpr int kLambdaEntries = 96 * 255 / 2 + 1;

    // 8-bit 4:2:0 frames of width x height; throws when the frame can't be analysed that way
    static SuperLayout Make(int width, int height, int blk, int overlap, int pad, bool pyramid);

    // The storage frames' sizes
    int LumaRows() const { return 4 * hp + 1; }
    int ChromaRows() const { return 8 * hc + 1; }
    int PyramidWidth() const { return wp; }
    int PyramidRows() const { return (pyramidBytes + wp - 1) / wp + 1; }

    bool operator==(const SuperLayout &o) const;
};

// The GPU frames mvgpu.Super attaches to a frame, each holding its own reference
struct SuperFrames {
    const VSFrame *luma = nullptr;
    const VSFrame *chroma = nullptr;
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

// The analysis description mvu.Analyse attaches, under the same names, plus the vector frame when
// there is one: a 32-bit record (x, y, SAD, 0) per block, vectors in half-pels, a row of records
// per row of blocks
void ExportAnalysis(VSFrame *dst, const SuperLayout &layout, int delta, const VSFrame *vectors, const std::string &prefix, const VSAPI *vsapi);

// The vector frame attached to an analysis frame, a new reference, or null
const VSFrame *GetAnalysisVectors(const VSFrame *frame, const std::string &prefix, const VSAPI *vsapi);

// mvu's argument helpers: an "h" or "h,v" list argument (absent -> the defaults, one value -> v = h),
// and the block size and overlap rules
void GetPairArgument(int &h, int &v, const char *name, int defaultH, int defaultV, const VSMap *in, const VSAPI *vsapi);
void CheckBlockSize(int blkX, int blkY, int overlapX, int overlapY, int subSamplingW, int subSamplingH);
