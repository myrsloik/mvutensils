#pragma once

// Host-side helpers the mvgpu filters share: mvu's overlap windows and scene change thresholds, and
// the taps of the zimg bilinear resizes through which mvu spreads the blocks' values (vectors,
// masks) over the pixels.

#include <cstdint>
#include <vector>

#include "SuperLayout.h"

// mvu's OverlapWindows::Init: the 9 windows of nx x ny blocks overlapping by ox, oy (top left, top
// middle, top right, middle left, ..., bottom right, nx * ny values each, 0 .. 2048), in the same
// float arithmetic
std::vector<int32_t> MakeOverlapWindows(int nx, int ny, int ox, int oy);

// mvu's GetThSCDScaleFactor: what a SAD threshold given for an 8x8 block of 8-bit luma is multiplied
// by for vectors of this analysis, by their block size, chroma and bit depth (float SADs are on the
// 16-bit scale)
double ThSCDScale(const VectorInfo &v);

// mvu's scene change test (ScaleThSCD, IsSceneChange) for vectors of this analysis: a block is badly
// matched when its SAD is above thscd1, scaled, and a frame's vectors are at a scene change when more
// of their blocks than limit are. Throws for arguments outside mvu's ranges.
struct SceneChange {
    int thscd1 = 0;
    int limit = 0;
};
SceneChange ScaleSceneChange(const VectorInfo &v, int64_t thscd1, float thscd2);

// An output sample's taps in a zimg bilinear resize: source samples left and left + 1, weighted c
// and (1 << 14) - c in a 16-bit integer resize, f0 and f1 in a float one
struct ResizeTap {
    int32_t left = 0, c = 0;
    float f0 = 0, f1 = 0;
};

// The taps of zimg's bilinear resize of srcDim samples to dstDim, reading the source from shift on,
// width samples of it: compute_filter and matrix_to_filter in zimg's resize/filter.cpp, in the same
// double arithmetic, with the matrix's sparse rows as RowMatrix keeps them. An upscale has two taps,
// which matrix_to_filter's corrections make add up to 1 << 14 (the float ones as near to 1 as their
// error diffusion gets). Throws for more than two (a downscale).
std::vector<ResizeTap> BilinearTaps(unsigned srcDim, unsigned dstDim, double shift, double width);

// mvu's MaskResizer (FlowInter, FlowFPS, Flow, FlowBlur) for one dimension: srcDim blocks, step
// apart, plus one overlap, laid over the samples they cover and resized to dstDim samples in tiles of
// 64, each tile a resize of its own; each output sample's taps packed as (left << 15) | c
std::vector<int32_t> TileTaps(int srcDim, int dstDim, int step, int overlap);

// mvu's PlaneResizer (the masks) for one dimension: srcDim blocks covering coverDim samples, (srcDim
// - 1) * step + blk, the dstDim of them in the frame resized from them as one plane
std::vector<ResizeTap> PlaneTaps(int srcDim, int dstDim, int coverDim);

// zimg's choice of the order of the passes (resize/resize.cpp's resize_h_first), for every size of
// tile MaskResizer makes of a dstW x dstH plane: all of them horizontally first, all of them
// vertically first, or some one way and some the other
enum class PassOrder { Horizontal, Vertical, Mixed };
PassOrder TilesPassOrder(int srcW, int dstW, int stepX, int overlapX, int srcH, int dstH, int stepY, int overlapY);
// and for PlaneResizer's resize of the whole plane: true when it resizes horizontally first
bool PlaneHorizontalFirst(int srcW, int dstW, int coverW, int srcH, int dstH, int coverH);
