
#include "MaskResize.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <utility>

// Between two neighbouring block centres there are exactly `step` pixels, all with the same weights, so a block row is
// expanded across the output span by span: one pair of values per span and one weight table for all of them. The
// output is made in strips at most Strip pixels wide: two expanded block rows are kept, and each output row is a blend
// of those two, so nothing image-sized is needed. Float arithmetic, integer outputs rounded to nearest.

namespace {

constexpr int Strip = 256;
constexpr int MaxStep = 128; // the largest block size, with no overlap

// the first pixel whose centre (x + 0.5) is at or past the centre of block b
int Start(const BlockGridAxis &g, int b) noexcept {
    return b * g.step + g.blkSize / 2;
}

// the weight of block b + 1 at pixel Start(b) + j
float Weight(const BlockGridAxis &g, int j) noexcept {
    return static_cast<float>(2 * j + 1 - (g.blkSize & 1)) * (0.5f / static_cast<float>(g.step));
}

// block values v[0, nBlk) expanded to the pixels [x0, x0 + w), plus bias; weights[j] = Weight(g, j)
template <typename T>
void ExpandRow(const BlockGridAxis &g, const float *weights, const T *v, int x0, int w, float bias, float *out) noexcept {
    const int end = x0 + w;
    int x = x0;
    for (; x < std::min(Start(g, 0), end); ++x)
        out[x - x0] = static_cast<float>(v[0]) + bias;
    for (int b = (x - Start(g, 0)) / g.step; b < g.nBlk - 1 && x < end; ++b) {
        const float a = static_cast<float>(v[b]) + bias;
        const float d = static_cast<float>(v[b + 1]) - static_cast<float>(v[b]);
        const float *MVU_RESTRICT wb = weights + (x - Start(g, b));
        float *MVU_RESTRICT o = out + (x - x0);
        const int n = std::min(Start(g, b + 1), end) - x;
        for (int k = 0; k < n; ++k)
            o[k] = a + d * wb[k];
        x += n;
    }
    for (; x < end; ++x)
        out[x - x0] = static_cast<float>(v[g.nBlk - 1]) + bias;
}

// x carries the rounding bias and lies within the inputs' range. The 8-bit clamp lets the conversion use saturating
// packs; a 16-bit one wouldn't (SSE2 has no unsigned 32 -> 16-bit pack, and LLVM folds any bias back into the clamp)
template <typename T>
T ToPixel(float x) noexcept {
    if constexpr (std::is_same_v<T, uint8_t>)
        return static_cast<uint8_t>(std::clamp(static_cast<int32_t>(x), 0, 255));
    else if constexpr (std::is_integral_v<T>)
        return static_cast<T>(static_cast<int32_t>(x));
    else
        return x;
}

// strides in elements
template <typename T>
void Upscale(const BlockGridAxis &gridX, const BlockGridAxis &gridY, int x0, int y0, int width, int height, const T *src, ptrdiff_t srcStride, T *dst, ptrdiff_t dstStride) noexcept {
    // local copies: without strict aliasing (clang-cl's default) any store could otherwise change the referenced grid,
    // making the loops reload it
    const BlockGridAxis gx = gridX, gy = gridY;
    const float bias = std::is_integral_v<T> ? 0.5f : 0.0f;
    const int yEnd = y0 + height;

    float weights[MaxStep];
    for (int j = 0; j < gx.step; ++j)
        weights[j] = Weight(gx, j);

    for (int sx = 0; sx < width; sx += Strip) {
        const int w = std::min(Strip, width - sx);
        float rows[2][Strip];
        float diff[Strip];
        float *top = rows[0], *bottom = rows[1];
        int topRow = -1;

        auto expand = [&](int by, float *out) {
            ExpandRow(gx, weights, src + by * srcStride, x0 + sx, w, bias, out);
        };
        auto flat = [&](int y) {
            T *MVU_RESTRICT d = dst + (y - y0) * dstStride + sx;
            for (int x = 0; x < w; ++x)
                d[x] = ToPixel<T>(top[x]);
        };

        int y = y0;
        // rows above the first block row's centre
        if (y < Start(gy, 0)) {
            expand(0, top);
            topRow = 0;
            for (; y < std::min(Start(gy, 0), yEnd); ++y)
                flat(y);
        }
        // rows between two block rows' centres
        for (int b = (y - Start(gy, 0)) / gy.step; b < gy.nBlk - 1 && y < yEnd; ++b) {
            if (topRow != b)
                expand(b, top);
            expand(b + 1, bottom);
            for (int x = 0; x < w; ++x)
                diff[x] = bottom[x] - top[x];
            const int start = Start(gy, b);
            for (; y < std::min(Start(gy, b + 1), yEnd); ++y) {
                const float f = Weight(gy, y - start);
                T *MVU_RESTRICT d = dst + (y - y0) * dstStride + sx;
                for (int x = 0; x < w; ++x)
                    d[x] = ToPixel<T>(top[x] + diff[x] * f);
            }
            std::swap(top, bottom);
            topRow = b + 1;
        }
        // rows below the last block row's centre
        if (y < yEnd) {
            if (topRow != gy.nBlk - 1)
                expand(gy.nBlk - 1, top);
            for (; y < yEnd; ++y)
                flat(y);
        }
    }
}

BlockGridAxis MakeAxis(int nBlk, int blkSize, int overlap) {
    const int step = blkSize - overlap;
    if (nBlk < 1 || step < 1 || step > MaxStep)
        throw std::runtime_error("unsupported block grid for mask resizing");
    return { nBlk, blkSize, step };
}

} // namespace

void UpscaleBlocks(const BlockGridAxis &gx, const BlockGridAxis &gy, int x, int y, int width, int height, const uint8_t *src, ptrdiff_t srcStride, uint8_t *dst, ptrdiff_t dstStride) noexcept {
    Upscale(gx, gy, x, y, width, height, src, srcStride, dst, dstStride);
}

void UpscaleBlocks(const BlockGridAxis &gx, const BlockGridAxis &gy, int x, int y, int width, int height, const uint16_t *src, ptrdiff_t srcStride, uint16_t *dst, ptrdiff_t dstStride) noexcept {
    Upscale(gx, gy, x, y, width, height, src, srcStride / static_cast<ptrdiff_t>(sizeof(uint16_t)), dst, dstStride / static_cast<ptrdiff_t>(sizeof(uint16_t)));
}

void UpscaleBlocks(const BlockGridAxis &gx, const BlockGridAxis &gy, int x, int y, int width, int height, const float *src, ptrdiff_t srcStride, float *dst, ptrdiff_t dstStride) noexcept {
    Upscale(gx, gy, x, y, width, height, src, srcStride / static_cast<ptrdiff_t>(sizeof(float)), dst, dstStride / static_cast<ptrdiff_t>(sizeof(float)));
}

void MaskResizer::Init(int nBlkX, int nBlkY, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int dstWidth, int dstHeight) {
    const BlockGridAxis gridX = MakeAxis(nBlkX, nBlkSizeX, nOverlapX);
    const BlockGridAxis gridY = MakeAxis(nBlkY, nBlkSizeY, nOverlapY);
    int nWidthTiles = (dstWidth + TileSize - 1) / TileSize;
    int nHeightTiles = (dstHeight + TileSize - 1) / TileSize;
    tiles.clear();
    tiles.reserve(nWidthTiles * nHeightTiles);

    for (int y = 0; y < nHeightTiles; y++) {
        for (int x = 0; x < nWidthTiles; x++) {
            MaskTile tile;
            tile.dstX = x * TileSize;
            tile.dstY = y * TileSize;
            tile.dstWidth = std::min(TileSize, dstWidth - tile.dstX);
            tile.dstHeight = std::min(TileSize, dstHeight - tile.dstY);
            tile.gridX = gridX;
            tile.gridY = gridY;
            tiles.push_back(tile);
        }
    }
}

void PlaneResizer::Init(int dstwidth, int dstheight, int nBlkX, int nBlkY, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int bitsPerSample_) {
    gridX = MakeAxis(nBlkX, nBlkSizeX, nOverlapX);
    gridY = MakeAxis(nBlkY, nBlkSizeY, nOverlapY);
    width = dstwidth;
    height = dstheight;
    bitsPerSample = bitsPerSample_;
}

void PlaneResizer::Process(uint8_t *dst, ptrdiff_t dststride, const void *src, ptrdiff_t srcstride) {
    if (bitsPerSample == 8)
        UpscaleBlocks(gridX, gridY, 0, 0, width, height, static_cast<const uint8_t *>(src), srcstride, dst, dststride);
    else if (bitsPerSample == 32)
        UpscaleBlocks(gridX, gridY, 0, 0, width, height, static_cast<const float *>(src), srcstride, reinterpret_cast<float *>(dst), dststride);
    else
        UpscaleBlocks(gridX, gridY, 0, 0, width, height, static_cast<const uint16_t *>(src), srcstride, reinterpret_cast<uint16_t *>(dst), dststride);
}
