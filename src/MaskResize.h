#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <array>
#include <utility>
#include "Common.h"

// Bilinear upscaling of block-level masks and vector fields to the frame. Block b covers pixels
// [b * step, b * step + blkSize) with step = blkSize - overlap, and its value sits at the block's centre,
// b * step + blkSize / 2, so neighbouring values are exactly `step` pixels apart. Pixels beyond the outermost
// centres repeat the edge block's value.

// The block grid along one axis
struct BlockGridAxis {
    int nBlk = 0;
    int blkSize = 0;
    int step = 0;
};

// Upscales the block values in src (one per block, the whole grid) to the window [x, x + width) x [y, y + height)
// of the plane, which dst points at. Strides in bytes.
void UpscaleBlocks(const BlockGridAxis &gx, const BlockGridAxis &gy, int x, int y, int width, int height, const uint8_t *src, ptrdiff_t srcStride, uint8_t *dst, ptrdiff_t dstStride) noexcept;
void UpscaleBlocks(const BlockGridAxis &gx, const BlockGridAxis &gy, int x, int y, int width, int height, const uint16_t *src, ptrdiff_t srcStride, uint16_t *dst, ptrdiff_t dstStride) noexcept;
void UpscaleBlocks(const BlockGridAxis &gx, const BlockGridAxis &gy, int x, int y, int width, int height, const float *src, ptrdiff_t srcStride, float *dst, ptrdiff_t dstStride) noexcept;

class MaskResizer {
public:
    struct BufferPair {
        const void *src;
        ptrdiff_t srcStride;
        void *dst;
        ptrdiff_t dstStride;
    };

    class MaskTile {
    public:
        int dstX;
        int dstY;
        int dstWidth;
        int dstHeight;
        BlockGridAxis gridX;
        BlockGridAxis gridY;

        template <typename... Pairs>
        void Process(const Pairs&... pairs) const {
            (UpscaleBlocks(gridX, gridY, dstX, dstY, dstWidth, dstHeight, static_cast<const uint16_t *>(pairs.src), pairs.srcStride, static_cast<uint16_t *>(pairs.dst), pairs.dstStride), ...);
        }
    };

    constexpr static int TileSize = 64;
    std::vector<MaskTile> tiles;

    void Init(int nBlkX, int nBlkY, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int dstWidth, int dstHeight);


    static BufferPair MakeBufferPair(const void *src, ptrdiff_t srcStride, void *dst, ptrdiff_t dstStride = 0) {
        return { src, srcStride ? srcStride : GetTileBufferStride(), dst, dstStride ? dstStride : GetTileBufferStride() };
    }

    static constexpr ptrdiff_t GetTileBufferStride() {
        return RoundUpToAlignment(TileSize * sizeof(uint16_t));
    }

    template <size_t N>
    static auto GetTileBuffers() {
        return GetTileBuffersImpl(std::make_index_sequence<N>{});
    }

private:
    static MvuAlignedPtr<uint16_t> GetTileBuffer() {
        return mvu_make_aligned<uint16_t>(GetTileBufferStride() * TileSize);
    }

    template <size_t... Is>
    static auto GetTileBuffersImpl(std::index_sequence<Is...>) {
        return std::array<MvuAlignedPtr<uint16_t>, sizeof...(Is)>{ ((void)Is, GetTileBuffer())... };
    }
};

class PlaneResizer {
public:
    void Init(int dstwidth, int dstheight, int nBlkX, int nBlkY, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int bitsPerSample);
    void Process(uint8_t *dst, ptrdiff_t dststride, const void *src, ptrdiff_t srcstride);
private:
    BlockGridAxis gridX;
    BlockGridAxis gridY;
    int width = 0;
    int height = 0;
    int bitsPerSample = 8;
};
