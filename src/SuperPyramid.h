#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <VapourSynth4.h>

class SuperPyramidError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};


// Reads an "h" or "h,v" list argument (T = int, int64_t or float): absent -> (defaultH, defaultV); a single
// value -> v = h; more than two values throws. Definition + instantiations live in SuperPyramid.cpp.
template <typename T>
void GetHVPairArgument(T &h, T &v, const char *name, std::type_identity_t<T> defaultH, std::type_identity_t<T> defaultV, const VSMap *in, const VSAPI *vsapi);
void CheckBlkSize(int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int subSamplingW, int subSamplingH, bool useSatd = false);

enum class SharpParam {
    Bilinear,
    Bicubic,
    Wiener
};

enum class RFilterParam {
    Simple,
    Bilinear,
    Cubic
};

// Important note on data storage: Pyramid level 0 isn't just the original unreduced frame with padding,
// if nPel > 1 it also stores the relevant offset images in pPlane[1..3] or pPlane[1..15] for pel 2 and 4 respectively
// If an external pel clip is supplied the data at (0, 0) is discarded and the original size level 0 frame data is used instead
// The pPlane allocations are completely independent of each other and can be exported

class PyramidPlane {
    friend class FramePyramid;
public:
    // Almost all manipulations are done on the VSFrame storage before assigning it here
    const uint8_t *pPlane[16] = {};

    // Both in bytes
    ptrdiff_t nPitch = -1;
    ptrdiff_t nOffsetPadding = -1;
    ptrdiff_t subPelPlaneOffset = -1; // Only used for pel 2 and 4, the offset between the planes in pPlane[1..3] or pPlane[1..15]

    // This is the original width and height, only used when generating planes and invalid when reonstructed from frame properties
    int nRealWidth = -1;
    int nRealHeight = -1;

    // This is the width and height of the plane after adding nBlkSizePadX and nBlkSizePadY
    int nWidth = -1;
    int nHeight = -1;

    int nPaddedWidth = -1;
    int nPaddedHeight = -1;

    int nHPadding = -1;
    int nVPadding = -1;

    int nHPaddingPel = -1;
    int nVPaddingPel = -1;

    int nPel = 1; // 1 on all planes except the topmost where it can be 1, 2 or 4

    template<typename PixelType>
    const uint8_t *GetAbsolutePointerPel1(int nX, int nY) const noexcept {
        return pPlane[0] + nX * sizeof(PixelType) + nY * nPitch;
    }

    template<typename PixelType>
    const uint8_t *GetAbsolutePointerPel2(int nX, int nY) const noexcept {
        int idx = (nX & 1) | ((nY & 1) << 1);
        nX >>= 1;
        nY >>= 1;
        return pPlane[idx] + nX * sizeof(PixelType) + nY * nPitch;
    }

    template<typename PixelType>
    const uint8_t *GetAbsolutePointerPel4(int nX, int nY) const noexcept {
        int idx = (nX & 3) | ((nY & 3) << 2);
        nX >>= 2;
        nY >>= 2;
        return pPlane[idx] + nX * sizeof(PixelType) + nY * nPitch;
    }

    template<typename PixelType>
    const uint8_t *GetAbsolutePointer(int nX, int nY) const noexcept {
        if (nPel == 1)
            return pPlane[0] + nX * sizeof(PixelType) + nY * nPitch;
        else if (nPel == 2) {
            int idx = (nX & 1) | ((nY & 1) << 1);
            nX >>= 1;
            nY >>= 1;
            return pPlane[idx] + nX * sizeof(PixelType) + nY * nPitch;
        } else { // nPel = 4
            int idx = (nX & 3) | ((nY & 3) << 2);
            nX >>= 2;
            nY >>= 2;
            return pPlane[idx] + nX * sizeof(PixelType) + nY * nPitch;
        }
    }

    template<typename PixelType>
    const uint8_t *GetPointer(int nX, int nY) const noexcept {
        assert(nHPaddingPel >= 0 && nVPaddingPel >= 0);
        return GetAbsolutePointer<PixelType>(nX + nHPaddingPel, nY + nVPaddingPel);
    }

private:
    const VSFrame *storage = nullptr;

    template<typename PixelType>
    void CopyAndPadPlane(const VSFrame *src, int plane, int hPad, int vPad, int nBlkSizePadX, int nBlkSizePadY, int nPel, VSCore *core, const VSAPI *vsapi) noexcept;

    template<typename PixelType>
    void ReducePlane(const PyramidPlane &src, int xRatioUV, int yRatioUV, RFilterParam rFilter, uint8_t *tempBuffer, VSCore *core, const VSAPI *vsapi) noexcept;

    template<typename PixelType>
    void GeneratePelPlanes(SharpParam sharp, const VSAPI *vsapi) noexcept;

    template<typename PixelType>
    void SetExternalPelPlanes(const VSFrame *pelFrame, int plane, const VSAPI *vsapi);

    void FromExternalPlane(const VSFrame *planeFrame, int hPad, int vPad, int tailGuardLines, const VSAPI *vsapi) noexcept;
    void FromExternalPelPlanes(const VSFrame *pelFrame, int pel, int hPad, int vPad, const VSAPI *vsapi);

    template<typename PixelType>
    void SetExtPel2(const VSFrame *pelFrame, int plane, const VSAPI *vsapi);

    template<typename PixelType>
    void SetExtPel4(const VSFrame *pelFrame, int plane, const VSAPI *vsapi);

    template<typename PixelType>
    void PadPlaneData(int plane) noexcept;
};

class FramePyramidLevel {
public:
    PyramidPlane planes[3] = {};
};

class FramePyramid {
public:
    std::vector<FramePyramidLevel> pyramidLevels; // 0 is the orignal padded frame, higher levels are n times reduced
    int nPel = 1; // Why is nPel stored here as well? It's trivial to get with an accessor from the top level plane

    int nWidth[3] = {}; // The original width of the frame, including padding to reach a multiple of blksize-overlap (true input dimensions are nWidth[plane] - nBlkSizePadX
    int nHeight[3] = {};

    int nRealWidth[3] = {}; // The original width of the input frame
    int nRealHeight[3] = {}; // The original height of the input frame

    int nHPad[3] = {};
    int nVPad[3] = {};

    int nBlkSizePadX[3] = {};
    int nBlkSizePadY[3] = {};

    int xRatioUV = 1;
    int yRatioUV = 1;

    int nBlkSizeX = -1;
    int nBlkSizeY = -1;
    int nOverlapX = -1;
    int nOverlapY = -1;

    int nLevels = 0;

    bool chroma;

    int bitsPerSample = -1;

private:
    const VSFrame *serializedData = nullptr;
    const VSAPI *vsapi;
    void FreeFrames() noexcept;
    void LoadFrameData(const VSFrame *srcFrame, int maxLevel, const std::string &prefix);
    void GeneratePelPlanes(SharpParam sharp, const VSAPI *vsapi);
    void SetExternalPelPlanes(const VSFrame *srcFrame, const VSFrame *pelFrame, const VSAPI *vsapi);
    void SharedInit(const VSFrame *srcFrame, int levels, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int hPad, int vPad, RFilterParam rFilter, int pel, VSCore *core, const VSAPI *vsapi);
public:
    // Constructor to build from source frame with generated subpel frame, does not take ownership of srcFrame
    FramePyramid(const VSFrame *srcFrame, int levels, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int hPad, int vPad, RFilterParam rFilter, int pel, SharpParam sharp, VSCore *core, const VSAPI *vsapi);
    // Constructor to build from source frame with supplied subpel frame, does not take ownership of srcFrame
    FramePyramid(const VSFrame *srcFrame, int levels, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int hPad, int vPad, RFilterParam rFilter, int pel, const VSFrame *pelFrame, VSCore *core, const VSAPI *vsapi);

    // Constructor to reconstruct from frame properties, takes ownership of srcFrame and free it even if the constructor throws
    // You can pass maxLevel = -1 to load all levels, maxLevel = 0 to load only metadata and no planes, maxLevel > 0 to load that many levels, note that only analyse uses more than 1 level
    FramePyramid(const VSFrame *srcFrame, int maxLevel, const std::string &prefix, const VSAPI *vsapi);

    // Constructor to load metadata from a node and do nothing else
    FramePyramid(VSNode *node, const std::string &prefix, const VSAPI *vsapi);

    ~FramePyramid();

    void ExportFrameData(VSFrame *dst, const std::string &prefix) const noexcept; // Stores all levels as frame properties of the output frame, note that each used plane is stored as a separate property
    const FramePyramidLevel &GetLevel(int level) const noexcept;
    bool IsCompatible(const FramePyramid &other) const noexcept;
    bool IsCompatibleWithSource(const VSVideoInfo *vi) const noexcept;
    static int GetMaxLevelsForBlockSize(int width, int height, int xRatioUV, int yRatioUV, int blkSizeX, int blkSizeY, int overlapX, int overlapY, int padX, int padY) noexcept;
};