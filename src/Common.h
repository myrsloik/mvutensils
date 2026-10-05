#pragma once

#include <vector>
#include <stdexcept>
#include <string>
#include <cstring>
#include <algorithm>
#include <bit>
#include <memory>
#include <VapourSynth4.h>

template<typename T, size_t U>
[[nodiscard]] constexpr int ARRAY_SIZE(const T (&)[U]) {
    return static_cast<int>(U);
}

template<typename T>
static void VS_CC filterFree(void *instanceData, [[maybe_unused]] VSCore *core, [[maybe_unused]] const VSAPI *vsapi) {
    delete reinterpret_cast<T *>(instanceData);
}

// Frame requests past the end of a clip are clamped to its last frame, so an input shorter than the clip it is
// paired with would silently deliver the wrong frames
inline void CheckClipLength(VSNode *node, const char *name, int minFrames, const char *reference, const VSAPI *vsapi) {
    if (vsapi->getVideoInfo(node)->numFrames < minFrames)
        throw std::runtime_error(std::string(name) + " must have at least as many frames as " + reference);
}

static constexpr const int MV_DEFAULT_SCD1 = 400;
static constexpr const float MV_DEFAULT_SCD2 = 51.0f;

static constexpr char DEFAULT_MVUTENSILS_PREFIX[] = "MVUtensils";
// mvgpu's frames are laid out for the GPU (a super frame is its storage, a vector frame its records),
// so they get their own prefix: an mvu clip and an mvgpu clip never mistake each other's properties.
// mvgpu.ToMVU and mvgpu.FromMVU convert vector clips between the two (VectorConvert.cpp).
static constexpr char DEFAULT_MVGPUTENSILS_PREFIX[] = "MVGPUtensils";

/* returns the biggest integer x such that 2^x <= i */
[[nodiscard]] static constexpr inline int ilog2(int i) noexcept {
    return std::bit_width(static_cast<unsigned>(i)) - 1;
}
