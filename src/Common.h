#pragma once

#include <vector>
#include <stdexcept>
#include <string>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <bit>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#ifdef _WIN32
#include <malloc.h>
#else 
#include <cstdlib>
#endif
#include <VapourSynth4.h>

#define MVU_RESTRICT __restrict

#ifdef _MSC_VER
#define MVU_FORCE_INLINE __forceinline
#define MVU_NOINLINE __declspec(noinline)
#else
#define MVU_FORCE_INLINE inline __attribute__((always_inline))
#define MVU_NOINLINE __attribute__((noinline))
#endif

static constexpr size_t MVU_MEMORY_ALIGN = 64;

// n / d rounded toward zero, exactly like C++'s / for every int n, for a divisor that stays the same for many divisions:
// a multiply and a shift instead of a division instruction (Granlund and Montgomery's round-up method; |n| <= 2^31 keeps
// the product below 2^64). Construct it in baseline-ISA code and pass it to AVX2/AVX-512 kernels: the constructor
// isn't force-inlined and std::bit_width compiled for AVX2 assumes lzcnt, so their files' copy could be the one the
// linker keeps.
class InvariantDivisor {
    int shift;
    uint64_t magic;

    static int ShiftFor(int d) noexcept {
        assert(d >= 1); // before the initialisers shift by 64 or divide by 0
        return 32 + std::bit_width(static_cast<unsigned>(d - 1));
    }
public:
    explicit InvariantDivisor(int d = 1) noexcept : shift(ShiftFor(d)), magic(((uint64_t(1) << shift) + d - 1) / d) {}

    [[nodiscard]] MVU_FORCE_INLINE int Divide(int n) const noexcept {
        const uint64_t a = n < 0 ? uint64_t(0) - static_cast<uint64_t>(n) : static_cast<uint64_t>(n);
        const uint32_t q = static_cast<uint32_t>((a * magic) >> shift);
        return static_cast<int>(n < 0 ? 0u - q : q); // negated unsigned: INT_MIN / 1 doesn't overflow
    }
};

class MVUtensilsError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

template<typename T, size_t U>
[[nodiscard]] constexpr int ARRAY_SIZE(const T (&)[U]) {
    return static_cast<int>(U);
}

template<typename T>
static void VS_CC filterFree(void *instanceData, [[maybe_unused]] VSCore *core, [[maybe_unused]] const VSAPI *vsapi) {
    delete reinterpret_cast<T *>(instanceData);
}

[[nodiscard]] constexpr int RoundUpToAlignment(int value, int alignment = MVU_MEMORY_ALIGN) {
    return ((value + alignment - 1) / alignment) * alignment;
}

constexpr int ERROR_SIZE = 1024;

static constexpr const int MV_DEFAULT_SCD1 = 400;
static constexpr const float MV_DEFAULT_SCD2 = 51.0f;

static constexpr char DEFAULT_MVUTENSILS_PREFIX[] = "MVUtensils";

static inline void mvu_bitblt(void *dstp, ptrdiff_t dst_stride, const void *srcp, ptrdiff_t src_stride, size_t row_size, size_t height) {
    if (height) {
        if (src_stride == dst_stride && src_stride == (ptrdiff_t)row_size) {
            memcpy(dstp, srcp, row_size * height);
        } else {
            const uint8_t *srcp8 = (const uint8_t *)srcp;
            uint8_t *dstp8 = (uint8_t *)dstp;
            size_t i;
            for (i = 0; i < height; i++) {
                memcpy(dstp8, srcp8, row_size);
                srcp8 += src_stride;
                dstp8 += dst_stride;
            }
        }
    }
}

/* returns the biggest integer x such that 2^x <= i */
[[nodiscard]] static constexpr inline int ilog2(int i) noexcept {
    return std::bit_width(static_cast<unsigned>(i)) - 1;
}

template<typename T>
[[nodiscard]] static inline T *mvu_aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return (T *)_aligned_malloc(size, alignment);
#else
    void *tmp = nullptr;
    if (posix_memalign(&tmp, alignment, size))
        tmp = nullptr;
    return (T *)tmp;
#endif
}

static inline void mvu_aligned_free(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

template<typename T> using MvuAlignedPtr = std::unique_ptr<T, decltype(&mvu_aligned_free)>;

// Owning aligned allocation (size in bytes, matching mvu_aligned_malloc); pairs with MvuAlignedPtr.
template<typename T>
[[nodiscard]] static inline MvuAlignedPtr<T> mvu_make_aligned(size_t size) {
    return { mvu_aligned_malloc<T>(size, MVU_MEMORY_ALIGN), mvu_aligned_free };
}

// Resolve the field order for frame n from its frame properties.
// Reads _Field from props. If the property is missing, throws std::runtime_error
// if requireField is true and tff_exists is false.
// When tff_exists is true, _Field is ignored entirely and tff XOR-flipped by frame
// parity is used instead.
[[nodiscard]] inline bool GetTopField(const VSFrame *propsSrc, int n, bool tff_exists, bool tff, bool requireField, const VSAPI *vsapi) {
    int err;
    const VSMap *props = vsapi->getFramePropertiesRO(propsSrc);
    bool top_field = !!vsapi->mapGetInt(props, "_Field", 0, &err);
    if (err && requireField && !tff_exists)
        throw std::runtime_error("_Field property not found in input frame. Therefore, you must pass tff argument");
    if (tff_exists)
        top_field = tff ^ ((n % 2) != 0);
    return top_field;
}

// Frame requests past the end of a clip are clamped to its last frame, so an input shorter than the clip it is
// paired with would silently deliver the wrong frames
inline void CheckClipLength(VSNode *node, const char *name, int minFrames, const char *reference, const VSAPI *vsapi) {
    if (vsapi->getVideoInfo(node)->numFrames < minFrames)
        throw std::runtime_error(std::string(name) + " must have at least as many frames as " + reference);
}

template <typename PixelType>
[[nodiscard]] static constexpr PixelType AveragePixels(PixelType p1, PixelType p2) noexcept {
    if constexpr (std::is_integral_v<PixelType>)
        return (p1 + p2 + 1) >> 1;
    else
        return (p1 + p2) * 0.5f;
}

template <typename PixelType>
[[nodiscard]] static constexpr PixelType AveragePixels(PixelType p1, PixelType p2, PixelType p3, PixelType p4) noexcept {
    if constexpr (std::is_integral_v<PixelType>)
        return (p1 + p2 + p3 + p4 + 2) >> 2;
    else
        return (p1 + p2 + p3 + p4) * 0.25f;
}

template <typename T>
[[nodiscard]] static constexpr T ClampIntToRange(T p, int maxVal) noexcept {
    if constexpr (std::is_integral_v<T>)
        return std::clamp<T>(p, 0, maxVal);
    else
        return p;
}

template <int rShift, int roundBias, typename T>
[[nodiscard]] static constexpr T ShiftDivide(T p) noexcept {
    if constexpr (std::is_integral_v<T>)
        return (p + roundBias) >> rShift;
    else
        return p / (1 << rShift);
}

template <typename PixelType>
[[nodiscard]] static constexpr int PixelMaxValue(int bitsPerSample) noexcept {
    if constexpr (std::is_integral_v<PixelType>)
        return (1 << bitsPerSample) - 1;
    else
        return 1;
}

template<typename T>
[[nodiscard]] static constexpr T SelectOnBitsPerSample(int bitsPerSample, T o8, T o16, T o32) noexcept {
    if (bitsPerSample == 8)
        return o8;
    else if (bitsPerSample == 32)
        return o32;
    else
        return o16;
}