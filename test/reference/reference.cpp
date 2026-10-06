// reference.cpp
//
// The CPU reference of mvgpu.Analyse and mvgpu.AnalyseMany: the GPU search step for step, in plain
// C++, so that test/check_reference.py can compare the plugin's vectors with it byte for byte. It
// reads raw planar frames of 8 to 16-bit samples and builds what mvgpu.Super builds from them
// (mvu.Super's planes:
// edge-padded, half-pel Wiener planes, quarter samples computed from them at pel 4; and the coarse
// pyramid, halved with rfilter 1's filter), then refines every field:
//
// - the coarse search on the pyramid: an exhaustive search of +-32 px at the smallest level searched
//   (--levels), then at every finer level down to 1/2 seeds from the level above and two pairs of
//   checkerboard passes under Analyse's cost, lambda times 2^(plevel * level), with 8x8 blocks at
//   every level (Pyramid);
// - a seed list per block of the full-size grid: zero, the field's median (but with --globalmv 0),
//   the chained and inverted vectors of fields refined before (AnalyseMany only) and the finest
//   level's vectors around the block (BuildSeeds);
// - the seeds measured, a pair of checkerboard passes, the wide fallback search for the blocks still
//   above badsad and one more pair, then (but at pel 1) the 8 half-pel positions around each vector,
//   and at pel 4 the 8 quarter-pel positions around that (Refine). The passes try the positions within
//   pelsearch / pel pixels around each block's winner (--pelsearch, pel by default: the 8 one pixel
//   around).
//
// mvu's penalties, in 256ths of a candidate's SAD, which its cost takes on besides: pzero on the zero
// seed, pglobal on the median (the global vector), pnew on the positions a search steps to (the
// passes' positions, the top level's window but zero, the fallback's, the sub-pel steps'); none on the
// predictors (the coarse vectors, the chained and inverted ones, the neighbours', the block's own).
// Every candidate counts with its own penalty, so a predictor equal to zero is a seed of its own, and
// of equally cheap candidates the earliest wins.
//
// The SAD is MVUtensils': luma plus U and V at the chroma block size, the chroma vector the luma
// vector divided by the subsampling, toward zero; with --chroma 0 luma only; with --satd luma's is
// mvu's SATD instead (BlockSatd), at the full-size grid (the coarse search keeps the SAD). The cost is
// SAD + ((lambda * |v - p|^2) >> 8), p the component-wise median of the four neighbours, lambda
// relaxed by (lsad / (lsad + worst neighbour SAD / 2))^2, the worst SAD / 2 taken in steps of
// 2^(bits - 8) (Analyse.cpp tabulates lambda by it). mvlambda, lsad and badsad are scaled to the bit
// depth as mvu.Analyse scales them, by (2^bits - 1) / 255. A SAD is unsigned, as mvu's is: the
// largest blocks' reach 2^32 (128x128 with 4:4:4 chroma at 16 bits, or with --satd). Float SADs, of
// samples outside their range, can pass the largest SAD a block's pixels make, which the worst
// neighbour SAD is capped at (the lambda tables' last entries), and 32 bits, at which they saturate.
//
// Build:  clang-cl /nologo /O2 /std:c++20 /EHsc reference.cpp   (or meson compile mvgpu_reference)
//
// Usage:  reference --src frames.yuv --size WxH --frames N --out vectors.bin [--format 420|422|440|444|gray]
//                   [--bits 8] [--blksize 16] [--blksizev 16] [--overlap 8] [--overlapv 8] [--pad 16]
//                   [--padv 16] [--superblksize 16] [--superblksizev 16] [--superoverlap 8]
//                   [--superoverlapv 8] [--onelevel] [--pel 2] [--radius 2] [--delta 1] [--standalone] [--chroma 1]
//                   [--plevel 1] [--mvlambda 1000] [--lsad 400] [--badsad 1000] [--badrange 40]
//                   [--badstep 2] [--satd] [--levels 0] [--globalmv 1] [--pnew 25] [--pzero 25] [--pglobal 0]
//                   [--pelsearch 2] [--threads 16]
//
// The arguments are mvgpu.Super's and mvgpu.AnalyseMany's (--blksizev, --overlapv and --padv the
// vertical ones, the horizontal ones by default; --superblksize and --superoverlap, and their
// vertical ones, the super's grid where it isn't the grid analysed; --onelevel: a super without the
// coarse levels, which a frame narrower than 192 pixels doesn't have either, so no coarse search, no
// median and no coarse seeds; --standalone: an Analyse per delta, without chained or inverted seeds);
// badsad is per 8x8 block, as mvgpu takes it. The frames are
// raw planar Y, U, V, at 4:2:0, 4:2:2, 4:4:0 or 4:4:4, or Y alone for gray (whose SADs are luma's
// alone, as with --chroma 0), bytes at 8 bits, 16-bit little-endian samples at 9 to 16 and 32-bit
// floats at 32 (--bits); with subsampled chroma the padding and the overlap must be even, as mvgpu
// has them. Float frames take mvu's float SADs (SadRawF, SatdRawF), its float operations in the order
// its AVX-512 build takes them, of the super's samples built in float as mvu.Super builds them, read
// where mvgpu reads what Super stored: the half-pel planes, and for subsampled chroma at pel 4 its
// quarter-pel image; each plane's SAD scaled to the 16-bit range as mvu scales it, everything after
// as at 16 bits. The output holds
// every field (n, d) as six 32-bit ints, n, d, nbx, nby, pel and 1, then the nbx * nby x
// components, the y components and the SADs.
//
// For tracking down a difference, two environment variables make it do something else once the
// frames are read: REFERENCE_PROBE=N,D,BX,BY,VX,VY prints the SAD of block (BX, BY) of frame N
// against frame N + D at vector (VX, VY), and REFERENCE_CHECK_SADS=FILE recomputes the SAD of every
// block of every field in a vector file (check_reference.py --dump writes mvgpu's) at its vector.

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#define REFERENCE_SSE2
#endif

// The float arithmetic unfused, each operation rounded, as the GPU's precise operations and
// mvu.Super's code do it
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

namespace {

[[noreturn]] void Die(const std::string &what) {
    fprintf(stderr, "error: %s\n", what.c_str());
    exit(1);
}

int gThreads = 16;

void ParallelFor(int count, const std::function<void(int)> &body) {
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    const int t = std::max(1, std::min(gThreads, count));
    for (int i = 0; i < t; ++i)
        pool.emplace_back([&] {
            for (int r; (r = next.fetch_add(1)) < count;)
                body(r);
        });
    for (auto &th : pool)
        th.join();
}

// The search's constants, mvgpu's (Analyse.cpp, SuperLayout.h)
constexpr int kPairs = 1;       // checkerboard pass pairs before the fallback
constexpr int kFinest = 1;      // the coarse level whose field seeds the full-size grid
constexpr int kTopRadius = 32;  // the exhaustive search's radius at the top level, px
constexpr int kLevelPairs = 2;  // checkerboard pass pairs at every coarse level
constexpr int kTopWidth = 96;   // a frame is halved while the next level stays at least this wide
constexpr int kMaxSeeds = 10;   // seed slots per block

// ---------------------------------------------------------------------------------------------
// The super's planes

// One plane of one frame as mvu.Super pel 2 holds it: edge-padded, plus the three half-pel planes;
// samples of type T, bytes, 16 bits or floats
template <typename T>
struct Plane {
    int w = 0, h = 0;    // padded size
    std::vector<T> p[4]; // full, x + 1/2, y + 1/2, both
};

int Avg(int a, int b) {
    return (a + b + 1) >> 1;
}

int Wiener(int m0, int m1, int m2, int m3, int m4, int m5, int pixelMax) {
    const int m = (((m2 + m3) * 4 - (m1 + m4)) * 5 + m0 + m5 + 16) >> 5;
    return std::clamp(m, 0, pixelMax);
}

// SuperPyramid.cpp's HorizontalWiener and VerticalWiener, border handling included
template <typename T>
void HorizontalWiener(T *dst, const T *src, int w, int h, int pixelMax) {
    for (int j = 0; j < h; ++j) {
        const T *s = src + static_cast<size_t>(j) * w;
        T *d = dst + static_cast<size_t>(j) * w;
        d[0] = static_cast<T>(Avg(s[0], s[1]));
        d[1] = static_cast<T>(Avg(s[1], s[2]));
        for (int i = 2; i < w - 4; ++i)
            d[i] = static_cast<T>(Wiener(s[i - 2], s[i - 1], s[i], s[i + 1], s[i + 2], s[i + 3], pixelMax));
        for (int i = w - 4; i < w - 1; ++i)
            d[i] = static_cast<T>(Avg(s[i], s[i + 1]));
        d[w - 1] = s[w - 1];
    }
}

template <typename T>
void VerticalWiener(T *dst, const T *src, int w, int h, int pixelMax) {
    auto row = [&](const T *base, int j) { return base + static_cast<size_t>(j) * w; };
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < w; ++i)
            dst[static_cast<size_t>(j) * w + i] = static_cast<T>(Avg(row(src, j)[i], row(src, j + 1)[i]));
    for (int j = 2; j < h - 4; ++j) {
        const T *r0 = row(src, j - 2), *r1 = row(src, j - 1), *r2 = row(src, j), *r3 = row(src, j + 1), *r4 = row(src, j + 2), *r5 = row(src, j + 3);
        T *d = dst + static_cast<size_t>(j) * w;
        for (int i = 0; i < w; ++i)
            d[i] = static_cast<T>(Wiener(r0[i], r1[i], r2[i], r3[i], r4[i], r5[i], pixelMax));
    }
    for (int j = h - 4; j < h - 1; ++j)
        for (int i = 0; i < w; ++i)
            dst[static_cast<size_t>(j) * w + i] = static_cast<T>(Avg(row(src, j)[i], row(src, j + 1)[i]));
    memcpy(dst + static_cast<size_t>(h - 1) * w, row(src, h - 1), w * sizeof(T));
}

// w x h pixels of src, extended to the block-aligned aw x ah and padded by padX and padY, repeating
// the edge pixels, as CopyAndPadPlane does; then its half-pel planes
template <typename T>
Plane<T> MakePlane(const T *src, int w, int h, int padX, int padY, int aw, int ah, int pixelMax) {
    Plane<T> pl;
    pl.w = aw + 2 * padX;
    pl.h = ah + 2 * padY;
    for (auto &p : pl.p)
        p.resize(static_cast<size_t>(pl.w) * pl.h);
    T *d = pl.p[0].data();
    for (int y = 0; y < pl.h; ++y) {
        const int sy = std::clamp(y - padY, 0, h - 1);
        for (int x = 0; x < pl.w; ++x)
            d[static_cast<size_t>(y) * pl.w + x] = src[static_cast<size_t>(sy) * w + std::clamp(x - padX, 0, w - 1)];
    }
    HorizontalWiener(pl.p[1].data(), pl.p[0].data(), pl.w, pl.h, pixelMax);
    VerticalWiener(pl.p[2].data(), pl.p[0].data(), pl.w, pl.h, pixelMax);
    HorizontalWiener(pl.p[3].data(), pl.p[2].data(), pl.w, pl.h, pixelMax);
    return pl;
}

// pel 4: the quarter-pel sample (X, Y) of a padded plane, in quarter pels, from its half-pel planes
// exactly as mvu.Super's quarter planes hold it (SuperPyramid.cpp's GeneratePelQuarters): on the
// half-pel grid that plane's sample; otherwise the rounded average of its two neighbours on the
// grid, diagonally the average of the two vertical averages
template <typename T>
int QuarterSample(const Plane<T> &pl, int X, int Y) {
    auto at = [&](int Xh, int Yh) { return static_cast<int>(pl.p[(Xh & 1) | ((Yh & 1) << 1)][static_cast<size_t>(Yh >> 1) * pl.w + (Xh >> 1)]); };
    const int Xa = X >> 1, Xb = Xa + (X & 1), Ya = Y >> 1, Yb = Ya + (Y & 1);
    return Avg(Avg(at(Xa, Ya), at(Xa, Yb)), Avg(at(Xb, Ya), at(Xb, Yb)));
}

// One plane of the pyramid: halved repeatedly, unpadded, edges repeated on access
template <typename T>
struct SmallPlane {
    int w = 0, h = 0;
    std::vector<T> p;
    T At(int x, int y) const { return p[static_cast<size_t>(std::clamp(y, 0, h - 1)) * w + std::clamp(x, 0, w - 1)]; }
};

// SuperPyramid.cpp's RB2BilinearFiltered (mvu.Super's default rfilter=1): 1/8, 3/8, 3/8, 1/8 in
// both directions, a plain average at the edges
template <typename T>
SmallPlane<T> Reduce(const SmallPlane<T> &src) {
    SmallPlane<T> d;
    d.w = (src.w + 1) / 2;
    d.h = (src.h + 1) / 2;
    d.p.resize(static_cast<size_t>(d.w) * d.h);
    std::vector<int> tmp(2 * d.w);
    for (int y = 0; y < d.h; ++y) {
        for (int x = 0; x < 2 * d.w; ++x) {
            if (y == 0 || y == d.h - 1)
                tmp[x] = (src.At(x, 2 * y) + src.At(x, 2 * y + 1) + 1) >> 1;
            else
                tmp[x] = (src.At(x, 2 * y - 1) + (src.At(x, 2 * y) + src.At(x, 2 * y + 1)) * 3 + src.At(x, 2 * y + 2) + 4) >> 3;
        }
        T *row = d.p.data() + static_cast<size_t>(y) * d.w;
        row[0] = static_cast<T>((tmp[0] + tmp[1] + 1) >> 1);
        for (int x = 1; x < d.w - 1; ++x)
            row[x] = static_cast<T>((tmp[2 * x - 1] + (tmp[2 * x] + tmp[2 * x + 1]) * 3 + tmp[2 * x + 2] + 4) >> 3);
        if (d.w > 1)
            row[d.w - 1] = static_cast<T>((tmp[2 * (d.w - 1)] + tmp[2 * (d.w - 1) + 1] + 1) >> 1);
    }
    return d;
}

template <typename T>
struct Frame {
    Plane<T> y, u, v;
    // levels[L - 1] holds Y, U, V at 1 / 2^L of the frame's size, L = 1 .. top level
    std::vector<std::array<SmallPlane<T>, 3>> levels;
    // Float frames with subsampled chroma at pel 4: U's and V's quarter-pel images as mvgpu.Super makes
    // them, sample (4x + fx, 4y + fy) of padded pixel (x, y) at row 4y + fy, 4 * qw wide (empty otherwise)
    std::vector<T> qimg[2];
    int qw = 0;
};

// ---------------------------------------------------------------------------------------------
// Float frames: mvu.Super's float planes

// mvu.Super's float AveragePixels and Wiener tap, as super_common.glsl's AvgF and WienerF
float AvgF(float a, float b) {
    return (a + b) * 0.5f;
}
float WienerF(float m0, float m1, float m2, float m3, float m4, float m5) {
    float t = (m2 + m3) * 4.0f;
    t = t - (m1 + m4);
    t = t * 5.0f;
    const float s = m0 + (m5 + t);
    return s * (1.0f / 32.0f);
}

void HorizontalWienerF(float *dst, const float *src, int w, int h) {
    for (int j = 0; j < h; ++j) {
        const float *s = src + static_cast<size_t>(j) * w;
        float *d = dst + static_cast<size_t>(j) * w;
        d[0] = AvgF(s[0], s[1]);
        d[1] = AvgF(s[1], s[2]);
        for (int i = 2; i < w - 4; ++i)
            d[i] = WienerF(s[i - 2], s[i - 1], s[i], s[i + 1], s[i + 2], s[i + 3]);
        for (int i = w - 4; i < w - 1; ++i)
            d[i] = AvgF(s[i], s[i + 1]);
        d[w - 1] = s[w - 1];
    }
}

void VerticalWienerF(float *dst, const float *src, int w, int h) {
    auto row = [&](const float *base, int j) { return base + static_cast<size_t>(j) * w; };
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < w; ++i)
            dst[static_cast<size_t>(j) * w + i] = AvgF(row(src, j)[i], row(src, j + 1)[i]);
    for (int j = 2; j < h - 4; ++j) {
        const float *r0 = row(src, j - 2), *r1 = row(src, j - 1), *r2 = row(src, j), *r3 = row(src, j + 1), *r4 = row(src, j + 2), *r5 = row(src, j + 3);
        float *d = dst + static_cast<size_t>(j) * w;
        for (int i = 0; i < w; ++i)
            d[i] = WienerF(r0[i], r1[i], r2[i], r3[i], r4[i], r5[i]);
    }
    for (int j = h - 4; j < h - 1; ++j)
        for (int i = 0; i < w; ++i)
            dst[static_cast<size_t>(j) * w + i] = AvgF(row(src, j)[i], row(src, j + 1)[i]);
    memcpy(dst + static_cast<size_t>(h - 1) * w, row(src, h - 1), w * sizeof(float));
}

// MakePlane in float
Plane<float> MakePlaneF(const float *src, int w, int h, int padX, int padY, int aw, int ah) {
    Plane<float> pl;
    pl.w = aw + 2 * padX;
    pl.h = ah + 2 * padY;
    for (auto &p : pl.p)
        p.resize(static_cast<size_t>(pl.w) * pl.h);
    float *d = pl.p[0].data();
    for (int y = 0; y < pl.h; ++y) {
        const int sy = std::clamp(y - padY, 0, h - 1);
        for (int x = 0; x < pl.w; ++x)
            d[static_cast<size_t>(y) * pl.w + x] = src[static_cast<size_t>(sy) * w + std::clamp(x - padX, 0, w - 1)];
    }
    HorizontalWienerF(pl.p[1].data(), pl.p[0].data(), pl.w, pl.h);
    VerticalWienerF(pl.p[2].data(), pl.p[0].data(), pl.w, pl.h);
    HorizontalWienerF(pl.p[3].data(), pl.p[2].data(), pl.w, pl.h);
    return pl;
}

// Reduce in float, as pyr_reduce.comp's float rfilter 1
SmallPlane<float> ReduceF(const SmallPlane<float> &src) {
    auto at = [&](int x, int y) { return src.p[static_cast<size_t>(std::clamp(y, 0, src.h - 1)) * src.w + std::clamp(x, 0, src.w - 1)]; };
    SmallPlane<float> d;
    d.w = (src.w + 1) / 2;
    d.h = (src.h + 1) / 2;
    d.p.resize(static_cast<size_t>(d.w) * d.h);
    std::vector<float> tmp(2 * d.w + 2);
    for (int y = 0; y < d.h; ++y) {
        for (int x = 0; x < 2 * d.w + 2; ++x) {
            if (y == 0 || y == d.h - 1)
                tmp[x] = (at(x, 2 * y) + at(x, 2 * y + 1)) * 0.5f;
            else
                tmp[x] = ((at(x, 2 * y - 1) + (at(x, 2 * y) + at(x, 2 * y + 1)) * 3.0f) + at(x, 2 * y + 2)) * 0.125f;
        }
        float *row = d.p.data() + static_cast<size_t>(y) * d.w;
        for (int x = 0; x < d.w; ++x) {
            const int c = 2 * x;
            if (x == 0 || x == d.w - 1)
                row[x] = (tmp[c] + tmp[c + 1]) * 0.5f;
            else
                row[x] = ((tmp[c - 1] + (tmp[c] + tmp[c + 1]) * 3.0f) + tmp[c + 2]) * 0.125f;
        }
    }
    return d;
}

// The quarter-pel sample (X, Y) of a float plane, as mvu.Super's float quarter planes hold it
// inside the plane (super_qpel.comp's image), reads clamped to it
float QuarterF(const Plane<float> &pl, int X, int Y) {
    auto at = [&](int Xh, int Yh) {
        const int x = std::clamp(Xh >> 1, 0, pl.w - 1), y = std::clamp(Yh >> 1, 0, pl.h - 1);
        return pl.p[(Xh & 1) | ((Yh & 1) << 1)][static_cast<size_t>(y) * pl.w + x];
    };
    const int Xa = X >> 1, Xb = Xa + (X & 1), Ya = Y >> 1, Yb = Ya + (Y & 1);
    return AvgF(AvgF(at(Xa, Ya), at(Xa, Yb)), AvgF(at(Xb, Ya), at(Xb, Yb)));
}

// ---------------------------------------------------------------------------------------------
// Float SADs: mvu's, its float operations in the order its AVX-512 build takes them, as
// sadf_common.glsl takes them (see there), of a w x h block of differences d (current minus
// reference, row-major)

// The raw float SAD: blocks 16 or more wide in 16 lanes, 8 wide in 8, lane l summing column l of every
// chunk of the row, the rows in 4 accumulators by their row modulo 4 (of fewer than 4 rows all in the
// first), (a0 + a1) + (a2 + a3), then the lanes halving, lane i with lane i + L/2 first; blocks 4 or 2
// wide each column's rows halving, then the columns (c0 + c2) + (c1 + c3)
float SadRawF(const float *d, int w, int h) {
    auto ad = [&](int x, int y) { return std::fabs(d[y * w + x]); };
    if (w <= 4) {
        auto column = [&](int x) {
            if (h == 2)
                return ad(x, 0) + ad(x, 1);
            if (h == 4)
                return (ad(x, 0) + ad(x, 2)) + (ad(x, 1) + ad(x, 3));
            return ((ad(x, 0) + ad(x, 4)) + (ad(x, 2) + ad(x, 6))) + ((ad(x, 1) + ad(x, 5)) + (ad(x, 3) + ad(x, 7)));
        };
        return w == 2 ? column(0) + column(1) : (column(0) + column(2)) + (column(1) + column(3));
    }
    const int L = w == 8 ? 8 : 16;
    float acc[4][16] = {};
    for (int y = 0; y < h; ++y) {
        float *a = acc[h < 4 ? 0 : y & 3];
        for (int c = 0; c < w; c += L)
            for (int l = 0; l < L; ++l)
                a[l] += ad(c + l, y);
    }
    float t[16];
    for (int l = 0; l < L; ++l)
        t[l] = (acc[0][l] + acc[1][l]) + (acc[2][l] + acc[3][l]);
    for (int half = L / 2; half >= 1; half /= 2)
        for (int i = 0; i < half; ++i)
            t[i] = t[i] + t[i + half];
    return t[0];
}

// mvu's mapping of a raw float SAD to the 16-bit range (scale_f32_sad; 32767.5 for the SATD, whose halving
// the compiler folded into the scale)
uint32_t ScaledF(float s, float scale) {
    const float v = s * scale;
    if (!(v > 0.0f))
        return 0;
    if (v >= 4294967040.0f)
        return 0xFFFFFFFFu;
    return static_cast<uint32_t>(v + 0.5f);
}

// The 16 absolute values of the 4x4 tile at (x0, y0), satd_f32_c's, c[4 * j + k] the k-th of column j
void TileF(const float *d, int w, int x0, int y0, float c[16]) {
    float t[16];
    for (int i = 0; i < 4; ++i) {
        const float *r = d + (y0 + i) * w + x0;
        const float a0 = r[0] + r[1], a1 = r[0] - r[1], a2 = r[2] + r[3], a3 = r[2] - r[3];
        t[4 * i] = a0 + a2;
        t[4 * i + 1] = a1 + a3;
        t[4 * i + 2] = a0 - a2;
        t[4 * i + 3] = a1 - a3;
    }
    for (int j = 0; j < 4; ++j) {
        const float a0 = t[j] + t[4 + j], a1 = t[j] - t[4 + j], a2 = t[8 + j] + t[12 + j], a3 = t[8 + j] - t[12 + j];
        c[4 * j] = std::fabs(a0 + a2);
        c[4 * j + 1] = std::fabs(a1 + a3);
        c[4 * j + 2] = std::fabs(a0 - a2);
        c[4 * j + 3] = std::fabs(a1 - a3);
    }
}

// A tile's sum as blocks up to 8 wide take it
float T4F(const float c[16]) {
    return (((c[5] + c[14]) + (c[15] + c[12])) + ((c[13] + c[6]) + (c[7] + c[4]))) + (((c[1] + c[10]) + (c[11] + c[8])) + ((c[9] + c[2]) + (c[3] + c[0])));
}

// A tile folded into a lane's running sum, as wider blocks take it
float T16F(const float c[16], float acc) {
    return c[15] + ((c[14] + (c[12] + (c[13] + (c[10] + c[11])))) + ((c[8] + (c[9] + (c[6] + c[7]))) + ((c[4] + (c[5] + c[3])) + ((c[2] + c[0]) + (c[1] + acc)))));
}

// The raw float SATD: up to 8 wide tile after tile, wider a row of tiles at a time in 4 lanes, tile t
// into lane t % 4, the first lane's sum starting from the rows before
float SatdRawF(const float *d, int w, int h) {
    float total = 0.0f, c[16];
    if (w <= 8) {
        for (int ty = 0; ty < h / 4; ++ty)
            for (int tx = 0; tx < w / 4; ++tx) {
                TileF(d, w, 4 * tx, 4 * ty, c);
                total += T4F(c);
            }
        return total;
    }
    for (int ty = 0; ty < h / 4; ++ty) {
        float l[4] = {total, 0.0f, 0.0f, 0.0f};
        for (int tx = 0; tx < w / 4; ++tx) {
            TileF(d, w, 4 * tx, 4 * ty, c);
            l[tx % 4] = T16F(c, l[tx % 4]);
        }
        total = (l[0] + l[2]) + (l[1] + l[3]);
    }
    return total;
}

// ---------------------------------------------------------------------------------------------
// SAD

// The SAD of two bw x bh blocks
int BlockSad(const uint8_t *a, ptrdiff_t pa, const uint8_t *b, ptrdiff_t pb, int bw, int bh) {
#ifdef REFERENCE_SSE2
    if (bw % 8 == 0) {
        __m128i acc = _mm_setzero_si128();
        for (int y = 0; y < bh; ++y) {
            const uint8_t *ra = a + y * pa, *rb = b + y * pb;
            int x = 0;
            for (; x + 16 <= bw; x += 16)
                acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ra + x)), _mm_loadu_si128(reinterpret_cast<const __m128i *>(rb + x))));
            if (x < bw)
                acc = _mm_add_epi64(acc, _mm_sad_epu8(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(ra + x)), _mm_loadl_epi64(reinterpret_cast<const __m128i *>(rb + x))));
        }
        // each half's sum whole: a 32-wide block's can pass 16 bits
        return _mm_cvtsi128_si32(acc) + _mm_cvtsi128_si32(_mm_unpackhi_epi64(acc, acc));
    }
#endif
    int sad = 0;
    for (int y = 0; y < bh; ++y)
        for (int x = 0; x < bw; ++x)
            sad += std::abs(a[y * pa + x] - b[y * pb + x]);
    return sad;
}

int BlockSad(const uint16_t *a, ptrdiff_t pa, const uint16_t *b, ptrdiff_t pb, int bw, int bh) {
#ifdef REFERENCE_SSE2
    if (bw % 8 == 0) {
        const __m128i zero = _mm_setzero_si128();
        __m128i acc = zero;
        for (int y = 0; y < bh; ++y) {
            const uint16_t *ra = a + y * pa, *rb = b + y * pb;
            for (int x = 0; x < bw; x += 8) {
                const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ra + x)), vb = _mm_loadu_si128(reinterpret_cast<const __m128i *>(rb + x));
                const __m128i d = _mm_or_si128(_mm_subs_epu16(va, vb), _mm_subs_epu16(vb, va)); // |a - b|
                acc = _mm_add_epi32(acc, _mm_add_epi32(_mm_unpacklo_epi16(d, zero), _mm_unpackhi_epi16(d, zero)));
            }
        }
        acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2)));
        acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1)));
        return _mm_cvtsi128_si32(acc);
    }
#endif
    int sad = 0;
    for (int y = 0; y < bh; ++y)
        for (int x = 0; x < bw; ++x)
            sad += std::abs(a[y * pa + x] - b[y * pb + x]);
    return sad;
}

// mvu's SATD (SADFunctions.cpp's Satd_C): the block in 4x4 tiles, each the sum of the absolute values
// of the 4x4 Hadamard transform of its differences, halved, which is exact, the sum always being even
template <typename T>
int BlockSatd(const T *a, ptrdiff_t pa, const T *b, ptrdiff_t pb, int bw, int bh) {
    int64_t satd = 0;
    for (int y = 0; y < bh; y += 4)
        for (int x = 0; x < bw; x += 4) {
            int64_t m[4][4];
            for (int i = 0; i < 4; ++i) {
                const T *ra = a + (y + i) * pa + x, *rb = b + (y + i) * pb + x;
                const int64_t d0 = ra[0] - rb[0], d1 = ra[1] - rb[1], d2 = ra[2] - rb[2], d3 = ra[3] - rb[3];
                const int64_t s0 = d0 + d1, s1 = d0 - d1, s2 = d2 + d3, s3 = d2 - d3;
                m[i][0] = s0 + s2;
                m[i][1] = s1 + s3;
                m[i][2] = s0 - s2;
                m[i][3] = s1 - s3;
            }
            int64_t sum = 0;
            for (int j = 0; j < 4; ++j) {
                const int64_t s0 = m[0][j] + m[1][j], s1 = m[0][j] - m[1][j], s2 = m[2][j] + m[3][j], s3 = m[2][j] - m[3][j];
                sum += std::abs(s0 + s2) + std::abs(s1 + s3) + std::abs(s0 - s2) + std::abs(s1 - s3);
            }
            satd += sum >> 1;
        }
    return static_cast<int>(satd);
}

// A block's SAD, or with satd its SATD
template <typename T>
int BlockMetric(const T *a, ptrdiff_t pa, const T *b, ptrdiff_t pb, int bw, int bh, bool satd) {
    return satd ? BlockSatd(a, pa, b, pb, bw, bh) : BlockSad(a, pa, b, pb, bw, bh);
}

using Key = std::pair<int, int>; // field (n, d)

struct Vec {
    int x = 0, y = 0; // 1 / pel pixels at full size, full pels of their level in the pyramid
    bool operator==(const Vec &o) const { return x == o.x && y == o.y; }
};

struct Field {
    int nbx = 0, nby = 0;
    std::vector<Vec> v;
    std::vector<uint32_t> sad;
};

template <typename T>
struct Search {
    // The clip: frames of w x h, chroma subsampled by xr and yr (1 or 2)
    int w = 0, h = 0, frames = 0, xr = 2, yr = 2;
    std::vector<Frame<T>> clip;
    // The grid: blk x blkY blocks, step and stepY apart, nbx x nby of them covering the block-aligned
    // aw x ah, extended by a column or row while they fall short of the frame, as mvu.Super and
    // mvu.Analyse have it; the planes padded by pad horizontally and padY vertically, chroma by padc
    // and padcY
    int blk = 16, blkY = 16, step = 8, stepY = 8, nbx = 0, nby = 0, aw = 0, ah = 0, pad = 16, padY = 16, padc = 8, padcY = 8;
    int pel = 2;
    int topLevel = 0; // pyramid levels built, 1 / 2^topLevel the smallest
    // The fields: radius of them either side, delta frames apart
    int radius = 2, delta = 1;
    bool standalone = false; // no chained or inverted seeds
    bool chroma = true;      // the SAD counts U and V
    bool satd = false;       // luma's SAD is its SATD (BlockSatd), at the full-size grid
    int plevel = 1;          // lambda * 2^(plevel * L) at coarse level L
    int64_t lambda0 = 0;     // mvlambda scaled to the block size and divided by pel squared, the full-size grid's
    int64_t lambdaBlock = 0; // mvlambda scaled to the block size, the coarse levels' before plevel
    int64_t lsad = 0;        // scaled to the block size
    int badSad = 0;          // scaled to the block size
    int fallbackRadius = 0;  // px; 0 = no fallback
    int fallbackStep = 1;    // px between the positions it searches, then the positions around the best
    bool global = true;      // the median seeds every level (globalmv)
    int pzero = 25, pglobal = 0, pnew = 25; // mvu's penalties, 256ths of the SAD (see the top of the file)
    int moveRadius = 1;      // the full-size passes' reach, pelsearch / pel px
    int depthShift = 0;      // bits - 8: lambda is relaxed by the worst SAD / 2 in steps of 2^depthShift,
    int areaShift = 0;       // and at full size 2^areaShift times that for blocks of more than 32x32 pixels
    // The largest SAD a block's pixels make, and a level block's: the worst neighbour SAD is capped at it
    // (the lambda tables' last entries), which only float SADs, of samples outside their range, pass
    uint32_t sadCap = UINT32_MAX, levelSadCap = UINT32_MAX;

    // A penalty on a SAD
    static int64_t Penalty(uint32_t sad, int pen) { return (static_cast<int64_t>(pen) * sad) >> 8; }

    // lambda relaxed by lsad for the worst neighbour SAD, which counts in steps of 2^(depthShift +
    // shift) halves (Analyse.cpp's tables; shift: the full-size grid's areaShift)
    int64_t Relaxed(int64_t lambda, uint32_t worst, int shift) const {
        const int64_t half = static_cast<int64_t>(worst >> (1 + depthShift + shift)) << (depthShift + shift);
        const double sc = static_cast<double>(lsad) / std::max<int64_t>(lsad + half, 1);
        return static_cast<int64_t>(lambda * sc * sc);
    }

    // Block b's position in the frame
    int BX(int b) const { return (b % nbx) * step; }
    int BY(int b) const { return (b / nbx) * stepY; }

    // The range ValidateVectors accepts for a block at (x, y): within the block-aligned frame and
    // its padding. ClampHalf keeps to the half-pel grid inside it, ClampFull to the full-pel grid,
    // the grids the seeds, passes and half-pel step (ClampHalf) and the fallback (ClampFull) stay on
    // at pel 4; at pel 2 all three are the same.
    Vec ClampTo(int x, int y, Vec v, int top) const {
        return {std::clamp(v.x, -pel * (x + pad), pel * (aw + pad - blk - x) - top), std::clamp(v.y, -pel * (y + padY), pel * (ah + padY - blkY - y) - top)};
    }
    Vec Clamp(int x, int y, Vec v) const { return ClampTo(x, y, v, 1); }
    Vec ClampHalf(int x, int y, Vec v) const { return ClampTo(x, y, v, pel == 4 ? 2 : 1); }
    Vec ClampFull(int x, int y, Vec v) const { return ClampTo(x, y, v, pel == 4 ? 4 : 1); }

    // At pel 4, a vector rounded to the half-pel grid, each component toward zero; at pel 2 itself
    Vec HalfGrid(Vec v) const {
        if (pel != 4)
            return v;
        auto round = [](int c) { return c < 0 ? -((-c) & ~1) : c & ~1; };
        return {round(v.x), round(v.y)};
    }

    // One plane's bw x bh block at (X, Y), 1 / pel pixels of the padded plane, against the current
    // block, its SAD or with satd its SATD: read in place when the position is on the half-pel grid
    // (at pel 1 the full-pel one), else computed (pel 4)
    int PlaneSad(const T *cur, ptrdiff_t curStride, const Plane<T> &ref, int X, int Y, int bw, int bh, bool satd = false) const {
        if (pel == 1)
            return BlockMetric(cur, curStride, ref.p[0].data() + static_cast<size_t>(Y) * ref.w + X, ref.w, bw, bh, satd);
        if (pel == 2 || ((X | Y) & 1) == 0) {
            const int Xh = pel == 2 ? X : X >> 1, Yh = pel == 2 ? Y : Y >> 1;
            return BlockMetric(cur, curStride, ref.p[(Xh & 1) | ((Yh & 1) << 1)].data() + static_cast<size_t>(Yh >> 1) * ref.w + (Xh >> 1), ref.w, bw, bh,
                               satd);
        }
        std::vector<T> tmp(static_cast<size_t>(bw) * bh);
        for (int j = 0; j < bh; ++j)
            for (int i = 0; i < bw; ++i)
                tmp[j * bw + i] = static_cast<T>(QuarterSample(ref, X + 4 * i, Y + 4 * j));
        return BlockMetric(cur, curStride, tmp.data(), bw, bw, bh, satd);
    }

    // mvu's float SAD (with satd its SATD) of one plane's bw x bh block of the current frame at (xs, ys)
    // of the padded plane against the reference's at (X, Y), 1 / pel pixels, scaled to the 16-bit range.
    // The reference's samples where the kernels read them: on the half-pel grid (at pel 1 the full-pel
    // one) that plane's, a row of the block a row of the plane; between its samples (pel 4) computed from
    // them as mvu.Super's float quarter planes hold them; or in the reference's quarter-pel image img
    // where it has one.
    uint32_t PlaneSadF(const Plane<float> &cur, int xs, int ys, const Plane<float> &ref, const std::vector<float> &img, int qw, int X, int Y, int bw, int bh,
                       bool satdF) const {
        thread_local std::vector<float> d;
        d.resize(static_cast<size_t>(bw) * bh);
        for (int j = 0; j < bh; ++j) {
            const float *c = cur.p[0].data() + static_cast<size_t>(ys + j) * cur.w + xs;
            float *dj = d.data() + static_cast<size_t>(j) * bw;
            const int Yj = Y + pel * j;
            if (!img.empty()) {
                const float *r = img.data() + static_cast<size_t>(Yj) * 4 * qw + X;
                for (int i = 0; i < bw; ++i)
                    dj[i] = c[i] - r[4 * i];
            } else if (pel == 4 && ((X | Yj) & 1)) {
                for (int i = 0; i < bw; ++i)
                    dj[i] = c[i] - QuarterF(ref, X + 4 * i, Yj);
            } else {
                const int Xh = pel == 4 ? X >> 1 : X, Yh = pel == 4 ? Yj >> 1 : Yj;
                const float *r = pel == 1 ? ref.p[0].data() + static_cast<size_t>(Yj) * ref.w + X
                                          : ref.p[(Xh & 1) | ((Yh & 1) << 1)].data() + static_cast<size_t>(Yh >> 1) * ref.w + (Xh >> 1);
                for (int i = 0; i < bw; ++i)
                    dj[i] = c[i] - r[i];
            }
        }
        return satdF ? ScaledF(SatdRawF(d.data(), bw, bh), 32767.5f) : ScaledF(SadRawF(d.data(), bw, bh), 65535.0f);
    }

    // The SAD of the block at (x, y) of frame n against frame r displaced by v (with satd luma's SATD);
    // each plane's fits an int, their sum 32 unsigned bits. Float SADs: each plane's scaled alone, then
    // added, as mvu adds them, saturated to the fields' 32 bits.
    uint32_t Sad(int n, int r, int x, int y, Vec v) const {
        const Frame<T> &cf = clip[n], &rf = clip[r];
        // Analyse divides the chroma vector by the subsampling, toward zero
        const int xc = x / xr + padc, yc = y / yr + padcY, Xc = pel * xc + v.x / xr, Yc = pel * yc + v.y / yr;
        if constexpr (std::is_same_v<T, float>) {
            static const std::vector<float> none;
            uint64_t sad = PlaneSadF(cf.y, x + pad, y + padY, rf.y, none, 0, pel * (x + pad) + v.x, pel * (y + padY) + v.y, blk, blkY, satd);
            if (chroma) {
                sad += PlaneSadF(cf.u, xc, yc, rf.u, rf.qimg[0], rf.qw, Xc, Yc, blk / xr, blkY / yr, false);
                sad += PlaneSadF(cf.v, xc, yc, rf.v, rf.qimg[1], rf.qw, Xc, Yc, blk / xr, blkY / yr, false);
            }
            return static_cast<uint32_t>(std::min<uint64_t>(sad, UINT32_MAX));
        } else {
            const T *cur = cf.y.p[0].data() + static_cast<size_t>(y + padY) * cf.y.w + (x + pad);
            uint32_t sad = static_cast<uint32_t>(PlaneSad(cur, cf.y.w, rf.y, pel * (x + pad) + v.x, pel * (y + padY) + v.y, blk, blkY, satd));
            if (!chroma)
                return sad;
            const size_t cc = static_cast<size_t>(yc) * cf.u.w + xc;
            sad += static_cast<uint32_t>(PlaneSad(cf.u.p[0].data() + cc, cf.u.w, rf.u, Xc, Yc, blk / xr, blkY / yr));
            sad += static_cast<uint32_t>(PlaneSad(cf.v.p[0].data() + cc, cf.v.w, rf.v, Xc, Yc, blk / xr, blkY / yr));
            return sad;
        }
    }

    // The SAD of the 8x8 block at (x, y) of pyramid level L against frame r displaced by v, in full
    // pels of that level: luma plus U and V at the chroma block size, edges repeated; float SADs as
    // Sad's, each plane's scaled alone
    uint32_t CoarseSad(int n, int r, int L, int x, int y, Vec v) const {
        const auto &cl = clip[n].levels[L - 1], &rl = clip[r].levels[L - 1];
        if constexpr (std::is_same_v<T, float>) {
            auto plane = [&](const SmallPlane<float> &c, const SmallPlane<float> &rp, int xs, int ys, Vec u, int bw, int bh) {
                float d[64];
                for (int j = 0; j < bh; ++j)
                    for (int i = 0; i < bw; ++i)
                        d[j * bw + i] = c.At(xs + i, ys + j) - rp.At(xs + i + u.x, ys + j + u.y);
                return ScaledF(SadRawF(d, bw, bh), 65535.0f);
            };
            uint64_t sad = plane(cl[0], rl[0], x, y, v, 8, 8);
            if (chroma) {
                const Vec cv{v.x / xr, v.y / yr};
                sad += plane(cl[1], rl[1], x / xr, y / yr, cv, 8 / xr, 8 / yr);
                sad += plane(cl[2], rl[2], x / xr, y / yr, cv, 8 / xr, 8 / yr);
            }
            return static_cast<uint32_t>(std::min<uint64_t>(sad, UINT32_MAX));
        } else {
            return CoarseSadInt(cl, rl, x, y, v);
        }
    }

    uint32_t CoarseSadInt(const std::array<SmallPlane<T>, 3> &cl, const std::array<SmallPlane<T>, 3> &rl, int x, int y, Vec v) const {
        const SmallPlane<T> &cy = cl[0], &ry = rl[0];
        int sad = 0;
        if (x + v.x >= 0 && y + v.y >= 0 && x + v.x + 8 <= ry.w && y + v.y + 8 <= ry.h && x + 8 <= cy.w && y + 8 <= cy.h) {
            sad = BlockSad(cy.p.data() + static_cast<size_t>(y) * cy.w + x, cy.w, ry.p.data() + static_cast<size_t>(y + v.y) * ry.w + x + v.x, ry.w, 8, 8);
        } else {
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 8; ++i)
                    sad += std::abs(cy.At(x + i, y + j) - ry.At(x + i + v.x, y + j + v.y));
        }
        if (!chroma)
            return static_cast<uint32_t>(sad);
        const int cvx = v.x / xr, cvy = v.y / yr, xc = x / xr, yc = y / yr;
        for (int p = 1; p <= 2; ++p)
            for (int j = 0; j < 8 / yr; ++j)
                for (int i = 0; i < 8 / xr; ++i)
                    sad += std::abs(cl[p].At(xc + i, yc + j) - rl[p].At(xc + i + cvx, yc + j + cvy));
        return static_cast<uint32_t>(sad);
    }

    static int Median(std::vector<int> &v) {
        if (v.empty())
            return 0;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }

    // The fields in the order AnalyseMany's seeds need them: -delta, +delta (inverting -delta),
    // -2 delta (chaining -delta), +2 delta (chaining +delta, inverting -2 delta), ...
    std::vector<Key> Order() const {
        std::vector<Key> keys;
        for (int k = 1; k <= radius; ++k)
            for (int d : {-k * delta, k * delta})
                for (int n = 0; n < frames; ++n)
                    if (n + d >= 0 && n + d < frames)
                        keys.push_back({n, d});
        return keys;
    }
};

// ---------------------------------------------------------------------------------------------
// The coarse search: an exhaustive search at the smallest pyramid level, then at each finer level
// seeds from the level above and checkerboard passes, down to kFinest. Blocks are 8x8 without
// overlap at every level; vectors are full pels of their level.

struct LevelField {
    int nbx = 0, nby = 0;
    std::vector<Vec> v;
    std::vector<uint32_t> sad;
};

template <typename T>
struct Pyramid {
    const Search<T> &s;
    int n, r;
    int PadAt(int L) const { return std::max(1, s.pad >> L); }
    int PadYAt(int L) const { return std::max(1, s.padY >> L); }
    const SmallPlane<T> &Luma(int L) const { return s.clip[n].levels[L - 1][0]; }
    // Keep the reference block within the level's (scaled-down) padding
    Vec Bound(int L, int x, int y, Vec v) const {
        const int pad = PadAt(L), padY = PadYAt(L);
        return {std::clamp(v.x, -(x + pad), Luma(L).w + pad - 8 - x), std::clamp(v.y, -(y + padY), Luma(L).h + padY - 8 - y)};
    }
    LevelField Blank(int L) const {
        LevelField f;
        f.nbx = std::max(1, Luma(L).w / 8);
        f.nby = std::max(1, Luma(L).h / 8);
        f.v.assign(static_cast<size_t>(f.nbx) * f.nby, Vec{});
        f.sad.assign(f.v.size(), UINT32_MAX);
        return f;
    }

    // Checkerboard passes under Analyse's cost at level L: lambda = mvlambda * 2^(plevel * L) at full
    // pel, relaxed by lsad; candidates are the four neighbours' vectors and the 8 positions one pixel
    // around the winner, those with pnew's penalty
    void Passes(int L, LevelField &f) const {
        const int64_t lambdaL = s.lambdaBlock << (s.plevel * L);
        for (int pass = 0; pass < 2 * kLevelPairs; ++pass) {
            const int colour = pass & 1;
            ParallelFor(f.nby, [&](int by) {
                for (int bx = (by + colour) & 1; bx < f.nbx; bx += 2) {
                    const int b = by * f.nbx + bx, x = bx * 8, y = by * 8;
                    const int nbs[4] = {bx > 0 ? b - 1 : -1, bx + 1 < f.nbx ? b + 1 : -1, by > 0 ? b - f.nbx : -1, by + 1 < f.nby ? b + f.nbx : -1};
                    int xs[4], ys[4];
                    uint32_t worst = 0;
                    for (int i = 0; i < 4; ++i) {
                        const int k = nbs[i] >= 0 ? nbs[i] : b;
                        xs[i] = f.v[k].x;
                        ys[i] = f.v[k].y;
                        if (nbs[i] >= 0)
                            worst = std::max(worst, f.sad[k]);
                    }
                    std::sort(xs, xs + 4);
                    std::sort(ys, ys + 4);
                    const Vec p{(xs[1] + xs[2]) / 2, (ys[1] + ys[2]) / 2};
                    const int64_t lambda = s.Relaxed(lambdaL, std::min(worst, s.levelSadCap), 0);
                    auto cost = [&](uint32_t sad, Vec v) {
                        const int64_t dx = v.x - p.x, dy = v.y - p.y;
                        return static_cast<int64_t>(sad) + ((lambda * (dx * dx + dy * dy)) >> 8);
                    };
                    Vec best = f.v[b];
                    uint32_t bestSad = f.sad[b];
                    int64_t bestCost = cost(bestSad, best);
                    auto consider = [&](Vec v, int pen) {
                        v = Bound(L, x, y, v);
                        if (v == best)
                            return;
                        const uint32_t sad = s.CoarseSad(n, r, L, x, y, v);
                        const int64_t c = cost(sad, v) + Search<T>::Penalty(sad, pen);
                        if (c < bestCost) {
                            bestCost = c;
                            best = v;
                            bestSad = sad;
                        }
                    };
                    for (int i : nbs)
                        if (i >= 0)
                            consider(f.v[i], 0);
                    const Vec c = best;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            if (dx || dy)
                                consider({c.x + dx, c.y + dy}, s.pnew);
                    f.v[b] = best;
                    f.sad[b] = bestSad;
                }
            });
        }
    }

    // The finest level's field
    LevelField Run() const {
        const int top = s.topLevel;
        LevelField cur = Blank(top);
        ParallelFor(cur.nby, [&](int by) {
            for (int bx = 0; bx < cur.nbx; ++bx) {
                const int b = by * cur.nbx + bx, x = bx * 8, y = by * 8;
                const Vec lo = Bound(top, x, y, {-kTopRadius, -kTopRadius}), hi = Bound(top, x, y, {kTopRadius, kTopRadius});
                int bestLen = INT32_MAX;
                int64_t bestCost = INT64_MAX;
                for (int vy = lo.y; vy <= hi.y; ++vy)
                    for (int vx = lo.x; vx <= hi.x; ++vx) {
                        // pnew's penalty but at zero, mvu's coarsest predictor
                        const uint32_t sad = s.CoarseSad(n, r, top, x, y, {vx, vy});
                        const int64_t cost = static_cast<int64_t>(sad) + (vx || vy ? Search<T>::Penalty(sad, s.pnew) : 0);
                        const int len = vx * vx + vy * vy;
                        if (cost < bestCost || (cost == bestCost && len < bestLen)) {
                            bestCost = cost;
                            cur.sad[b] = sad;
                            cur.v[b] = {vx, vy};
                            bestLen = len;
                        }
                    }
            }
        });
        Passes(top, cur);
        for (int L = top - 1; L >= kFinest; --L) {
            LevelField next = Blank(L);
            std::vector<int> mx, my;
            for (const Vec &v : cur.v) {
                mx.push_back(2 * v.x);
                my.push_back(2 * v.y);
            }
            const Vec global{Search<T>::Median(mx), Search<T>::Median(my)};
            ParallelFor(next.nby, [&](int by) {
                for (int bx = 0; bx < next.nbx; ++bx) {
                    const int b = by * next.nbx + bx, x = bx * 8, y = by * 8;
                    const int px = std::min(bx / 2, cur.nbx - 1), py = std::min(by / 2, cur.nby - 1);
                    // the candidates with their penalties: the parents none, zero pzero's, the median
                    // pglobal's; the earliest of the cheapest wins
                    int64_t bestCost = INT64_MAX;
                    auto add = [&](Vec v, int pen) {
                        v = Bound(L, x, y, v);
                        const uint32_t sad = s.CoarseSad(n, r, L, x, y, v);
                        const int64_t c = static_cast<int64_t>(sad) + Search<T>::Penalty(sad, pen);
                        if (c < bestCost) {
                            bestCost = c;
                            next.sad[b] = sad;
                            next.v[b] = v;
                        }
                    };
                    static const int around[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                    for (const auto &o : around) {
                        const int ax = px + o[0], ay = py + o[1];
                        if (ax >= 0 && ay >= 0 && ax < cur.nbx && ay < cur.nby) {
                            const Vec v = cur.v[static_cast<size_t>(ay) * cur.nbx + ax];
                            add({2 * v.x, 2 * v.y}, 0);
                        }
                    }
                    add(Vec{}, s.pzero);
                    if (s.global)
                        add(global, s.pglobal);
                }
            });
            Passes(L, next);
            cur = std::move(next);
        }
        return cur;
    }
};

// ---------------------------------------------------------------------------------------------
// The full-size grid

// A seed and the penalty its cost takes on (see the top of the file)
struct Seed {
    Vec v;
    int pen = 0;
};

// The seeds of every block of field (n, d): zero (pzero's penalty), the field's median (pglobal's,
// but with --globalmv 0), then the predictors without a penalty: the chained and inverted vectors of
// the fields already refined, the finest coarse level's vectors around the block; clamped, on the
// half-pel grid, the predictors without duplicates among them
template <typename T>
std::vector<std::vector<Seed>> BuildSeeds(const Search<T> &s, int n, int d, const LevelField &co, const std::map<Key, Field> &done) {
    const int nb = s.nbx * s.nby;
    const int scale = s.pel << kFinest; // the finest level's full pels in the grid's units
    std::vector<std::vector<Seed>> seeds(nb);
    const size_t first = co.v.empty() || !s.global ? 1 : 2; // the predictors' first place
    auto add = [&](int b, Vec v) {
        v = s.ClampHalf(s.BX(b), s.BY(b), v);
        auto &list = seeds[b];
        for (size_t i = first; i < list.size(); ++i)
            if (list[i].v == v)
                return;
        list.push_back({v, 0});
    };
    std::vector<int> mx, my;
    for (const Vec &v : co.v) {
        mx.push_back(v.x * scale);
        my.push_back(v.y * scale);
    }
    const Vec global{Search<T>::Median(mx), Search<T>::Median(my)};

    // (n, d) chains (n, u) and (n + u, d - u), u the step toward d, and for d > 0 inverts (n + d, -d)
    const int u = d > 0 ? s.delta : -s.delta;
    const Field *stepF = nullptr, *restF = nullptr, *invF = nullptr;
    if (std::abs(d) >= 2 * s.delta && !s.standalone) {
        auto a = done.find({n, u}), b = done.find({n + u, d - u});
        if (a != done.end() && b != done.end()) {
            stepF = &a->second;
            restF = &b->second;
        }
    }
    if (d > 0 && !s.standalone)
        if (auto it = done.find({n + d, -d}); it != done.end())
            invF = &it->second;
    std::vector<Vec> inv;
    std::vector<uint32_t> invSad;
    std::vector<char> hasInv; // (any SAD, 2^32 - 1 too, which a float SAD saturates at)
    if (invF) {
        // The block at p' in frame n + d matched p' + u in frame n: the block of frame n nearest to
        // that position moves by -u; of several, the lowest SAD's, of equal ones the first
        inv.assign(nb, Vec{});
        invSad.assign(nb, UINT32_MAX);
        hasInv.assign(nb, 0);
        for (int b = 0; b < nb; ++b) {
            const Vec w = invF->v[b];
            const double unit = s.pel;
            const int tx = static_cast<int>(std::lround((s.BX(b) + w.x / unit) / s.step)), ty = static_cast<int>(std::lround((s.BY(b) + w.y / unit) / s.stepY));
            if (tx < 0 || ty < 0 || tx >= s.nbx || ty >= s.nby)
                continue;
            const int t = ty * s.nbx + tx;
            if (!hasInv[t] || invF->sad[b] < invSad[t]) {
                hasInv[t] = 1;
                invSad[t] = invF->sad[b];
                inv[t] = {-w.x, -w.y};
            }
        }
    }
    for (int b = 0; b < nb; ++b) {
        const int x = s.BX(b), y = s.BY(b);
        seeds[b].push_back({Vec{}, s.pzero});
        if (first == 2)
            seeds[b].push_back({s.ClampHalf(x, y, global), s.pglobal});
        if (stepF) {
            const Vec v1 = stepF->v[b];
            const double unit = s.pel;
            const int tx = std::clamp(static_cast<int>(std::lround((x + v1.x / unit) / s.step)), 0, s.nbx - 1);
            const int ty = std::clamp(static_cast<int>(std::lround((y + v1.y / unit) / s.stepY)), 0, s.nby - 1);
            const Vec v2 = restF->v[ty * s.nbx + tx];
            add(b, s.HalfGrid(Vec{v1.x + v2.x, v1.y + v2.y}));
        }
        if (invF && hasInv[b])
            add(b, s.HalfGrid(inv[b]));
        const int span = 8 << kFinest;
        const int cx = std::min((x + s.blk / 2) / span, co.nbx - 1), cy = std::min((y + s.blkY / 2) / span, co.nby - 1);
        static const int around[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (const auto &o : around) {
            const int ax = cx + o[0], ay = cy + o[1];
            if (ax >= 0 && ay >= 0 && ax < co.nbx && ay < co.nby) {
                const Vec v = co.v[static_cast<size_t>(ay) * co.nbx + ax];
                add(b, Vec{v.x * scale, v.y * scale});
            }
        }
        if (seeds[b].size() > static_cast<size_t>(kMaxSeeds))
            Die("more seeds than the GPU kernels take");
    }
    return seeds;
}

// The seeds measured, then pairs of checkerboard passes, the fallback, the half-pel step and at
// pel 4 the quarter-pel step (see the top of the file)
template <typename T>
Field Refine(const Search<T> &s, int n, int d, const std::vector<std::vector<Seed>> &seeds) {
    const int nb = s.nbx * s.nby;
    const int r = n + d;
    Field f;
    f.nbx = s.nbx;
    f.nby = s.nby;
    f.v.assign(nb, Vec{});
    f.sad.assign(nb, UINT32_MAX);
    std::vector<std::vector<uint32_t>> seedSad(nb);
    ParallelFor(s.nby, [&](int by) {
        for (int bx = 0; bx < s.nbx; ++bx) {
            const int b = by * s.nbx + bx;
            int64_t bestCost = INT64_MAX;
            for (const Seed &seed : seeds[b]) {
                const uint32_t sad = s.Sad(n, r, s.BX(b), s.BY(b), seed.v);
                seedSad[b].push_back(sad);
                const int64_t c = sad + Search<T>::Penalty(sad, seed.pen);
                if (c < bestCost) {
                    bestCost = c;
                    f.sad[b] = sad;
                    f.v[b] = seed.v;
                }
            }
        }
    });

    // The lambda of a worst neighbour SAD, as Analyse.cpp tabulates it, capped at the table's last entry
    auto lambdaOf = [&](uint32_t worst) { return s.Relaxed(s.lambda0, std::min(worst, s.sadCap), s.areaShift); };
    // Predictor and relaxed lambda of block b from its four neighbours
    auto context = [&](int b, Vec &p, int64_t &lambda) {
        const int bx = b % s.nbx, by = b / s.nbx;
        const int nbs[4] = {bx > 0 ? b - 1 : b, bx + 1 < s.nbx ? b + 1 : b, by > 0 ? b - s.nbx : b, by + 1 < s.nby ? b + s.nbx : b};
        int xs[4], ys[4];
        uint32_t worst = 0;
        for (int i = 0; i < 4; ++i) {
            xs[i] = f.v[nbs[i]].x;
            ys[i] = f.v[nbs[i]].y;
            if (nbs[i] != b)
                worst = std::max(worst, f.sad[nbs[i]]);
        }
        std::sort(xs, xs + 4);
        std::sort(ys, ys + 4);
        p = {(xs[1] + xs[2]) / 2, (ys[1] + ys[2]) / 2};
        lambda = lambdaOf(worst);
    };
    auto cost = [](uint32_t sad, Vec v, Vec p, int64_t lambda) {
        const int64_t dx = v.x - p.x, dy = v.y - p.y;
        return static_cast<int64_t>(sad) + ((lambda * (dx * dx + dy * dy)) >> 8);
    };

    // The blocks still above badSad: a full-pel search of +-fallbackRadius px around their
    // predictor, every fallbackStep px, then the positions up to fallbackStep - 1 px around the best,
    // every position with pnew's penalty
    auto fallback = [&]() {
        const int R = s.fallbackRadius;
        std::vector<Vec> nv(f.v);
        std::vector<uint32_t> ns(f.sad);
        ParallelFor(s.nby, [&](int by) {
            for (int bx = 0; bx < s.nbx; ++bx) {
                const int b = by * s.nbx + bx, x = s.BX(b), y = s.BY(b);
                if (static_cast<int64_t>(f.sad[b]) <= s.badSad)
                    continue;
                Vec p;
                int64_t lambda;
                context(b, p, lambda);
                Vec best = f.v[b];
                uint32_t bestSad = f.sad[b];
                int64_t bestCost = cost(bestSad, best, p, lambda);
                // full-pel positions around the predictor, rounded to a full pel
                const int pel = s.pel, cx = p.x & ~(pel - 1), cy = p.y & ~(pel - 1), stepH = pel * s.fallbackStep;
                const Vec lo = s.ClampFull(x, y, {cx - pel * R, cy - pel * R}), hi = s.ClampFull(x, y, {cx + pel * R, cy + pel * R});
                auto tryAt = [&](Vec v) {
                    const uint32_t sad = s.Sad(n, r, x, y, v);
                    const int64_t c = cost(sad, v, p, lambda) + Search<T>::Penalty(sad, s.pnew);
                    if (c < bestCost) {
                        bestCost = c;
                        best = v;
                        bestSad = sad;
                    }
                };
                for (int vy = (lo.y + pel - 1) & ~(pel - 1); vy <= hi.y; vy += stepH)
                    for (int vx = (lo.x + pel - 1) & ~(pel - 1); vx <= hi.x; vx += stepH)
                        tryAt({vx, vy});
                if (s.fallbackStep > 1) {
                    // at pel 4 around the best rounded down to the full-pel grid: the block's own
                    // vector, kept when nothing in the window beat it, may be on the half-pel grid
                    const Vec c = pel == 4 ? Vec{best.x & ~3, best.y & ~3} : best;
                    for (int dy = -pel * (s.fallbackStep - 1); dy <= pel * (s.fallbackStep - 1); dy += pel)
                        for (int dx = -pel * (s.fallbackStep - 1); dx <= pel * (s.fallbackStep - 1); dx += pel)
                            if (dx || dy) {
                                const Vec v = s.ClampFull(x, y, {c.x + dx, c.y + dy});
                                if (!(v == c))
                                    tryAt(v);
                            }
                }
                nv[b] = best;
                ns[b] = bestSad;
            }
        });
        f.v = std::move(nv);
        f.sad = std::move(ns);
    };

    // The checkerboard passes: each block of the pass's colour tries its seeds (with their penalties),
    // its neighbours' vectors and the positions within moveRadius px around the winner, rows top to
    // bottom, then left to right (with pnew's penalty)
    const int totalPairs = kPairs + (s.fallbackRadius > 0 ? 1 : 0);
    for (int pass = 0; pass < 2 * totalPairs; ++pass) {
        if (s.fallbackRadius > 0 && pass == 2 * kPairs)
            fallback();
        const int colour = pass & 1;
        ParallelFor(s.nby, [&](int by) {
            for (int bx = (by + colour) & 1; bx < s.nbx; bx += 2) {
                const int b = by * s.nbx + bx, x = s.BX(b), y = s.BY(b);
                Vec p;
                int64_t lambda;
                context(b, p, lambda);
                Vec best = f.v[b];
                uint32_t bestSad = f.sad[b];
                int64_t bestCost = cost(bestSad, best, p, lambda);
                auto consider = [&](Vec v, uint32_t sad, int pen) {
                    const int64_t c = cost(sad, v, p, lambda) + Search<T>::Penalty(sad, pen);
                    if (c < bestCost) {
                        bestCost = c;
                        best = v;
                        bestSad = sad;
                    }
                };
                for (size_t i = 0; i < seeds[b].size(); ++i)
                    consider(seeds[b][i].v, seedSad[b][i], seeds[b][i].pen);
                // every neighbour, even one equal to the best so far, which may be a seed with a penalty
                const int nbs[4] = {bx > 0 ? b - 1 : -1, bx + 1 < s.nbx ? b + 1 : -1, by > 0 ? b - s.nbx : -1, by + 1 < s.nby ? b + s.nbx : -1};
                for (int i : nbs) {
                    if (i < 0)
                        continue;
                    const Vec v = s.ClampHalf(x, y, f.v[i]);
                    consider(v, s.Sad(n, r, x, y, v), 0);
                }
                const Vec c = best;
                const int R = s.moveRadius;
                for (int dy = -R; dy <= R; ++dy)
                    for (int dx = -R; dx <= R; ++dx)
                        if (dx || dy) {
                            const Vec v = s.ClampHalf(x, y, {c.x + s.pel * dx, c.y + s.pel * dy});
                            if (!(v == c))
                                consider(v, s.Sad(n, r, x, y, v), s.pnew);
                        }
                f.v[b] = best;
                f.sad[b] = bestSad;
            }
        });
    }

    // Sub-pel refinement, every block independently against its neighbours as the step before left
    // them: the 8 positions half a pel around, kept to the half-pel grid; at pel 4 then the 8 a
    // quarter pel around, against the half-pel step's field, anywhere in range; with pnew's penalty
    auto subPelStep = [&](int unit, bool halfGrid) {
        std::vector<Vec> outV(f.v);
        std::vector<uint32_t> outSad(f.sad);
        ParallelFor(s.nby, [&](int by) {
            for (int bx = 0; bx < s.nbx; ++bx) {
                const int b = by * s.nbx + bx, x = s.BX(b), y = s.BY(b);
                Vec p;
                int64_t lambda;
                context(b, p, lambda);
                const Vec c = f.v[b];
                Vec best = c;
                uint32_t bestSad = f.sad[b];
                int64_t bestCost = cost(bestSad, best, p, lambda);
                for (int dy = -unit; dy <= unit; dy += unit)
                    for (int dx = -unit; dx <= unit; dx += unit)
                        if (dx || dy) {
                            const Vec at{c.x + dx, c.y + dy};
                            const Vec v = halfGrid ? s.ClampHalf(x, y, at) : s.Clamp(x, y, at);
                            if (v == c)
                                continue;
                            const uint32_t sad = s.Sad(n, r, x, y, v);
                            const int64_t cc = cost(sad, v, p, lambda) + Search<T>::Penalty(sad, s.pnew);
                            if (cc < bestCost) {
                                bestCost = cc;
                                best = v;
                                bestSad = sad;
                            }
                        }
                outV[b] = best;
                outSad[b] = bestSad;
            }
        });
        f.v = std::move(outV);
        f.sad = std::move(outSad);
    };
    if (s.pel >= 2)
        subPelStep(s.pel / 2, true);
    if (s.pel == 4)
        subPelStep(1, false);
    return f;
}

void WriteField(FILE *f, int n, int d, int pel, const Field &fl) {
    const int32_t hdr[6] = {n, d, fl.nbx, fl.nby, pel, 1};
    fwrite(hdr, sizeof(hdr), 1, f);
    std::vector<int32_t> a(fl.v.size());
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = fl.v[i].x;
    fwrite(a.data(), 4, a.size(), f);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = fl.v[i].y;
    fwrite(a.data(), 4, a.size(), f);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<int32_t>(fl.sad[i]); // the unsigned SAD's bits
    fwrite(a.data(), 4, a.size(), f);
}

// The arguments, as main parses them
struct Options {
    std::string srcPath, outPath, format = "420";
    int w = 0, h = 0, frames = 0, bits = 8, blk = 16, overlap = 8, pad = 16, pel = 2, radius = 2, delta = 1, plevel = 1, badrange = 40, badstep = 2;
    int blkY = -1, overlapY = -1, padY = -1; // the vertical ones, the horizontal ones unless given
    int superBlk = -1, superBlkY = -1, superOverlap = -1, superOverlapY = -1; // the super's grid, the one analysed unless given
    int64_t mvlambda = 1000, lsad = 400, badsad = 1000;
    bool standalone = false, chroma = true, onelevel = false, satd = false;
    int levels = 0, pnew = 25, pzero = -1, pglobal = 0, pelsearch = -1; // pzero -1: pnew's; pelsearch -1: pel
    bool globalmv = true;
};

// The search over samples of type T: bytes at 8 bits, 16 bits at 9 to 16, floats at 32
template <typename T>
int Run(const Options &o) {
    const int w = o.w, h = o.h, frames = o.frames, blk = o.blk, blkY = o.blkY, overlap = o.overlap, overlapY = o.overlapY, pad = o.pad, padY = o.padY,
              pel = o.pel;
    Search<T> s;
    s.w = w;
    s.h = h;
    s.frames = frames;
    const bool gray = o.format == "gray";
    s.xr = o.format == "420" || o.format == "422" ? 2 : 1;
    s.yr = o.format == "420" || o.format == "440" ? 2 : 1;
    if (w % s.xr || h % s.yr)
        Die("subsampled chroma needs the frame's size divisible by the subsampling");
    if (pad % s.xr || padY % s.yr)
        Die("with subsampled chroma the padding must be even across the subsampling, or chroma would be read outside its padding");
    s.blk = blk;
    s.blkY = blkY;
    s.step = blk - overlap;
    s.stepY = blkY - overlapY;
    s.pad = pad;
    s.padY = padY;
    s.padc = pad / s.xr;
    s.padcY = padY / s.yr;
    // MVUtensils' grids: the blocks that fit, plus one more column or row while they fall short. The
    // super's block-aligned frame, the size of its planes, follows its grid; the grid analysed lies in
    // it (mvu's MotionBlockPyramid), and the vectors stay within it and its padding (Clamp).
    auto aligned = [&](int size, int step, int overlap) {
        const int b = step * ((size - overlap) / step) + overlap;
        return b < size ? b + step : b;
    };
    s.aw = aligned(w, o.superBlk - o.superOverlap, o.superOverlap);
    s.ah = aligned(h, o.superBlkY - o.superOverlapY, o.superOverlapY);
    auto count = [](int size, int step, int overlap) {
        int n = std::max(0, (size - overlap) / step);
        while (step * n + overlap < size)
            ++n;
        return n;
    };
    s.nbx = count(w, s.step, overlap);
    s.nby = count(h, s.stepY, overlapY);
    if (s.step * s.nbx + overlap > s.aw || s.stepY * s.nby + overlapY > s.ah)
        Die("the grid analysed doesn't fit the super's block-aligned frame (mvu: \"The chosen block size has no multiple ...\")");
    if (s.nbx < 1 || s.nby < 1)
        Die("the frame is too small to hold a single block");
    s.pel = pel;
    s.radius = o.radius;
    s.delta = o.delta;
    s.standalone = o.standalone;
    s.chroma = o.chroma && !gray;
    s.satd = o.satd;
    s.plevel = o.plevel;
    // mvu.Analyse's scaling: to the bit depth (floats searched as 16-bit samples), rounded, then to
    // the block size; lambda divided by pel squared at full size
    const bool isFloat = o.bits == 32;
    const int searchBits = isFloat ? 16 : o.bits;
    const int pixelMax = (1 << searchBits) - 1;
    auto toDepth = [&](int64_t v) { return static_cast<int64_t>(static_cast<double>(v) * pixelMax / 255.0 + 0.5); };
    const int64_t mvlambda = toDepth(o.mvlambda), lsad = toDepth(o.lsad), badsad = toDepth(o.badsad);
    const int64_t area = static_cast<int64_t>(blk) * blkY;
    s.lambda0 = mvlambda * area / 64 / (pel * pel);
    s.lambdaBlock = mvlambda * area / 64;
    s.lsad = lsad * area / 64;
    s.badSad = static_cast<int>(std::min<int64_t>(badsad * area / 64, INT32_MAX));
    s.fallbackRadius = std::abs(o.badrange);
    s.fallbackStep = o.badstep;
    s.depthShift = searchBits - 8;
    const int areaPixels = blk * blkY;
    s.areaShift = areaPixels > 8192 ? 4 : areaPixels > 4096 ? 3 : areaPixels > 2048 ? 2 : areaPixels > 1024 ? 1 : 0;
    // (the pixels of a block's SAD, luma's twice with satd, whose SATD reaches at most twice the SAD)
    const int64_t blockPixels = static_cast<int64_t>(areaPixels) * (s.satd ? 2 : 1) + (s.chroma ? 2 * (blk / s.xr) * (blkY / s.yr) : 0);
    s.sadCap = static_cast<uint32_t>(blockPixels * pixelMax);
    s.levelSadCap = static_cast<uint32_t>((64 + (s.chroma ? 2 * (8 / s.xr) * (8 / s.yr) : 0)) * static_cast<int64_t>(pixelMax));
    for (int cw = w; cw / 2 >= kTopWidth && !o.onelevel; cw = (cw + 1) / 2)
        ++s.topLevel;
    // levels: as Analyse.cpp counts them, the full-size grid's and the coarse ones, mvu's messages
    const int available = s.topLevel + 1;
    if (o.levels > available)
        Die("levels is " + std::to_string(o.levels) + " but the super clip only has " + std::to_string(available) + (available == 1 ? " level" : " levels"));
    const int levelCount = o.levels > 0 ? o.levels : available + o.levels;
    if (levelCount < 1)
        Die("the levels argument resolves to a non-positive level count");
    s.topLevel = levelCount - 1;
    s.global = o.globalmv;
    s.pnew = o.pnew;
    s.pzero = o.pzero < 0 ? o.pnew : o.pzero;
    s.pglobal = o.pglobal;
    s.moveRadius = (o.pelsearch < 0 ? pel : o.pelsearch) / pel;

    // The frames: the super's planes and the pyramid
    const int wc = w / s.xr, hc = h / s.yr;
    const size_t lumaSamples = static_cast<size_t>(w) * h, chromaSamples = gray ? 0 : static_cast<size_t>(wc) * hc,
                 frameSamples = lumaSamples + 2 * chromaSamples;
    if constexpr (std::is_same_v<T, float>) {
        std::vector<std::vector<float>> raw(frames, std::vector<float>(frameSamples));
        FILE *f = fopen(o.srcPath.c_str(), "rb");
        if (!f)
            Die("cannot open " + o.srcPath);
        for (auto &r : raw)
            if (fread(r.data(), sizeof(float), frameSamples, f) != frameSamples)
                Die("the source holds fewer frames");
        fclose(f);
        s.clip.resize(frames);
        const bool image = pel == 4 && !gray && (s.xr > 1 || s.yr > 1);
        ParallelFor(frames, [&](int i) {
            const float *py = raw[i].data(), *pu = py + lumaSamples, *pv = pu + chromaSamples;
            Frame<float> &fr = s.clip[i];
            fr.y = MakePlaneF(py, w, h, pad, padY, s.aw, s.ah);
            if (!gray) {
                fr.u = MakePlaneF(pu, wc, hc, s.padc, s.padcY, s.aw / s.xr, s.ah / s.yr);
                fr.v = MakePlaneF(pv, wc, hc, s.padc, s.padcY, s.aw / s.xr, s.ah / s.yr);
                if (image) {
                    fr.qw = fr.u.w;
                    for (int p = 0; p < 2; ++p) {
                        const Plane<float> &fp = p ? fr.v : fr.u;
                        fr.qimg[p].resize(static_cast<size_t>(16) * fp.w * fp.h);
                        for (int Y = 0; Y < 4 * fp.h; ++Y)
                            for (int X = 0; X < 4 * fp.w; ++X)
                                fr.qimg[p][static_cast<size_t>(Y) * 4 * fp.w + X] = QuarterF(fp, X, Y);
                    }
                }
            }
            std::array<SmallPlane<float>, 3> level;
            const float *src[3] = {py, pu, pv};
            for (int p = 0; p < (gray ? 1 : 3); ++p) {
                level[p].w = p ? wc : w;
                level[p].h = p ? hc : h;
                level[p].p.assign(src[p], src[p] + static_cast<size_t>(level[p].w) * level[p].h);
            }
            for (int L = 1; L <= s.topLevel; ++L) {
                for (int p = 0; p < (gray ? 1 : 3); ++p)
                    level[p] = ReduceF(level[p]);
                fr.levels.push_back(level);
            }
        });
    } else {
        std::vector<std::vector<T>> raw(frames, std::vector<T>(frameSamples));
        FILE *f = fopen(o.srcPath.c_str(), "rb");
        if (!f)
            Die("cannot open " + o.srcPath);
        for (auto &r : raw)
            if (fread(r.data(), sizeof(T), frameSamples, f) != frameSamples)
                Die("the source holds fewer frames");
        fclose(f);
        for (const auto &r : raw)
            for (T v : r)
                if (v > pixelMax)
                    Die("the source holds samples above " + std::to_string(pixelMax) + ", more than --bits " + std::to_string(o.bits) + " take");
        s.clip.resize(frames);
        ParallelFor(frames, [&](int i) {
            const T *py = raw[i].data(), *pu = py + lumaSamples, *pv = pu + chromaSamples;
            Frame<T> &fr = s.clip[i];
            fr.y = MakePlane(py, w, h, pad, padY, s.aw, s.ah, pixelMax);
            if (!gray) {
                fr.u = MakePlane(pu, wc, hc, s.padc, s.padcY, s.aw / s.xr, s.ah / s.yr, pixelMax);
                fr.v = MakePlane(pv, wc, hc, s.padc, s.padcY, s.aw / s.xr, s.ah / s.yr, pixelMax);
            }
            std::array<SmallPlane<T>, 3> level;
            const T *src[3] = {py, pu, pv};
            for (int p = 0; p < (gray ? 1 : 3); ++p) {
                level[p].w = p ? wc : w;
                level[p].h = p ? hc : h;
                level[p].p.assign(src[p], src[p] + static_cast<size_t>(level[p].w) * level[p].h);
            }
            for (int L = 1; L <= s.topLevel; ++L) {
                for (int p = 0; p < (gray ? 1 : 3); ++p)
                    level[p] = Reduce(level[p]);
                fr.levels.push_back(level);
            }
        });
    }

    // For tracking down a difference (see the top of the file)
    if (const char *probe = getenv("REFERENCE_PROBE")) {
        int pn, pd, pbx, pby, pvx, pvy;
        if (sscanf(probe, "%d,%d,%d,%d,%d,%d", &pn, &pd, &pbx, &pby, &pvx, &pvy) != 6)
            Die("REFERENCE_PROBE takes N,D,BX,BY,VX,VY");
        printf("SAD %u\n", s.Sad(pn, pn + pd, pbx * s.step, pby * s.stepY, {pvx, pvy}));
        return 0;
    }
    if (const char *path = getenv("REFERENCE_CHECK_SADS")) {
        FILE *f = fopen(path, "rb");
        if (!f)
            Die(std::string("cannot open ") + path);
        int32_t hdr[6];
        int64_t blocks = 0, bad = 0;
        while (fread(hdr, sizeof(hdr), 1, f) == 1) {
            const int count = hdr[2] * hdr[3];
            std::vector<int32_t> a(3 * static_cast<size_t>(count));
            if (fread(a.data(), 4, a.size(), f) != a.size())
                Die("truncated vector file");
            for (int b = 0; b < count; ++b) {
                const uint32_t sad = s.Sad(hdr[0], hdr[0] + hdr[1], s.BX(b), s.BY(b), {a[b], a[count + b]});
                ++blocks;
                if (sad != static_cast<uint32_t>(a[2 * count + b])) {
                    if (bad < 10)
                        printf("field (%d, %+d) block (%d, %d) vector (%d, %d): SAD %u in the file, %u here\n", hdr[0], hdr[1], b % s.nbx, b / s.nbx, a[b],
                               a[count + b], static_cast<uint32_t>(a[2 * count + b]), sad);
                    ++bad;
                }
            }
        }
        fclose(f);
        printf("%lld of %lld SADs differ\n", static_cast<long long>(bad), static_cast<long long>(blocks));
        return 0;
    }

    FILE *out = fopen(o.outPath.c_str(), "wb");
    if (!out)
        Die("cannot write " + o.outPath);
    std::map<Key, Field> done;
    for (const Key &k : s.Order()) {
        // (without coarse levels, no coarse field: no median and no coarse seeds)
        const LevelField co = s.topLevel >= kFinest ? Pyramid<T>{s, k.first, k.first + k.second}.Run() : LevelField{};
        Field f = Refine(s, k.first, k.second, BuildSeeds(s, k.first, k.second, co, done));
        WriteField(out, k.first, k.second, pel, f);
        if (!o.standalone)
            done[k] = std::move(f);
    }
    fclose(out);
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                Die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--src") o.srcPath = next();
        else if (a == "--size") { if (sscanf(next().c_str(), "%dx%d", &o.w, &o.h) != 2) Die("--size takes WxH"); }
        else if (a == "--frames") o.frames = atoi(next().c_str());
        else if (a == "--out") o.outPath = next();
        else if (a == "--format") o.format = next();
        else if (a == "--bits") o.bits = atoi(next().c_str());
        else if (a == "--blksize") o.blk = atoi(next().c_str());
        else if (a == "--blksizev") o.blkY = atoi(next().c_str());
        else if (a == "--overlap") o.overlap = atoi(next().c_str());
        else if (a == "--overlapv") o.overlapY = atoi(next().c_str());
        else if (a == "--pad") o.pad = atoi(next().c_str());
        else if (a == "--padv") o.padY = atoi(next().c_str());
        else if (a == "--superblksize") o.superBlk = atoi(next().c_str());
        else if (a == "--superblksizev") o.superBlkY = atoi(next().c_str());
        else if (a == "--superoverlap") o.superOverlap = atoi(next().c_str());
        else if (a == "--superoverlapv") o.superOverlapY = atoi(next().c_str());
        else if (a == "--pel") o.pel = atoi(next().c_str());
        else if (a == "--radius") o.radius = atoi(next().c_str());
        else if (a == "--delta") o.delta = atoi(next().c_str());
        else if (a == "--standalone") o.standalone = true;
        else if (a == "--onelevel") o.onelevel = true;
        else if (a == "--chroma") o.chroma = atoi(next().c_str()) != 0;
        else if (a == "--plevel") o.plevel = atoi(next().c_str());
        else if (a == "--mvlambda") o.mvlambda = atoll(next().c_str());
        else if (a == "--lsad") o.lsad = atoll(next().c_str());
        else if (a == "--badsad") o.badsad = atoll(next().c_str());
        else if (a == "--badrange") o.badrange = atoi(next().c_str());
        else if (a == "--badstep") o.badstep = atoi(next().c_str());
        else if (a == "--satd") o.satd = true;
        else if (a == "--levels") o.levels = atoi(next().c_str());
        else if (a == "--globalmv") o.globalmv = atoi(next().c_str()) != 0;
        else if (a == "--pnew") o.pnew = atoi(next().c_str());
        else if (a == "--pzero") o.pzero = atoi(next().c_str());
        else if (a == "--pglobal") o.pglobal = atoi(next().c_str());
        else if (a == "--pelsearch") o.pelsearch = atoi(next().c_str());
        else if (a == "--threads") gThreads = std::max(1, atoi(next().c_str()));
        else Die("unknown option " + a);
    }
    if (o.srcPath.empty() || o.outPath.empty() || o.w < 1 || o.h < 1 || o.frames < 1)
        Die("usage: reference --src frames.yuv --size WxH --frames N --out vectors.bin [options]; see the top of reference.cpp");
    if (o.format != "420" && o.format != "422" && o.format != "440" && o.format != "444" && o.format != "gray")
        Die("--format takes 420, 422, 440, 444 or gray");
    if ((o.bits < 8 || o.bits > 16) && o.bits != 32)
        Die("--bits takes 8 to 16, or 32 for floats");
    if (o.blkY < 0)
        o.blkY = o.blk;
    if (o.overlapY < 0)
        o.overlapY = o.overlap;
    if (o.padY < 0)
        o.padY = o.pad;
    if (o.superBlk < 0)
        o.superBlk = o.blk;
    if (o.superBlkY < 0)
        o.superBlkY = o.superBlk == o.blk ? o.blkY : o.superBlk;
    if (o.superOverlap < 0)
        o.superOverlap = o.overlap;
    if (o.superOverlapY < 0)
        o.superOverlapY = o.superOverlap == o.overlap ? o.overlapY : o.superOverlap;
    if (!((o.blk == 4 && o.blkY == 4) || (o.blk == 8 && (o.blkY == 8 || o.blkY == 4)) || (o.blk == 16 && (o.blkY == 16 || o.blkY == 8 || o.blkY == 2)) ||
          (o.blk == 32 && (o.blkY == 32 || o.blkY == 16)) || (o.blk == 64 && (o.blkY == 64 || o.blkY == 32)) || (o.blk == 128 && (o.blkY == 128 || o.blkY == 64))))
        Die("--blksize and --blksizev take mvu's sizes: 4x4, 8x4, 8x8, 16x2, 16x8, 16x16, 32x16, 32x32, 64x32, 64x64, 128x64 or 128x128");
    if (o.satd && o.blk == 16 && o.blkY == 2)
        Die("satd cannot work with 16x2 blocks");
    const int xr = o.format == "420" || o.format == "422" ? 2 : 1, yr = o.format == "420" || o.format == "440" ? 2 : 1;
    if (o.overlap < 0 || o.overlap > o.blk / 2 || o.overlapY < 0 || o.overlapY > o.blkY / 2 || o.overlap % xr || o.overlapY % yr)
        Die("the overlap must be at most half the block size, and divisible by chroma's subsampling");
    if (o.superBlk < 1 || o.superBlkY < 1 || o.superOverlap < 0 || o.superOverlap > o.superBlk / 2 || o.superOverlapY < 0 || o.superOverlapY > o.superBlkY / 2)
        Die("the super's grid is invalid");
    if (o.pel != 1 && o.pel != 2 && o.pel != 4)
        Die("--pel takes 1, 2 or 4");
    if (o.radius < 1 || o.delta < 1)
        Die("--radius and --delta must be positive");
    if (o.plevel < 0 || o.plevel > 2)
        Die("--plevel takes 0, 1 or 2");
    if (o.pad < 1 || o.padY < 1)
        Die("--pad and --padv must be positive");
    if (o.badstep < 1 || o.badstep > 8)
        Die("--badstep takes 1 to 8");
    if (o.pnew < 0 || o.pnew > 256 || o.pzero > 256 || o.pglobal < 0 || o.pglobal > 256)
        Die("--pnew, --pzero and --pglobal take 0 to 256");
    if (o.pelsearch == 0 || o.pelsearch < -1 || (o.pelsearch > 0 && o.pelsearch / o.pel > 32))
        Die("--pelsearch must be positive, at most 33 * pel - 1");
    return o.bits == 8 ? Run<uint8_t>(o) : o.bits == 32 ? Run<float>(o) : Run<uint16_t>(o);
}
