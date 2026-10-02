// reference.cpp
//
// The CPU reference of mvgpu.Analyse and mvgpu.AnalyseMany: the GPU search step for step, in plain
// C++, so that test/check_reference.py can compare the plugin's vectors with it byte for byte. It
// reads raw 8-bit planar frames and builds what mvgpu.Super builds from them (mvu.Super's planes:
// edge-padded, half-pel Wiener planes, quarter samples computed from them at pel 4; and the coarse
// pyramid, halved with rfilter 1's filter), then refines every field:
//
// - the coarse search on the pyramid: an exhaustive search of +-32 px at the smallest level, then
//   at every finer level down to 1/2 seeds from the level above and two pairs of checkerboard
//   passes under Analyse's cost, lambda times 2^(plevel * level), with 8x8 blocks at every level
//   (Pyramid);
// - a seed list per block of the full-size grid: zero, the field's median, the chained and inverted
//   vectors of fields refined before (AnalyseMany only) and the finest level's vectors around the
//   block (BuildSeeds);
// - the seeds measured, a pair of checkerboard passes, the wide fallback search for the blocks still
//   above badsad and one more pair, then the 8 half-pel positions around each vector, and at pel 4
//   the 8 quarter-pel positions around that (Refine).
//
// The SAD is MVUtensils': luma plus U and V at the chroma block size, the chroma vector the luma
// vector divided by the subsampling, toward zero; with --chroma 0 luma only. The cost is
// SAD + ((lambda * |v - p|^2) >> 8), p the component-wise median of the four neighbours, lambda
// relaxed by (lsad / (lsad + worst neighbour SAD / 2))^2.
//
// Build:  clang-cl /nologo /O2 /std:c++20 /EHsc reference.cpp   (or meson compile mvgpu_reference)
//
// Usage:  reference --src frames.yuv --size WxH --frames N --out vectors.bin [--format 420|444]
//                   [--blksize 16] [--overlap 8] [--pad 16] [--pel 2] [--radius 2] [--delta 1]
//                   [--standalone] [--chroma 1] [--plevel 1] [--mvlambda 1000] [--lsad 400]
//                   [--badsad 1000] [--badrange 40] [--badstep 2] [--threads 16]
//
// The arguments are mvgpu.Super's and mvgpu.AnalyseMany's (--standalone: an Analyse per delta,
// without chained or inverted seeds); badsad is per 8x8 block, as mvgpu takes it. The frames are
// raw 8-bit planar Y, U, V, at 4:2:0 or 4:4:4 (mvgpu.Analyse searches 4:2:0 so far); with
// subsampled chroma the padding must be even, as mvgpu has it. The output holds every field (n, d)
// as six 32-bit ints, n, d, nbx, nby, pel and 1, then the nbx * nby x components, the y components
// and the SADs.
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
#include <vector>

#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#define REFERENCE_SSE2
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

// One plane of one frame as mvu.Super pel 2 holds it: edge-padded, plus the three half-pel planes
struct Plane {
    int w = 0, h = 0;          // padded size
    std::vector<uint8_t> p[4]; // full, x + 1/2, y + 1/2, both
};

uint8_t Avg(int a, int b) {
    return static_cast<uint8_t>((a + b + 1) >> 1);
}

uint8_t Wiener(int m0, int m1, int m2, int m3, int m4, int m5) {
    const int m = (((m2 + m3) * 4 - (m1 + m4)) * 5 + m0 + m5 + 16) >> 5;
    return static_cast<uint8_t>(std::clamp(m, 0, 255));
}

// SuperPyramid.cpp's HorizontalWiener and VerticalWiener, border handling included
void HorizontalWiener(uint8_t *dst, const uint8_t *src, int w, int h) {
    for (int j = 0; j < h; ++j) {
        const uint8_t *s = src + static_cast<size_t>(j) * w;
        uint8_t *d = dst + static_cast<size_t>(j) * w;
        d[0] = Avg(s[0], s[1]);
        d[1] = Avg(s[1], s[2]);
        for (int i = 2; i < w - 4; ++i)
            d[i] = Wiener(s[i - 2], s[i - 1], s[i], s[i + 1], s[i + 2], s[i + 3]);
        for (int i = w - 4; i < w - 1; ++i)
            d[i] = Avg(s[i], s[i + 1]);
        d[w - 1] = s[w - 1];
    }
}

void VerticalWiener(uint8_t *dst, const uint8_t *src, int w, int h) {
    auto row = [&](const uint8_t *base, int j) { return base + static_cast<size_t>(j) * w; };
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < w; ++i)
            dst[static_cast<size_t>(j) * w + i] = Avg(row(src, j)[i], row(src, j + 1)[i]);
    for (int j = 2; j < h - 4; ++j) {
        const uint8_t *r0 = row(src, j - 2), *r1 = row(src, j - 1), *r2 = row(src, j), *r3 = row(src, j + 1), *r4 = row(src, j + 2), *r5 = row(src, j + 3);
        uint8_t *d = dst + static_cast<size_t>(j) * w;
        for (int i = 0; i < w; ++i)
            d[i] = Wiener(r0[i], r1[i], r2[i], r3[i], r4[i], r5[i]);
    }
    for (int j = h - 4; j < h - 1; ++j)
        for (int i = 0; i < w; ++i)
            dst[static_cast<size_t>(j) * w + i] = Avg(row(src, j)[i], row(src, j + 1)[i]);
    memcpy(dst + static_cast<size_t>(h - 1) * w, row(src, h - 1), w);
}

// w x h pixels of src, extended to the block-aligned aw x ah and padded by padX and padY, repeating
// the edge pixels, as CopyAndPadPlane does; then its half-pel planes
Plane MakePlane(const uint8_t *src, int w, int h, int padX, int padY, int aw, int ah) {
    Plane pl;
    pl.w = aw + 2 * padX;
    pl.h = ah + 2 * padY;
    for (auto &p : pl.p)
        p.resize(static_cast<size_t>(pl.w) * pl.h);
    uint8_t *d = pl.p[0].data();
    for (int y = 0; y < pl.h; ++y) {
        const int sy = std::clamp(y - padY, 0, h - 1);
        for (int x = 0; x < pl.w; ++x)
            d[static_cast<size_t>(y) * pl.w + x] = src[static_cast<size_t>(sy) * w + std::clamp(x - padX, 0, w - 1)];
    }
    HorizontalWiener(pl.p[1].data(), pl.p[0].data(), pl.w, pl.h);
    VerticalWiener(pl.p[2].data(), pl.p[0].data(), pl.w, pl.h);
    HorizontalWiener(pl.p[3].data(), pl.p[2].data(), pl.w, pl.h);
    return pl;
}

// pel 4: the quarter-pel sample (X, Y) of a padded plane, in quarter pels, from its half-pel planes
// exactly as mvu.Super's quarter planes hold it (SuperPyramid.cpp's GeneratePelQuarters): on the
// half-pel grid that plane's sample; otherwise the rounded average of its two neighbours on the
// grid, diagonally the average of the two vertical averages
uint8_t QuarterSample(const Plane &pl, int X, int Y) {
    auto at = [&](int Xh, int Yh) { return static_cast<int>(pl.p[(Xh & 1) | ((Yh & 1) << 1)][static_cast<size_t>(Yh >> 1) * pl.w + (Xh >> 1)]); };
    const int Xa = X >> 1, Xb = Xa + (X & 1), Ya = Y >> 1, Yb = Ya + (Y & 1);
    return Avg(Avg(at(Xa, Ya), at(Xa, Yb)), Avg(at(Xb, Ya), at(Xb, Yb)));
}

// One plane of the pyramid: halved repeatedly, unpadded, edges repeated on access
struct SmallPlane {
    int w = 0, h = 0;
    std::vector<uint8_t> p;
    uint8_t At(int x, int y) const { return p[static_cast<size_t>(std::clamp(y, 0, h - 1)) * w + std::clamp(x, 0, w - 1)]; }
};

// SuperPyramid.cpp's RB2BilinearFiltered (mvu.Super's default rfilter=1): 1/8, 3/8, 3/8, 1/8 in
// both directions, a plain average at the edges
SmallPlane Reduce(const SmallPlane &src) {
    SmallPlane d;
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
        uint8_t *row = d.p.data() + static_cast<size_t>(y) * d.w;
        row[0] = static_cast<uint8_t>((tmp[0] + tmp[1] + 1) >> 1);
        for (int x = 1; x < d.w - 1; ++x)
            row[x] = static_cast<uint8_t>((tmp[2 * x - 1] + (tmp[2 * x] + tmp[2 * x + 1]) * 3 + tmp[2 * x + 2] + 4) >> 3);
        if (d.w > 1)
            row[d.w - 1] = static_cast<uint8_t>((tmp[2 * (d.w - 1)] + tmp[2 * (d.w - 1) + 1] + 1) >> 1);
    }
    return d;
}

struct Frame {
    Plane y, u, v;
    // levels[L - 1] holds Y, U, V at 1 / 2^L of the frame's size, L = 1 .. top level
    std::vector<std::array<SmallPlane, 3>> levels;
};

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

using Key = std::pair<int, int>; // field (n, d)

struct Vec {
    int x = 0, y = 0; // 1 / pel pixels at full size, full pels of their level in the pyramid
    bool operator==(const Vec &o) const { return x == o.x && y == o.y; }
};

struct Field {
    int nbx = 0, nby = 0;
    std::vector<Vec> v;
    std::vector<int> sad;
};

struct Search {
    // The clip: frames of w x h, chroma subsampled by xr and yr (1 or 2)
    int w = 0, h = 0, frames = 0, xr = 2, yr = 2;
    std::vector<Frame> clip;
    // The grid: blk x blk blocks, step apart, nbx x nby of them covering the block-aligned aw x ah,
    // extended by a column or row while they fall short of the frame, as mvu.Super and mvu.Analyse
    // have it; the planes padded by pad, chroma by padc horizontally and padcY vertically
    int blk = 16, step = 8, nbx = 0, nby = 0, aw = 0, ah = 0, pad = 16, padc = 8, padcY = 8;
    int pel = 2;
    int topLevel = 0; // pyramid levels built, 1 / 2^topLevel the smallest
    // The fields: radius of them either side, delta frames apart
    int radius = 2, delta = 1;
    bool standalone = false; // no chained or inverted seeds
    bool chroma = true;      // the SAD counts U and V
    int plevel = 1;          // lambda * 2^(plevel * L) at coarse level L
    int64_t lambda0 = 0;     // mvlambda scaled to the block size and divided by pel squared, the full-size grid's
    int64_t lambdaBlock = 0; // mvlambda scaled to the block size, the coarse levels' before plevel
    int64_t lsad = 0;        // scaled to the block size
    int badSad = 0;          // scaled to the block size
    int fallbackRadius = 0;  // px; 0 = no fallback
    int fallbackStep = 1;    // px between the positions it searches, then the positions around the best

    // Block b's position in the frame
    int BX(int b) const { return (b % nbx) * step; }
    int BY(int b) const { return (b / nbx) * step; }

    // The range ValidateVectors accepts for a block at (x, y): within the block-aligned frame and
    // its padding. ClampHalf keeps to the half-pel grid inside it, ClampFull to the full-pel grid,
    // the grids the seeds, passes and half-pel step (ClampHalf) and the fallback (ClampFull) stay on
    // at pel 4; at pel 2 all three are the same.
    Vec ClampTo(int x, int y, Vec v, int top) const {
        return {std::clamp(v.x, -pel * (x + pad), pel * (aw + pad - blk - x) - top), std::clamp(v.y, -pel * (y + pad), pel * (ah + pad - blk - y) - top)};
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
    // block: read in place when the position is on the half-pel grid, else computed (pel 4)
    int PlaneSad(const uint8_t *cur, ptrdiff_t curStride, const Plane &ref, int X, int Y, int bw, int bh) const {
        if (pel == 2 || ((X | Y) & 1) == 0) {
            const int Xh = pel == 2 ? X : X >> 1, Yh = pel == 2 ? Y : Y >> 1;
            return BlockSad(cur, curStride, ref.p[(Xh & 1) | ((Yh & 1) << 1)].data() + static_cast<size_t>(Yh >> 1) * ref.w + (Xh >> 1), ref.w, bw, bh);
        }
        std::vector<uint8_t> tmp(static_cast<size_t>(bw) * bh);
        for (int j = 0; j < bh; ++j)
            for (int i = 0; i < bw; ++i)
                tmp[j * bw + i] = QuarterSample(ref, X + 4 * i, Y + 4 * j);
        return BlockSad(cur, curStride, tmp.data(), bw, bw, bh);
    }

    // The SAD of the block at (x, y) of frame n against frame r displaced by v
    int Sad(int n, int r, int x, int y, Vec v) const {
        const Frame &cf = clip[n], &rf = clip[r];
        const uint8_t *cur = cf.y.p[0].data() + static_cast<size_t>(y + pad) * cf.y.w + (x + pad);
        int sad = PlaneSad(cur, cf.y.w, rf.y, pel * (x + pad) + v.x, pel * (y + pad) + v.y, blk, blk);
        if (!chroma)
            return sad;
        // Analyse divides the chroma vector by the subsampling, toward zero
        const int xc = x / xr + padc, yc = y / yr + padcY, Xc = pel * xc + v.x / xr, Yc = pel * yc + v.y / yr;
        const size_t cc = static_cast<size_t>(yc) * cf.u.w + xc;
        sad += PlaneSad(cf.u.p[0].data() + cc, cf.u.w, rf.u, Xc, Yc, blk / xr, blk / yr);
        sad += PlaneSad(cf.v.p[0].data() + cc, cf.v.w, rf.v, Xc, Yc, blk / xr, blk / yr);
        return sad;
    }

    // The SAD of the 8x8 block at (x, y) of pyramid level L against frame r displaced by v, in full
    // pels of that level: luma plus U and V at the chroma block size, edges repeated
    int CoarseSad(int n, int r, int L, int x, int y, Vec v) const {
        const auto &cl = clip[n].levels[L - 1], &rl = clip[r].levels[L - 1];
        const SmallPlane &cy = cl[0], &ry = rl[0];
        int sad = 0;
        if (x + v.x >= 0 && y + v.y >= 0 && x + v.x + 8 <= ry.w && y + v.y + 8 <= ry.h && x + 8 <= cy.w && y + 8 <= cy.h) {
            sad = BlockSad(cy.p.data() + static_cast<size_t>(y) * cy.w + x, cy.w, ry.p.data() + static_cast<size_t>(y + v.y) * ry.w + x + v.x, ry.w, 8, 8);
        } else {
            for (int j = 0; j < 8; ++j)
                for (int i = 0; i < 8; ++i)
                    sad += std::abs(cy.At(x + i, y + j) - ry.At(x + i + v.x, y + j + v.y));
        }
        if (!chroma)
            return sad;
        const int cvx = v.x / xr, cvy = v.y / yr, xc = x / xr, yc = y / yr;
        for (int p = 1; p <= 2; ++p)
            for (int j = 0; j < 8 / yr; ++j)
                for (int i = 0; i < 8 / xr; ++i)
                    sad += std::abs(cl[p].At(xc + i, yc + j) - rl[p].At(xc + i + cvx, yc + j + cvy));
        return sad;
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
    std::vector<int> sad;
};

struct Pyramid {
    const Search &s;
    int n, r;
    int PadAt(int L) const { return std::max(1, s.pad >> L); }
    const SmallPlane &Luma(int L) const { return s.clip[n].levels[L - 1][0]; }
    // Keep the reference block within the level's (scaled-down) padding
    Vec Bound(int L, int x, int y, Vec v) const {
        const int pad = PadAt(L);
        return {std::clamp(v.x, -(x + pad), Luma(L).w + pad - 8 - x), std::clamp(v.y, -(y + pad), Luma(L).h + pad - 8 - y)};
    }
    LevelField Blank(int L) const {
        LevelField f;
        f.nbx = std::max(1, Luma(L).w / 8);
        f.nby = std::max(1, Luma(L).h / 8);
        f.v.assign(static_cast<size_t>(f.nbx) * f.nby, Vec{});
        f.sad.assign(f.v.size(), INT32_MAX);
        return f;
    }

    // Checkerboard passes under Analyse's cost at level L: lambda = mvlambda * 2^(plevel * L) at full
    // pel, relaxed by lsad; candidates are the four neighbours' vectors and the 8 positions one pixel
    // around the winner
    void Passes(int L, LevelField &f) const {
        const int64_t lambdaL = s.lambdaBlock << (s.plevel * L);
        for (int pass = 0; pass < 2 * kLevelPairs; ++pass) {
            const int colour = pass & 1;
            ParallelFor(f.nby, [&](int by) {
                for (int bx = (by + colour) & 1; bx < f.nbx; bx += 2) {
                    const int b = by * f.nbx + bx, x = bx * 8, y = by * 8;
                    const int nbs[4] = {bx > 0 ? b - 1 : -1, bx + 1 < f.nbx ? b + 1 : -1, by > 0 ? b - f.nbx : -1, by + 1 < f.nby ? b + f.nbx : -1};
                    int xs[4], ys[4], worst = 0;
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
                    const double sc = static_cast<double>(s.lsad) / std::max<int64_t>(s.lsad + (worst >> 1), 1);
                    const int64_t lambda = static_cast<int64_t>(lambdaL * sc * sc);
                    auto cost = [&](int sad, Vec v) {
                        const int64_t dx = v.x - p.x, dy = v.y - p.y;
                        return sad + ((lambda * (dx * dx + dy * dy)) >> 8);
                    };
                    Vec best = f.v[b];
                    int bestSad = f.sad[b];
                    int64_t bestCost = cost(bestSad, best);
                    auto consider = [&](Vec v) {
                        v = Bound(L, x, y, v);
                        if (v == best)
                            return;
                        const int sad = s.CoarseSad(n, r, L, x, y, v);
                        const int64_t c = cost(sad, v);
                        if (c < bestCost) {
                            bestCost = c;
                            best = v;
                            bestSad = sad;
                        }
                    };
                    for (int i : nbs)
                        if (i >= 0)
                            consider(f.v[i]);
                    const Vec c = best;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            if (dx || dy)
                                consider({c.x + dx, c.y + dy});
                    f.v[b] = best;
                    f.sad[b] = bestSad;
                }
            });
        }
    }

    // The finest level's field
    LevelField Run() const {
        const int T = s.topLevel;
        LevelField cur = Blank(T);
        ParallelFor(cur.nby, [&](int by) {
            for (int bx = 0; bx < cur.nbx; ++bx) {
                const int b = by * cur.nbx + bx, x = bx * 8, y = by * 8;
                const Vec lo = Bound(T, x, y, {-kTopRadius, -kTopRadius}), hi = Bound(T, x, y, {kTopRadius, kTopRadius});
                int bestLen = INT32_MAX;
                for (int vy = lo.y; vy <= hi.y; ++vy)
                    for (int vx = lo.x; vx <= hi.x; ++vx) {
                        const int sad = s.CoarseSad(n, r, T, x, y, {vx, vy});
                        const int len = vx * vx + vy * vy;
                        if (sad < cur.sad[b] || (sad == cur.sad[b] && len < bestLen)) {
                            cur.sad[b] = sad;
                            cur.v[b] = {vx, vy};
                            bestLen = len;
                        }
                    }
            }
        });
        Passes(T, cur);
        for (int L = T - 1; L >= kFinest; --L) {
            LevelField next = Blank(L);
            std::vector<int> mx, my;
            for (const Vec &v : cur.v) {
                mx.push_back(2 * v.x);
                my.push_back(2 * v.y);
            }
            const Vec global{Search::Median(mx), Search::Median(my)};
            ParallelFor(next.nby, [&](int by) {
                for (int bx = 0; bx < next.nbx; ++bx) {
                    const int b = by * next.nbx + bx, x = bx * 8, y = by * 8;
                    const int px = std::min(bx / 2, cur.nbx - 1), py = std::min(by / 2, cur.nby - 1);
                    std::vector<Vec> cand;
                    auto add = [&](Vec v) {
                        v = Bound(L, x, y, v);
                        if (std::find(cand.begin(), cand.end(), v) == cand.end())
                            cand.push_back(v);
                    };
                    static const int around[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                    for (const auto &o : around) {
                        const int ax = px + o[0], ay = py + o[1];
                        if (ax >= 0 && ay >= 0 && ax < cur.nbx && ay < cur.nby) {
                            const Vec v = cur.v[static_cast<size_t>(ay) * cur.nbx + ax];
                            add({2 * v.x, 2 * v.y});
                        }
                    }
                    add(Vec{});
                    add(global);
                    for (const Vec &v : cand) {
                        const int sad = s.CoarseSad(n, r, L, x, y, v);
                        if (sad < next.sad[b]) {
                            next.sad[b] = sad;
                            next.v[b] = v;
                        }
                    }
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

// The seeds of every block of field (n, d): zero, the field's median, the chained and inverted
// vectors of the fields already refined, the finest coarse level's vectors around the block;
// clamped, on the half-pel grid, without duplicates
std::vector<std::vector<Vec>> BuildSeeds(const Search &s, int n, int d, const LevelField &co, const std::map<Key, Field> &done) {
    const int nb = s.nbx * s.nby;
    const int scale = s.pel << kFinest; // the finest level's full pels in the grid's units
    std::vector<std::vector<Vec>> seeds(nb);
    auto add = [&](int b, Vec v) {
        v = s.ClampHalf(s.BX(b), s.BY(b), v);
        auto &list = seeds[b];
        if (std::find(list.begin(), list.end(), v) == list.end())
            list.push_back(v);
    };
    std::vector<int> mx, my;
    for (const Vec &v : co.v) {
        mx.push_back(v.x * scale);
        my.push_back(v.y * scale);
    }
    const Vec global{Search::Median(mx), Search::Median(my)};

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
    std::vector<int> invSad;
    if (invF) {
        // The block at p' in frame n + d matched p' + u in frame n: the block of frame n nearest to
        // that position moves by -u
        inv.assign(nb, Vec{});
        invSad.assign(nb, INT32_MAX);
        for (int b = 0; b < nb; ++b) {
            const Vec w = invF->v[b];
            const double unit = s.pel;
            const int tx = static_cast<int>(std::lround((s.BX(b) + w.x / unit) / s.step)), ty = static_cast<int>(std::lround((s.BY(b) + w.y / unit) / s.step));
            if (tx < 0 || ty < 0 || tx >= s.nbx || ty >= s.nby)
                continue;
            const int t = ty * s.nbx + tx;
            if (invF->sad[b] < invSad[t]) {
                invSad[t] = invF->sad[b];
                inv[t] = {-w.x, -w.y};
            }
        }
    }
    for (int b = 0; b < nb; ++b) {
        const int x = s.BX(b), y = s.BY(b);
        add(b, Vec{});
        add(b, global);
        if (stepF) {
            const Vec v1 = stepF->v[b];
            const double unit = s.pel;
            const int tx = std::clamp(static_cast<int>(std::lround((x + v1.x / unit) / s.step)), 0, s.nbx - 1);
            const int ty = std::clamp(static_cast<int>(std::lround((y + v1.y / unit) / s.step)), 0, s.nby - 1);
            const Vec v2 = restF->v[ty * s.nbx + tx];
            add(b, s.HalfGrid(Vec{v1.x + v2.x, v1.y + v2.y}));
        }
        if (invF && invSad[b] != INT32_MAX)
            add(b, s.HalfGrid(inv[b]));
        const int span = 8 << kFinest;
        const int cx = std::min((x + s.blk / 2) / span, co.nbx - 1), cy = std::min((y + s.blk / 2) / span, co.nby - 1);
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
Field Refine(const Search &s, int n, int d, const std::vector<std::vector<Vec>> &seeds) {
    const int nb = s.nbx * s.nby;
    const int r = n + d;
    Field f;
    f.nbx = s.nbx;
    f.nby = s.nby;
    f.v.assign(nb, Vec{});
    f.sad.assign(nb, INT32_MAX);
    std::vector<std::vector<int>> seedSad(nb);
    ParallelFor(s.nby, [&](int by) {
        for (int bx = 0; bx < s.nbx; ++bx) {
            const int b = by * s.nbx + bx;
            for (const Vec &v : seeds[b]) {
                const int sad = s.Sad(n, r, s.BX(b), s.BY(b), v);
                seedSad[b].push_back(sad);
                if (sad < f.sad[b]) {
                    f.sad[b] = sad;
                    f.v[b] = v;
                }
            }
        }
    });

    // The lambda of every (worst neighbour SAD) >> 1, as Analyse.cpp tabulates it
    auto lambdaOf = [&](int worst) {
        const double scale = static_cast<double>(s.lsad) / std::max<int64_t>(s.lsad + (worst >> 1), 1);
        return static_cast<int64_t>(s.lambda0 * scale * scale);
    };
    // Predictor and relaxed lambda of block b from its four neighbours
    auto context = [&](int b, Vec &p, int64_t &lambda) {
        const int bx = b % s.nbx, by = b / s.nbx;
        const int nbs[4] = {bx > 0 ? b - 1 : b, bx + 1 < s.nbx ? b + 1 : b, by > 0 ? b - s.nbx : b, by + 1 < s.nby ? b + s.nbx : b};
        int xs[4], ys[4], worst = 0;
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
    auto cost = [](int sad, Vec v, Vec p, int64_t lambda) {
        const int64_t dx = v.x - p.x, dy = v.y - p.y;
        return sad + ((lambda * (dx * dx + dy * dy)) >> 8);
    };

    // The blocks still above badSad: a full-pel search of +-fallbackRadius px around their
    // predictor, every fallbackStep px, then the positions up to fallbackStep - 1 px around the best
    auto fallback = [&]() {
        const int R = s.fallbackRadius;
        std::vector<Vec> nv(f.v);
        std::vector<int> ns(f.sad);
        ParallelFor(s.nby, [&](int by) {
            for (int bx = 0; bx < s.nbx; ++bx) {
                const int b = by * s.nbx + bx, x = s.BX(b), y = s.BY(b);
                if (f.sad[b] <= s.badSad)
                    continue;
                Vec p;
                int64_t lambda;
                context(b, p, lambda);
                Vec best = f.v[b];
                int bestSad = f.sad[b];
                int64_t bestCost = cost(bestSad, best, p, lambda);
                // full-pel positions around the predictor, rounded to a full pel
                const int pel = s.pel, cx = p.x & ~(pel - 1), cy = p.y & ~(pel - 1), stepH = pel * s.fallbackStep;
                const Vec lo = s.ClampFull(x, y, {cx - pel * R, cy - pel * R}), hi = s.ClampFull(x, y, {cx + pel * R, cy + pel * R});
                auto tryAt = [&](Vec v) {
                    const int sad = s.Sad(n, r, x, y, v);
                    const int64_t c = cost(sad, v, p, lambda);
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

    // The checkerboard passes: each block of the pass's colour tries its seeds, its neighbours'
    // vectors and the 8 positions one pixel around the winner
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
                int bestSad = f.sad[b];
                int64_t bestCost = cost(bestSad, best, p, lambda);
                auto consider = [&](Vec v, int sad) {
                    const int64_t c = cost(sad, v, p, lambda);
                    if (c < bestCost) {
                        bestCost = c;
                        best = v;
                        bestSad = sad;
                    }
                };
                for (size_t i = 0; i < seeds[b].size(); ++i)
                    consider(seeds[b][i], seedSad[b][i]);
                const int nbs[4] = {bx > 0 ? b - 1 : -1, bx + 1 < s.nbx ? b + 1 : -1, by > 0 ? b - s.nbx : -1, by + 1 < s.nby ? b + s.nbx : -1};
                for (int i : nbs) {
                    if (i < 0)
                        continue;
                    const Vec v = s.ClampHalf(x, y, f.v[i]);
                    if (!(v == best))
                        consider(v, s.Sad(n, r, x, y, v));
                }
                const Vec c = best;
                for (int dy = -s.pel; dy <= s.pel; dy += s.pel)
                    for (int dx = -s.pel; dx <= s.pel; dx += s.pel)
                        if (dx || dy) {
                            const Vec v = s.ClampHalf(x, y, {c.x + dx, c.y + dy});
                            if (!(v == c))
                                consider(v, s.Sad(n, r, x, y, v));
                        }
                f.v[b] = best;
                f.sad[b] = bestSad;
            }
        });
    }

    // Sub-pel refinement, every block independently against its neighbours as the step before left
    // them: the 8 positions half a pel around, kept to the half-pel grid; at pel 4 then the 8 a
    // quarter pel around, against the half-pel step's field, anywhere in range
    auto subPelStep = [&](int unit, bool halfGrid) {
        std::vector<Vec> outV(f.v);
        std::vector<int> outSad(f.sad);
        ParallelFor(s.nby, [&](int by) {
            for (int bx = 0; bx < s.nbx; ++bx) {
                const int b = by * s.nbx + bx, x = s.BX(b), y = s.BY(b);
                Vec p;
                int64_t lambda;
                context(b, p, lambda);
                const Vec c = f.v[b];
                Vec best = c;
                int bestSad = f.sad[b];
                int64_t bestCost = cost(bestSad, best, p, lambda);
                for (int dy = -unit; dy <= unit; dy += unit)
                    for (int dx = -unit; dx <= unit; dx += unit)
                        if (dx || dy) {
                            const Vec at{c.x + dx, c.y + dy};
                            const Vec v = halfGrid ? s.ClampHalf(x, y, at) : s.Clamp(x, y, at);
                            if (v == c)
                                continue;
                            const int sad = s.Sad(n, r, x, y, v);
                            const int64_t cc = cost(sad, v, p, lambda);
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
        a[i] = fl.sad[i];
    fwrite(a.data(), 4, a.size(), f);
}

} // namespace

int main(int argc, char **argv) {
    std::string srcPath, outPath, format = "420";
    int w = 0, h = 0, frames = 0, blk = 16, overlap = 8, pad = 16, pel = 2, radius = 2, delta = 1, plevel = 1, badrange = 40, badstep = 2;
    int64_t mvlambda = 1000, lsad = 400, badsad = 1000;
    bool standalone = false, chroma = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                Die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--src") srcPath = next();
        else if (a == "--size") { if (sscanf(next().c_str(), "%dx%d", &w, &h) != 2) Die("--size takes WxH"); }
        else if (a == "--frames") frames = atoi(next().c_str());
        else if (a == "--out") outPath = next();
        else if (a == "--format") format = next();
        else if (a == "--blksize") blk = atoi(next().c_str());
        else if (a == "--overlap") overlap = atoi(next().c_str());
        else if (a == "--pad") pad = atoi(next().c_str());
        else if (a == "--pel") pel = atoi(next().c_str());
        else if (a == "--radius") radius = atoi(next().c_str());
        else if (a == "--delta") delta = atoi(next().c_str());
        else if (a == "--standalone") standalone = true;
        else if (a == "--chroma") chroma = atoi(next().c_str()) != 0;
        else if (a == "--plevel") plevel = atoi(next().c_str());
        else if (a == "--mvlambda") mvlambda = atoll(next().c_str());
        else if (a == "--lsad") lsad = atoll(next().c_str());
        else if (a == "--badsad") badsad = atoll(next().c_str());
        else if (a == "--badrange") badrange = atoi(next().c_str());
        else if (a == "--badstep") badstep = atoi(next().c_str());
        else if (a == "--threads") gThreads = std::max(1, atoi(next().c_str()));
        else Die("unknown option " + a);
    }
    if (srcPath.empty() || outPath.empty() || w < 1 || h < 1 || frames < 1)
        Die("usage: reference --src frames.yuv --size WxH --frames N --out vectors.bin [options]; see the top of reference.cpp");
    if (format != "420" && format != "444")
        Die("--format takes 420 or 444");
    if (blk != 8 && blk != 16 && blk != 32)
        Die("--blksize takes 8, 16 or 32");
    if (overlap < 0 || overlap > blk / 2 || (format == "420" && overlap % 2))
        Die("the overlap must be at most half the block size, and even at 4:2:0");
    if (pel != 2 && pel != 4)
        Die("--pel takes 2 or 4");
    if (radius < 1 || delta < 1)
        Die("--radius and --delta must be positive");
    if (plevel < 0 || plevel > 2)
        Die("--plevel takes 0, 1 or 2");
    if (pad < 1)
        Die("--pad must be positive");
    if (badstep < 1 || badstep > 8)
        Die("--badstep takes 1 to 8");

    Search s;
    s.w = w;
    s.h = h;
    s.frames = frames;
    s.xr = s.yr = format == "420" ? 2 : 1;
    if (w % s.xr || h % s.yr)
        Die("4:2:0 frames need even dimensions");
    if (pad % s.xr || pad % s.yr)
        Die("with subsampled chroma the padding must be even, or chroma would be read outside its padding");
    s.blk = blk;
    s.step = blk - overlap;
    s.pad = pad;
    s.padc = pad / s.xr;
    s.padcY = pad / s.yr;
    // MVUtensils' grid: the blocks that fit, plus one more column or row while they fall short
    auto aligned = [&](int size) {
        const int b = s.step * ((size - overlap) / s.step) + overlap;
        return b < size ? b + s.step : b;
    };
    s.aw = aligned(w);
    s.ah = aligned(h);
    s.nbx = (s.aw - overlap) / s.step;
    s.nby = (s.ah - overlap) / s.step;
    if (s.nbx < 1 || s.nby < 1)
        Die("the frame is too small to hold a single block");
    s.pel = pel;
    s.radius = radius;
    s.delta = delta;
    s.standalone = standalone;
    s.chroma = chroma;
    s.plevel = plevel;
    // mvu.Analyse's scaling for 8-bit and the block size, lambda divided by pel squared at full size
    const int64_t area = static_cast<int64_t>(blk) * blk;
    s.lambda0 = mvlambda * area / 64 / (pel * pel);
    s.lambdaBlock = mvlambda * area / 64;
    s.lsad = lsad * area / 64;
    s.badSad = static_cast<int>(std::min<int64_t>(badsad * area / 64, INT32_MAX));
    s.fallbackRadius = std::abs(badrange);
    s.fallbackStep = badstep;
    for (int cw = w; cw / 2 >= kTopWidth; cw = (cw + 1) / 2)
        ++s.topLevel;
    if (s.topLevel < kFinest)
        Die("the frame is narrower than " + std::to_string(2 * kTopWidth) + " pixels, too small for the coarse search");

    // The frames: the super's planes and the pyramid
    const int wc = w / s.xr, hc = h / s.yr;
    const size_t lumaBytes = static_cast<size_t>(w) * h, chromaBytes = static_cast<size_t>(wc) * hc, frameBytes = lumaBytes + 2 * chromaBytes;
    {
        std::vector<std::vector<uint8_t>> raw(frames, std::vector<uint8_t>(frameBytes));
        FILE *f = fopen(srcPath.c_str(), "rb");
        if (!f)
            Die("cannot open " + srcPath);
        for (auto &r : raw)
            if (fread(r.data(), 1, frameBytes, f) != frameBytes)
                Die("the source holds fewer frames");
        fclose(f);
        s.clip.resize(frames);
        ParallelFor(frames, [&](int i) {
            const uint8_t *py = raw[i].data(), *pu = py + lumaBytes, *pv = pu + chromaBytes;
            Frame &fr = s.clip[i];
            fr.y = MakePlane(py, w, h, pad, pad, s.aw, s.ah);
            fr.u = MakePlane(pu, wc, hc, s.padc, s.padcY, s.aw / s.xr, s.ah / s.yr);
            fr.v = MakePlane(pv, wc, hc, s.padc, s.padcY, s.aw / s.xr, s.ah / s.yr);
            std::array<SmallPlane, 3> level;
            const uint8_t *src[3] = {py, pu, pv};
            for (int p = 0; p < 3; ++p) {
                level[p].w = p ? wc : w;
                level[p].h = p ? hc : h;
                level[p].p.assign(src[p], src[p] + static_cast<size_t>(level[p].w) * level[p].h);
            }
            for (int L = 1; L <= s.topLevel; ++L) {
                for (auto &p : level)
                    p = Reduce(p);
                fr.levels.push_back(level);
            }
        });
    }

    // For tracking down a difference (see the top of the file)
    if (const char *probe = getenv("REFERENCE_PROBE")) {
        int pn, pd, pbx, pby, pvx, pvy;
        if (sscanf(probe, "%d,%d,%d,%d,%d,%d", &pn, &pd, &pbx, &pby, &pvx, &pvy) != 6)
            Die("REFERENCE_PROBE takes N,D,BX,BY,VX,VY");
        printf("SAD %d\n", s.Sad(pn, pn + pd, pbx * s.step, pby * s.step, {pvx, pvy}));
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
                const int sad = s.Sad(hdr[0], hdr[0] + hdr[1], s.BX(b), s.BY(b), {a[b], a[count + b]});
                ++blocks;
                if (sad != a[2 * count + b]) {
                    if (bad < 10)
                        printf("field (%d, %+d) block (%d, %d) vector (%d, %d): SAD %d in the file, %d here\n", hdr[0], hdr[1], b % s.nbx, b / s.nbx, a[b],
                               a[count + b], a[2 * count + b], sad);
                    ++bad;
                }
            }
        }
        fclose(f);
        printf("%lld of %lld SADs differ\n", static_cast<long long>(bad), static_cast<long long>(blocks));
        return 0;
    }

    FILE *out = fopen(outPath.c_str(), "wb");
    if (!out)
        Die("cannot write " + outPath);
    std::map<Key, Field> done;
    for (const Key &k : s.Order()) {
        const LevelField co = Pyramid{s, k.first, k.first + k.second}.Run();
        Field f = Refine(s, k.first, k.second, BuildSeeds(s, k.first, k.second, co, done));
        WriteField(out, k.first, k.second, pel, f);
        if (!standalone)
            done[k] = std::move(f);
    }
    fclose(out);
    return 0;
}
