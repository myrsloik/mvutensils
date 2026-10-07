#include "FilterShared.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <string>

std::string CheckChromaStrides(std::initializer_list<const VSFrame *> frames, const VSAPI *vsapi) {
    for (const VSFrame *f : frames)
        if (vsapi->getVideoFrameFormat(f)->numPlanes == 3 && vsapi->getStride(f, 1) != vsapi->getStride(f, 2))
            return "a frame's U and V planes have different strides";
    return {};
}

std::vector<int32_t> MakeOverlapWindows(int nx, int ny, int ox, int oy) {
    std::vector<float> fWin1UVx(nx), fWin1UVxfirst(nx), fWin1UVxlast(nx);
    for (int i = 0; i < ox; i++) {
        fWin1UVx[i] = cosf(std::numbers::pi_v<float> * (i - ox + 0.5f) / (ox * 2));
        fWin1UVx[i] = fWin1UVx[i] * fWin1UVx[i];
        fWin1UVxfirst[i] = 1;
        fWin1UVxlast[i] = fWin1UVx[i];
    }
    for (int i = ox; i < nx - ox; i++) {
        fWin1UVx[i] = 1;
        fWin1UVxfirst[i] = 1;
        fWin1UVxlast[i] = 1;
    }
    for (int i = nx - ox; i < nx; i++) {
        fWin1UVx[i] = cosf(std::numbers::pi_v<float> * (i - nx + ox + 0.5f) / (ox * 2));
        fWin1UVx[i] = fWin1UVx[i] * fWin1UVx[i];
        fWin1UVxfirst[i] = fWin1UVx[i];
        fWin1UVxlast[i] = 1;
    }
    std::vector<float> fWin1UVy(ny), fWin1UVyfirst(ny), fWin1UVylast(ny);
    for (int i = 0; i < oy; i++) {
        fWin1UVy[i] = cosf(std::numbers::pi_v<float> * (i - oy + 0.5f) / (oy * 2));
        fWin1UVy[i] = fWin1UVy[i] * fWin1UVy[i];
        fWin1UVyfirst[i] = 1;
        fWin1UVylast[i] = fWin1UVy[i];
    }
    for (int i = oy; i < ny - oy; i++) {
        fWin1UVy[i] = 1;
        fWin1UVyfirst[i] = 1;
        fWin1UVylast[i] = 1;
    }
    for (int i = ny - oy; i < ny; i++) {
        fWin1UVy[i] = cosf(std::numbers::pi_v<float> * (i - ny + oy + 0.5f) / (oy * 2));
        fWin1UVy[i] = fWin1UVy[i] * fWin1UVy[i];
        fWin1UVyfirst[i] = fWin1UVy[i];
        fWin1UVylast[i] = 1;
    }
    const std::vector<float> *ys[3] = {&fWin1UVyfirst, &fWin1UVy, &fWin1UVylast}, *xs[3] = {&fWin1UVxfirst, &fWin1UVx, &fWin1UVxlast};
    std::vector<int32_t> windows(static_cast<size_t>(9) * nx * ny);
    for (int w = 0; w < 9; ++w)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                windows[(static_cast<size_t>(w) * ny + j) * nx + i] = static_cast<int16_t>((int)((*ys[w / 3])[j] * (*xs[w % 3])[i] * 2048 + 0.5f));
    return windows;
}

double ThSCDScale(const VectorInfo &v) {
    return static_cast<double>(v.blkX * v.blkY) / (8.0 * 8.0) * (v.chroma ? (1.0 + 2.0 / (v.xRatio * v.yRatio)) : 1.0) *
           (((1 << std::min(16, v.bits)) - 1) / 255.0);
}

SceneChange ScaleSceneChange(const VectorInfo &v, int64_t thscd1, float thscd2) {
    constexpr int maxSAD = 8 * 8 * 255;
    if (thscd1 < 0 || thscd1 > maxSAD)
        throw std::runtime_error("thscd1 must be between 0 and " + std::to_string(maxSAD));
    if (!std::isfinite(thscd2) || thscd2 < 0.0f || thscd2 > 100.0f)
        throw std::runtime_error("thscd2 must be a percentage between 0 and 100");
    SceneChange s;
    s.thscd1 = static_cast<int>(static_cast<int64_t>(thscd1 * ThSCDScale(v) + 0.5));
    // IsSceneChange compares the count with the fraction of blocks as a float: a count above it is above its floor
    const float blocks = static_cast<float>(static_cast<double>(thscd2) * v.nbx * v.nby / 100.0);
    s.limit = static_cast<int>(std::floor(blocks));
    return s;
}

std::vector<ResizeTap> BilinearTaps(unsigned srcDim, unsigned dstDim, double shift, double width) {
    const double scale = static_cast<double>(dstDim) / width;
    if (scale < 1.0)
        throw std::runtime_error("the blocks are too small for the resize's two taps");
    const double step = std::min(scale, 1.0);
    const double support = static_cast<double>(1) / step;
    const unsigned filterSize = std::max(static_cast<unsigned>(std::ceil(support)) * 2U, 1U);
    auto filter = [](double x) { return std::max(1.0 - std::abs(x), 0.0); };
    auto roundHalfup = [](double x) { return x < 0 ? std::floor(x + 0.5) : std::floor(x + 0.49999999999999994); };

    // RowMatrix's rows: an entry is allocated when a value other than the one there is stored
    struct Row {
        size_t left = 0;
        std::vector<double> data;
        double Get(size_t j) const { return j < left || j >= left + data.size() ? 0.0 : data[j - left]; }
        void Set(size_t j, double v) {
            if (Get(j) == v)
                return;
            if (data.empty()) {
                data.assign(1, 0.0);
                left = j;
            } else if (j < left) {
                data.insert(data.begin(), left - j, 0.0);
                left = j;
            } else if (j >= left + data.size()) {
                data.insert(data.end(), j - (left + data.size()) + 1, 0.0);
            }
            data[j - left] = v;
        }
    };
    std::vector<Row> m(dstDim);
    for (unsigned i = 0; i < dstDim; ++i) {
        const double pos = (i + 0.5) / scale + shift;
        const double beginPos = roundHalfup(pos - filterSize / 2.0) + 0.5;
        double total = 0.0;
        for (unsigned j = 0; j < filterSize; ++j) {
            const double xpos = beginPos + j;
            total += filter((xpos - pos) * step);
        }
        size_t left = SIZE_MAX;
        for (unsigned j = 0; j < filterSize; ++j) {
            const double xpos = beginPos + j;
            double realPos = xpos < 0.0 ? -xpos : xpos >= srcDim ? 2.0 * srcDim - xpos : xpos;
            realPos = std::clamp(realPos, 0.0, std::nextafter(static_cast<double>(srcDim), -INFINITY));
            const size_t idx = static_cast<size_t>(std::floor(realPos));
            m[i].Set(idx, m[i].Get(idx) + filter((xpos - pos) * step) / total);
            left = std::min(left, idx);
        }
        if (m[i].Get(left) == 0.0) {
            m[i].Set(left, DBL_EPSILON);
            m[i].Set(left, 0.0);
        }
    }

    size_t taps = 0;
    for (const Row &r : m)
        taps = std::max(taps, r.data.size());
    if (taps > 2)
        throw std::runtime_error("the resize has more than two taps");
    std::vector<ResizeTap> out(dstDim);
    for (unsigned i = 0; i < dstDim; ++i) {
        const size_t left = std::min(m[i].left, static_cast<size_t>(srcDim) - taps);
        double err = 0, errF = 0;
        int16_t sum = 0, greatest = 0;
        size_t greatestIdx = 0;
        int16_t c[2] = {};
        float f[2] = {};
        for (size_t j = 0; j < taps; ++j) {
            const double coeff = m[i].Get(left + j);
            const double expected = coeff * (1 << 14) - err;
            const int16_t coeffI = static_cast<int16_t>(std::lrint(expected));
            err = static_cast<double>(coeffI) - expected;
            const double expectedF = coeff - errF;
            f[j] = static_cast<float>(expectedF);
            errF = static_cast<double>(f[j]) - expectedF;
            if (std::abs(coeffI) > greatest) {
                greatest = coeffI;
                greatestIdx = j;
            }
            sum += coeffI;
            c[j] = coeffI;
        }
        c[greatestIdx] += (1 << 14) - sum;
        if (c[0] < 0 || c[0] + (taps > 1 ? c[1] : 0) != (1 << 14))
            throw std::runtime_error("the resize's taps don't add up");
        out[i] = {static_cast<int32_t>(left), c[0], f[0], f[1]};
    }
    return out;
}

std::vector<int32_t> TileTaps(int srcDim, int dstDim, int step, int overlap) {
    const double srcScale = static_cast<double>(srcDim) / (step * srcDim + overlap);
    std::vector<int32_t> taps(dstDim);
    for (int t = 0; t < dstDim; t += 64) {
        const int n = std::min(64, dstDim - t);
        // The tile's active region, which zimg's graph builder turns into the resize's shift and width
        const double left = t * srcScale, width = n * srcScale;
        const double scaleW = static_cast<double>(n) / width;
        const std::vector<ResizeTap> tile =
            BilinearTaps(static_cast<unsigned>(srcDim), static_cast<unsigned>(n), left - 0.0 / scaleW, width * (static_cast<double>(n) / static_cast<double>(n)));
        for (int i = 0; i < n; ++i)
            taps[t + i] = static_cast<int32_t>((static_cast<uint32_t>(tile[i].left) << 15) | static_cast<uint32_t>(tile[i].c));
    }
    return taps;
}

std::vector<ResizeTap> PlaneTaps(int srcDim, int dstDim, int coverDim) {
    // The active region PlaneResizer gives the source, the blocks' cover cut to the frame, and the
    // shift and width zimg's graph builder makes of it
    const double active = (static_cast<double>(dstDim) / coverDim) * srcDim;
    const double scale = static_cast<double>(dstDim) / active;
    return BilinearTaps(static_cast<unsigned>(srcDim), static_cast<unsigned>(dstDim), 0.0 - 0.0 / scale, active * (static_cast<double>(dstDim) / static_cast<double>(dstDim)));
}

namespace {

bool ResizeHFirst(double xscale, double yscale) {
    const double hFirstCost = std::max(xscale, 1.0) * 2.0 + xscale * std::max(yscale, 1.0);
    const double vFirstCost = std::max(yscale, 1.0) + yscale * std::max(xscale, 1.0) * 2.0;
    return hFirstCost < vFirstCost;
}

} // namespace

PassOrder TilesPassOrder(int srcW, int dstW, int stepX, int overlapX, int srcH, int dstH, int stepY, int overlapY) {
    const double sx = static_cast<double>(srcW) / (stepX * srcW + overlapX), sy = static_cast<double>(srcH) / (stepY * srcH + overlapY);
    int horizontal = 0, vertical = 0;
    for (int w : {std::min(64, dstW), (dstW - 1) % 64 + 1}) {
        for (int h : {std::min(64, dstH), (dstH - 1) % 64 + 1}) {
            const double xscale = static_cast<double>(w) / (w * sx), yscale = static_cast<double>(h) / (h * sy);
            ++(ResizeHFirst(xscale, yscale) ? horizontal : vertical);
        }
    }
    return !vertical ? PassOrder::Horizontal : !horizontal ? PassOrder::Vertical : PassOrder::Mixed;
}

bool PlaneHorizontalFirst(int srcW, int dstW, int coverW, int srcH, int dstH, int coverH) {
    const double activeW = (static_cast<double>(dstW) / coverW) * srcW, activeH = (static_cast<double>(dstH) / coverH) * srcH;
    return ResizeHFirst(static_cast<double>(dstW) / activeW, static_cast<double>(dstH) / activeH);
}
