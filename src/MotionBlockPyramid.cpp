#include "MotionBlockPyramid.h"

#include <functional>
#include <algorithm>
#include <limits>
#include <memory>
#include <cmath>
#include "Common.h"


///////////////////////////////////

void MotionBlockLevel::InterpolatePredictorsFromParent(const MotionBlockLevel &parentLevel) noexcept {
    int normFactor = 3 - nLogPel + parentLevel.nLogPel;
    int mulFactor = (normFactor < 0) ? -normFactor : 0;
    normFactor = (normFactor < 0) ? 0 : normFactor;
    int normov = (nBlkSizeX - nOverlapX) * (nBlkSizeY - nOverlapY);
    int aoddx = (nBlkSizeX * 3 - nOverlapX * 2);
    int aevenx = (nBlkSizeX * 3 - nOverlapX * 4);
    int aoddy = (nBlkSizeY * 3 - nOverlapY * 2);
    int aeveny = (nBlkSizeY * 3 - nOverlapY * 4);

    double scaleov = 1.0 / normov;
    for (int l = 0, index = 0; l < nBlkY; l++) {
        for (int k = 0; k < nBlkX; k++, index++) {
            VECTOR v1, v2, v3, v4;
            int i = k;
            int j = l;
            if (i >= 2 * parentLevel.nBlkX)
                i = 2 * parentLevel.nBlkX - 1;
            if (j >= 2 * parentLevel.nBlkY)
                j = 2 * parentLevel.nBlkY - 1;
            int offy = -1 + 2 * (j % 2);
            int offx = -1 + 2 * (i % 2);

            if ((i == 0) || (i >= 2 * parentLevel.nBlkX - 1)) {
                if ((j == 0) || (j >= 2 * parentLevel.nBlkY - 1)) {
                    v1 = v2 = v3 = v4 = parentLevel.vectors[i / 2 + (j / 2) * parentLevel.nBlkX];
                } else {
                    v1 = v2 = parentLevel.vectors[i / 2 + (j / 2) * parentLevel.nBlkX];
                    v3 = v4 = parentLevel.vectors[i / 2 + (j / 2 + offy) * parentLevel.nBlkX];
                }
            } else if ((j == 0) || (j >= 2 * parentLevel.nBlkY - 1)) {
                v1 = v2 = parentLevel.vectors[i / 2 + (j / 2) * parentLevel.nBlkX];
                v3 = v4 = parentLevel.vectors[i / 2 + offx + (j / 2) * parentLevel.nBlkX];
            } else {
                v1 = parentLevel.vectors[i / 2 + (j / 2) * parentLevel.nBlkX];
                v2 = parentLevel.vectors[i / 2 + offx + (j / 2) * parentLevel.nBlkX];
                v3 = parentLevel.vectors[i / 2 + (j / 2 + offy) * parentLevel.nBlkX];
                v4 = parentLevel.vectors[i / 2 + offx + (j / 2 + offy) * parentLevel.nBlkX];
            }

            int64_t temp_sad;

            if (nOverlapX == 0 && nOverlapY == 0) {
                vectors[index].x = 9 * v1.x + 3 * v2.x + 3 * v3.x + v4.x;
                vectors[index].y = 9 * v1.y + 3 * v2.y + 3 * v3.y + v4.y;
                temp_sad = 9 * v1.sad + 3 * v2.sad + 3 * v3.sad + v4.sad + 8;
            } else {
                int ax1 = (offx > 0) ? aoddx : aevenx;
                int ax2 = (nBlkSizeX - nOverlapX) * 4 - ax1;
                int ay1 = (offy > 0) ? aoddy : aeveny;
                int ay2 = (nBlkSizeY - nOverlapY) * 4 - ay1;
                // 64 bit so that the multiplications by the SADs don't overflow with 16 bit input.
                int64_t a11 = ax1 * ay1, a12 = ax1 * ay2, a21 = ax2 * ay1, a22 = ax2 * ay2;
                vectors[index].x = (int)((a11 * v1.x + a21 * v2.x + a12 * v3.x + a22 * v4.x) * scaleov);
                vectors[index].y = (int)((a11 * v1.y + a21 * v2.y + a12 * v3.y + a22 * v4.y) * scaleov);
                temp_sad = static_cast<int64_t>((a11 * v1.sad + a21 * v2.sad + a12 * v3.sad + a22 * v4.sad) * scaleov);
            }
            vectors[index].x = (vectors[index].x >> normFactor) * (1 << mulFactor);
            vectors[index].y = (vectors[index].y >> normFactor) * (1 << mulFactor);
            vectors[index].sad = temp_sad >> 4;
        }
    }
}

void MotionBlockLevel::EstimateGlobalMVDoubled(VECTOR &globalMVec) const noexcept{
    constexpr int MAX_DISTINCT_MOTION_VALUES = 64;

    struct ValCount { int val, count; };

    auto findMode = [](const ValCount *pairs, int numPairs) {
        int bestVal = pairs[0].val, bestCount = pairs[0].count;
        for (int i = 1; i < numPairs; i++) {
            if (pairs[i].count > bestCount) {
                bestCount = pairs[i].count;
                bestVal = pairs[i].val;
            }
        }
        return bestVal;
        };

    auto accumulate = [this](ValCount *pairs, int &numPairs, auto getMV) {
        numPairs = 0;
        for (int i = 0; i < nBlkCount; i++) {
            int v = getMV(vectors[i]);
            // Search existing pairs
            bool found = false;
            for (int j = 0; j < numPairs; j++) {
                if (pairs[j].val == v) {
                    pairs[j].count++;
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (numPairs >= MAX_DISTINCT_MOTION_VALUES)
                    return false;
                pairs[numPairs++] = { v, 1 };
            }
        }
        return true;
        };

    ValCount pairsX[MAX_DISTINCT_MOTION_VALUES];
    ValCount pairsY[MAX_DISTINCT_MOTION_VALUES];
    int numX = 0, numY = 0;

    if (!accumulate(pairsX, numX, [](const VECTOR &v) { return v.x; }) || !accumulate(pairsY, numY, [](const VECTOR &v) { return v.y; })) {
        EstimateGlobalMVDoubledFallback(globalMVec);
        return;
    }

    int medianx = findMode(pairsX, numX);
    int mediany = findMode(pairsY, numY);

    // Refinement: average vectors near the mode
    int64_t meanvx = 0, meanvy = 0;
    int num = 0;
    for (int i = 0; i < nBlkCount; i++) {
        if (abs(vectors[i].x - medianx) < 6 && abs(vectors[i].y - mediany) < 6) {
            meanvx += vectors[i].x;
            meanvy += vectors[i].y;
            num++;
        }
    }

    // Output vectors must be doubled for next (finer) scale level
    if (num > 0) {
        globalMVec.x = static_cast<int>(2 * meanvx / num);
        globalMVec.y = static_cast<int>(2 * meanvy / num);
    } else {
        globalMVec.x = 2 * medianx;
        globalMVec.y = 2 * mediany;
    }
}

void MotionBlockLevel::EstimateGlobalMVDoubledFallback(VECTOR &globalMVec) const noexcept {
    // Find mode of x and y components by sorting and finding the longest run.
    // This uses O(nBlkCount) memory instead of O(nPel * 16384).
    // It's a slower fallback for the unlikely case when motion vectors have many distinct values

    std::vector<int> vals(nBlkCount);

    // Find mode of x
    for (int i = 0; i < nBlkCount; i++)
        vals[i] = vectors[i].x;
    std::sort(vals.begin(), vals.end());

    int medianx = vals[0];
    int bestCount = 1, curCount = 1;
    for (int i = 1; i < nBlkCount; i++) {
        if (vals[i] == vals[i - 1]) {
            if (++curCount > bestCount) {
                bestCount = curCount;
                medianx = vals[i];
            }
        } else {
            curCount = 1;
        }
    }

    // Find mode of y, reuse the buffer
    for (int i = 0; i < nBlkCount; i++)
        vals[i] = vectors[i].y;
    std::sort(vals.begin(), vals.end());

    int mediany = vals[0];
    bestCount = 1; curCount = 1;
    for (int i = 1; i < nBlkCount; i++) {
        if (vals[i] == vals[i - 1]) {
            if (++curCount > bestCount) {
                bestCount = curCount;
                mediany = vals[i];
            }
        } else {
            curCount = 1;
        }
    }

    // Refinement: average vectors near the mode
    int64_t meanvx = 0, meanvy = 0;
    int num = 0;
    for (int i = 0; i < nBlkCount; i++) {
        if (abs(vectors[i].x - medianx) < 6 && abs(vectors[i].y - mediany) < 6) {
            meanvx += vectors[i].x;
            meanvy += vectors[i].y;
            num++;
        }
    }

    // Output vectors must be doubled for next (finer) scale level
    if (num > 0) {
        globalMVec.x = static_cast<int>(2 * meanvx / num);
        globalMVec.y = static_cast<int>(2 * meanvy / num);
    } else {
        globalMVec.x = 2 * medianx;
        globalMVec.y = 2 * mediany;
    }
}

/////////////////////////////////////////////////////////////




/* fetch the block in the reference frame, which is pointed by the vector (vx, vy) */
template <int nLogPelT, typename PixelType>
const uint8_t *MotionBlockLevel::GetRefBlock(int nVx, int nVy) const noexcept {
    if constexpr (nLogPelT == 0)
        return pRefFrame->planes[0].GetAbsolutePointerPel1<PixelType>(x[0] + nVx, y[0] + nVy);
    else if constexpr (nLogPelT == 1)
        return pRefFrame->planes[0].GetAbsolutePointerPel2<PixelType>(x[0] * 2 + nVx, y[0] * 2 + nVy);
    else if constexpr (nLogPelT == 2)
        return pRefFrame->planes[0].GetAbsolutePointerPel4<PixelType>(x[0] * 4 + nVx, y[0] * 4 + nVy);
    return nullptr;
}


template <int nLogPelT, typename PixelType>
const uint8_t *MotionBlockLevel::GetRefBlockU(int nVx, int nVy) const noexcept {
    int xbias = (nVx < 0) * ((1 << nLogxRatioUV) - 1);
    int ybias = (nVy < 0) * ((1 << nLogyRatioUV) - 1);

    if constexpr (nLogPelT == 0)
        return pRefFrame->planes[1].GetAbsolutePointerPel1<PixelType>(
            x[1] + ((nVx + xbias) >> nLogxRatioUV),
            y[1] + ((nVy + ybias) >> nLogyRatioUV));
    else if constexpr (nLogPelT == 1)
        return pRefFrame->planes[1].GetAbsolutePointerPel2<PixelType>(
            x[1] * 2 + ((nVx + xbias) >> nLogxRatioUV),
            y[1] * 2 + ((nVy + ybias) >> nLogyRatioUV));
    else if constexpr (nLogPelT == 2)
        return pRefFrame->planes[1].GetAbsolutePointerPel4<PixelType>(
            x[1] * 4 + ((nVx + xbias) >> nLogxRatioUV),
            y[1] * 4 + ((nVy + ybias) >> nLogyRatioUV));
    return nullptr;
}


template <int nLogPelT, typename PixelType>
const uint8_t *MotionBlockLevel::GetRefBlockV(int nVx, int nVy) const noexcept {
    int xbias = (nVx < 0) * ((1 << nLogxRatioUV) - 1);
    int ybias = (nVy < 0) * ((1 << nLogyRatioUV) - 1);

    if constexpr (nLogPelT == 0)
        return pRefFrame->planes[2].GetAbsolutePointerPel1<PixelType>(
            x[2] + ((nVx + xbias) >> nLogxRatioUV),
            y[2] + ((nVy + ybias) >> nLogyRatioUV));
    else if constexpr (nLogPelT == 1)
        return pRefFrame->planes[2].GetAbsolutePointerPel2<PixelType>(
            x[2] * 2 + ((nVx + xbias) >> nLogxRatioUV),
            y[2] * 2 + ((nVy + ybias) >> nLogyRatioUV));
    else if constexpr (nLogPelT == 2)
        return pRefFrame->planes[2].GetAbsolutePointerPel4<PixelType>(
            x[2] * 4 + ((nVx + xbias) >> nLogxRatioUV),
            y[2] * 4 + ((nVy + ybias) >> nLogyRatioUV));
    return nullptr;
}


/* computes square distance between two vectors */
static int64_t SquareDifferenceNorm(const VECTOR &v1, const int v2x, const int v2y) noexcept {
    int64_t dx = v1.x - v2x;
    int64_t dy = v1.y - v2y;
    return dx * dx + dy * dy;
}

/* computes the cost of a vector (vx, vy) */
int64_t MotionBlockLevel::MotionDistorsion(int vx, int vy) const noexcept {
    int64_t dist = SquareDifferenceNorm(predictor, vx, vy);
    return (int64_t)((nLambda * dist) >> 8);
}

/* check if a vector is inside search boundaries */
bool MotionBlockLevel::IsVectorOK(int vx, int vy) const noexcept {
    return ((vx >= nDxMin) &&
        (vy >= nDyMin) &&
        (vx < nDxMax) &&
        (vy < nDyMax));
}


#define CHECKMV_PENALTYNEW (1 << 1)
#define CHECKMV_UPDATEDIR (1 << 2)
#define CHECKMV_UPDATEBESTMV (1 << 3)

template <int nLogPel, int flags, typename PixelType>
void MotionBlockLevel::CheckMV_Template(int vx, int vy, int *dir, int val) noexcept {
    if (IsVectorOK(vx, vy)) {
        int64_t cost = MotionDistorsion(vx, vy);
        if (cost >= nMinCost)
            return;
        CheckMV_Slow<nLogPel, flags, PixelType>(vx, vy, cost, dir, val);
    }
}

template <int nLogPel, int flags, typename PixelType>
void MotionBlockLevel::CheckMV_Slow(int vx, int vy, int64_t cost, int *dir, int val) noexcept {
    int64_t sad = SAD(pSrc_temp[0], nSrcPitch_temp[0], GetRefBlock<nLogPel, PixelType>(vx, vy), nRefPitch[0]);
    cost += sad + ((flags & CHECKMV_PENALTYNEW) ? ((penaltyNew * sad) >> 8) : 0);
    if (cost >= nMinCost)
        return;

    int64_t saduv = 0;
    if (chroma) {
        saduv += SADCHROMA(pSrc_temp[1], nSrcPitch_temp[1], GetRefBlockU<nLogPel, PixelType>(vx, vy), nRefPitch[1]);
        saduv += SADCHROMA(pSrc_temp[2], nSrcPitch_temp[2], GetRefBlockV<nLogPel, PixelType>(vx, vy), nRefPitch[2]);

        cost += saduv + ((flags & CHECKMV_PENALTYNEW) ? ((penaltyNew * saduv) >> 8) : 0);
        if (cost >= nMinCost)
            return;
    }

    if constexpr (flags & CHECKMV_UPDATEBESTMV) {
        bestMV.x = vx;
        bestMV.y = vy;
    }
    nMinCost = cost;
    bestMV.sad = sad + saduv;
    if constexpr (flags & CHECKMV_UPDATEDIR)
        *dir = val;
}


/* check if the vector (vx, vy) is better than the best vector found so far without penalty new - renamed in v.2.11*/
template <int nLogPel, typename PixelType>
void MotionBlockLevel::CheckMV0(int vx, int vy) noexcept { //here the chance for default values are high especially for zeroMVfieldShifted (on left/top border)
    CheckMV_Template<nLogPel, CHECKMV_UPDATEBESTMV, PixelType>(vx, vy, 0, 0);
}


/* check if the vector (vx, vy) is better than the best vector found so far */
template <int nLogPel, typename PixelType>
void MotionBlockLevel::CheckMV(int vx, int vy) noexcept { //here the chance for default values are high especially for zeroMVfieldShifted (on left/top border)
    CheckMV_Template<nLogPel, CHECKMV_PENALTYNEW | CHECKMV_UPDATEBESTMV, PixelType>(vx, vy, 0, 0);
}


/* check if the vector (vx, vy) is better, and update dir accordingly */
template <int nLogPel, typename PixelType>
void MotionBlockLevel::CheckMV2(int vx, int vy, int *dir, int val) noexcept {
    CheckMV_Template<nLogPel, CHECKMV_PENALTYNEW | CHECKMV_UPDATEDIR | CHECKMV_UPDATEBESTMV, PixelType>(vx, vy, dir, val);
}


/* check if the vector (vx, vy) is better, and update dir accordingly, but not bestMV.x, y */
template <int nLogPel, typename PixelType>
void MotionBlockLevel::CheckMVdir(int vx, int vy, int *dir, int val) noexcept {
    CheckMV_Template<nLogPel, CHECKMV_PENALTYNEW | CHECKMV_UPDATEDIR, PixelType>(vx, vy, dir, val);
}


/* clip a vector to the search boundaries */
VECTOR MotionBlockLevel::ClipMV(VECTOR v) const noexcept {
    VECTOR v2;
    v2.x = std::min(std::max(v.x, nDxMin), nDxMax - 1);
    v2.y = std::min(std::max(v.y, nDyMin), nDyMax - 1);
    v2.sad = v.sad;
    return v2;
}


/* find the median between a, b and c */
static int Median(int a, int b, int c) {
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

void MotionBlockLevel::Initialize(int _nBlkX, int _nBlkY, int _nBlkSizeX, int _nBlkSizeY, int nPel_, int _nLevel, bool smallestPlane_, bool chroma_, int _nOverlapX, int _nOverlapY, int _xRatioUV, int _yRatioUV, int bitsPerSample) noexcept {
    nPel = nPel_;
    nLogPel = ilog2(nPel);
    nLogScale = _nLevel;
    nScale = 1 << nLogScale;

    nBlkSizeX = _nBlkSizeX;
    nBlkSizeY = _nBlkSizeY;
    nOverlapX = _nOverlapX;
    nOverlapY = _nOverlapY;

    nBlkX = _nBlkX;
    nBlkY = _nBlkY;
    nBlkCount = nBlkX * nBlkY;

    xRatioUV = _xRatioUV;
    yRatioUV = _yRatioUV;
    nLogxRatioUV = ilog2(xRatioUV);
    nLogyRatioUV = ilog2(yRatioUV);

    smallestPlane = smallestPlane_;
    chroma = chroma_;

    globalMVPredictor = zeroMV;

    vectors.resize(nBlkCount);

    // float SAD is mapped to the 16-bit scale
    verybigSAD = (int64_t)nBlkSizeX * nBlkSizeY * (static_cast<int64_t>(1) << std::min(16, bitsPerSample));
}

MotionBlockLevel::~MotionBlockLevel() {
    mvu_aligned_free(pSrc_temp[0]);
    mvu_aligned_free(pSrc_temp[1]);
    mvu_aligned_free(pSrc_temp[2]);
}

void MotionBlockLevel::FetchPredictors(int blkidx, int blkx, int blky, int blkScanDir, VECTOR predictors[4]) noexcept {
    // Left (or right) predictor
    if ((blkScanDir == 1 && blkx > 0) || (blkScanDir == -1 && blkx < nBlkX - 1))
        predictors[1] = ClipMV(vectors[blkidx - blkScanDir]);
    else
        predictors[1] = ClipMV(zeroMVfieldShifted); // v1.11.1 - values instead of pointer

    // Up predictor
    if (blky > 0)
        predictors[2] = ClipMV(vectors[blkidx - nBlkX]);
    else
        predictors[2] = ClipMV(zeroMVfieldShifted);

    // bottom-right pridictor (from coarse level)
    if ((blky < nBlkY - 1) && ((blkScanDir == 1 && blkx < nBlkX - 1) || (blkScanDir == -1 && blkx > 0)))
        predictors[3] = ClipMV(vectors[blkidx + nBlkX + blkScanDir]);
    else
        // Up-right predictor
        if ((blky > 0) && ((blkScanDir == 1 && blkx < nBlkX - 1) || (blkScanDir == -1 && blkx > 0)))
            predictors[3] = ClipMV(vectors[blkidx - nBlkX + blkScanDir]);
        else
            predictors[3] = ClipMV(zeroMVfieldShifted);

    // Median predictor
    if (blky > 0) {
        predictors[0].x = Median(predictors[1].x, predictors[2].x, predictors[3].x);
        predictors[0].y = Median(predictors[1].y, predictors[2].y, predictors[3].y);
        //      predictors[0].sad = Median(predictors[1].sad, predictors[2].sad, predictors[3].sad);
        // but it is not true median vector (x and y may be mixed) and not its sad ?!
        // we really do not know SAD, here is more safe estimation especially for phaseshift method - v1.6.0
        predictors[0].sad = std::max(predictors[1].sad, std::max(predictors[2].sad, predictors[3].sad));
    } else {
        //        predictors[0].x = (predictors[1].x + predictors[2].x + predictors[3].x);
        //        predictors[0].y = (predictors[1].y + predictors[2].y + predictors[3].y);
        //      predictors[0].sad = (predictors[1].sad + predictors[2].sad + predictors[3].sad);
        // but for top line we have only left predictor[1] - v1.6.0
        predictors[0] = predictors[1];
    }

    // if there are no other planes, predictor is the median
    if (smallestPlane)
        predictor = predictors[0];
    double scale = (double)LSAD / std::max<int64_t>(LSAD + (predictor.sad >> 1), 1);
    nLambda = static_cast<int64_t>(nLambda * scale * scale);
}

void MotionBlockLevel::InitMotionEstimationFields(bool useSatd, bool chroma, int bytesPerSample) {
    this->chroma = chroma;

    if (useSatd && (nBlkSizeX == 16 && nBlkSizeY == 2))
        throw MotionBlockPyramidError("Unsupported block size for SATD");

    if (useSatd)
        SAD = selectSATDFunction(nBlkSizeX, nBlkSizeY, bytesPerSample * 8);
    else
        SAD = selectSADFunction(nBlkSizeX, nBlkSizeY, bytesPerSample * 8);
    BLITLUMA = selectCopyFunction(nBlkSizeX, nBlkSizeY, bytesPerSample * 8);
    if (chroma) {
        SADCHROMA = selectSADFunction(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bytesPerSample * 8);
        BLITCHROMA = selectCopyFunction(nBlkSizeX / xRatioUV, nBlkSizeY / yRatioUV, bytesPerSample * 8);
    }

    // 64-byte alignment for the block buffers (cache-line aligned for the SIMD SAD/SATD loads)
    static constexpr size_t ALIGN_PLANES = 64;

    nSrcPitch_temp[0] = nBlkSizeX * bytesPerSample;
    nSrcPitch_temp[1] = nBlkSizeX / xRatioUV * bytesPerSample;
    nSrcPitch_temp[2] = nSrcPitch_temp[1];

    pSrc_temp[0] = mvu_aligned_malloc<uint8_t>(nBlkSizeY * nSrcPitch_temp[0], ALIGN_PLANES);
    pSrc_temp[1] = mvu_aligned_malloc<uint8_t>(nBlkSizeY / yRatioUV * nSrcPitch_temp[1], ALIGN_PLANES);
    pSrc_temp[2] = mvu_aligned_malloc<uint8_t>(nBlkSizeY / yRatioUV * nSrcPitch_temp[2], ALIGN_PLANES);

#undef ALIGN_PLANES
}

template <int nLogPel, typename PixelType>
void MotionBlockLevel::DiamondSearch(int length) noexcept {
    enum Direction {
        Right = 1,
        Left = 2,
        Down = 4,
        Up = 8,
    };

    int dx;
    int dy;

    // We begin by making no assumption on which direction to search.
    int direction = 15;

    int lastDirection;

    while (direction > 0) {
        dx = bestMV.x;
        dy = bestMV.y;
        lastDirection = direction;
        direction = 0;

        // First, we look the directions that were hinted by the previous step
        // of the algorithm. If we find one, we add it to the set of directions
        // we'll test next
        if (lastDirection & Right)
            CheckMV2<nLogPel, PixelType>(dx + length, dy, &direction, Right);
        if (lastDirection & Left)
            CheckMV2<nLogPel, PixelType>(dx - length, dy, &direction, Left);
        if (lastDirection & Down)
            CheckMV2<nLogPel, PixelType>(dx, dy + length, &direction, Down);
        if (lastDirection & Up)
            CheckMV2<nLogPel, PixelType>(dx, dy - length, &direction, Up);

        // If one of the directions improves the SAD, we make further tests
        // on the diagonals
        if (direction) {
            lastDirection = direction;
            dx = bestMV.x;
            dy = bestMV.y;

            if (lastDirection & (Right + Left)) {
                CheckMV2<nLogPel, PixelType>(dx, dy + length, &direction, Down);
                CheckMV2<nLogPel, PixelType>(dx, dy - length, &direction, Up);
            } else {
                CheckMV2<nLogPel, PixelType>(dx + length, dy, &direction, Right);
                CheckMV2<nLogPel, PixelType>(dx - length, dy, &direction, Left);
            }
        }

        // If not, we do not stop here. We infer from the last direction the
        // diagonals to be checked, because we might be lucky.
        else {
            switch (lastDirection) {
                case Right:
                    CheckMV2<nLogPel, PixelType>(dx + length, dy + length, &direction, Right + Down);
                    CheckMV2<nLogPel, PixelType>(dx + length, dy - length, &direction, Right + Up);
                    break;
                case Left:
                    CheckMV2<nLogPel, PixelType>(dx - length, dy + length, &direction, Left + Down);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy - length, &direction, Left + Up);
                    break;
                case Down:
                    CheckMV2<nLogPel, PixelType>(dx + length, dy + length, &direction, Right + Down);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy + length, &direction, Left + Down);
                    break;
                case Up:
                    CheckMV2<nLogPel, PixelType>(dx + length, dy - length, &direction, Right + Up);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy - length, &direction, Left + Up);
                    break;
                case Right + Down:
                    CheckMV2<nLogPel, PixelType>(dx + length, dy + length, &direction, Right + Down);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy + length, &direction, Left + Down);
                    CheckMV2<nLogPel, PixelType>(dx + length, dy - length, &direction, Right + Up);
                    break;
                case Left + Down:
                    CheckMV2<nLogPel, PixelType>(dx + length, dy + length, &direction, Right + Down);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy + length, &direction, Left + Down);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy - length, &direction, Left + Up);
                    break;
                case Right + Up:
                    CheckMV2<nLogPel, PixelType>(dx + length, dy + length, &direction, Right + Down);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy - length, &direction, Left + Up);
                    CheckMV2<nLogPel, PixelType>(dx + length, dy - length, &direction, Right + Up);
                    break;
                case Left + Up:
                    CheckMV2<nLogPel, PixelType>(dx - length, dy - length, &direction, Left + Up);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy + length, &direction, Left + Down);
                    CheckMV2<nLogPel, PixelType>(dx + length, dy - length, &direction, Right + Up);
                    break;
                default:
                    // Even the default case may happen, in the first step of the
                    // algorithm for example.
                    CheckMV2<nLogPel, PixelType>(dx + length, dy + length, &direction, Right + Down);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy + length, &direction, Left + Down);
                    CheckMV2<nLogPel, PixelType>(dx + length, dy - length, &direction, Right + Up);
                    CheckMV2<nLogPel, PixelType>(dx - length, dy - length, &direction, Left + Up);
                    break;
            }
        }
    }
}


template <int nLogPel, typename PixelType>
void MotionBlockLevel::ExpandingSearch(int r, int s, int mvx, int mvy) noexcept {
    // diameter = 2*r + 1, step=s
    // part of true enhaustive search (thin expanding square) around mvx, mvy

    // sides of square without corners
    for (int i = -r + s; i < r; i += s) // without corners! - v2.1
    {
        CheckMV<nLogPel, PixelType>(mvx + i, mvy - r);
        CheckMV<nLogPel, PixelType>(mvx + i, mvy + r);
    }

    for (int j = -r + s; j < r; j += s) {
        CheckMV<nLogPel, PixelType>(mvx - r, mvy + j);
        CheckMV<nLogPel, PixelType>(mvx + r, mvy + j);
    }

    // then corners - they are more far from cenrer
    CheckMV<nLogPel, PixelType>(mvx - r, mvy - r);
    CheckMV<nLogPel, PixelType>(mvx - r, mvy + r);
    CheckMV<nLogPel, PixelType>(mvx + r, mvy - r);
    CheckMV<nLogPel, PixelType>(mvx + r, mvy + r);
}


/* (x-1)%6 */
static constexpr int mod6m1[8] = { 5, 0, 1, 2, 3, 4, 5, 0 };
/* radius 2 hexagon. repeated entries are to avoid having to compute mod6 every time. */
static constexpr int hex2[8][2] = { { -1, -2 }, { -2, 0 }, { -1, 2 }, { 1, 2 }, { 2, 0 }, { 1, -2 }, { -1, -2 }, { -2, 0 } };

template <int nLogPel, typename PixelType>
void MotionBlockLevel::Hex2Search(int i_me_range) noexcept { //adopted from x264
    int dir = -2;
    int bmx = bestMV.x;
    int bmy = bestMV.y;

    if (i_me_range > 1) {
        /* hexagon */
        //        COST_MV_X3_DIR( -2,0, -1, 2,  1, 2, costs   );
        //        COST_MV_X3_DIR(  2,0,  1,-2, -1,-2, costs+3 );
        //        COPY2_IF_LT( bcost, costs[0], dir, 0 );
        //        COPY2_IF_LT( bcost, costs[1], dir, 1 );
        //        COPY2_IF_LT( bcost, costs[2], dir, 2 );
        //        COPY2_IF_LT( bcost, costs[3], dir, 3 );
        //        COPY2_IF_LT( bcost, costs[4], dir, 4 );
        //        COPY2_IF_LT( bcost, costs[5], dir, 5 );
        CheckMVdir<nLogPel, PixelType>(bmx - 2, bmy, &dir, 0);
        CheckMVdir<nLogPel, PixelType>(bmx - 1, bmy + 2, &dir, 1);
        CheckMVdir<nLogPel, PixelType>(bmx + 1, bmy + 2, &dir, 2);
        CheckMVdir<nLogPel, PixelType>(bmx + 2, bmy, &dir, 3);
        CheckMVdir<nLogPel, PixelType>(bmx + 1, bmy - 2, &dir, 4);
        CheckMVdir<nLogPel, PixelType>(bmx - 1, bmy - 2, &dir, 5);


        if (dir != -2) {
            bmx += hex2[dir + 1][0];
            bmy += hex2[dir + 1][1];
            /* half hexagon, not overlapping the previous iteration */
            for (int i = 1; i < i_me_range / 2 && IsVectorOK(bmx, bmy); i++) {
                const int odir = mod6m1[dir + 1];
                //                COST_MV_X3_DIR( hex2[odir+0][0], hex2[odir+0][1],
                //                                hex2[odir+1][0], hex2[odir+1][1],
                //                                hex2[odir+2][0], hex2[odir+2][1],
                //                                costs );

                dir = -2;
                //                COPY2_IF_LT( bcost, costs[0], dir, odir-1 );
                //                COPY2_IF_LT( bcost, costs[1], dir, odir   );
                //                COPY2_IF_LT( bcost, costs[2], dir, odir+1 );

                CheckMVdir<nLogPel, PixelType>(bmx + hex2[odir + 0][0], bmy + hex2[odir + 0][1], &dir, odir - 1);
                CheckMVdir<nLogPel, PixelType>(bmx + hex2[odir + 1][0], bmy + hex2[odir + 1][1], &dir, odir);
                CheckMVdir<nLogPel, PixelType>(bmx + hex2[odir + 2][0], bmy + hex2[odir + 2][1], &dir, odir + 1);
                if (dir == -2)
                    break;
                bmx += hex2[dir + 1][0];
                bmy += hex2[dir + 1][1];
            }
        }

        bestMV.x = bmx;
        bestMV.y = bmy;
    }
    /* square refine */
    //        omx = bmx; omy = bmy;
    //        COST_MV_X4(  0,-1,  0,1, -1,0, 1,0 );
    //        COST_MV_X4( -1,-1, -1,1, 1,-1, 1,1 );
    ExpandingSearch<nLogPel, PixelType>(1, 1, bmx, bmy);
}


template <int nLogPel, typename PixelType>
void MotionBlockLevel::CrossSearch(int start, int x_max, int y_max, int mvx, int mvy) noexcept { // part of umh  search

    for (int i = start; i < x_max; i += 2) {
        CheckMV<nLogPel, PixelType>(mvx - i, mvy);
        CheckMV<nLogPel, PixelType>(mvx + i, mvy);
    }

    for (int j = start; j < y_max; j += 2) {
        CheckMV<nLogPel, PixelType>(mvx, mvy - j);
        CheckMV<nLogPel, PixelType>(mvx, mvy + j);
    }
}


template <int nLogPel, typename PixelType>
void MotionBlockLevel::UMHSearch(int i_me_range, int omx, int omy) noexcept { // radius
    // Uneven-cross Multi-Hexagon-grid Search (see x264)
    /* hexagon grid */

    //            int omx = bestMV.x;
    //            int omy = bestMV.y;
    // my mod: do not shift the center after Cross
    CrossSearch<nLogPel, PixelType>(1, i_me_range, i_me_range, omx, omy);


    int i = 1;
    do {
        static const int hex4[16][2] = {
            { -4, 2 }, { -4, 1 }, { -4, 0 }, { -4, -1 }, { -4, -2 }, { 4, -2 }, { 4, -1 }, { 4, 0 }, { 4, 1 }, { 4, 2 }, { 2, 3 }, { 0, 4 }, { -2, 3 }, { -2, -3 }, { 0, -4 }, { 2, -3 },
        };

        for (int j = 0; j < 16; j++) {
            int mx = omx + hex4[j][0] * i;
            int my = omy + hex4[j][1] * i;
            CheckMV<nLogPel, PixelType>(mx, my);
        }
    } while (++i <= i_me_range / 4);

    //            if( bmy <= mv_y_max )
    //                goto me_hex2;
    Hex2Search<nLogPel, PixelType>(i_me_range);
}


template <int nLogPel, typename PixelType>
void MotionBlockLevel::Refine() noexcept {
    // then, we refine, according to the search type
    if (searchType == SearchType::Logarithmic) {
        for (int i = nSearchParam; i > 0; i /= 2)
            DiamondSearch<nLogPel, PixelType>(i);
    } else if (searchType == SearchType::Exhaustive) {
        int mvx = bestMV.x;
        int mvy = bestMV.y;
        for (int i = 1; i <= nSearchParam; i++) // region is same as enhausted, but ordered by radius (from near to far)
            ExpandingSearch<nLogPel, PixelType>(i, 1, mvx, mvy);
    } else if (searchType == SearchType::Hex2) {
        Hex2Search<nLogPel, PixelType>(nSearchParam);

    } else if (searchType == SearchType::UnevenMultiHexagon) {
        UMHSearch<nLogPel, PixelType>(nSearchParam, bestMV.x, bestMV.y);

    } else if (searchType == SearchType::Horizontal) {
        int mvx = bestMV.x;
        int mvy = bestMV.y;
        for (int i = 1; i <= nSearchParam; i++) {
            CheckMV<nLogPel, PixelType>(mvx - i, mvy);
            CheckMV<nLogPel, PixelType>(mvx + i, mvy);
        }
    } else if (searchType == SearchType::Vertical) {
        int mvx = bestMV.x;
        int mvy = bestMV.y;
        for (int i = 1; i <= nSearchParam; i++) {
            CheckMV<nLogPel, PixelType>(mvx, mvy - i);
            CheckMV<nLogPel, PixelType>(mvx, mvy + i);
        }
    }
}


template <int nLogPel, typename PixelType>
void MotionBlockLevel::PseudoEPZSearch(int blkIdx, int blkx, int blky, int blkScanDir, int64_t badSAD, int badrange, bool tryMany, int &badcount) noexcept {

    // We treat zero alone
    // Do we bias zero with not taking into account distorsion ?
    bestMV.x = zeroMVfieldShifted.x;
    bestMV.y = zeroMVfieldShifted.y;

    auto rawSad = [&](int mvx, int mvy) noexcept -> int64_t {
        int64_t s = SAD(pSrc_temp[0], nSrcPitch_temp[0], GetRefBlock<nLogPel, PixelType>(mvx, mvy), nRefPitch[0]);
        if (chroma) {
            s += SADCHROMA(pSrc_temp[1], nSrcPitch_temp[1], GetRefBlockU<nLogPel, PixelType>(mvx, mvy), nRefPitch[1]);
            s += SADCHROMA(pSrc_temp[2], nSrcPitch_temp[2], GetRefBlockV<nLogPel, PixelType>(mvx, mvy), nRefPitch[2]);
        }
        return s;
    };

    // Zero vector
    int64_t sadZero = rawSad(zeroMVfieldShifted.x, zeroMVfieldShifted.y);
    bestMV.sad = sadZero;
    nMinCost = sadZero + ((penaltyZero * sadZero) >> 8); // v.1.11.0.2

    VECTOR bestMVMany[8];
    int64_t nMinCostMany[8] = { 0 };

    if (tryMany) {
        //  refine around zero
        Refine<nLogPel, PixelType>();
        bestMVMany[0] = bestMV; // save bestMV
        nMinCostMany[0] = nMinCost;
    }

    // Global MV predictor (reuse zero's SAD if it clips to the same position)
    globalMVPredictor = ClipMV(globalMVPredictor);
    int64_t sadGlobal = (globalMVPredictor.x == zeroMVfieldShifted.x && globalMVPredictor.y == zeroMVfieldShifted.y)
        ? sadZero
        : rawSad(globalMVPredictor.x, globalMVPredictor.y);
    int64_t cost = sadGlobal + ((pglobal * sadGlobal) >> 8);

    if (cost < nMinCost || tryMany) {
        bestMV.x = globalMVPredictor.x;
        bestMV.y = globalMVPredictor.y;
        bestMV.sad = sadGlobal;
        nMinCost = cost;
    }
    if (tryMany) {
        // refine around global
        Refine<nLogPel, PixelType>();               // reset bestMV
        bestMVMany[1] = bestMV; // save bestMV
        nMinCostMany[1] = nMinCost;
    }

    VECTOR predictors[4]; /* set of predictors for the current block */
    FetchPredictors(blkIdx, blkx, blky, blkScanDir, predictors);
    // Median/parent predictor (reuse zero's or global's SAD when it coincides with either)
    int64_t sadPred = (predictor.x == zeroMVfieldShifted.x && predictor.y == zeroMVfieldShifted.y) ? sadZero
        : (predictor.x == globalMVPredictor.x && predictor.y == globalMVPredictor.y) ? sadGlobal
        : rawSad(predictor.x, predictor.y);
    cost = sadPred;

    if (cost < nMinCost || tryMany) {
        bestMV.x = predictor.x;
        bestMV.y = predictor.y;
        bestMV.sad = sadPred;
        nMinCost = cost;
    }
    if (tryMany) {
        // refine around predictor
        Refine<nLogPel, PixelType>();               // reset bestMV
        bestMVMany[2] = bestMV; // save bestMV
        nMinCostMany[2] = nMinCost;
    }

    // then all the other predictors, skipping any already tested above
    int npred = 4;

    for (auto &p : predictors) {
        if ((p.x == zeroMVfieldShifted.x && p.y == zeroMVfieldShifted.y) ||
            (p.x == globalMVPredictor.x && p.y == globalMVPredictor.y) ||
            (p.x == predictor.x && p.y == predictor.y))
            p.x = std::numeric_limits<int>::min();
    }

    for (int i = 0; i < npred; i++) {
        if (predictors[i].x == std::numeric_limits<int>::min()) {
            if (tryMany)
                nMinCostMany[i + 3] = verybigSAD + 1;
            continue;
        }
        for (int j = i + 1; j < npred; j++) {
            if (predictors[j].x == predictors[i].x && predictors[j].y == predictors[i].y)
                predictors[j].x = std::numeric_limits<int>::min();
        }
        if (tryMany)
            nMinCost = verybigSAD + 1;
        CheckMV0<nLogPel, PixelType>(predictors[i].x, predictors[i].y);
        if (tryMany) {
            // refine around predictor
            Refine<nLogPel, PixelType>();                   // reset bestMV
            bestMVMany[i + 3] = bestMV; // save bestMV
            nMinCostMany[i + 3] = nMinCost;
        }
    }


    if (tryMany) { // select best of multi best
        nMinCost = verybigSAD + 1;
        for (int i = 0; i < npred + 3; i++) {
            if (nMinCostMany[i] < nMinCost) {
                bestMV = bestMVMany[i];
                nMinCost = nMinCostMany[i];
            }
        }
    } else {
        // then, we refine, according to the search type
        Refine<nLogPel, PixelType>();
    }

    int64_t foundSAD = bestMV.sad;

#define BADCOUNT_LIMIT 16

    if (blkIdx > 1 && foundSAD > (badSAD + badSAD * badcount / BADCOUNT_LIMIT)) {
        // bad vector, try wide search
        // with some soft limit (BADCOUNT_LIMIT) of bad cured vectors (time consumed)
        badcount++;

        if (badrange > 0) { // UMH
            // rathe good is not found, lets try around zero
            UMHSearch<nLogPel, PixelType>(badrange * (1 << nLogPel), 0, 0);
        } else if (badrange < 0) { // ESA
            for (int i = 1; i < -badrange * (1 << nLogPel); i += (1 << nLogPel)) { // at radius
                ExpandingSearch<nLogPel, PixelType>(i, 1 << nLogPel, 0, 0);
                if (bestMV.sad < foundSAD / 4)
                    break; // stop search if rathe good is found
            }
        }

        int mvx = bestMV.x; // refine in small area
        int mvy = bestMV.y;
        for (int i = 1; i < (1 << nLogPel); i++) { // small radius
            ExpandingSearch<nLogPel, PixelType>(i, 1, mvx, mvy);
        }
    }


    // we store the result
    vectors[blkIdx] = bestMV;
}


template <int nLogPel, typename PixelType>
void MotionBlockLevel::DoSearchMVs(const FramePyramidLevel &pSrcFrame, const FramePyramidLevel &pRefFrame,
    SearchType st, int stp, int64_t lambda, int64_t lsad, int pnew,
    int plevel, VECTOR *globalMVec, int fieldShift,
    int pzero, int pglobal, int64_t badSAD, int badrange, bool meander, bool tryMany, bool chroma) noexcept {

    zeroMVfieldShifted.x = 0;
    zeroMVfieldShifted.y = fieldShift;
    zeroMVfieldShifted.sad = 0;
    globalMVPredictor.x = (1 << nLogPel) * globalMVec->x; // v1.8.2
    globalMVPredictor.y = (1 << nLogPel) * globalMVec->y + fieldShift;
    globalMVPredictor.sad = globalMVec->sad;
    penaltyNew = pnew; // penalty for new vector
    LSAD = lsad;       // SAD limit for lambda using
    // may be they must be scaled by nPel ?

    VECTOR *pBlkData = vectors.data();

    this->pRefFrame = &pRefFrame;

    y[0] = pSrcFrame.planes[0].nVPadding;

    if (chroma) {
        y[1] = pSrcFrame.planes[1].nVPadding;
        y[2] = pSrcFrame.planes[2].nVPadding;
    }

    nRefPitch[0] = pRefFrame.planes[0].nPitch;
    if (chroma) {
        nRefPitch[1] = pRefFrame.planes[1].nPitch;
        nRefPitch[2] = pRefFrame.planes[2].nPitch;
    }

    searchType = st;
    nSearchParam = stp;

    int64_t nLambdaLevel = lambda / ((1 << nLogPel) * (1 << nLogPel));
    if (plevel == 1)
        nLambdaLevel = nLambdaLevel * nScale;
    else if (plevel == 2)
        nLambdaLevel = nLambdaLevel * nScale * nScale;

    penaltyZero = pzero;
    this->pglobal = pglobal;
    int badcount = 0;
    // Functions using float must not be used here

    for (int blky = 0; blky < nBlkY; blky++) {
        int blkScanDir = (blky % 2 == 0 || !meander) ? 1 : -1;
        // meander (alternate) scan blocks (even row left to right, odd row right to left)
        int blkxStart = (blky % 2 == 0 || !meander) ? 0 : nBlkX - 1;
        if (blkScanDir == 1) { // start with leftmost block
            x[0] = pSrcFrame.planes[0].nHPadding;
            if (chroma) {
                x[1] = pSrcFrame.planes[1].nHPadding;
                x[2] = pSrcFrame.planes[2].nHPadding;
            }
        } else { // start with rightmost block, but it is already set at prev row
            x[0] = pSrcFrame.planes[0].nHPadding + (nBlkSizeX - nOverlapX) * (nBlkX - 1);
            if (chroma) {
                x[1] = pSrcFrame.planes[1].nHPadding + ((nBlkSizeX - nOverlapX) / xRatioUV) * (nBlkX - 1);
                x[2] = pSrcFrame.planes[2].nHPadding + ((nBlkSizeX - nOverlapX) / xRatioUV) * (nBlkX - 1);
            }
        }
        for (int iblkx = 0; iblkx < nBlkX; iblkx++) {
            int blkx = blkxStart + iblkx * blkScanDir;
            int blkIdx = blky * nBlkX + blkx;

            //create aligned copy
            BLITLUMA(pSrc_temp[0], nSrcPitch_temp[0], pSrcFrame.planes[0].GetAbsolutePointerPel1<PixelType>(x[0], y[0]), pSrcFrame.planes[0].nPitch);
            if (chroma) {
                BLITCHROMA(pSrc_temp[1], nSrcPitch_temp[1], pSrcFrame.planes[1].GetAbsolutePointerPel1<PixelType>(x[1], y[1]), pSrcFrame.planes[1].nPitch);
                BLITCHROMA(pSrc_temp[2], nSrcPitch_temp[2], pSrcFrame.planes[2].GetAbsolutePointerPel1<PixelType>(x[2], y[2]), pSrcFrame.planes[2].nPitch);
            }

            if (blky == 0)
                nLambda = 0;
            else
                nLambda = nLambdaLevel;

            // decreased padding of coarse levels
            int nHPaddingScaled = pSrcFrame.planes[0].nHPadding >> nLogScale;
            int nVPaddingScaled = pSrcFrame.planes[0].nVPadding >> nLogScale;
            /* computes search boundaries */
            nDxMax = (pSrcFrame.planes[0].nPaddedWidth - x[0] - nBlkSizeX - pSrcFrame.planes[0].nHPadding + nHPaddingScaled) << nLogPel;
            nDyMax = (pSrcFrame.planes[0].nPaddedHeight - y[0] - nBlkSizeY - pSrcFrame.planes[0].nVPadding + nVPaddingScaled) << nLogPel;
            nDxMin = -((x[0] - pSrcFrame.planes[0].nHPadding + nHPaddingScaled) << nLogPel);
            nDyMin = -((y[0] - pSrcFrame.planes[0].nVPadding + nVPaddingScaled) << nLogPel);

            /* search the mv */
            predictor = ClipMV(vectors[blkIdx]);
            PseudoEPZSearch<nLogPel, PixelType>(blkIdx, blkx, blky, blkScanDir, badSAD, badrange, tryMany, badcount);

            /* write the results */
            pBlkData[blkx] = bestMV;

            /* increment indexes & pointers */
            if (iblkx < nBlkX - 1) {
                x[0] += (nBlkSizeX - nOverlapX) * blkScanDir;
                if (chroma) {
                    x[1] += ((nBlkSizeX - nOverlapX) >> nLogxRatioUV) * blkScanDir;
                    x[2] += ((nBlkSizeX - nOverlapX) >> nLogxRatioUV) * blkScanDir;
                }
            }
        }
        pBlkData += nBlkX;

        y[0] += (nBlkSizeY - nOverlapY);
        if (chroma) {
            y[1] += ((nBlkSizeY - nOverlapY) >> nLogyRatioUV);
            y[2] += ((nBlkSizeY - nOverlapY) >> nLogyRatioUV);
        }
    }
}


void MotionBlockLevel::SearchMVs(const FramePyramidLevel &pSrcFrame, const FramePyramidLevel &pRefFrame,
    SearchType st, int stp, int64_t lambda, int64_t lsad, int pnew,
    int plevel, VECTOR *globalMVec, int fieldShift, bool useSatd,
    int pzero, int pglobal, int64_t badSAD, int badrange, bool meander, bool tryMany, bool chroma, int bytesPerSample) {

    InitMotionEstimationFields(useSatd, chroma, bytesPerSample);

    if (bytesPerSample == 1) {
        if (nLogPel == 0)
            DoSearchMVs<0, uint8_t>(pSrcFrame, pRefFrame, st, stp, lambda, lsad, pnew, plevel, globalMVec, fieldShift, pzero, pglobal, badSAD, badrange, meander, tryMany, chroma);
        else if (nLogPel == 1)
            DoSearchMVs<1, uint8_t>(pSrcFrame, pRefFrame, st, stp, lambda, lsad, pnew, plevel, globalMVec, fieldShift, pzero, pglobal, badSAD, badrange, meander, tryMany, chroma);
        else
            DoSearchMVs<2, uint8_t>(pSrcFrame, pRefFrame, st, stp, lambda, lsad, pnew, plevel, globalMVec, fieldShift, pzero, pglobal, badSAD, badrange, meander, tryMany, chroma);
    } else if (bytesPerSample == 2) {
        if (nLogPel == 0)
            DoSearchMVs<0, uint16_t>(pSrcFrame, pRefFrame, st, stp, lambda, lsad, pnew, plevel, globalMVec, fieldShift, pzero, pglobal, badSAD, badrange, meander, tryMany, chroma);
        else if (nLogPel == 1)
            DoSearchMVs<1, uint16_t>(pSrcFrame, pRefFrame, st, stp, lambda, lsad, pnew, plevel, globalMVec, fieldShift, pzero, pglobal, badSAD, badrange, meander, tryMany, chroma);
        else
            DoSearchMVs<2, uint16_t>(pSrcFrame, pRefFrame, st, stp, lambda, lsad, pnew, plevel, globalMVec, fieldShift, pzero, pglobal, badSAD, badrange, meander, tryMany, chroma);
    } else {
        if (nLogPel == 0)
            DoSearchMVs<0, float>(pSrcFrame, pRefFrame, st, stp, lambda, lsad, pnew, plevel, globalMVec, fieldShift, pzero, pglobal, badSAD, badrange, meander, tryMany, chroma);
        else if (nLogPel == 1)
            DoSearchMVs<1, float>(pSrcFrame, pRefFrame, st, stp, lambda, lsad, pnew, plevel, globalMVec, fieldShift, pzero, pglobal, badSAD, badrange, meander, tryMany, chroma);
        else
            DoSearchMVs<2, float>(pSrcFrame, pRefFrame, st, stp, lambda, lsad, pnew, plevel, globalMVec, fieldShift, pzero, pglobal, badSAD, badrange, meander, tryMany, chroma);
    }
}


template <int nLogPel, typename PixelType>
void MotionBlockLevel::DoRecalculateMVs(const FramePyramidLevel &pSrcFrame, const FramePyramidLevel &pRefFrame,
    int nBlkSizeX_, int nBlkSizeY_, int nOverlapX_, int nOverlapY_, bool chroma_,
    SearchType st, int stp, int64_t lambda, int pnew,
    int fieldShift, int64_t thSAD, bool smooth, bool meander, bool useSatd) {
                                    
    zeroMVfieldShifted.x = 0;
    zeroMVfieldShifted.y = fieldShift;
    zeroMVfieldShifted.sad = 0;
    globalMVPredictor.x = 0;          
    globalMVPredictor.y = fieldShift;
    globalMVPredictor.sad = 9999999;
    penaltyNew = pnew;

    int nBlkXold = nBlkX;
    int nBlkYold = nBlkY;
    int nBlkSizeXold = nBlkSizeX;
    int nBlkSizeYold = nBlkSizeY;
    int nOverlapXold = nOverlapX;
    int nOverlapYold = nOverlapY;
    int nStepXold = nBlkSizeXold - nOverlapXold;
    int nStepYold = nBlkSizeYold - nOverlapYold;
    int nLogPelold = ilog2(nPel);

    std::vector<VECTOR> oldVectors;
    std::swap(oldVectors, vectors);
    
    nBlkSizeX = nBlkSizeX_;
    nBlkSizeY = nBlkSizeY_;
    nOverlapX = nOverlapX_;
    nOverlapY = nOverlapY_;
    nPel = 1 << nLogPel;
    chroma = chroma_;

    xRatioUV = 1;
    yRatioUV = 1;
    nLogxRatioUV = 0;
    nLogyRatioUV = 0;
    if (chroma) {
        xRatioUV = pSrcFrame.planes[0].nRealWidth / pSrcFrame.planes[1].nRealWidth;
        yRatioUV = pSrcFrame.planes[0].nRealHeight / pSrcFrame.planes[1].nRealHeight;
        nLogxRatioUV = ilog2(xRatioUV);
        nLogyRatioUV = ilog2(yRatioUV);
    }

    // Calculate new block count
    int nRealWidth = pSrcFrame.planes[0].nRealWidth;
    int nRealHeight = pSrcFrame.planes[0].nRealHeight;
    int nWidth = pSrcFrame.planes[0].nWidth;
    int nHeight = pSrcFrame.planes[0].nHeight;
    nBlkX = (nRealWidth - nOverlapX) / (nBlkSizeX - nOverlapX);
    nBlkY = (nRealHeight - nOverlapY) / (nBlkSizeY - nOverlapY);

    int nWidth_B = (nBlkSizeX - nOverlapX) * nBlkX + nOverlapX;
    int nHeight_B = (nBlkSizeY - nOverlapY) * nBlkY + nOverlapY;

    while (nWidth_B < nRealWidth) {
        nBlkX++;
        nWidth_B = (nBlkSizeX - nOverlapX) * nBlkX + nOverlapX;
    }

    while (nHeight_B < nRealHeight) {
        nBlkY++;
        nHeight_B = (nBlkSizeY - nOverlapY) * nBlkY + nOverlapY;
    }

    if (nWidth_B > nWidth || nHeight_B > nHeight)
        throw MotionBlockPyramidError("The chosen block size has no multiple that will process the entire frame without exceeding the super clip padding, derive a new suitable super clip and try again");

    // Early exit when there are no vectors to recalculate, this usually happens due to Analyse not being able to derive vectors because the reference frame (n+delta) is out of bounds
    if (oldVectors.empty())
        return;

    vectors.resize(nBlkX * nBlkY);

    VECTOR *pBlkData = vectors.data();

    this->pRefFrame = &pRefFrame;

    InitMotionEstimationFields(useSatd, chroma, sizeof(PixelType));

    x[0] = pSrcFrame.planes[0].nHPadding;
    y[0] = pSrcFrame.planes[0].nVPadding;
    if (chroma) {
        x[1] = pSrcFrame.planes[1].nHPadding;
        x[2] = pSrcFrame.planes[2].nHPadding;
        y[1] = pSrcFrame.planes[1].nVPadding;
        y[2] = pSrcFrame.planes[2].nVPadding;
    }

    nRefPitch[0] = pRefFrame.planes[0].nPitch;
    if (chroma) {
        nRefPitch[1] = pRefFrame.planes[1].nPitch;
        nRefPitch[2] = pRefFrame.planes[2].nPitch;
    }

    searchType = st;
    nSearchParam = stp;

    int64_t nLambdaLevel = lambda / ((1 << nLogPel) * (1 << nLogPel));

    // Functions using float must not be used here
    for (int blky = 0; blky < nBlkY; blky++) {
        int blkScanDir = (blky % 2 == 0 || !meander) ? 1 : -1;
        // meander (alternate) scan blocks (even row left to right, odd row right to left)
        int blkxStart = (blky % 2 == 0 || !meander) ? 0 : nBlkX - 1;
        if (blkScanDir == 1) { // start with leftmost block
            x[0] = pSrcFrame.planes[0].nHPadding;
            if (chroma) {
                x[1] = pSrcFrame.planes[1].nHPadding;
                x[2] = pSrcFrame.planes[2].nHPadding;
            }
        } else { // start with rightmost block, but it is already set at prev row
            x[0] = pSrcFrame.planes[0].nHPadding + (nBlkSizeX - nOverlapX) * (nBlkX - 1);
            if (chroma) {
                x[1] = pSrcFrame.planes[1].nHPadding + ((nBlkSizeX - nOverlapX) / xRatioUV) * (nBlkX - 1);
                x[2] = pSrcFrame.planes[2].nHPadding + ((nBlkSizeX - nOverlapX) / xRatioUV) * (nBlkX - 1);
            }
        }
        for (int iblkx = 0; iblkx < nBlkX; iblkx++) {
            int blkx = blkxStart + iblkx * blkScanDir;

            const uint8_t *pRealSrc[3] = {};
            pRealSrc[0] = pSrcFrame.planes[0].GetAbsolutePointerPel1<PixelType>(x[0], y[0]);
            if (chroma) {
                pRealSrc[1] = pSrcFrame.planes[1].GetAbsolutePointerPel1<PixelType>(x[1], y[1]);
                pRealSrc[2] = pSrcFrame.planes[2].GetAbsolutePointerPel1<PixelType>(x[2], y[2]);
            }

            BLITLUMA(pSrc_temp[0], nSrcPitch_temp[0], pRealSrc[0], pSrcFrame.planes[0].nPitch);
            if (chroma) {
                BLITCHROMA(pSrc_temp[1], nSrcPitch_temp[1], pRealSrc[1], pSrcFrame.planes[1].nPitch);
                BLITCHROMA(pSrc_temp[2], nSrcPitch_temp[2], pRealSrc[2], pSrcFrame.planes[2].nPitch);
            }

            if (blky == 0)
                nLambda = 0;
            else
                nLambda = nLambdaLevel;


            // may be they must be scaled by nPel ?

            /* computes search boundaries */
            nDxMax = (pSrcFrame.planes[0].nPaddedWidth - x[0] - nBlkSizeX) << nLogPel;
            nDyMax = (pSrcFrame.planes[0].nPaddedHeight - y[0] - nBlkSizeY) << nLogPel;
            nDxMin = -(x[0] << nLogPel);
            nDyMin = -(y[0] << nLogPel);

            // get and interplolate old vectors
            int centerX = nBlkSizeX / 2 + (nBlkSizeX - nOverlapX) * blkx; // center of new block
            int blkxold = (centerX - nBlkSizeXold / 2) / nStepXold;       // centerXold less or equal to new
            int centerY = nBlkSizeY / 2 + (nBlkSizeY - nOverlapY) * blky;
            int blkyold = (centerY - nBlkSizeYold / 2) / nStepYold;

            int deltaX = std::max(0, centerX - (nBlkSizeXold / 2 + nStepXold * blkxold)); // distance from old to new
            int deltaY = std::max(0, centerY - (nBlkSizeYold / 2 + nStepYold * blkyold));

            int blkxold1 = std::min(nBlkXold - 1, std::max(0, blkxold));
            int blkxold2 = std::min(nBlkXold - 1, std::max(0, blkxold + 1));
            int blkyold1 = std::min(nBlkYold - 1, std::max(0, blkyold));
            int blkyold2 = std::min(nBlkYold - 1, std::max(0, blkyold + 1));

            VECTOR vectorOld; // interpolated or nearest

            if (smooth) { // interpolate
                VECTOR vectorOld1 = oldVectors[blkxold1 + blkyold1 * nBlkXold]; // 4 old nearest vectors (may coinside)
                VECTOR vectorOld2 = oldVectors[blkxold2 + blkyold1 * nBlkXold];
                VECTOR vectorOld3 = oldVectors[blkxold1 + blkyold2 * nBlkXold];
                VECTOR vectorOld4 = oldVectors[blkxold2 + blkyold2 * nBlkXold];

                // interpolate
                int vector1_x = vectorOld1.x * nStepXold + deltaX * (vectorOld2.x - vectorOld1.x); // scaled by nStepXold to skip slow division
                int vector1_y = vectorOld1.y * nStepXold + deltaX * (vectorOld2.y - vectorOld1.y);
                int64_t vector1_sad = vectorOld1.sad * nStepXold + deltaX * (vectorOld2.sad - vectorOld1.sad);

                int vector2_x = vectorOld3.x * nStepXold + deltaX * (vectorOld4.x - vectorOld3.x);
                int vector2_y = vectorOld3.y * nStepXold + deltaX * (vectorOld4.y - vectorOld3.y);
                int64_t vector2_sad = vectorOld3.sad * nStepXold + deltaX * (vectorOld4.sad - vectorOld3.sad);

                vectorOld.x = (vector1_x + deltaY * (vector2_x - vector1_x) / nStepYold) / nStepXold;
                vectorOld.y = (vector1_y + deltaY * (vector2_y - vector1_y) / nStepYold) / nStepXold;
                vectorOld.sad = (vector1_sad + deltaY * (vector2_sad - vector1_sad) / nStepYold) / nStepXold;

            } else { // nearest
                if (deltaX * 2 < nStepXold && deltaY * 2 < nStepYold)
                    vectorOld = oldVectors[blkxold1 + blkyold1 * nBlkXold];
                else if (deltaX * 2 >= nStepXold && deltaY * 2 < nStepYold)
                    vectorOld = oldVectors[blkxold2 + blkyold1 * nBlkXold];
                else if (deltaX * 2 < nStepXold && deltaY * 2 >= nStepYold)
                    vectorOld = oldVectors[blkxold1 + blkyold2 * nBlkXold];
                else //(deltaX*2>=nStepXold && deltaY*2>=nStepYold )
                    vectorOld = oldVectors[blkxold2 + blkyold2 * nBlkXold];
            }

            // scale vector to new nPel
            vectorOld.x = (vectorOld.x << nLogPel) >> nLogPelold;
            vectorOld.y = (vectorOld.y << nLogPel) >> nLogPelold;

            predictor = ClipMV(vectorOld);                                                                    // predictor
            predictor.sad = vectorOld.sad * (nBlkSizeX * nBlkSizeY) / (nBlkSizeXold * nBlkSizeYold); // normalized to new block size

            bestMV = predictor;

            int64_t sad = SAD(pSrc_temp[0], nSrcPitch_temp[0], GetRefBlock<nLogPel, PixelType>(predictor.x, predictor.y), nRefPitch[0]);
            if (chroma) {
                sad += SADCHROMA(pSrc_temp[1], nSrcPitch_temp[1], GetRefBlockU<nLogPel, PixelType>(predictor.x, predictor.y), nRefPitch[1]);
                sad += SADCHROMA(pSrc_temp[2], nSrcPitch_temp[2], GetRefBlockV<nLogPel, PixelType>(predictor.x, predictor.y), nRefPitch[2]);
            }
            bestMV.sad = sad;
            nMinCost = sad;

            if (bestMV.sad > thSAD) { // if old interpolated vector is bad
                // then, we refine, according to the search type

                if (searchType == SearchType::Logarithmic) {
                    for (int i = nSearchParam; i > 0; i /= 2)
                        DiamondSearch<nLogPel, PixelType>(i);

                } else if (searchType == SearchType::Exhaustive) {
                    int mvx = bestMV.x;
                    int mvy = bestMV.y;
                    for (int i = 1; i <= nSearchParam; i++) // region is same as exhaustive, but ordered by radius (from near to far)
                        ExpandingSearch<nLogPel, PixelType>(i, 1, mvx, mvy);
                } else if (searchType == SearchType::Hex2) {
                    Hex2Search<nLogPel, PixelType>(nSearchParam);

                } else if (searchType == SearchType::UnevenMultiHexagon) {
                    UMHSearch<nLogPel, PixelType>(nSearchParam, bestMV.x, bestMV.y);

                } else if (searchType == SearchType::Horizontal) {
                    int mvx = bestMV.x;
                    int mvy = bestMV.y;
                    for (int i = 1; i <= nSearchParam; i++) {
                        CheckMV<nLogPel, PixelType>(mvx - i, mvy);
                        CheckMV<nLogPel, PixelType>(mvx + i, mvy);
                    }
                } else if (searchType == SearchType::Vertical) {
                    int mvx = bestMV.x;
                    int mvy = bestMV.y;
                    for (int i = 1; i <= nSearchParam; i++) {
                        CheckMV<nLogPel, PixelType>(mvx, mvy - i);
                        CheckMV<nLogPel, PixelType>(mvx, mvy + i);
                    }
                }
            }

            /* write the results */
            pBlkData[blkx] = bestMV;

            if (iblkx < nBlkX - 1) {
                x[0] += (nBlkSizeX - nOverlapX) * blkScanDir;
                if (chroma) {
                    x[1] += ((nBlkSizeX - nOverlapX) >> nLogxRatioUV) * blkScanDir;
                    x[2] += ((nBlkSizeX - nOverlapX) >> nLogxRatioUV) * blkScanDir;
                }
            }
        }
        pBlkData += nBlkX;

        y[0] += (nBlkSizeY - nOverlapY);
        if (chroma) {
            y[1] += ((nBlkSizeY - nOverlapY) >> nLogyRatioUV);
            y[2] += ((nBlkSizeY - nOverlapY) >> nLogyRatioUV);
        }
    }
}

void MotionBlockLevel::RecalculateMVs(const FramePyramidLevel &pSrcFrame, const FramePyramidLevel &pRefFrame,
    int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, bool chroma,
    SearchType st, int stp, int64_t lambda, int pnew,
    int fieldShift, int64_t thSAD, bool useSatd, bool smooth, bool meander, int bytesPerSample) {

    if (bytesPerSample == 1) {
        if (nLogPel == 0)
            DoRecalculateMVs<0, uint8_t>(pSrcFrame, pRefFrame, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma, st, stp, lambda, pnew, fieldShift, thSAD, smooth, meander, useSatd);
        else if (nLogPel == 1)
            DoRecalculateMVs<1, uint8_t>(pSrcFrame, pRefFrame, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma, st, stp, lambda, pnew, fieldShift, thSAD, smooth, meander, useSatd);
        else
            DoRecalculateMVs<2, uint8_t>(pSrcFrame, pRefFrame, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma, st, stp, lambda, pnew, fieldShift, thSAD, smooth, meander, useSatd);
    } else if (bytesPerSample == 2) {
        if (nLogPel == 0)
            DoRecalculateMVs<0, uint16_t>(pSrcFrame, pRefFrame, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma, st, stp, lambda, pnew, fieldShift, thSAD, smooth, meander, useSatd);
        else if (nLogPel == 1)
            DoRecalculateMVs<1, uint16_t>(pSrcFrame, pRefFrame, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma, st, stp, lambda, pnew, fieldShift, thSAD, smooth, meander, useSatd);
        else
            DoRecalculateMVs<2, uint16_t>(pSrcFrame, pRefFrame, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma, st, stp, lambda, pnew, fieldShift, thSAD, smooth, meander, useSatd);
    } else {
        if (nLogPel == 0)
            DoRecalculateMVs<0, float>(pSrcFrame, pRefFrame, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma, st, stp, lambda, pnew, fieldShift, thSAD, smooth, meander, useSatd);
        else if (nLogPel == 1)
            DoRecalculateMVs<1, float>(pSrcFrame, pRefFrame, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma, st, stp, lambda, pnew, fieldShift, thSAD, smooth, meander, useSatd);
        else
            DoRecalculateMVs<2, float>(pSrcFrame, pRefFrame, nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma, st, stp, lambda, pnew, fieldShift, thSAD, smooth, meander, useSatd);
    }
}


bool MotionBlockLevel::IsSceneChange(int64_t nTh1, float nTh2) const noexcept {
    int sum = 0;
    for (int i = 0; i < nBlkCount; i++)
        sum += (vectors[i].sad > nTh1) ? 1 : 0;

    return (sum > nTh2);
}

// levels > 0 means use that many levels, levels < 0 means use all valid levels except abs(levels) from the end, levels = 0 means use all valid levels
// this was mostly inherited from old mvtools code
MotionBlockPyramid::MotionBlockPyramid(const FramePyramid &src, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int nLevels, bool chroma, int deltaFrame) {
    this->nBlkSizeX = nBlkSizeX;
    this->nBlkSizeY = nBlkSizeY;
    this->nOverlapX = nOverlapX;
    this->nOverlapY = nOverlapY;
    this->nDeltaFrame = deltaFrame;
    xRatioUV = src.xRatioUV;
    yRatioUV = src.yRatioUV;
    this->chroma = src.chroma && chroma;
    nWidth = src.nWidth[0];
    nHeight = src.nHeight[0];
    nRealWidth = src.nRealWidth[0];
    nRealHeight = src.nRealHeight[0];
    nHPadding = src.nHPad[0];
    nVPadding = src.nVPad[0];
    nPel = src.nPel;
    nBlkX = (nRealWidth - nOverlapX) / (nBlkSizeX - nOverlapX);
    nBlkY = (nRealHeight - nOverlapY) / (nBlkSizeY - nOverlapY);
    bitsPerSample = src.bitsPerSample;

    int nWidth_B = (nBlkSizeX - nOverlapX) * nBlkX + nOverlapX;
    int nHeight_B = (nBlkSizeY - nOverlapY) * nBlkY + nOverlapY;

    while (nWidth_B < nRealWidth) {
        nBlkX++;
        nWidth_B = (nBlkSizeX - nOverlapX) * nBlkX + nOverlapX;
    }

    while (nHeight_B < nRealHeight) {
        nBlkY++;
        nHeight_B = (nBlkSizeY - nOverlapY) * nBlkY + nOverlapY;
    }

    if (nWidth_B > nWidth || nHeight_B > nHeight)
        throw MotionBlockPyramidError("The chosen block size has no multiple that will process the entire frame without exceeding the super clip padding, derive a new suitable super clip and try again");

    int nLevelsMax = FramePyramid::GetMaxLevelsForBlockSize(nWidth, nHeight, xRatioUV, yRatioUV, nBlkSizeX, nBlkSizeY, nHPadding, nVPadding);
    
    nLevelCount = nLevels > 0 ? nLevels : nLevelsMax + nLevels;
    nLevelCount = std::min(nLevelCount, static_cast<int>(src.nLevels));

    if (nLevelCount < 1)
        throw MotionBlockPyramidError("the levels argument resolves to a non-positive level count");

    pyramidLevels.resize(nLevelCount);

    for (int i = 0; i < nLevelCount; i++) {
        const bool smallestPlane = (i == nLevelCount - 1);
        int nBlkXCurrent = ((nWidth_B >> i) - nOverlapX) / (nBlkSizeX - nOverlapX);
        int nBlkYCurrent = ((nHeight_B >> i) - nOverlapY) / (nBlkSizeY - nOverlapY);

        pyramidLevels[i].Initialize(nBlkXCurrent, nBlkYCurrent, nBlkSizeX, nBlkSizeY, (i == 0) ? nPel : 1, i, smallestPlane, chroma, nOverlapX, nOverlapY, xRatioUV, yRatioUV, bitsPerSample);
    }

    state = State::ReadyForSearch;
}

void MotionBlockPyramid::LoadFrameData(const VSFrame *srcFrame, bool loadVectors, const std::string &prefix, const VSAPI *vsapi) {
    sourceFrame = srcFrame;

    if (!srcFrame)
        return;

    auto props = vsapi->getFramePropertiesRO(srcFrame);
    int err;
    nWidth = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisWidth").c_str(), 0, &err);
    nHeight = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisHeight").c_str(), 0, &err);
    nRealWidth = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisRealWidth").c_str(), 0, &err);
    nRealHeight = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisRealHeight").c_str(), 0, &err);
    nHPadding = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisHPad").c_str(), 0, &err);
    nVPadding = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisVPad").c_str(), 0, &err);
    nPel = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisPel").c_str(), 0, &err);
    nLevelCount = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisLevels").c_str(), 0, &err);
    chroma = !!vsapi->mapGetInt(props, (prefix + "AnalysisChroma").c_str(), 0, &err);
    xRatioUV = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisXRatioUV").c_str(), 0, &err);
    yRatioUV = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisYRatioUV").c_str(), 0, &err);

    nBlkSizeX = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisBlkSizeX").c_str(), 0, &err);
    nBlkSizeY = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisBlkSizeY").c_str(), 0, &err);

    nOverlapX = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisOverlapX").c_str(), 0, &err);
    nOverlapY = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisOverlapY").c_str(), 0, &err);

    nBlkX = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisNBlkX").c_str(), 0, &err);
    nBlkY = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisNBlkY").c_str(), 0, &err);

    nDeltaFrame = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisDeltaFrame").c_str(), 0, &err);

    bitsPerSample = vsapi->mapGetIntSaturated(props, (prefix + "AnalysisBitsPerSample").c_str(), 0, &err);

    if (xRatioUV < 1 || yRatioUV < 1 || xRatioUV > 2 || yRatioUV > 2 || nRealWidth > nWidth || nRealHeight > nHeight || nVPadding < 0 || nHPadding < 0
        || nRealHeight < 1 || nRealWidth < 1 || nBlkSizeX < 2 || nBlkSizeY < 2 || nOverlapX < 0 || nOverlapY < 0
        || nOverlapX > nBlkSizeX / 2 || nOverlapY > nBlkSizeY / 2 || nLevelCount < 1 || (nPel != 1 && nPel != 2 && nPel != 4)
        || bitsPerSample < 8 || (bitsPerSample > 16 && bitsPerSample != 32) || nBlkX < 1 || nBlkY < 1)
        return;

    if (loadVectors) {
        std::string vectorsProp = prefix + "AnalysisVectors";
        std::string sadProp = prefix + "AnalysisSAD";

        int vectSize = vsapi->mapNumElements(props, vectorsProp.c_str());
        int sadSize = vsapi->mapNumElements(props, sadProp.c_str());

        if (vectSize != sadSize || static_cast<int64_t>(vectSize) != static_cast<int64_t>(nBlkX) * nBlkY) {
            state = State::MetadataOnly;
            return;
        }

        pyramidLevels.resize(1);
        pyramidLevels[0].Initialize(nBlkX, nBlkY, nBlkSizeX, nBlkSizeY, nPel, 0, true, chroma, nOverlapX, nOverlapY, xRatioUV, yRatioUV, bitsPerSample);

        const int64_t *vectData = vsapi->mapGetIntArray(props, vectorsProp.c_str(), nullptr);
        const int64_t *sadData = vsapi->mapGetIntArray(props, sadProp.c_str(), nullptr);

        auto &dest = pyramidLevels[0].vectors;
        dest.resize(nBlkX * nBlkY);
        for (int i = 0; i < vectSize; ++i) {
            auto &destBlock = dest[i];
            destBlock.x = static_cast<int>(vectData[i] & 0xFFFFFFFF);
            destBlock.y = static_cast<int>((vectData[i] >> 32) & 0xFFFFFFFF);
            destBlock.sad = sadData[i];
        }

        // Reject corrupt/hostile vector data before any consumer dereferences it for motion compensation.
        ValidateVectors();

        if (state != State::MetadataOnly)
            state = State::ReadyForRecalculate;
    } else {
        state = State::MetadataOnly;
    }
}

void MotionBlockPyramid::ValidateVectors() const {
    if (pyramidLevels.empty())
        return;

    const auto &vectors = pyramidLevels[0].vectors;

    // The grid must describe exactly one vector per block; otherwise the metadata and the vector array disagree
    // and we cannot derive safe per-block bounds. (LoadFrameData already enforces this, checked again defensively.)
    if (static_cast<int64_t>(vectors.size()) != static_cast<int64_t>(nBlkX) * nBlkY)
        throw MotionBlockPyramidError("Motion vector count does not match the block grid");

    // Bounds computed in int64 to stay safe even for hostile metadata. The safe vector range is exactly the one
    // DoSearchMVs (at level 0) and DoRecalculateMVs enforce, so any vector produced by Analyse/Recalculate passes;
    // only corrupt or hand-crafted data that would read outside the padded reference plane is rejected.
    const int64_t nLogPel = ilog2(nPel);
    const int64_t nPaddedWidth = static_cast<int64_t>(nWidth) + 2 * nHPadding;
    const int64_t nPaddedHeight = static_cast<int64_t>(nHeight) + 2 * nVPadding;
    const int64_t stepX = nBlkSizeX - nOverlapX;
    const int64_t stepY = nBlkSizeY - nOverlapY;

    for (int blky = 0; blky < nBlkY; ++blky) {
        // Block top-left in the padded plane (matches MotionBlockLevel x[0]/y[0]).
        const int64_t y0 = static_cast<int64_t>(nVPadding) + stepY * blky;
        const int64_t nDyMin = -(y0 << nLogPel);
        const int64_t nDyMax = (nPaddedHeight - y0 - nBlkSizeY) << nLogPel;

        for (int blkx = 0; blkx < nBlkX; ++blkx) {
            const int64_t x0 = static_cast<int64_t>(nHPadding) + stepX * blkx;
            const int64_t nDxMin = -(x0 << nLogPel);
            const int64_t nDxMax = (nPaddedWidth - x0 - nBlkSizeX) << nLogPel;

            const VECTOR &v = vectors[static_cast<size_t>(blky) * nBlkX + blkx];

            if (v.sad < 0)
                throw MotionBlockPyramidError("Motion vector data contains a negative SAD value");

            if (v.x < nDxMin || v.x >= nDxMax || v.y < nDyMin || v.y >= nDyMax)
                throw MotionBlockPyramidError("Motion vector out of bounds; it would cause an out-of-bounds reference read");
        }
    }
}

MotionBlockPyramid::MotionBlockPyramid(const VSFrame *src, bool loadVectors, const std::string &prefix, const VSAPI *vsapi) :
    vsapi(vsapi) {
    try {
        LoadFrameData(src, loadVectors, prefix, vsapi);
    } catch (...) {
        vsapi->freeFrame(sourceFrame);
        throw;
    }
}

MotionBlockPyramid::MotionBlockPyramid(VSNode *node, const std::string &prefix, const VSAPI *vsapi) :
    vsapi(vsapi) {
    char errorMsg[ERROR_SIZE] = {};
    const VSFrame *srcFrame = vsapi->getFrame(0, node, errorMsg, ERROR_SIZE);
    if (!srcFrame)
        throw std::runtime_error("Failed to retrieve first frame from super clip. Error message: " + std::string(errorMsg));
    try {
        LoadFrameData(srcFrame, false, prefix, vsapi);
    } catch (...) {
        vsapi->freeFrame(sourceFrame); // see the note above; LoadFrameData took ownership of srcFrame
        throw;
    }
}

MotionBlockPyramid::~MotionBlockPyramid() {
    if (vsapi)
        vsapi->freeFrame(sourceFrame);
}

void MotionBlockPyramid::SearchMVs(const FramePyramid &pSrcGOF, const FramePyramid &pRefGOF,
    SearchType searchType, int nSearchParam, int nPelSearch, int64_t nLambda,
    int64_t lsad, int pnew, int plevel, bool global, int fieldShift, bool useSatd,
    int pzero, int pglobal, int64_t badSAD, int badrange, bool meander, TryManyLevels tryMany,
    bool chroma) {

    if (state != State::ReadyForSearch)
        throw MotionBlockPyramidError("MotionBlockPyramid isn't in an appropriate state for searching motion vectors");

    if (!pSrcGOF.IsCompatible(pRefGOF))
        throw MotionBlockPyramidError("The two reference frames don't have the same format");

    if (!IsCompatibleWithAnalysis(pSrcGOF) || !IsCompatibleWithAnalysis(pRefGOF))
        throw MotionBlockPyramidError("Incompatible frame format for motion vector search, bitdepth must match");

    int fieldShiftCur = (nLevelCount - 1 == 0) ? fieldShift : 0; // may be non zero for finest level only

    VECTOR globalMV = zeroMV; // create and init global motion vector as zero

    if (!global)
        pglobal = pzero;

    int bytesPerSample = SelectOnBitsPerSample(pSrcGOF.bitsPerSample, 1, 2, 4);

    auto tryManyLevel = [tryMany] (bool FinestLevel) {
        return (tryMany == TryManyLevels::All || (tryMany == TryManyLevels::AllExceptFinest && !FinestLevel));
    };

    // Search the motion vectors, for the low details interpolations first
    SearchType searchTypeSmallest = (nLevelCount == 1 || searchType == SearchType::Horizontal || searchType == SearchType::Vertical) ? searchType : SearchType::Exhaustive; // full search for smallest coarse plane
    int nSearchParamSmallest = (nLevelCount == 1) ? nPelSearch : nSearchParam;
    pyramidLevels[nLevelCount - 1].SearchMVs(
        pSrcGOF.GetLevel(nLevelCount - 1),
        pRefGOF.GetLevel(nLevelCount - 1),
        searchTypeSmallest, nSearchParamSmallest, nLambda, lsad, pnew, plevel,
        &globalMV, fieldShiftCur, useSatd,
        pzero, pglobal, badSAD, badrange, meander, tryManyLevel(nLevelCount == 1), chroma, bytesPerSample);
    // Refining the search until we reach the highest detail interpolation.

    for (int i = nLevelCount - 2; i >= 0; i--) {
        SearchType searchTypeLevel = (i == 0 || searchType == SearchType::Horizontal || searchType == SearchType::Vertical) ? searchType : SearchType::Exhaustive; // full search for coarse planes
        int nSearchParamLevel = (i == 0) ? nPelSearch : nSearchParam;
        if (global) {
            pyramidLevels[i + 1].EstimateGlobalMVDoubled(globalMV); // get updated global MV (doubled)
        }
        pyramidLevels[i].InterpolatePredictorsFromParent(pyramidLevels[i + 1]);
        fieldShiftCur = (i == 0) ? fieldShift : 0; // may be non zero for finest level only
        pyramidLevels[i].SearchMVs(pSrcGOF.GetLevel(i), pRefGOF.GetLevel(i),
            searchTypeLevel, nSearchParamLevel, nLambda, lsad, pnew, plevel,
            &globalMV, fieldShiftCur, useSatd,
            pzero, pglobal, badSAD, badrange, meander, tryManyLevel(i == 0), chroma, bytesPerSample);
    }

    state = State::AnalysisDone;
}


void MotionBlockPyramid::RecalculateMVs(const FramePyramid &pSrcGOF, const FramePyramid &pRefGOF,
    int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, bool chroma,
    SearchType searchType, int nSearchParam, int64_t nLambda, int pnew,
    int fieldShift, int64_t thSAD, bool useSatd, bool smooth, bool meander, int deltaFrame) {

    if (!IsCompatibleForRecalc(pSrcGOF) || !IsCompatibleForRecalc(pRefGOF))
        throw MotionBlockPyramidError("Incompatible frame format for motion vector recalculation, bitdepth must match");

    if (!pSrcGOF.IsCompatible(pRefGOF))
        throw MotionBlockPyramidError("The two reference frames don't have the same format");

    if (pyramidLevels.empty()) {
        if (state != State::MetadataOnly)
            throw MotionBlockPyramidError("The vectors passed are not a valid analysis result");
        pyramidLevels.resize(1);
        pyramidLevels[0].Initialize(this->nBlkX, this->nBlkY, this->nBlkSizeX, this->nBlkSizeY, this->nPel, 0,
            true, this->chroma, this->nOverlapX, this->nOverlapY, this->xRatioUV, this->yRatioUV, this->bitsPerSample);
    }

    int bytesPerSample = SelectOnBitsPerSample(pSrcGOF.bitsPerSample, 1, 2, 4);

    pyramidLevels[0].RecalculateMVs(pSrcGOF.GetLevel(0), pRefGOF.GetLevel(0),
        nBlkSizeX, nBlkSizeY, nOverlapX, nOverlapY, chroma,
        searchType, nSearchParam, nLambda, pnew,
        fieldShift, thSAD, useSatd, smooth, meander, bytesPerSample);

    // Update from level 0 plane properties afterwards for proper export
    this->nBlkSizeX = pyramidLevels[0].nBlkSizeX;
    this->nBlkSizeY = pyramidLevels[0].nBlkSizeY;
    this->nOverlapX = pyramidLevels[0].nOverlapX;
    this->nOverlapY = pyramidLevels[0].nOverlapY;
    this->xRatioUV = pSrcGOF.xRatioUV;
    this->yRatioUV = pSrcGOF.yRatioUV;
    nBlkX = pyramidLevels[0].nBlkX;
    nBlkY = pyramidLevels[0].nBlkY;
    nPel = pyramidLevels[0].nPel;
    bitsPerSample = pSrcGOF.bitsPerSample;
    nWidth = pSrcGOF.nWidth[0];
    nHeight = pSrcGOF.nHeight[0];
    nRealWidth = pSrcGOF.nRealWidth[0];
    nRealHeight = pSrcGOF.nRealHeight[0];
    nHPadding = pSrcGOF.nHPad[0];
    nVPadding = pSrcGOF.nVPad[0];
    this->chroma = chroma;
    this->nDeltaFrame = deltaFrame;

    state = State::AnalysisDone;
}

void MotionBlockPyramid::ExportFrameData(VSFrame *dst, const std::string &prefix, const VSAPI *vsapi) const noexcept {
    auto props = vsapi->getFramePropertiesRW(dst);

    vsapi->mapSetInt(props, (prefix + "AnalysisWidth").c_str(), nWidth, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisHeight").c_str(), nHeight, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisRealWidth").c_str(), nRealWidth, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisRealHeight").c_str(), nRealHeight, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisHPad").c_str(), nHPadding, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisVPad").c_str(), nVPadding, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisPel").c_str(), nPel, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisLevels").c_str(), pyramidLevels.size(), maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisChroma").c_str(), chroma, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisXRatioUV").c_str(), xRatioUV, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisYRatioUV").c_str(), yRatioUV, maReplace);

    vsapi->mapSetInt(props, (prefix + "AnalysisBlkSizeX").c_str(), nBlkSizeX, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisBlkSizeY").c_str(), nBlkSizeY, maReplace);

    vsapi->mapSetInt(props, (prefix + "AnalysisOverlapX").c_str(), nOverlapX, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisOverlapY").c_str(), nOverlapY, maReplace);

    vsapi->mapSetInt(props, (prefix + "AnalysisNBlkX").c_str(), nBlkX, maReplace);
    vsapi->mapSetInt(props, (prefix + "AnalysisNBlkY").c_str(), nBlkY, maReplace);

    vsapi->mapSetInt(props, (prefix + "AnalysisDeltaFrame").c_str(), nDeltaFrame, maReplace);

    vsapi->mapSetInt(props, (prefix + "AnalysisBitsPerSample").c_str(), bitsPerSample, maReplace);

    if (HasMotionVectors()) {
        std::string vectorsProp = prefix + "AnalysisVectors";
        std::vector<int64_t> tmp;
        tmp.resize(pyramidLevels[0].vectors.size());
        const auto &source = pyramidLevels[0].vectors;
        for (size_t i = 0; i < source.size(); i++)
            tmp[i] = (static_cast<int64_t>(source[i].x) & 0xFFFFFFFF) | ((static_cast<int64_t>(source[i].y) & 0xFFFFFFFF) << 32);

        vsapi->mapSetIntArray(props, (prefix + "AnalysisVectors").c_str(), tmp.data(), static_cast<int>(tmp.size()));

        for (size_t i = 0; i < source.size(); i++)
            tmp[i] = source[i].sad;

        vsapi->mapSetIntArray(props, (prefix + "AnalysisSAD").c_str(), tmp.data(), static_cast<int>(tmp.size()));
    }
}

bool MotionBlockPyramid::IsUsable(int64_t thscd1, float thscd2) const noexcept {
    return HasMotionVectors() && !pyramidLevels[0].IsSceneChange(thscd1, thscd2);
}

BlockData MotionBlockPyramid::GetBlock(int nBlk) const noexcept {
    int bY = nBlk / nBlkX;
    int bX = nBlk % nBlkX;

    return { bX * (nBlkSizeX - nOverlapX), bY * (nBlkSizeY - nOverlapY), pyramidLevels[0].vectors[nBlk] };
}

double MotionBlockPyramid::GetThSCDScaleFactor(int bitsPerSample) const {
    // thscd1 is defined as SAD per reference 8x8 luma block at 8-bit depth.
    // Scale to the actual block geometry, chroma contribution, and bit depth.
    const double lumaScale = (double)(nBlkSizeX * nBlkSizeY) / (8.0 * 8.0);
    const double chromaFactor = chroma ? (1.0 + 2.0 / (xRatioUV * yRatioUV)) : 1.0;
    const double depthScale = ((1 << std::min(16, bitsPerSample)) - 1) / 255.0; // float SAD uses the 16-bit scale

    return lumaScale * chromaFactor * depthScale;
}

void MotionBlockPyramid::ScaleThSCD(int64_t &thscd1, float &thscd2, int bitsPerSample) const {
    constexpr int maxSAD = 8 * 8 * 255;

    if (thscd1 < 0 || thscd1 > maxSAD)
        throw MotionBlockPyramidError("thscd1 must be between 0 and " + std::to_string(maxSAD));

    if (!std::isfinite(thscd2) || thscd2 < 0.0f || thscd2 > 100.0f)
        throw MotionBlockPyramidError("thscd2 must be a percentage between 0 and 100");

    // SCD thresholds
    thscd1 = (int64_t)(thscd1 * GetThSCDScaleFactor(bitsPerSample) + 0.5);

    // thscd2 is a percentage (0-100) of the blocks; convert to an absolute (fractional) block count.
    thscd2 = static_cast<float>((double)thscd2 * nBlkX * nBlkY / 100.0);
}

MotionBlockPyramid::State MotionBlockPyramid::GetState() const noexcept {
    return state;
}

bool MotionBlockPyramid::HasMotionVectors() const noexcept {
    return state == State::ReadyForRecalculate || state == State::AnalysisDone;
}

bool MotionBlockPyramid::IsCompatible(const MotionBlockPyramid &other) const noexcept {
    if (nWidth != other.nWidth || nHeight != other.nHeight || nRealWidth != other.nRealWidth || nRealHeight != other.nRealHeight)
        return false;

    if (nBlkSizeX != other.nBlkSizeX || nBlkSizeY != other.nBlkSizeY || nOverlapX != other.nOverlapX || nOverlapY != other.nOverlapY)
        return false;

    if (nBlkX != other.nBlkX || nBlkY != other.nBlkY)
        return false;

    if (nPel != other.nPel)
        return false;

    if (chroma && other.chroma && (xRatioUV != other.xRatioUV || yRatioUV != other.yRatioUV))
        return false;

    if (nHPadding != other.nHPadding || nVPadding != other.nVPadding)
        return false;

    if (bitsPerSample != other.bitsPerSample)
        return false;

    return true;
}

bool MotionBlockPyramid::IsCompatibleWithAnalysis(const FramePyramid &other) const noexcept {
    if (nWidth != other.nWidth[0] || nHeight != other.nHeight[0] || nRealWidth != other.nRealWidth[0] || nRealHeight != other.nRealHeight[0])
        return false;

    if (chroma && (xRatioUV != other.xRatioUV || yRatioUV != other.yRatioUV))
        return false;

    if (nHPadding != other.nHPad[0] || nVPadding != other.nVPad[0])
        return false;

    if (nPel != other.nPel)
        return false;

    return true;
}

bool MotionBlockPyramid::IsCompatibleForRecalc(const FramePyramid &other) const noexcept {
    if (bitsPerSample != other.bitsPerSample)
        return false;

    return true;
}

//////////////////////////
// Mask related functions

template<typename PixelType>
static PixelType MaskLength(VECTOR v, uint8_t pel, float fMaskNormFactor2, float fHalfGamma, int maxVal) noexcept {
    const double dx = (double)v.x / pel;
    const double dy = (double)v.y / pel;
    const double norme = dx * dx + dy * dy;
    const double l = maxVal * pow(norme * fMaskNormFactor2, (double)fHalfGamma);
    return static_cast<PixelType>((l > maxVal) ? maxVal : l);
}

template<typename PixelType>
std::unique_ptr<BlockMask<PixelType>> MotionBlockPyramid::MakeVectorLengthMask(float normFactor, float fGamma) const noexcept {
    auto RetMask = std::make_unique<BlockMask<PixelType>>(nBlkX, nBlkY);

    ptrdiff_t MaskPitch = RetMask->stride;
    PixelType *Mask = RetMask->mask;

    float halfGamma = fGamma / 2;
    normFactor = normFactor * normFactor;

    int maxVal = PixelMaxValue<PixelType>(bitsPerSample);
    MaskPitch /= sizeof(PixelType);

    for (int by = 0; by < nBlkY; by++) {
        for (int bx = 0; bx < nBlkX; bx++)
            Mask[bx] = MaskLength<PixelType>(GetBlock(bx + by * nBlkX).vector, nPel, normFactor, halfGamma, maxVal);
        Mask += MaskPitch;
    }

    return RetMask;
}

template<typename PixelType>
static PixelType PixelNorm(int64_t sad, float dSADNormFactor, float fGamma, int maxVal) {
    float l = maxVal * pow(sad * dSADNormFactor, fGamma);
    return static_cast<PixelType>((l > maxVal) ? maxVal : l);
}

template<typename PixelType>
std::unique_ptr<BlockMask<PixelType>> MotionBlockPyramid::MakeSADMask(float dSADNormFactor, float fGamma, int time256) const noexcept {
    auto RetMask = std::make_unique<BlockMask<PixelType>>(nBlkX, nBlkY);

    ptrdiff_t MaskPitch = RetMask->stride;
    PixelType *Mask = RetMask->mask;

    int nBlkStepX = nBlkSizeX - nOverlapX;
    int nBlkStepY = nBlkSizeY - nOverlapY;
    dSADNormFactor = 4.0f * dSADNormFactor / (nBlkSizeX * nBlkSizeY);
    memset(Mask, 0, nBlkY * MaskPitch);
    int time4096X = (256 - time256) * 16 / (nBlkStepX * nPel);
    int time4096Y = (256 - time256) * 16 / (nBlkStepY * nPel);

    int maxVal = PixelMaxValue<PixelType>(bitsPerSample);
    MaskPitch /= sizeof(PixelType);

    for (int by = 0; by < nBlkY; by++) {
        for (int bx = 0; bx < nBlkX; bx++) {
            int i = bx + by * nBlkX;
            const BlockData block = GetBlock(i);
            int vx = block.vector.x;
            int vy = block.vector.y;
            int bxi = bx - vx * time4096X / 4096;
            int byi = by - vy * time4096Y / 4096;
            if (bxi < 0 || bxi >= nBlkX || byi < 0 || byi >= nBlkY) {
                bxi = bx;
                byi = by;
            }
            int i1 = bxi + byi * nBlkX;
            int64_t sad = GetBlock(i1).vector.sad >> (std::min(16, bitsPerSample) - 8);
            Mask[bx] = PixelNorm<PixelType>(sad, dSADNormFactor, fGamma, maxVal);
        }
        Mask += MaskPitch;
    }

    return RetMask;

}

template<typename PixelType>
static void ByteOccMask(PixelType &occMask, int occlusion, float occnorm, float fGamma, int maxVal) {
    float maxValF = static_cast<float>(maxVal);
    if (fGamma == 1.0f)
        occMask = std::max<PixelType>(occMask, static_cast<PixelType>(std::min<float>(maxValF * occlusion * occnorm, maxValF)));
    else
        occMask = std::max<PixelType>(occMask, static_cast<PixelType>(std::min<float>(maxValF * pow(occlusion * occnorm, fGamma), maxValF)));
}

template<typename PixelType>
std::unique_ptr<BlockMask<PixelType>> MotionBlockPyramid::MakeVectorOcclusionMask(float dMaskNormDivider, float fGamma, int time256, bool force8bitRange) const noexcept {
    auto RetMask = std::make_unique<BlockMask<PixelType>>(nBlkX, nBlkY);

    int nBlkStepX = nBlkSizeX - nOverlapX;
    int nBlkStepY = nBlkSizeY - nOverlapY;
    bool isBackward = nDeltaFrame > 0;

    ptrdiff_t MaskPitch = RetMask->stride;
    PixelType *Mask = RetMask->mask;

    memset(Mask, 0, MaskPitch * nBlkY);
    int time4096X = time256 * 16 / (nBlkStepX * nPel);
    int time4096Y = time256 * 16 / (nBlkStepY * nPel);
    float occnormX = (80.0f  * dMaskNormDivider) / (nBlkStepX * nPel);
    float occnormY = (80.0f * dMaskNormDivider) / (nBlkStepY * nPel);

    int maxVal = force8bitRange ? 255 : PixelMaxValue<PixelType>(bitsPerSample);
    MaskPitch /= sizeof(PixelType);

    for (int by = 0; by < nBlkY; by++) {
        for (int bx = 0; bx < nBlkX; bx++) {
            int i = bx + by * nBlkX;
            const BlockData block = GetBlock(i);
            int vx = block.vector.x;
            int vy = block.vector.y;
            if (bx < nBlkX - 1) {
                int i1 = i + 1;
                const BlockData block1 = GetBlock(i1);
                int vx1 = block1.vector.x;
                if (vx1 < vx) {
                    int occlusion = vx - vx1;
                    int minb = isBackward ? std::max(0, bx + 1 - occlusion * time4096X / 4096) : bx;
                    int maxb = isBackward ? bx + 1 : std::min(bx + 1 - occlusion * time4096X / 4096, nBlkX - 1);
                    for (int bxi = minb; bxi <= maxb; bxi++)
                        ByteOccMask<PixelType>(Mask[bxi + by * MaskPitch], occlusion, occnormX, fGamma, maxVal);
                }
            }
            if (by < nBlkY - 1) {
                int i1 = i + nBlkX;
                const BlockData block1 = GetBlock(i1);
                int vy1 = block1.vector.y;
                if (vy1 < vy) {
                    int occlusion = vy - vy1;
                    int minb = isBackward ? std::max(0, by + 1 - occlusion * time4096Y / 4096) : by;
                    int maxb = isBackward ? by + 1 : std::min(by + 1 - occlusion * time4096Y / 4096, nBlkY - 1);
                    for (int byi = minb; byi <= maxb; byi++)
                        ByteOccMask<PixelType>(Mask[bx + byi * MaskPitch], occlusion, occnormY, fGamma, maxVal);
                }
            }
        }
    }

    return RetMask;
}

std::unique_ptr<SmallVectorMasks> MotionBlockPyramid::MakeSmallVectorMasks(int fieldOffset) const noexcept {
    std::unique_ptr<SmallVectorMasks> masks = std::make_unique<SmallVectorMasks>(nBlkX, nBlkY);
    ptrdiff_t pitchVSmallY = masks->pitchVSmallY / sizeof(uint16_t);

    for (int by = 0; by < nBlkY; by++) {
        for (int bx = 0; bx < nBlkX; bx++) {
            int i = bx + by * nBlkX;
            const BlockData block = GetBlock(i);
            int vx = block.vector.x;
            int vy = block.vector.y;
            // Bias into uint16_t for the resizer (see SmallVectorMasks). Saturate so a vector that doesn't
            // fit the biased range clamps to the farthest representable offset instead of wrapping to a
            // bogus (possibly out-of-bounds) one on large frames / high pel.
            masks->VXSmallY[bx + by * pitchVSmallY] = static_cast<uint16_t>(std::clamp(vx + (1 << 15), 0, 0xFFFF));
            masks->VYSmallY[bx + by * pitchVSmallY] = static_cast<uint16_t>(std::clamp(vy + (1 << 15) + fieldOffset, 0, 0xFFFF));
        }
    }

    return masks;
}

void SmallVectorMasks::AdjustSmallVectorMaskSubSampling(int nBlkX, int nBlkY, int subSamplingW, int subSamplingH) noexcept {
    ptrdiff_t pitch = pitchVSmallY / sizeof(uint16_t);

    if (subSamplingW > 0) {
        uint16_t *cVXSmallY = VXSmallY;
        for (int by = 0; by < nBlkY; by++) {
            for (int bx = 0; bx < nBlkX; bx++)
                cVXSmallY[bx] = ((static_cast<int>(cVXSmallY[bx]) - (1 << 15)) >> subSamplingW) + (1 << 15);
            cVXSmallY += pitch;
        }
    }

    if (subSamplingH > 0) {
        uint16_t *cVYSmallY = VYSmallY;
        for (int by = 0; by < nBlkY; by++) {
            for (int bx = 0; bx < nBlkX; bx++)
                cVYSmallY[bx] = ((static_cast<int>(cVYSmallY[bx]) - (1 << 15)) >> subSamplingH) + (1 << 15);
            cVYSmallY += pitch;
        }
    }
}

SmallVectorMasks::SmallVectorMasks(int nBlkX, int nBlkY) {
    pitchVSmallY = RoundUpToAlignment(nBlkX * sizeof(uint16_t));
    VXSmallY = mvu_aligned_malloc<uint16_t>(pitchVSmallY * nBlkY, MVU_MEMORY_ALIGN);
    VYSmallY = mvu_aligned_malloc<uint16_t>(pitchVSmallY * nBlkY, MVU_MEMORY_ALIGN);
}

SmallVectorMasks::~SmallVectorMasks() {
    mvu_aligned_free(VXSmallY);
    mvu_aligned_free(VYSmallY);
}

// Explicit instantiations to keep the headers somewhat clean and readable

template std::unique_ptr<BlockMask<uint8_t>> MotionBlockPyramid::MakeVectorLengthMask(float normFactor, float fGamma) const noexcept;
template std::unique_ptr<BlockMask<uint16_t>> MotionBlockPyramid::MakeVectorLengthMask(float normFactor, float fGamma) const noexcept;
template std::unique_ptr<BlockMask<float>> MotionBlockPyramid::MakeVectorLengthMask(float normFactor, float fGamma) const noexcept;

template std::unique_ptr<BlockMask<uint8_t>> MotionBlockPyramid::MakeSADMask(float dSADNormFactor, float fGamma, int time256) const noexcept;
template std::unique_ptr<BlockMask<uint16_t>> MotionBlockPyramid::MakeSADMask(float dSADNormFactor, float fGamma, int time256) const noexcept;
template std::unique_ptr<BlockMask<float>> MotionBlockPyramid::MakeSADMask(float dSADNormFactor, float fGamma, int time256) const noexcept;

template std::unique_ptr<BlockMask<uint8_t>> MotionBlockPyramid::MakeVectorOcclusionMask(float dMaskNormDivider, float fGamma, int time256, bool force8bitRange) const noexcept;
template std::unique_ptr<BlockMask<uint16_t>> MotionBlockPyramid::MakeVectorOcclusionMask(float dMaskNormDivider, float fGamma, int time256, bool force8bitRange) const noexcept;
template std::unique_ptr<BlockMask<float>> MotionBlockPyramid::MakeVectorOcclusionMask(float dMaskNormDivider, float fGamma, int time256, bool force8bitRange) const noexcept;
