#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <VapourSynth4.h>
#include "SuperPyramid.h"
#include "CopyCode.h"
#include "SADFunctions.h"
#include "Common.h"

class MotionBlockPyramidError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Note that to accomodate resizing the masks are offset by +(1 << 15)
struct SmallVectorMasks {
    uint16_t *VXSmallY;
    uint16_t *VYSmallY;
    ptrdiff_t pitchVSmallY;

    void AdjustSmallVectorMaskSubSampling(int nBlkX, int nBlkY, int subSamplingW, int subSamplingH) noexcept;
    SmallVectorMasks(int nBlkX, int nBlkY);
    ~SmallVectorMasks();
};

template<typename PixelType>
struct BlockMask {
    PixelType *mask;
    ptrdiff_t stride;

    BlockMask(int nBlkX, int nBlkY) {
        stride = RoundUpToAlignment(nBlkX * sizeof(PixelType));
        mask = mvu_aligned_malloc<PixelType>(stride * nBlkY, MVU_MEMORY_ALIGN);
    }

    ~BlockMask() {
        mvu_aligned_free(mask);
    }
};

struct VECTOR {
    int x;
    int y;
    int64_t sad;
};

struct BlockData {
    int x;
    int y;
    VECTOR vector;
};

enum class SearchType {
    Logarithmic,
    Exhaustive,
    Hex2,
    UnevenMultiHexagon,
    Horizontal,
    Vertical
};

enum class TryManyLevels {
    None = 0,
    AllExceptFinest = 1,
    All = 2
};

static constexpr const VECTOR zeroMV = { 0, 0, -1 };


// A motion block level is the motion estimation context euivalent
// to a level of the super pyramid, it contains all the data and parameters needed to perform motion estimation on this level

class MotionBlockLevel {
    friend class MotionBlockPyramid;
private:
    /* fields set at initialization */

    int nBlkX;        /* width in number of blocks */
    int nBlkY;        /* height in number of blocks */
    int nBlkSizeX;    /* size of a block */
    int nBlkSizeY;    /* size of a block */
    int nBlkCount;    /* number of blocks in the plane */
    int nPel;         /* pel refinement accuracy */
    int nLogPel;      /* logarithm of the pel refinement accuracy */
    int nScale;       /* scaling factor of the plane */
    int nLogScale;    /* logarithm of the scaling factor */
    int nOverlapX; // overlap size
    int nOverlapY; // overlap size
    int xRatioUV;
    int yRatioUV;
    int nLogxRatioUV; // log of xRatioUV (0 for 1 and 1 for 2)
    int nLogyRatioUV; // log of yRatioUV (0 for 1 and 1 for 2)

    // static parameters during the whole search process
    SearchType searchType; /* search type used */
    int nSearchParam;      /* additionnal parameter for this search */
    int64_t LSAD;          // SAD limit for lambda usage
    int penaltyNew;        // cost penalty factor for new candidates
    int penaltyZero;       // cost penalty factor for zero vector
    int pglobal;           // cost penalty factor for global predictor
    int64_t verybigSAD;

    SADFunction SAD = nullptr;   /* function which computes the sad */
    SADFunction SADCHROMA = nullptr;
    COPYFunction BLITLUMA = nullptr;
    COPYFunction BLITCHROMA = nullptr;

    std::vector<VECTOR> vectors; /* motion vectors of the blocks */
    /* before the search, contains the hierachal predictor */
    /* after the search, contains the best motion vector */

    bool smallestPlane; /* say whether vectors can used predictors from a smaller plane */
    bool chroma;        /* do we do chroma me */

    /* working fields */
    // the frame to compare to, may be possible to pass prefframe instead of whole pob to some functions
    // constant the whole time
    const FramePyramidLevel *pRefFrame = nullptr;
    ptrdiff_t nRefPitch[3] = {};

    // These are all used deep in motion estimation
    VECTOR bestMV;    /* best vector found so far during the search */
    int64_t nMinCost; /* minimum cost ( sad + mv cost ) found so far */
    VECTOR predictor; /* best predictor for the current vector */

    // Maximum vector bounds possible
    int nDxMin; /* minimum x coordinate for the vector */
    int nDyMin; /* minimum y coordinate for the vector */
    int nDxMax; /* maximum x corrdinate for the vector */
    int nDyMax; /* maximum y coordinate for the vector */
    int x[3];       /* absolute x coordinate of the origin of the block in the reference frame */
    int y[3];       /* absolute y coordinate of the origin of the block in the reference frame */
    int64_t nLambda;       /* vector cost factor */

    // unknown
    VECTOR globalMVPredictor;  // predictor of global motion vector

    // always constant? constant for the whole frame?
    VECTOR zeroMVfieldShifted; // zero motion vector for fieldbased video at finest level pel2

    // Set for each block
    int nSrcPitch_temp[3];
    uint8_t *pSrc_temp[3] = {}; //for easy WRITE access to temp block
private:
    template <int nLogPel, typename PixelType>
    const uint8_t *GetRefBlock(int nVx, int nVy) const noexcept;

    template <int nLogPel, typename PixelType>
    const uint8_t *GetRefBlockU(int nVx, int nVy) const noexcept;

    template <int nLogPel, typename PixelType>
    const uint8_t *GetRefBlockV(int nVx, int nVy) const noexcept;

    template <int nLogPel, int flags, typename PixelType>
    void CheckMV_Template(int vx, int vy, int *dir, int val) noexcept;

    template <int nLogPel, int flags, typename PixelType>
    MVU_NOINLINE void CheckMV_Slow(int vx, int vy, int64_t cost, int *dir, int val) noexcept;

    template <int nLogPel, typename PixelType>
    void CheckMV0(int vx, int vy) noexcept;

    template <int nLogPel, typename PixelType>
    void CheckMV(int vx, int vy) noexcept;

    template <int nLogPel, typename PixelType>
    void CheckMV2(int vx, int vy, int *dir, int val) noexcept;

    template <int nLogPel, typename PixelType>
    void CheckMVdir(int vx, int vy, int *dir, int val) noexcept;

    template <int nLogPel, typename PixelType>
    void DiamondSearch(int length) noexcept;

    template <int nLogPel, typename PixelType>
    void ExpandingSearch(int r, int s, int mvx, int mvy) noexcept;

    template <int nLogPel, typename PixelType>
    void Hex2Search(int i_me_range) noexcept;

    template <int nLogPel, typename PixelType>
    void CrossSearch(int start, int x_max, int y_max, int mvx, int mvy) noexcept;

    template <int nLogPel, typename PixelType>
    void UMHSearch(int i_me_range, int omx, int omy) noexcept;

    template <int nLogPel, typename PixelType>
    void Refine() noexcept;

    template <int nLogPel, typename PixelType>
    void PseudoEPZSearch(int blkIdx, int blkx, int blky, int blkScanDir, int64_t badSAD, int badrange, bool tryMany, int &badcount) noexcept;

    template <int nLogPel, typename PixelType>
    void DoRecalculateMVs(const FramePyramidLevel &pSrcFrame, const FramePyramidLevel &pRefFrame,
        int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, bool chroma,
        SearchType st, int stp, int64_t lambda, int pnew,
        int fieldShift, int64_t thSAD, bool smooth, bool meander, bool useSatd);

    template <int nLogPel, typename PixelType>
    void DoSearchMVs(const FramePyramidLevel &pSrcFrame, const FramePyramidLevel &pRefFrame,
        SearchType st, int stp, int64_t lambda, int64_t lsad, int pnew,
        int plevel, VECTOR *globalMVec, int fieldShift,
        int pzero, int pglobal, int64_t badSAD, int badrange, bool meander, bool tryMany, bool chroma) noexcept;

    bool IsVectorOK(int vx, int vy) const noexcept;
    int64_t MotionDistorsion(int vx, int vy) const noexcept;
    VECTOR ClipMV(VECTOR v) const noexcept;
    void FetchPredictors(int blkidx, int blkx, int blky, int blkScanDir, VECTOR predictors[4]) noexcept;
    void InitMotionEstimationFields(bool useSatd, bool chroma, int bytesPerSample);
    void EstimateGlobalMVDoubledFallback(VECTOR &globalMVec) const noexcept;
public:
    void SearchMVs(const FramePyramidLevel &pSrcFrame, const FramePyramidLevel &pRefFrame,
        SearchType st, int stp, int64_t lambda, int64_t lsad, int pnew,
        int plevel, VECTOR *globalMVec, int fieldShift, bool useSatd,
        int pzero, int pglobal, int64_t badSAD, int badrange, bool meander, bool tryMany, bool chroma, int bytesPerSample);

    void RecalculateMVs(const FramePyramidLevel &pSrcFrame, const FramePyramidLevel &pRefFrame,
        int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, bool chroma,
        SearchType st, int stp, int64_t lambda, int pnew,
        int fieldShift, int64_t thSAD, bool useSatd, bool smooth, bool meander, int bytesPerSample);

    void EstimateGlobalMVDoubled(VECTOR &globalMVec) const noexcept;
    void InterpolatePredictorsFromParent(const MotionBlockLevel &parentLevel) noexcept;

    bool IsSceneChange(int64_t nTh1, float nTh2) const noexcept;

    void Initialize(int _nBlkX, int _nBlkY, int _nBlkSizeX, int _nBlkSizeY, int _nPel, int _nLevel, bool smallestPlane, bool chroma, int _nOverlapX, int _nOverlapY, int _xRatioUV, int _yRatioUV, int bitsPerSample) noexcept;

    ~MotionBlockLevel();
};

// The parameters a vector clip was made with. Consumers size their kernels and buffers, scale their SAD thresholds
// (block size and chroma) and pick their reference frames (delta) from frame 0 when they are created, and check every
// frame that carries vectors against it: a spliced clip can mix different analyses. MotionBlockPyramid::IsCompatible
// compares all but the delta, which the clips passed to one filter together differ in by design.
struct AnalysisGeometry {
    int width = 0, height = 0, realWidth = 0, realHeight = 0, hPad = 0, vPad = 0, pel = 0, bitsPerSample = 0;
    int blkSizeX = 0, blkSizeY = 0, overlapX = 0, overlapY = 0, blkX = 0, blkY = 0;
    bool chroma = false;
    int xRatioUV = 0, yRatioUV = 0; // the subsampling chroma was analysed at, 0 for luma-only vectors
    int deltaFrame = 0; // the reference's offset, positive for backward vectors
    bool operator==(const AnalysisGeometry &) const = default;
};

class MotionBlockPyramid {
public:
    // Note that the implementation doesn't quite handle multiple search/recalculate operations
    // on the same object, this may be improved in the future if it's actually deemed to be useful but for
    // now create a new object every time

    enum class State {
        Invalid,
        MetadataOnly,
        ReadyForSearch,
        ReadyForRecalculate,
        AnalysisDone
    };

    // exported
    int nBlkSizeX;
    int nBlkSizeY;
    int nLevelCount;
    int nOverlapX;
    int nOverlapY;
    int xRatioUV;
    int yRatioUV;
    int nPel;
    int nDeltaFrame;
    bool chroma;
    int nRealWidth;
    int nRealHeight;
    int nWidth;
    int nHeight;
    int nBlkX;
    int nBlkY;
    int nHPadding;
    int nVPadding;

    int bitsPerSample;
private:
    State state = State::Invalid;
    std::vector<MotionBlockLevel> pyramidLevels;
    const VSFrame *sourceFrame = nullptr;
    const VSAPI *vsapi = nullptr;
    void LoadFrameData(const VSFrame *srcFrame, bool loadVectors, const std::string &prefix, const VSAPI *vsapi);
    void ValidateVectors() const;
public:
    // The FramePyramid in src is only used as a template for the internal data structures, the actual motion estimation is performed on the frames passed to SearchMVs
    MotionBlockPyramid(const FramePyramid &src, int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, int nLevels, bool chroma, int deltaFrame);

    // de-serialization from a frame, can choose to omit some levels by setting maxLevel, if -1 all levels are loaded, 0 means only metadata and no vectors are loaded,
    // positive numbers mean that many levels are loaded. When loading from clips passing 1 is usually enough.
    // Object can be in an invalid or limited state after this constructor
    MotionBlockPyramid(const VSFrame *src, bool loadVectors, const std::string &prefix, const VSAPI *vsapi);
    MotionBlockPyramid(const VSFrame *src, bool loadVectors, const std::string &prefix, const VSAPI *vsapi, const AnalysisGeometry &expected);

    // Constructor to load metadata and do nothing else
    MotionBlockPyramid(VSNode *node, const std::string &prefix, const VSAPI *vsapi);

    ~MotionBlockPyramid();

    void ExportFrameData(VSFrame *dst, const std::string &prefix, const VSAPI *vsapi) const noexcept; // serialization to a frame, oneLevel means that only the finest level is exported, otherwise all levels are exported as separate properties

    void SearchMVs(const FramePyramid &pSrcGOF, const FramePyramid &pRefGOF,
        SearchType searchType, int nSearchParam, int nPelSearch, int64_t nLambda,
        int64_t lsad, int pnew, int plevel, bool global, int fieldShift, bool useSatd,
        int pzero, int pglobal, int64_t badSAD, int badrange, bool meander, TryManyLevels tryMany,
        bool chroma);

    // FIXME, currently you can't SearchMVs and then RecalculateMVs on the same object and instead it needs to be exported and imported into a new one
    // this should probably be improved
    void RecalculateMVs(const FramePyramid &pSrcGOF, const FramePyramid &pRefGOF,
        int nBlkSizeX, int nBlkSizeY, int nOverlapX, int nOverlapY, bool chroma,
        SearchType searchType, int nSearchParam, int64_t nLambda, int pnew,
        int fieldShift, int64_t thSAD, bool useSatd, bool smooth, bool meander, int deltaFrame);

    [[nodiscard]] bool IsUsable(int64_t thscd1, float thscd2) const noexcept;
    [[nodiscard]] BlockData GetBlock(int nBlk) const noexcept;
    [[nodiscard]] double GetThSCDScaleFactor(int bitsPerSample) const;
    void ScaleThSCD(int64_t &thscd1, float &thscd2, int bitsPerSample) const;
    [[nodiscard]] State GetState() const noexcept;
    [[nodiscard]] bool HasMotionVectors() const noexcept;

    [[nodiscard]] AnalysisGeometry Geometry() const noexcept;

    [[nodiscard]] bool IsCompatible(const MotionBlockPyramid &other) const noexcept;
    [[nodiscard]] bool IsCompatibleWithAnalysis(const FramePyramid &other) const noexcept;
    [[nodiscard]] bool IsCompatibleForRecalc(const FramePyramid &other) const noexcept;

    template<typename PixelType>
    [[nodiscard]] std::unique_ptr<BlockMask<PixelType>> MakeVectorLengthMask(float normFactor, float fGamma) const noexcept;
    template<typename PixelType>
    [[nodiscard]] std::unique_ptr<BlockMask<PixelType>> MakeSADMask(float dSADNormFactor, float fGamma, int time256) const noexcept;
    template<typename PixelType>
    [[nodiscard]] std::unique_ptr<BlockMask<PixelType>> MakeVectorOcclusionMask(float dMaskNormDivider, float fGamma, int time256, bool force8bitRange) const noexcept;

    [[nodiscard]] std::unique_ptr<SmallVectorMasks> MakeSmallVectorMasks(int fieldOffset = 0) const noexcept;
};
