#include <cstdint>
#include <memory>
#include <optional>
#include <VapourSynth4.h>

#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"
#include "Common.h"

struct RecalculateData {
    VSNode *super = nullptr;
    VSNode *vectors = nullptr;

    const VSVideoInfo *vi;

    int deltaFrame;

    int64_t nLambda;

    int nBlkSizeX;
    int nBlkSizeY;
    int nOverlapX;
    int nOverlapY;

    SearchType searchType;

    int pnew;  
    bool meander;

    bool useSatd;

    int searchparam;
    bool chroma;
    bool smooth;
    int64_t thSAD;

    std::string prefix;
    AnalysisGeometry geometry;
    SuperGeometry superGeometry;

    const VSAPI *vsapi;

    RecalculateData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~RecalculateData() {
        vsapi->freeNode(super);
        vsapi->freeNode(vectors);
    }
};

static const VSFrame *VS_CC recalculateGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    RecalculateData *d = reinterpret_cast<RecalculateData *>(instanceData);

    // As in Analyse, a frame whose reference lies outside the super clip gets no vectors, also when the vector clip, which
    // may be longer, has some for it: they belong to a reference this super doesn't have
    const int nref = n + d->deltaFrame;
    const bool hasRef = nref >= 0 && nref < d->vi->numFrames;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->vectors, frameCtx);
        if (hasRef) {
            vsapi->requestFrameFilter(std::min(n, nref), d->super, frameCtx);
            vsapi->requestFrameFilter(std::max(n, nref), d->super, frameCtx);
        } else {
            vsapi->requestFrameFilter(n, d->super, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        try {
            const VSFrame *src = vsapi->getFrameFilter(n, d->super, frameCtx);
            FramePyramid pSrcGOF(src, 1, d->prefix, vsapi, d->superGeometry);

            MotionBlockPyramid fgop(vsapi->getFrameFilter(n, d->vectors, frameCtx), hasRef, d->prefix, vsapi, d->geometry);

            std::optional<FramePyramid> pRefGOF;
            if (hasRef)
                pRefGOF.emplace(vsapi->getFrameFilter(nref, d->super, frameCtx), 1, d->prefix, vsapi, d->superGeometry);

            // without a reference the old vectors weren't loaded, and recalculating only the metadata never reads it
            fgop.RecalculateMVs(pSrcGOF, pRefGOF ? *pRefGOF : pSrcGOF, d->nBlkSizeX, d->nBlkSizeY, d->nOverlapX, d->nOverlapY, d->chroma,
                d->searchType, d->searchparam, d->nLambda, d->pnew, d->thSAD, d->useSatd, d->smooth, d->meander, d->deltaFrame);

            VSFrame *dst = vsapi->copyFrame(src, core);
            fgop.ExportFrameData(dst, d->prefix, vsapi);
            return dst;
        } catch (const std::exception &e) {
            vsapi->setFilterError(("Recalculate: " + std::string(e.what())).c_str(), frameCtx);
            return nullptr;
        }
    }

    return nullptr;
}

static void recalculateCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<RecalculateData> d = std::make_unique<RecalculateData>(vsapi);
    int err;

    try {
        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        if (prefix)
            d->prefix = prefix;
        else
            d->prefix = DEFAULT_MVUTENSILS_PREFIX;

        d->super = vsapi->mapGetNode(in, "super", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->super);
        FramePyramid super(d->super, d->prefix, vsapi);
        d->superGeometry = super.Geometry();

        GetHVPairArgument(d->nBlkSizeX, d->nBlkSizeY, "blksize", super.nBlkSizeX, super.nBlkSizeY, in, vsapi);
        GetHVPairArgument(d->nOverlapX, d->nOverlapY, "overlap", super.nOverlapX, super.nOverlapY, in, vsapi);

        d->useSatd = !!vsapi->mapGetInt(in, "satd", 0, &err);

        CheckBlkSize(d->nBlkSizeX, d->nBlkSizeY, d->nOverlapX, d->nOverlapY, d->vi->format.subSamplingW, d->vi->format.subSamplingH, d->useSatd);

        d->thSAD = vsapi->mapGetIntSaturated(in, "thsad", 0, &err); // saturated so the pixelMax/blockArea scaling below can't overflow int64
        if (err)
            d->thSAD = 200;

        d->smooth = !!vsapi->mapGetIntSaturated(in, "smooth", 0, &err);
        if (err)
            d->smooth = true;

        d->searchType = static_cast<SearchType>(vsapi->mapGetIntSaturated(in, "search", 0, &err));
        if (err)
            d->searchType = SearchType::Hex2;

        d->searchparam = std::max(vsapi->mapGetIntSaturated(in, "searchparam", 0, &err), 1);
        if (err)
            d->searchparam = 2;

        d->chroma = !!vsapi->mapGetInt(in, "chroma", 0, &err);
        if (err)
            d->chroma = true;

        if (d->vi->format.colorFamily == cfGray)
            d->chroma = false;

        d->nLambda = vsapi->mapGetIntSaturated(in, "mvlambda", 0, &err);
        if (err)
            d->nLambda = 1000;
        if (d->nLambda < 0)
            throw std::runtime_error("mvlambda must be non-negative");

        // Read saturated, so the later blockArea/64 and pixelMax/255 scalings can never overflow int64.
        // Multiply before dividing so small blocks keep a non-zero lambda.
        d->nLambda = d->nLambda * (d->nBlkSizeX * d->nBlkSizeY) / 64;

        d->pnew = vsapi->mapGetIntSaturated(in, "pnew", 0, &err);
        if (err)
            d->pnew = 25;

        d->meander = !!vsapi->mapGetInt(in, "meander", 0, &err);
        if (err)
            d->meander = true;

        if (d->searchType != SearchType::Logarithmic && d->searchType != SearchType::Exhaustive && d->searchType != SearchType::Hex2 && d->searchType != SearchType::UnevenMultiHexagon && d->searchType != SearchType::Horizontal && d->searchType != SearchType::Vertical)
            throw std::runtime_error("search must be between 0 and 5");

        if (d->pnew < 0 || d->pnew > 256)
            throw std::runtime_error("pnew must be between 0 and 256");

        d->vectors = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        CheckClipLength(d->vectors, "vectors", d->vi->numFrames, "super", vsapi); // the output is as long as super

        MotionBlockPyramid vectors(d->vectors, d->prefix, vsapi);

        d->deltaFrame = vectors.nDeltaFrame;
        d->geometry = vectors.Geometry();

        int pixelMax = (1 << std::min(16, d->vi->format.bitsPerSample)) - 1; // float SAD uses the 16-bit scale
        d->thSAD = (int64_t)((double)d->thSAD * pixelMax / 255.0 + 0.5);
        d->nLambda = (int64_t)((double)d->nLambda * pixelMax / 255.0 + 0.5);

        const int referenceBlockSize = 8 * 8;
        d->thSAD = d->thSAD * (d->nBlkSizeX * d->nBlkSizeY) / referenceBlockSize;
        if (d->chroma)
            d->thSAD += d->thSAD / (super.xRatioUV * super.yRatioUV) * 2;

        if (!vectors.IsCompatibleForRecalc(super))
            throw std::runtime_error("wrong source or super clip frame size");

    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("Recalculate: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[2] = { 
        {d->super, rpGeneral},
        {d->vectors, rpStrictSpatial},
    };

    vsapi->createVideoFilter(out, "Recalculate", d->vi, recalculateGetFrame, filterFree<RecalculateData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

static void VS_CC recalculateCreateWrapper(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    int numVectors = vsapi->mapNumElements(in, "vectors");
    if (numVectors == 1) {
        recalculateCreate(in, out, userData, core, vsapi);
    } else {
        VSMap *inArgs = vsapi->createMap();
        vsapi->copyMap(in, inArgs);

        for (int i = 0; i < numVectors; ++i) {
            vsapi->mapConsumeNode(inArgs, "vectors", vsapi->mapGetNode(in, "vectors", i, nullptr), maReplace);
            recalculateCreate(inArgs, out, userData, core, vsapi);
            const char *error = vsapi->mapGetError(out);
            if (error) {
                std::string errMsg = "Recalculate: Error when recalculating vector " + std::to_string(i) + ": " + error;
                vsapi->clearMap(out);
                vsapi->mapSetError(out, errMsg.c_str());
                break;
            }
        }

        vsapi->freeMap(inArgs);
    }
}

void recalculateRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Recalculate",
                 "super:vnode;"
                 "vectors:vnode[];"
                 "thsad:int:opt;"
                 "smooth:int:opt;"
                 "blksize:int[]:opt;"
                 "search:int:opt;"
                 "searchparam:int:opt;"
                 "mvlambda:int:opt;"
                 "chroma:int:opt;"
                 "pnew:int:opt;"
                 "overlap:int[]:opt;"
                 "meander:int:opt;"
                 "satd:int:opt;"
                 "prefix:data:opt;",
                 "clip:vnode[];",
                 recalculateCreateWrapper, nullptr, plugin);
}
