#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include <VapourSynth4.h>

#include "Common.h"
#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"


struct AnalyseData {
    VSNode *node = nullptr;

    const VSVideoInfo *vi = nullptr;

    int nBlkSizeX;
    int nBlkSizeY;
    int nOverlapX;
    int nOverlapY;

    int deltaFrame;

    int64_t nLambda;

    SearchType searchType;

    int nSearchParam; 
    int nPelSearch; 

    int64_t lsad; 
    int pnew;   
    int plevel;    
    bool global;    
    int pglobal;  
    int pzero;       
    int64_t badSAD; 
    int badrange;   
    bool meander;   
    TryManyLevels tryMany;  
    bool useSatd;

    int levels;
    bool chroma;

    bool fields;
    bool tff;
    bool tff_exists;

    std::string prefix;
    SuperGeometry superGeometry;

    const VSAPI *vsapi;

    AnalyseData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~AnalyseData() {
        vsapi->freeNode(node);
    }
};

static const VSFrame *VS_CC analyseGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    AnalyseData *d = reinterpret_cast<AnalyseData *>(instanceData);

    int nref = n + d->deltaFrame;

    if (activationReason == arInitial) {
        if (nref >= 0 && nref < d->vi->numFrames) {
            vsapi->requestFrameFilter(std::min(n, nref), d->node, frameCtx);
            vsapi->requestFrameFilter(std::max(n, nref), d->node, frameCtx);
        } else {
            // too close to beginning/end of clip
            vsapi->requestFrameFilter(n, d->node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        try {
            const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
            FramePyramid srcFramePyramid(src, -1, d->prefix, vsapi, d->superGeometry);

            bool src_top_field = GetTopField(src, n, d->tff_exists, d->tff, d->fields, vsapi);

            MotionBlockPyramid vectorFields(srcFramePyramid, d->nBlkSizeX, d->nBlkSizeY, d->nOverlapX, d->nOverlapY, d->levels, d->chroma, d->deltaFrame);

            if (nref >= 0 && nref < d->vi->numFrames) {
                const VSFrame *ref = vsapi->getFrameFilter(nref, d->node, frameCtx);
                FramePyramid refFramePyramid(ref, -1, d->prefix, vsapi, d->superGeometry);

                bool ref_top_field = GetTopField(ref, nref, d->tff_exists, d->tff, d->fields, vsapi);

                int fieldShift = 0;
                if (d->fields && srcFramePyramid.nPel > 1 && (d->deltaFrame % 2))
                    fieldShift = ComputeFieldShift(src_top_field, ref_top_field, srcFramePyramid.nPel);

                vectorFields.SearchMVs(srcFramePyramid, refFramePyramid, d->searchType, d->nSearchParam, d->nPelSearch, d->nLambda, d->lsad, d->pnew, d->plevel, d->global, fieldShift, d->useSatd, d->pzero, d->pglobal, d->badSAD, d->badrange, d->meander, d->tryMany, d->chroma);
            }

            VSFrame *dst = vsapi->copyFrame(src, core);
            vectorFields.ExportFrameData(dst, d->prefix, vsapi);

            return dst;

        } catch (const std::exception &e) {
            // Note that exceptions can only happen before SearchMVs MMX code
            vsapi->setFilterError(("Analyse: " + std::string(e.what())).c_str(), frameCtx);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC analyseCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<AnalyseData> d = std::make_unique<AnalyseData>(vsapi);
    int err;

    try {
        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        if (prefix)
            d->prefix = prefix;
        else
            d->prefix = DEFAULT_MVUTENSILS_PREFIX;

        d->node = vsapi->mapGetNode(in, "super", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->node);

        FramePyramid super(d->node, d->prefix, vsapi);
        d->superGeometry = super.Geometry();

        GetHVPairArgument(d->nBlkSizeX, d->nBlkSizeY, "blksize", super.nBlkSizeX, super.nBlkSizeY, in, vsapi);
        GetHVPairArgument(d->nOverlapX, d->nOverlapY, "overlap", super.nOverlapX, super.nOverlapY, in, vsapi);

        d->useSatd = !!vsapi->mapGetInt(in, "satd", 0, &err);

        CheckBlkSize(d->nBlkSizeX, d->nBlkSizeY, d->nOverlapX, d->nOverlapY, d->vi->format.subSamplingW, d->vi->format.subSamplingH, d->useSatd);

        d->levels = vsapi->mapGetIntSaturated(in, "levels", 0, &err);

        d->searchType = static_cast<SearchType>(vsapi->mapGetIntSaturated(in, "search", 0, &err));
        if (err)
            d->searchType = SearchType::Hex2;

        int searchparam = vsapi->mapGetIntSaturated(in, "searchparam", 0, &err);
        if (err)
            searchparam = 2;

        d->nPelSearch = vsapi->mapGetIntSaturated(in, "pelsearch", 0, &err);
        if (err)
            d->nPelSearch = super.nPel;

        if (d->nPelSearch <= 0)
            throw std::runtime_error("pelsearch must be positive");

        d->chroma = !!vsapi->mapGetInt(in, "chroma", 0, &err);
        if (err)
            d->chroma = true;

        d->deltaFrame = vsapi->mapGetIntSaturated(in, "delta", 0, &err);
        if (err)
            d->deltaFrame = 1;

        d->nLambda = vsapi->mapGetIntSaturated(in, "mvlambda", 0, &err);
        if (err)
            d->nLambda = 1000;
        if (d->nLambda < 0)
            throw std::runtime_error("mvlambda must be non-negative");

        // Read saturated (like lsad/badSAD), so the later blockArea/64 and pixelMax/255 scalings can never
        // overflow int64. Multiply before dividing so small blocks keep a non-zero lambda.
        d->nLambda = d->nLambda * (d->nBlkSizeX * d->nBlkSizeY) / 64;

        d->lsad = vsapi->mapGetIntSaturated(in, "lsad", 0, &err);
        if (err)
            d->lsad = 400; // truemotion was 1200 here

        d->plevel = vsapi->mapGetIntSaturated(in, "plevel", 0, &err);
        if (err)
            d->plevel = 1;

        d->global = !!vsapi->mapGetInt(in, "globalmv", 0, &err);
        if (err)
            d->global = true;

        d->pnew = vsapi->mapGetIntSaturated(in, "pnew", 0, &err);
        if (err)
            d->pnew = 25;

        d->pzero = vsapi->mapGetIntSaturated(in, "pzero", 0, &err);
        if (err)
            d->pzero = d->pnew;

        d->pglobal = vsapi->mapGetIntSaturated(in, "pglobal", 0, &err);

        d->badSAD = vsapi->mapGetIntSaturated(in, "badsad", 0, &err);
        if (err)
            d->badSAD = 10000;

        d->badrange = vsapi->mapGetIntSaturated(in, "badrange", 0, &err);
        if (err)
            d->badrange = 24;

        d->meander = !!vsapi->mapGetInt(in, "meander", 0, &err);
        if (err)
            d->meander = true;

        d->tryMany = static_cast<TryManyLevels>(vsapi->mapGetIntSaturated(in, "trymany", 0, &err));
        if (d->tryMany != TryManyLevels::None && d->tryMany != TryManyLevels::All && d->tryMany != TryManyLevels::AllExceptFinest)
            throw std::runtime_error("trymany must be between 0 and 2");

        d->fields = !!vsapi->mapGetInt(in, "fields", 0, &err);

        d->tff = !!vsapi->mapGetInt(in, "tff", 0, &err);
        d->tff_exists = !err;

        if (d->searchType != SearchType::Logarithmic && d->searchType != SearchType::Exhaustive && d->searchType != SearchType::Hex2 && d->searchType != SearchType::UnevenMultiHexagon && d->searchType != SearchType::Horizontal && d->searchType != SearchType::Vertical)
            throw std::runtime_error("search must be between 0 and 5");

        if (d->plevel < 0 || d->plevel > 2)
            throw std::runtime_error("plevel must be between 0 and 2");

        if (d->pnew < 0 || d->pnew > 256)
            throw std::runtime_error("pnew must be between 0 and 256");

        if (d->pzero < 0 || d->pzero > 256)
            throw std::runtime_error("pzero must be between 0 and 256");

        if (d->pglobal < 0 || d->pglobal > 256)
            throw std::runtime_error("pglobal must be between 0 and 256");

        d->nSearchParam = std::max(searchparam, 1);

        if (d->vi->format.colorFamily == cfGray)
            d->chroma = false;

        int pixelMax = (1 << std::min(16, d->vi->format.bitsPerSample)) - 1; // float SAD uses the 16-bit scale
        d->lsad = (int64_t)((double)d->lsad * pixelMax / 255.0 + 0.5);
        d->badSAD = (int64_t)((double)d->badSAD * pixelMax / 255.0 + 0.5);
        d->nLambda = (int64_t)((double)d->nLambda * pixelMax / 255.0 + 0.5);

        d->lsad = (int64_t)d->lsad * (d->nBlkSizeX * d->nBlkSizeY) / 64;
        d->badSAD = d->badSAD * (d->nBlkSizeX * d->nBlkSizeY) / 64;

        if (d->deltaFrame == 0)
            throw std::runtime_error("delta can't be 0");

        MotionBlockPyramid DryRun(super, d->nBlkSizeX, d->nBlkSizeY, d->nOverlapX, d->nOverlapY, d->levels, d->chroma, d->deltaFrame);

    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("Analyse: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[1] = { 
        {d->node, rpGeneral}
    };

    vsapi->createVideoFilter(out, "Analyse", d->vi, analyseGetFrame, filterFree<AnalyseData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}


static void VS_CC analyseManyCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    int err;

    int radius = vsapi->mapGetIntSaturated(in, "radius", 0, &err);
    if (err)
        radius = 1;
    if (radius < 1) {
        vsapi->mapSetError(out, "AnalyseMany: radius must be a positive number");
        return;
    }

    int delta = vsapi->mapGetIntSaturated(in, "delta", 0, &err);
    if (err)
        delta = 1;

    if (delta < 1) {
        vsapi->mapSetError(out, "AnalyseMany: delta must be a positive number");
        return;
    }

    VSMap *args = vsapi->createMap();
    vsapi->copyMap(in, args);
    vsapi->mapDeleteKey(args, "radius");

    VSPlugin *thisPlugin = vsapi->getPluginByID("com.vapoursynth.mvutensils", core);

    for (int r = 1; r <= radius; ++r) {
        auto InvokeAnalyse = [&](int delta) {
            vsapi->mapSetInt(args, "delta", delta, maReplace);
            VSMap *ret = vsapi->invoke(thisPlugin, "Analyse", args);
            if (vsapi->mapGetError(ret)) {
                vsapi->mapSetError(out, ("AnalyseMany: " + std::string(vsapi->mapGetError(ret))).c_str());
                vsapi->freeMap(ret);
                return false;
            } else {
                vsapi->mapConsumeNode(out, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maAppend);
                vsapi->freeMap(ret);
                return true;
            }
        };

        if (!InvokeAnalyse(r * delta))
            break;
        if (!InvokeAnalyse(-r * delta))
            break;
    }

    vsapi->freeMap(args);
}

void analyseRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("Analyse",
                 "super:vnode;"
                 "blksize:int[]:opt;"
                 "levels:int:opt;"
                 "search:int:opt;"
                 "searchparam:int:opt;"
                 "pelsearch:int:opt;"
                 "mvlambda:int:opt;"
                 "chroma:int:opt;"
                 "delta:int:opt;"
                 "lsad:int:opt;"
                 "plevel:int:opt;"
                 "globalmv:int:opt;"
                 "pnew:int:opt;"
                 "pzero:int:opt;"
                 "pglobal:int:opt;"
                 "overlap:int[]:opt;"
                 "badsad:int:opt;"
                 "badrange:int:opt;"
                 "meander:int:opt;"
                 "trymany:int:opt;"
                 "fields:int:opt;"
                 "tff:int:opt;"
                 "satd:int:opt;"
                 "prefix:data:opt;",
                 "clip:vnode;",
                 analyseCreate, nullptr, plugin);
    vspapi->registerFunction("AnalyseMany",
                 "super:vnode;"
                 "blksize:int[]:opt;"
                 "levels:int:opt;"
                 "search:int:opt;"
                 "searchparam:int:opt;"
                 "pelsearch:int:opt;"
                 "mvlambda:int:opt;"
                 "chroma:int:opt;"
                 "delta:int:opt;"
                 "lsad:int:opt;"
                 "plevel:int:opt;"
                 "globalmv:int:opt;"
                 "pnew:int:opt;"
                 "pzero:int:opt;"
                 "pglobal:int:opt;"
                 "overlap:int[]:opt;"
                 "badsad:int:opt;"
                 "badrange:int:opt;"
                 "meander:int:opt;"
                 "trymany:int:opt;"
                 "fields:int:opt;"
                 "tff:int:opt;"
                 "satd:int:opt;"
                 "radius:int:opt;"
                 "prefix:data:opt;",
                 "clip:vnode[];",
                 analyseManyCreate, nullptr, plugin);
}
