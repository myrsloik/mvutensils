#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "Common.h"
#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"
#include "DepanShared.h"

constexpr char prop_DepanAnalyse_info[] = "DepanAnalyse_info";


struct DepanAnalyseData {
    VSNode *clip = nullptr;
    VSNode *vectors = nullptr;
    VSNode *mask = nullptr;
    bool zoom = false;
    bool rot = false;
    float pixaspect = 0.0f;
    float error = 0.0f;
    bool info = false;
    float wrong = 0.0f;
    float zerow = 0.0f;
    int64_t thscd1 = 0;
    float thscd2 = 0;
    AnalysisGeometry geometry;
    bool fields = false;
    bool tff = false;
    bool tff_exists = false;

    int deltaFrame = 0;

    std::string prefix;

    const VSVideoInfo *vi = nullptr;

    const VSAPI *vsapi = nullptr;

    DepanAnalyseData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~DepanAnalyseData() {
        vsapi->freeNode(clip);
        vsapi->freeNode(vectors);
        vsapi->freeNode(mask);
    }
};


static void TrasformUpdate(transform *tr, const float *blockDx, const float *blockDy, const int *blockX, const int *blockY, const float *blockWeight, int nBlkX, int nBlkY, float safety, bool ifZoom1, bool ifRot1, float *error1, float pixaspect) {
    transform trderiv;
    int n = nBlkX * nBlkY;
    trderiv.dxc = 0;
    trderiv.dxx = 0;
    trderiv.dxy = 0;
    trderiv.dyc = 0;
    trderiv.dyx = 0;
    trderiv.dyy = 0;
    float norm = 0.1f;
    float x2 = 0.1f;
    float y2 = 0.1f;
    float error2 = 0.1f;
    for (int i = 0; i < n; i++) {
        float bw = blockWeight[i];
        float xdif = (tr->dxc + tr->dxx * blockX[i] + tr->dxy * blockY[i] - blockX[i] - blockDx[i]);
        trderiv.dxc += 2 * xdif * bw;
        if (ifZoom1)
            trderiv.dxx += 2 * blockX[i] * xdif * bw;
        if (ifRot1)
            trderiv.dxy += 2 * blockY[i] * xdif * bw;
        float ydif = (tr->dyc + tr->dyx * blockX[i] + tr->dyy * blockY[i] - blockY[i] - blockDy[i]);
        trderiv.dyc += 2 * ydif * bw;
        if (ifRot1)
            trderiv.dyx += 2 * blockX[i] * ydif * bw;
        if (ifZoom1)
            trderiv.dyy += 2 * blockY[i] * ydif * bw;
        norm += bw;
        x2 += blockX[i] * blockX[i] * bw;
        y2 += blockY[i] * blockY[i] * bw;
        error2 += (xdif * xdif + ydif * ydif) * bw;
    }
    trderiv.dxc /= norm * 2;
    trderiv.dxx /= x2 * 2 * 1.5f; // with additional safety factors
    trderiv.dxy /= y2 * 2 * 3;
    trderiv.dyc /= norm * 2;
    trderiv.dyx /= x2 * 2 * 3;
    trderiv.dyy /= y2 * 2 * 1.5f;

    error2 /= norm;
    *error1 = sqrtf(error2);

    tr->dxc -= safety * trderiv.dxc;
    if (ifZoom1)
        tr->dxx -= safety * 0.5f * (trderiv.dxx + trderiv.dyy);

    tr->dxy -= safety * 0.5f * (trderiv.dxy - trderiv.dyx / (pixaspect * pixaspect));
    tr->dyc -= safety * trderiv.dyc;
    //    tr->dyx -= safety*trderiv.dyx;
    //    tr->dyy -= safety*trderiv.dyy;
    if (ifZoom1)
        tr->dyy = tr->dxx;
    //    float pixaspect=1; // was for test and forgot remove?! disabled in v1.2.5
    tr->dyx = -pixaspect * pixaspect * tr->dxy;
}


static void RejectBadBlocks(const transform *tr, const float *blockDx, const float *blockDy, const int64_t *blockSAD, const int *blockX, const int *blockY, float *blockWeight, int nBlkX, int nBlkY, float wrongDif, float globalDif, int64_t thSCD1, float zeroWeight, const float *blockWeightMask, int ignoredBorder) {
    for (int j = 0; j < nBlkY; j++) {
        for (int i = 0; i < nBlkX; i++) {
            int n = j * nBlkX + i;
            if (i < ignoredBorder || i >= nBlkX - ignoredBorder || j < ignoredBorder || j >= nBlkY - ignoredBorder) {
                blockWeight[n] = 0; // disable  blocks near frame borders
            } else if (blockSAD[n] > thSCD1) {
                blockWeight[n] = 0; // disable bad block with big SAD
            } else if (i > 0 && i < (nBlkX - 1) && j > 0 && j < (nBlkY - 1) && (fabsf((blockDx[n - 1 - nBlkX] + blockDx[n - nBlkX] + blockDx[n + 1 - nBlkX] +
                                                          blockDx[n - 1] + blockDx[n + 1] +
                                                          blockDx[n - 1 + nBlkX] + blockDx[n + nBlkX] + blockDx[n + 1 + nBlkX]) /
                                                             8 -
                                                         blockDx[n]) > wrongDif)) {
                blockWeight[n] = 0; // disable blocks very different from neighbours
            } else if (i > 0 && i < (nBlkX - 1) && j > 0 && j < (nBlkY - 1) && (fabsf((blockDy[n - 1 - nBlkX] + blockDy[n - nBlkX] + blockDy[n + 1 - nBlkX] +
                                                          blockDy[n - 1] + blockDy[n + 1] +
                                                          blockDy[n - 1 + nBlkX] + blockDy[n + nBlkX] + blockDy[n + 1 + nBlkX]) /
                                                             8 -
                                                         blockDy[n]) > wrongDif)) {
                blockWeight[n] = 0; // disable blocks very different from neighbours
            } else if (fabsf(tr->dxc + tr->dxx * blockX[n] + tr->dxy * blockY[n] - blockX[n] - blockDx[n]) > globalDif) {
                blockWeight[n] = 0; // disable blocks very different from global
            } else if (fabsf(tr->dyc + tr->dyx * blockX[n] + tr->dyy * blockY[n] - blockY[n] - blockDy[n]) > globalDif) {
                blockWeight[n] = 0; // disable blocks very different from global
            } else if (blockDx[n] == 0.0f && blockDy[n] == 0.0f) {
                blockWeight[n] = zeroWeight * blockWeightMask[n];
            } else {
                blockWeight[n] = blockWeightMask[n]; // good block
            }
        }
    }
}


static const VSFrame *VS_CC depanAnalyseGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    DepanAnalyseData *d = reinterpret_cast<DepanAnalyseData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter((d->deltaFrame > 0) ? std::max(0, n - 1) : n, d->vectors, frameCtx);
        vsapi->requestFrameFilter(n, d->clip, frameCtx);

        if (d->mask)
            vsapi->requestFrameFilter(n, d->mask, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->clip, frameCtx);
        VSFrame *dst = vsapi->copyFrame(src, core);
        vsapi->freeFrame(src);

        const VSFrame *mask = nullptr;

        try {
            VSMap *dst_props = vsapi->getFramePropertiesRW(dst);

            const int nFields = d->fields ? 2 : 1;

            const uint8_t *maskp = nullptr;
            ptrdiff_t mask_pitch = 0;
            if (d->mask) {
                mask = vsapi->getFrameFilter(n, d->mask, frameCtx);
                maskp = vsapi->getReadPtr(mask, 0);
                mask_pitch = vsapi->getStride(mask, 0);
            }

            const bool backward = d->deltaFrame > 0;
            int nframemv = backward ? std::max(0, n - 1) : n; // set prev frame number as data frame if backward

            const VSFrame *mvn = vsapi->getFrameFilter(nframemv, d->vectors, frameCtx);
            MotionBlockPyramid vectors(mvn, 1, d->prefix, vsapi, d->geometry);

            const size_t num_blocks = (size_t)vectors.nBlkX * (size_t)vectors.nBlkY;

            std::vector<float> blockDx(num_blocks); // dx vector
            std::vector<float> blockDy(num_blocks); // dy
            std::vector<int64_t> blockSAD(num_blocks);
            std::vector<int> blockX(num_blocks); // block x position
            std::vector<int> blockY(num_blocks);
            std::vector<float> blockWeight(num_blocks);
            std::vector<float> blockWeightMask(num_blocks);

            transform tr;

            float errorcur = d->error * 2;
            int iter = 0; // start iteration


            if (vectors.IsUsable(d->thscd1, d->thscd2)) {
                const float dPel = 1.0f / vectors.nPel; // subpixel precision value

                for (int j = 0; j < vectors.nBlkY; j++) {
                    for (int i = 0; i < vectors.nBlkX; i++) {
                        int nb = j * vectors.nBlkX + i;
                        const auto fbd = vectors.GetBlock(nb);
                        blockDx[nb] = fbd.vector.x * dPel;
                        blockDy[nb] = fbd.vector.y * dPel;
                        blockSAD[nb] = fbd.vector.sad;
                        blockX[nb] = fbd.x + vectors.nBlkSizeX / 2;
                        blockY[nb] = fbd.y + vectors.nBlkSizeY / 2;
                        if (d->mask && blockX[nb] < d->vi->width && blockY[nb] < d->vi->height)
                            blockWeightMask[nb] = maskp[blockX[nb] + blockY[nb] * mask_pitch];
                        else
                            blockWeightMask[nb] = 1.0f;
                        blockWeight[nb] = blockWeightMask[nb];
                    }
                }

                // begin with translation only
                float safety = 0.3f; // begin with small safety factor
                bool ifRot0 = false;
                bool ifZoom0 = false;
                float globalDif0 = 1000.0f;
                int ignoredBorder = mask ? 0 : 4;


                for (; iter < 5; iter++) {
                    TrasformUpdate(&tr, blockDx.data(), blockDy.data(), blockX.data(), blockY.data(), blockWeight.data(), vectors.nBlkX, vectors.nBlkY, safety, ifZoom0, ifRot0, &errorcur, d->pixaspect / nFields);
                    RejectBadBlocks(&tr, blockDx.data(), blockDy.data(), blockSAD.data(), blockX.data(), blockY.data(), blockWeight.data(), vectors.nBlkX, vectors.nBlkY, d->wrong, globalDif0, d->thscd1, d->zerow, blockWeightMask.data(), ignoredBorder);
                }


                const float errordif = 0.01f;   // error difference to terminate iterations

                for (; iter < 100; iter++) {
                    if (iter < 8)
                        safety = 0.3f; // use for safety
                    else if (iter < 10)
                        safety = 0.6f;
                    else
                        safety = 1.0f;
                    float errorprev = errorcur;
                    TrasformUpdate(&tr, blockDx.data(), blockDy.data(), blockX.data(), blockY.data(), blockWeight.data(), vectors.nBlkX, vectors.nBlkY, safety, d->zoom, d->rot, &errorcur, d->pixaspect / nFields);
                    if (((errorprev - errorcur) < errordif * 0.5f && iter > 9) || errorcur < errordif)
                        break; // check convergence, accuracy increased in v1.2.5
                    float globalDif = errorcur * 2;
                    RejectBadBlocks(&tr, blockDx.data(), blockDy.data(), blockSAD.data(), blockX.data(), blockY.data(), blockWeight.data(), vectors.nBlkX, vectors.nBlkY, d->wrong, globalDif, d->thscd1, d->zerow, blockWeightMask.data(), ignoredBorder);
                }
            }

            // mask data has been fully consumed, release it as early as possible
            vsapi->freeFrame(mask);
            mask = nullptr;

            // we get transform (null if scenechange)

            float xcenter = (float)d->vi->width / 2;
            float ycenter = (float)d->vi->height / 2;

            MotionData m;
            m.badMotion = true; // bad until a valid transform is computed below

            if (errorcur < d->error) { // if not bad result
                m.badMotion = false;
                // convert transform data to ordinary motion format
                if (backward) {
                    transform trinv;
                    inversetransform(&tr, &trinv);
                    transform2motion(&trinv, 0, xcenter, ycenter, d->pixaspect / nFields, &m.dx, &m.dy, &m.rot, &m.zoom);
                } else
                    transform2motion(&tr, 1, xcenter, ycenter, d->pixaspect / nFields, &m.dx, &m.dy, &m.rot, &m.zoom);

                // fieldbased correction
                if (d->fields) {
                    // dst is a copy of the source frame, so it carries the same _Field
                    bool top_field = GetTopField(dst, n, d->tff_exists, d->tff, true, vsapi);

                    float yadd = top_field ? 0.5f : -0.5f;

                    // scale dy for fieldbased frame by factor 2 (for compatibility)
                    yadd = yadd * 2;
                    m.dy += yadd;
                }
            }

            if (d->info) {
#define INFO_SIZE 128
                char info[INFO_SIZE + 1] = { 0 };

                snprintf(info, INFO_SIZE, "fn=%d iter=%d error=%.3f dx=%.2f dy=%.2f rot=%.3f zoom=%.5f bad=%d", n, iter, errorcur, m.dx, m.dy, m.rot, m.zoom, m.badMotion);
#undef INFO_SIZE

                vsapi->mapSetData(dst_props, prop_DepanAnalyse_info, info, -1, dtUtf8, maReplace);
            }

            mapSetMotion(dst_props, m, vsapi);

            return dst;
        } catch (const std::exception &e) {
            vsapi->setFilterError(("DepanAnalyse: " + std::string(e.what())).c_str(), frameCtx);
            vsapi->freeFrame(mask);
            vsapi->freeFrame(dst);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC depanAnalyseCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<DepanAnalyseData> d = std::make_unique<DepanAnalyseData>(vsapi);

    try {
        int err;

        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        if (prefix)
            d->prefix = prefix;
        else
            d->prefix = DEFAULT_MVUTENSILS_PREFIX;

        d->zoom = !!vsapi->mapGetInt(in, "zoom", 0, &err);
        if (err)
            d->zoom = true;

        d->rot = !!vsapi->mapGetInt(in, "rot", 0, &err);
        if (err)
            d->rot = true;

        d->pixaspect = vsapi->mapGetFloatSaturated(in, "pixaspect", 0, &err);
        if (err)
            d->pixaspect = 1.0f;

        d->error = vsapi->mapGetFloatSaturated(in, "error", 0, &err);
        if (err)
            d->error = 15.0f;

        d->info = !!vsapi->mapGetInt(in, "info", 0, &err);

        d->wrong = vsapi->mapGetFloatSaturated(in, "wrong", 0, &err);
        if (err)
            d->wrong = 10.0f;

        d->zerow = vsapi->mapGetFloatSaturated(in, "zerow", 0, &err);
        if (err)
            d->zerow = 0.05f;

        d->thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
        if (err)
            d->thscd1 = MV_DEFAULT_SCD1;

        d->thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
        if (err)
            d->thscd2 = MV_DEFAULT_SCD2;

        d->fields = !!vsapi->mapGetInt(in, "fields", 0, &err);

        d->tff = !!vsapi->mapGetInt(in, "tff", 0, &err);
        d->tff_exists = !err;

        if (!std::isfinite(d->pixaspect) || d->pixaspect <= 0.0f)
            throw std::runtime_error("pixaspect must be a finite positive value");

        if (!std::isfinite(d->error) || !std::isfinite(d->wrong) || !std::isfinite(d->zerow))
            throw std::runtime_error("error, wrong and zerow must be finite values");

        d->clip = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->clip);

        if (!vsh::isConstantVideoFormat(d->vi))
            throw std::runtime_error("clip must have constant format and dimensions");

        d->vectors = vsapi->mapGetNode(in, "vectors", 0, nullptr);

        MotionBlockPyramid vectors(d->vectors, d->prefix, vsapi);

        // FIXME, check vector clip compatibility

        CheckClipLength(d->vectors, "vectors", d->vi->numFrames, "clip", vsapi);

        d->mask = vsapi->mapGetNode(in, "mask", 0, &err);

        if (d->mask) {
            const VSVideoInfo *maskvi = vsapi->getVideoInfo(d->mask);

            CheckClipLength(d->mask, "mask", d->vi->numFrames, "clip", vsapi);


            if (!vsh::isConstantVideoFormat(maskvi) ||
                maskvi->width != d->vi->width ||
                maskvi->height != d->vi->height ||
                maskvi->format.bitsPerSample > 8) {
                throw std::runtime_error("mask must have constant format, the same dimensions as clip, and no more than 8 bits per sample");
            }
        }

        vectors.ScaleThSCD(d->thscd1, d->thscd2, vectors.bitsPerSample);
        d->geometry = vectors.Geometry();

        if (abs(vectors.nDeltaFrame) != 1)
            throw std::runtime_error("vectors clip must be created with delta=1 or -1");

        d->deltaFrame = vectors.nDeltaFrame;

    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("DepanAnalyse: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[3] = {
        {d->clip, rpStrictSpatial},
        {d->vectors, rpGeneral},
    };
    int numDeps = 2;
    if (d->mask)
        deps[numDeps++] = { d->mask, rpStrictSpatial }; // optional: only a real node may be a dependency

    bool info = d->info;

    vsapi->createVideoFilter(out, "DepanAnalyse", d->vi, depanAnalyseGetFrame, filterFree<DepanAnalyseData>, fmParallel, deps, numDeps, d.get(), core);
    d.release();

    if (vsapi->mapGetError(out))
        return;

    if (info) {
        if (!invokeFrameProps(prop_DepanAnalyse_info, out, core, vsapi)) {
            vsapi->mapSetError(out, std::string("DepanAnalyse: failed to invoke text.FrameProps: ").append(vsapi->mapGetError(out)).c_str());
            return;
        }
    }
}


void depanAnalyseRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->registerFunction("DepanAnalyse",
                 "clip:vnode;"
                 "vectors:vnode;"
                 "mask:vnode:opt;"
                 "zoom:int:opt;"
                 "rot:int:opt;"
                 "pixaspect:float:opt;"
                 "error:float:opt;"
                 "info:int:opt;"
                 "wrong:float:opt;"
                 "zerow:float:opt;"
                 "thscd1:int:opt;"
                 "thscd2:float:opt;"
                 "fields:int:opt;"
                 "tff:int:opt;",
                 "clip:vnode;",
                 depanAnalyseCreate, nullptr, plugin);
}
