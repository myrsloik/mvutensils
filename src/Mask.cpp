// Create an overlay mask with the motion vectors
// Author: Manao
// Copyright(c)2006 A.G.Balakhnin aka Fizick - YUY2, occlusion
// See legal notice in Copying.txt for more information
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .

#include <cmath>
#include <memory>

#include <VapourSynth4.h>
#include <VSConstants4.h>

#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"
#include "MaskResize.h"
#include "Common.h"

struct MaskData {
    VSNode *node = nullptr;
    VSVideoInfo vi;

    float fGamma;
    intptr_t kind;
    int time256;
    int nSceneChangeValue;
    float fSceneChangeValue;
    int64_t thscd1;
    float thscd2;

    float fMaskNormFactor;

    PlaneResizer maskResizerFull;
    AnalysisGeometry geometry;

    std::string prefix;

    std::string filterName;

    const VSAPI *vsapi;

    MaskData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~MaskData() {
        vsapi->freeNode(node);
    }
};


template<typename PixelType>
static const VSFrame *VS_CC maskGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    MaskData *d =  reinterpret_cast<MaskData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        VSFrame *dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, nullptr, core);

        try {
            MotionBlockPyramid vectors(vsapi->getFrameFilter(n, d->node, frameCtx), 1, d->prefix, vsapi, d->geometry);

            if (vectors.IsUsable(d->thscd1, d->thscd2)) {
                std::unique_ptr<BlockMask<PixelType>> Mask;
                if (d->kind == 0) {
                    Mask = vectors.MakeVectorLengthMask<PixelType>(d->fMaskNormFactor, d->fGamma);
                } else if (d->kind == 1) {
                    Mask = vectors.MakeSADMask<PixelType>(d->fMaskNormFactor, d->fGamma, d->time256);
                } else if (d->kind == 2) {
                    Mask = vectors.MakeVectorOcclusionMask<PixelType>(d->fMaskNormFactor, d->fGamma, d->time256, false);
                }

                d->maskResizerFull.Process(vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0), Mask->mask, Mask->stride);
            } else {
                PixelType *dstPtr = reinterpret_cast<PixelType *>(vsapi->getWritePtr(dst, 0));
                size_t off = (vsapi->getStride(dst, 0) / sizeof(PixelType)) * vsapi->getFrameHeight(dst, 0);
                if constexpr (std::is_floating_point_v<PixelType>)
                    std::fill(dstPtr, dstPtr + off, static_cast<PixelType>(d->fSceneChangeValue)); // mask range [0,1]
                else
                    std::fill(dstPtr, dstPtr + off, static_cast<PixelType>(d->nSceneChangeValue));
            }

            vsapi->mapSetInt(vsapi->getFramePropertiesRW(dst), "_Range", VSC_RANGE_FULL, maAppend);

            return dst;

        } catch (const std::exception &e) {
            vsapi->freeFrame(dst);
            vsapi->setFilterError((d->filterName + ": " + e.what()).c_str(), frameCtx);
        }
    }

    return nullptr;
}

static void VS_CC maskCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<MaskData> d = std::make_unique<MaskData>(vsapi);

    int err;

    d->kind = (intptr_t)userData;
    
    if (d->kind == 0)
        d->filterName = "VectorLengthMask";
    else if (d->kind == 1)
        d->filterName = "SADMask";
    else if (d->kind == 2)
        d->filterName = "OcclusionMask";

    float ml = vsapi->mapGetFloatSaturated(in, "ml", 0, &err);
    if (err)
        ml = 100.0f;

    d->fGamma = vsapi->mapGetFloatSaturated(in, "gamma", 0, &err);
    if (err)
        d->fGamma = 1.0f;

    double time = vsapi->mapGetFloat(in, "time", 0, &err);
    if (err)
        time = 100.0;

    d->fSceneChangeValue = vsapi->mapGetFloatSaturated(in, "scval", 0, &err);

    d->thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
    if (err)
        d->thscd1 = MV_DEFAULT_SCD1;

    d->thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
    if (err)
        d->thscd2 = MV_DEFAULT_SCD2;

    try {

        if (!std::isfinite(d->fGamma) || d->fGamma < 0.0f)
            throw std::runtime_error("gamma must be a finite non-negative value");

        if (!std::isfinite(ml) || ml <= 0.0f)
            throw std::runtime_error("ml must be a finite value greater than 0");

        if (!std::isfinite(time) || time < 0.0 || time > 100.0)
            throw std::runtime_error("time must be between 0.0 and 100.0");

        const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
        if (prefix)
            d->prefix = prefix;
        else
            d->prefix = DEFAULT_MVUTENSILS_PREFIX;

        d->node = vsapi->mapGetNode(in, "vectors", 0, nullptr);
        MotionBlockPyramid vectors(d->node, d->prefix, vsapi);
        d->geometry = vectors.Geometry();

        d->vi = *vsapi->getVideoInfo(d->node);
        d->vi.width = vectors.nRealWidth;
        d->vi.height = vectors.nRealHeight;
        VSSampleType sampleType = vectors.bitsPerSample == 32 ? stFloat : stInteger;
        vsapi->queryVideoFormat(&d->vi.format, cfGray, sampleType, vectors.bitsPerSample, 0, 0, core);

        if (sampleType == stInteger) {
            if (!std::isfinite(d->fSceneChangeValue))
                throw std::runtime_error("scval must be a finite value for integer formats");

            d->nSceneChangeValue = static_cast<int>(d->fSceneChangeValue + 0.5f); // round to nearest integer
            int maxVal = (1 << std::min(16, vectors.bitsPerSample)) - 1;
            if (d->nSceneChangeValue < 0 || d->nSceneChangeValue > maxVal)
                throw std::runtime_error("scval must be between 0 and " + std::to_string(maxVal));
        }

        vectors.ScaleThSCD(d->thscd1, d->thscd2, vectors.bitsPerSample);

        d->fMaskNormFactor = 1.0f / ml;

        d->time256 = (int)(time * 256 / 100);

        d->maskResizerFull.Init(d->vi.width, d->vi.height, vectors.nBlkX, vectors.nBlkY, vectors.nBlkSizeX, vectors.nBlkSizeY,
            vectors.nOverlapX, vectors.nOverlapY, vectors.bitsPerSample);

    } catch (const std::exception &e) {
        vsapi->mapSetError(out, (d->filterName + ": " + e.what()).c_str());
        return;
    }

    VSFilterDependency deps[1] = { 
        {d->node, rpStrictSpatial}, 
    };

    vsapi->createVideoFilter(out, d->filterName.c_str(), &d->vi, SelectOnBitsPerSample(d->vi.format.bitsPerSample, maskGetFrame<uint8_t>, maskGetFrame<uint16_t>, maskGetFrame<float>), filterFree<MaskData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

static constexpr char filterArgs[] =
    "vectors:vnode;"
    "ml:float:opt;"
    "gamma:float:opt;"
    "time:float:opt;"
    "scval:float:opt;"
    "thscd1:int:opt;"
    "thscd2:float:opt;"
    "prefix:data:opt;";

void maskRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("VectorLengthMask",
        filterArgs,
        "clip:vnode;",
        maskCreate, nullptr, plugin);
    vspapi->registerFunction("SADMask",
        filterArgs,
        "clip:vnode;",
        maskCreate, (void *)1, plugin);
    vspapi->registerFunction("OcclusionMask",
        filterArgs,
        "clip:vnode;",
        maskCreate, (void *)2, plugin);
}

