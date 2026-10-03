// Author: Manao
// Copyright(c)2006 A.G.Balakhnin aka Fizick - YUY2
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

#include <VapourSynth4.h>
#include <memory>
#include <stdexcept>

#include "SuperPyramid.h"
#include "MotionBlockPyramid.h"
#include "Common.h"


struct SCDetectionData {
    VSNode *node = nullptr;
    VSNode *vectors = nullptr;

    const VSVideoInfo *vi;

    int64_t thscd1;
    float thscd2;
    AnalysisGeometry geometry;

    std::string prefix;

    const VSAPI *vsapi;

    SCDetectionData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~SCDetectionData() {
        vsapi->freeNode(node);
        vsapi->freeNode(vectors);
    }
};

static const VSFrame *VS_CC scdetectionGetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    SCDetectionData *d = reinterpret_cast<SCDetectionData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        vsapi->requestFrameFilter(n, d->vectors, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        VSFrame *dst = vsapi->copyFrame(src, core);
        vsapi->freeFrame(src);

        try {
            MotionBlockPyramid vectors(vsapi->getFrameFilter(n, d->vectors, frameCtx), 1, d->prefix, vsapi, d->geometry);

            constexpr const char *propNames[2] = { "_SceneChangePrev", "_SceneChangeNext" };
            VSMap *props = vsapi->getFramePropertiesRW(dst);
            vsapi->mapSetInt(props, propNames[vectors.nDeltaFrame > 0], !vectors.IsUsable(d->thscd1, d->thscd2), maReplace);
        } catch (const std::exception &e) {
            vsapi->setFilterError(("SCDetection: " + std::string(e.what())).c_str(), frameCtx);
            vsapi->freeFrame(dst);
            return nullptr;
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC scdetectionCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<SCDetectionData> d = std::make_unique<SCDetectionData>(vsapi);

    int err;

    d->thscd1 = vsapi->mapGetInt(in, "thscd1", 0, &err);
    if (err)
        d->thscd1 = MV_DEFAULT_SCD1;

    d->thscd2 = vsapi->mapGetFloatSaturated(in, "thscd2", 0, &err);
    if (err)
        d->thscd2 = MV_DEFAULT_SCD2;

    const char *prefix = vsapi->mapGetData(in, "prefix", 0, &err);
    if (prefix)
        d->prefix = prefix;
    else
        d->prefix = DEFAULT_MVUTENSILS_PREFIX;

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vectors = vsapi->mapGetNode(in, "vectors", 0, nullptr);

    d->vi = vsapi->getVideoInfo(d->node);

    try {
        CheckClipLength(d->vectors, "vectors", d->vi->numFrames, "clip", vsapi);

        MotionBlockPyramid vectors(d->vectors, d->prefix, vsapi);

        vectors.ScaleThSCD(d->thscd1, d->thscd2, vectors.bitsPerSample);
        d->geometry = vectors.Geometry();
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("SCDetection: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[2] = { 
        {d->node, rpStrictSpatial},
        {d->vectors, rpStrictSpatial}
    };

    vsapi->createVideoFilter(out, "SCDetection", d->vi, scdetectionGetFrame, filterFree<SCDetectionData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();
}

void scdetectionRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) noexcept {
    vspapi->registerFunction("SCDetection",
                 "clip:vnode;"
                 "vectors:vnode;"
                 "thscd1:int:opt;"
                 "thscd2:float:opt;"
                 "prefix:data:opt;",
                 "clip:vnode;",
                 scdetectionCreate, nullptr, plugin);
}
