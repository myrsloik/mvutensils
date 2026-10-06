#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "Common.h"
#include "DepanShared.h"
#include "DepanCompensatePlane.h"

constexpr char prop_DepanCompensate_info[] = "DepanCompensate_info";


struct DepanCompensateData {
    VSNode *clip = nullptr;
    VSNode *data = nullptr;
    float offset;
    int subpixel;
    float pixaspect;
    bool matchfields;
    int mirror;
    int blur;
    bool info;
    bool fields;
    bool tff;
    bool tff_exists;

    const VSVideoInfo *vi;
    int intoffset;
    float xcenter;
    float ycenter;

    int pixel_max;

    const VSAPI *vsapi;

    DepanCompensateData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~DepanCompensateData() {
        vsapi->freeNode(clip);
        vsapi->freeNode(data);
    }
};


static const VSFrame *VS_CC depanCompensateGetFrame(int ndest, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    const DepanCompensateData *d = reinterpret_cast<const DepanCompensateData *>(instanceData);

    if (activationReason == arInitial) {
        int nsrc = ndest - d->intoffset;

        if (d->intoffset == 0 || nsrc < 0 || nsrc > d->vi->numFrames - 1) {
            vsapi->requestFrameFilter(ndest, d->clip, frameCtx);
            return nullptr;
        }

        int start = std::min(nsrc, ndest);
        int end = std::max(nsrc, ndest);

        vsapi->requestFrameFilter(start, d->clip, frameCtx);

        for (int n = start + 1; n <= end; n++)
            vsapi->requestFrameFilter(n, d->data, frameCtx);

        vsapi->requestFrameFilter(end, d->clip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        int nsrc = ndest - d->intoffset;

        if (d->intoffset == 0 || nsrc < 0 || nsrc > d->vi->numFrames - 1) //  nullptr transform, return source
            return vsapi->getFrameFilter(ndest, d->clip, frameCtx);

        bool forward = d->intoffset > 0;

        float fractoffset = d->offset;
        fractoffset += forward ? 1 : -1;
        fractoffset -= d->intoffset;

        int start = std::min(nsrc, ndest);
        int end = std::max(nsrc, ndest);

        int nfields = d->fields ? 2 : 1;

        transform trsum;

        const VSFrame *src = nullptr;
        VSFrame *dst = nullptr;
        const VSFrame *temp = nullptr;

        try {
            for (int n = start + 1; n <= end; n++) {
                const VSFrame *dataframe = vsapi->getFrameFilter(n, d->data, frameCtx);
                const VSMap *data_props = vsapi->getFramePropertiesRO(dataframe);

                MotionData m;
                bool ok = mapGetMotion(m, data_props, vsapi);

                vsapi->freeFrame(dataframe);

                if (!ok)
                    throw std::runtime_error("required frame properties not found in data clip. Did data clip really come from DepanAnalyse or DepanEstimate?");

                if (m.badMotion) {
                    trsum.setNull();
                    break;
                }

                transform tr;

                motion2transform(m.dx, m.dy, m.rot, m.zoom, d->pixaspect / nfields, d->xcenter, d->ycenter, forward, fractoffset, &tr);
                sumtransform(&trsum, &tr, &trsum);
            }

            if (d->fields && d->matchfields) {
                temp = vsapi->getFrameFilter(ndest, d->clip, frameCtx);
                bool top_field = GetTopField(temp, ndest, d->tff_exists, d->tff, true, vsapi);
                vsapi->freeFrame(temp);
                temp = nullptr;

                // reverse 1 line motion correction if matchfields mode
                trsum.dyc += top_field ? -0.5f : 0.5f;
            }


            src = vsapi->getFrameFilter(nsrc, d->clip, frameCtx);
            dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core);

            std::vector<int> work2width4356(2 * d->vi->width + 4356);

            int border[3] = { 0, 1 << (d->vi->format.bitsPerSample - 1), border[1] };
            int blur[3] = { d->blur, d->blur, d->blur };
            transform tr[3];

            tr[0] = tr[1] = trsum;
            if (d->vi->format.subSamplingW == 1 && d->vi->format.subSamplingH == 1) { // 420
                tr[1].dxc /= 2;
                tr[1].dyc /= 2;

                blur[1] = blur[2] = blur[0] / 2;
            } else if (d->vi->format.subSamplingW == 1 && d->vi->format.subSamplingH == 0) { // 422
                tr[1].dxc /= 2;
                tr[1].dxy /= 2;
                tr[1].dyx *= 2;

                blur[1] = blur[2] = blur[0] / 2;
            }
            tr[2] = tr[1];

            for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
                const uint8_t *srcp = vsapi->getReadPtr(src, plane);
                int src_width = vsapi->getFrameWidth(src, plane);
                int src_height = vsapi->getFrameHeight(src, plane);
                ptrdiff_t src_pitch = vsapi->getStride(src, plane);

                uint8_t *dstp = vsapi->getWritePtr(dst, plane);

                if (d->vi->format.bytesPerSample == 1) {
                    if (d->subpixel == 0)
                        compensate_plane_nearest<uint8_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror, border[plane], work2width4356.data(), blur[plane]);
                    else if (d->subpixel == 1)
                        compensate_plane_bilinear<uint8_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror, border[plane], work2width4356.data(), blur[plane]);
                    else
                        compensate_plane_bicubic<uint8_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror, border[plane], work2width4356.data(), blur[plane], d->pixel_max);
                } else {
                    if (d->subpixel == 0)
                        compensate_plane_nearest<uint16_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror, border[plane], work2width4356.data(), blur[plane]);
                    else if (d->subpixel == 1)
                        compensate_plane_bilinear<uint16_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror, border[plane], work2width4356.data(), blur[plane]);
                    else
                        compensate_plane_bicubic<uint16_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror, border[plane], work2width4356.data(), blur[plane], d->pixel_max);
                }
            }

            vsapi->freeFrame(src);
            src = nullptr;

            if (d->info) {
                float dxsum = 0.0f, dysum = 0.0f, rotsum = 0.0f, zoomsum = 1.0f;
                transform2motion(&trsum, forward, d->xcenter, d->ycenter, d->pixaspect / nfields, &dxsum, &dysum, &rotsum, &zoomsum);

#define INFO_SIZE 128
                char info[INFO_SIZE + 1] = { 0 };

                snprintf(info, INFO_SIZE, "offset=%.2f, %d to %d, dx=%.2f, dy=%.2f, rot=%.3f zoom=%.5f", d->offset, ndest - d->intoffset, ndest, dxsum, dysum, rotsum, zoomsum);
#undef INFO_SIZE

                VSMap *dst_props = vsapi->getFramePropertiesRW(dst);
                vsapi->mapSetData(dst_props, prop_DepanCompensate_info, info, -1, dtUtf8, maReplace);
            }

            return dst;
        } catch (const std::exception &e) {
            vsapi->setFilterError(("DepanCompensate: " + std::string(e.what())).c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            vsapi->freeFrame(temp);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC depanCompensateCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<DepanCompensateData> d = std::make_unique<DepanCompensateData>(vsapi);

    int err;

    d->offset = vsapi->mapGetFloatSaturated(in, "offset", 0, &err);

    d->subpixel = vsapi->mapGetIntSaturated(in, "subpixel", 0, &err);
    if (err)
        d->subpixel = 2;

    d->pixaspect = vsapi->mapGetFloatSaturated(in, "pixaspect", 0, &err);
    if (err)
        d->pixaspect = 1.0f;

    d->matchfields = !!vsapi->mapGetInt(in, "matchfields", 0, &err);
    if (err)
        d->matchfields = true;

    d->mirror = vsapi->mapGetIntSaturated(in, "mirror", 0, &err);

    d->blur = vsapi->mapGetIntSaturated(in, "blur", 0, &err);

    d->info = !!vsapi->mapGetInt(in, "info", 0, &err);

    d->fields = !!vsapi->mapGetInt(in, "fields", 0, &err);

    d->tff = !!vsapi->mapGetInt(in, "tff", 0, &err);
    d->tff_exists = !err;


    try {
        if (d->offset < -10.0f || d->offset > 10.0f)
            throw std::runtime_error("offset must be between -10.0 and 10.0 (inclusive)");

        if (d->subpixel < 0 || d->subpixel > 2)
            throw std::runtime_error("subpixel must be between 0 and 2 (inclusive)");

        if (!std::isfinite(d->pixaspect) || d->pixaspect <= 0.0f)
            throw std::runtime_error("pixaspect must be a finite value greater than 0");

        if (d->mirror < 0 || d->mirror > 15)
            throw std::runtime_error("mirror must be between 0 and 15 (inclusive)");

        if (d->blur < 0)
            throw std::runtime_error("blur must not be negative");


        d->clip = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = vsapi->getVideoInfo(d->clip);

        if (!vsh::isConstantVideoFormat(d->vi) ||
            (d->vi->format.colorFamily != cfYUV && d->vi->format.colorFamily != cfGray) ||
            d->vi->format.sampleType != stInteger ||
            d->vi->format.bitsPerSample > 16 ||
            d->vi->format.subSamplingW > 1 ||
            d->vi->format.subSamplingH > 1 ||
            (d->vi->format.subSamplingW == 0 && d->vi->format.subSamplingH == 1))
            throw std::runtime_error("clip must have constant format and dimensions, integer sample type, bit depth up to 16, and it must be Gray, 420, 422, or 444, and not RGB");

        d->data = vsapi->mapGetNode(in, "data", 0, nullptr);

        CheckClipLength(d->data, "data", d->vi->numFrames, "clip", vsapi);


        if (!std::isfinite(d->offset))
            throw std::runtime_error("offset must be a finite number");

        if (d->offset > 0.0f)
            d->intoffset = (int)ceilf(d->offset);
        else
            d->intoffset = (int)floorf(d->offset);

        d->xcenter = d->vi->width / 2.0f; // center of frame
        d->ycenter = d->vi->height / 2.0f;

        d->pixel_max = (1 << d->vi->format.bitsPerSample) - 1;
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("DepanCompensate: " + std::string(e.what())).c_str());
        return;
    }

    VSFilterDependency deps[2] = {
        {d->clip, rpGeneral},
        {d->data, rpGeneral},
    };

    bool info = d->info;

    vsapi->createVideoFilter(out, "DepanCompensate", d->vi, depanCompensateGetFrame, filterFree<DepanCompensateData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();

    if (vsapi->mapGetError(out))
        return;

    if (info) {
        if (!invokeFrameProps(prop_DepanCompensate_info, out, core, vsapi)) {
            vsapi->mapSetError(out, std::string("DepanCompensate: failed to invoke text.FrameProps: ").append(vsapi->mapGetError(out)).c_str());
            return;
        }
    }
}


void depanCompensateRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->registerFunction("DepanCompensate",
                 "clip:vnode;"
                 "data:vnode;"
                 "offset:float:opt;"
                 "subpixel:int:opt;"
                 "pixaspect:float:opt;"
                 "matchfields:int:opt;"
                 "mirror:int:opt;"
                 "blur:int:opt;"
                 "info:int:opt;"
                 "fields:int:opt;"
                 "tff:int:opt;",
                 "clip:vnode;",
                 depanCompensateCreate, nullptr, plugin);
}
