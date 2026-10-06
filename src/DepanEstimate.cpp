#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include <fftw3.h>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "Common.h"
#include "DepanShared.h"

static std::mutex g_fftw_plans_mutex;

constexpr char prop_DepanEstimate_info[] = "DepanEstimate_info";

// stage 1 to stage 2
constexpr char prop_DepanEstimateFFT[] = "DepanEstimateFFT";
constexpr char prop_DepanEstimateFFT2[] = "DepanEstimateFFT2";

// stage 2 to stage 3
constexpr char prop_DepanEstimateX[] = "DepanEstimateX";
constexpr char prop_DepanEstimateY[] = "DepanEstimateY";
constexpr char prop_DepanEstimateZoom[] = "DepanEstimateZoom";
constexpr char prop_DepanEstimateGood[] = "DepanEstimateGood";
constexpr char prop_DepanEstimateTrust[] = "DepanEstimateTrust";


struct DepanEstimateData {
    VSNode *clip = nullptr;
    float trust_limit = 0.0f;
    int winx = 0;
    int winy = 0;
    int wleft = 0;
    int wtop = 0;
    int dxmax = 0;
    int dymax = 0;
    float zoommax = 0.0f;
    float stab = 0.0f;
    float pixaspect = 0.0f;
    bool info = false;
    bool show = false;
    bool fields = false;
    bool tff = false;
    bool tff_exists = false;

    const VSVideoInfo *vi = nullptr;

    int pixel_max = 0;

    size_t fftsize = 0;

    fftwf_plan plan = nullptr;
    
    fftwf_plan planinv = nullptr;

    const VSAPI *vsapi;

    DepanEstimateData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~DepanEstimateData() {
        vsapi->freeNode(clip);
        {
            std::scoped_lock guard(g_fftw_plans_mutex);
            if (plan)
                fftwf_destroy_plan(plan);
            if (planinv)
                fftwf_destroy_plan(planinv);
        }
    }
};


// put source data to real array for FFT
static void frame_data2d(const uint8_t *srcp, ptrdiff_t pitch, float * MVU_RESTRICT realdata, int winx, int winy, int winleft, int h0, int bytes_per_sample) {
    int winxpadded = (winx / 2 + 1) * 2;

    srcp += pitch * h0 + winleft * bytes_per_sample; // offset of window data
    for (int j = 0; j < winy; j++) {
        if (bytes_per_sample == 1) {
            for (int i = 0; i < winx; i += 2) {
                realdata[i] = srcp[i];         // real part
                realdata[i + 1] = srcp[i + 1]; // real part
            }
        } else if (bytes_per_sample == 4) {
            for (int i = 0; i < winx; i += 2) {
                const float *srcpf = (const float *)srcp;
                realdata[i] = srcpf[i];         // real part
                realdata[i + 1] = srcpf[i + 1]; // real part
            }
        } else {
            for (int i = 0; i < winx; i += 2) {
                const uint16_t *srcp16 = (const uint16_t *)srcp;
                realdata[i] = srcp16[i];         // real part
                realdata[i + 1] = srcp16[i + 1]; // real part
            }
        }
        srcp += pitch;
        realdata += winxpadded;
    }
}


static void mult_conj_data2d(const fftwf_complex * MVU_RESTRICT fftnext, const fftwf_complex * MVU_RESTRICT fftsrc, fftwf_complex * MVU_RESTRICT mult, int winx, int winy) {
    // multiply complex conj. *next to src
    // (hermit)
    int nx = winx / 2 + 1; // half-complex row width

    int total = winy * nx; // may be odd (odd winy with a winx divisible by 4), so the buffer holds an odd count

    auto conjMul = [&](int k) {
        // real part
        mult[k][0] = fftnext[k][0] * fftsrc[k][0] + fftnext[k][1] * fftsrc[k][1];
        // imaginary part
        mult[k][1] = fftnext[k][0] * fftsrc[k][1] - fftnext[k][1] * fftsrc[k][0];
    };

    int k = 0;
    for (; k + 1 < total; k += 2) // paired for speed; guard so k + 1 never runs past the buffer
        conjMul(k), conjMul(k + 1);
    if (k < total) // odd total: process the trailing element on its own
        conjMul(k);
}


static void get_motion_vector(const float * MVU_RESTRICT correl, int winx, int winy, float trust_limit, int dxmax, int dymax, float stab, bool fieldbased, bool top_field, float pixaspect, MotionData &md, float *trust) {
    int winxpadded = (winx / 2 + 1) * 2;

    // find global max on real part of correlation surface
    // new version: search only at 4 corners with ranges dxmax, dymax
    float correlmax = correl[0];
    float correlmean = 0.0f;
    int count = 0;
    int imax = 0, jmax = 0;
    const float *correlp = correl;
    for (int j = 0; j <= dymax; j++) { // top
        for (int i = 0; i <= dxmax; i++) { //left
            float cur = correlp[i]; // real part
            correlmean += cur;
            count += 1;
            if (correlmax < cur) {
                correlmax = cur;
                imax = i;
                jmax = j;
            }
        }
        for (int i = winx - dxmax; i < winx; i++) { //right
            float cur = correlp[i]; // real part
            correlmean += cur;
            count += 1;
            if (correlmax < cur) {
                correlmax = cur;
                imax = i;
                jmax = j;
            }
        }
        correlp += winxpadded;
    }
    correlp = correl + (winy - dymax) * winxpadded;
    for (int j = winy - dymax; j < winy; j++) { // bottom
        for (int i = 0; i <= dxmax; i++) { //left
            float cur = correlp[i]; // real part
            correlmean += cur;
            count += 1;
            if (correlmax < cur) {
                correlmax = cur;
                imax = i;
                jmax = j;
            }
        }
        for (int i = winx - dxmax; i < winx; i++) { //right
            float cur = correlp[i]; // real part
            correlmean += cur;
            count += 1;
            if (correlmax < cur) {
                correlmax = cur;
                imax = i;
                jmax = j;
            }
        }
        correlp += winxpadded;
    }

    correlmean = correlmean / count; // mean value

    correlmax = correlmax / (winx * winy);   // normalize value
    correlmean = correlmean / (winx * winy); // normalize value

    *trust = (correlmax - correlmean) * 100.0f / (correlmax + 0.1f); // +0.1 for safe divide


    // get correct shift values on periodic surface (adjusted borders)
    int dx = (imax * 2 < winx) ? imax : (imax - winx);
    int dy = (jmax * 2 < winy) ? jmax : (jmax - winy);

    // some trust decreasing for large shifts

    *trust *= (dxmax + 1) / (dxmax + 1 + stab * abs(dx)) * (dymax + 1) / (dymax + 1 + stab * abs(dy));

    // reject if relative diffference correlmax from correlmean is small
    // probably due to scene change
    if (*trust < trust_limit) {
        md.badMotion = true; // scene change / unreliable
    } else {
        // normal, no scene change
        // get more precise float dx, dy by interpolation
        // get i, j, of left and right of max
        int imaxp1 = (imax + 1 < winx) ? (imax + 1) : (imax + 1 - winx); // plus 1, over period
        int imaxm1 = (imax - 1 >= 0) ? (imax - 1) : (imax - 1 + winx);   // minus 1
        int jmaxp1 = (jmax + 1 < winy) ? (jmax + 1) : (jmax + 1 - winy);
        int jmaxm1 = (jmax - 1 >= 0) ? (jmax - 1) : (jmax - 1 + winy);

        // first and second differential
        float f1 = (correl[jmax * winxpadded + imaxp1] - correl[jmax * winxpadded + imaxm1]) / 2.0f;
        float f2 = correl[jmax * winxpadded + imaxp1] + correl[jmax * winxpadded + imaxm1] - correl[jmax * winxpadded + imax] * 2.0f;

        float xadd;
        if (f2 == 0.0f)
            xadd = 0.0f;
        else {
            xadd = -f1 / f2;
            if (xadd > 1.0f)
                xadd = 1.0f;
            else if (xadd < -1.0f)
                xadd = -1.0f;
        }

        if (fabsf(dx + xadd) > dxmax)
            xadd = 0.0f;

        f1 = (correl[jmaxp1 * winxpadded + imax] - correl[jmaxm1 * winxpadded + imax]) / 2.0f;
        f2 = correl[jmaxp1 * winxpadded + imax] + (correl[jmaxm1 * winxpadded + imax]) - correl[jmax * winxpadded + imax] * 2.0f;

        float yadd;
        if (f2 == 0.0f)
            yadd = 0.0f;
        else {
            yadd = -f1 / f2;
            if (yadd > 1.0f)
                yadd = 1.0f; // limit addition for stability
            else if (yadd < -1.0f)
                yadd = -1.0f;
        }

        if (fabsf(dy + yadd) > dymax)
            yadd = 0.0f;

        if (fieldbased) { // correct line shift for fields
            if (top_field)
                yadd += 0.5f;
            else
                yadd += -0.5f;
            // scale dy for fieldbased frame by factor 2
            yadd = yadd * 2.0f;
            dy = dy * 2;
        }


        md.dx = (float)dx + xadd;
        md.dy = (float)dy + yadd;

        md.dy = md.dy / pixaspect;
    }
}


// get forward fft of src frame plane
static void get_plane_fft(const uint8_t *srcp, ptrdiff_t src_pitch, fftwf_complex *fftsrc, int winx, int winy, int winleft, int wintop, fftwf_plan plan, int bytes_per_sample) {
    // prepare 2d data for fft
    frame_data2d(srcp, src_pitch, (float *)fftsrc, winx, winy, winleft, wintop, bytes_per_sample);
    // make forward fft of data
    fftwf_execute_dft_r2c(plan, (float *)fftsrc, fftsrc);
}


static void showcorrelation(const float * MVU_RESTRICT correl, int winx, int winy, uint8_t *dstp, ptrdiff_t dst_pitch, int winleft, int wintop, int pixel_max) {
    int winxpadded = (winx / 2 + 1) * 2;

    // find max and min
    float correlmax = correl[0];
    float correlmin = correl[0];
    const float *correlp = correl;
    for (int j = 0; j < winy; j++) {
        for (int i = 0; i < winx; i++) {
            float cur = correlp[i];
            if (correlmax < cur) {
                correlmax = cur;
            }
            if (correlmin > cur) {
                correlmin = cur;
            }
        }
        correlp += winxpadded;
    }

    // normalize
    float norm = (float)pixel_max / (correlmax - correlmin);

    dstp += wintop * dst_pitch; // go to first line of window

    int bytes_per_sample = (pixel_max == 255) ? 1 : (pixel_max == 1) ? 4 : 2;

    dstp += winleft * bytes_per_sample;
    correlp = correl;
    for (int j = 0; j < winy; j++) {
        if (pixel_max == 255) {
            for (int i = 0; i < winx; i++)
                dstp[i] = (int)((correlp[i] - correlmin) * norm); // real part
        } else if (pixel_max == 1) {
            for (int i = 0; i < winx; i++) {
                float *dstpf = (float *)dstp;
                dstpf[i] = (correlp[i] - correlmin) * norm;
            }
        } else {
            for (int i = 0; i < winx; i++) {
                uint16_t *dstp16 = (uint16_t *)dstp;
                dstp16[i] = (int)((correlp[i] - correlmin) * norm);
            }
        }

        dstp += dst_pitch;
        correlp += winxpadded;
    }
}


static const VSFrame *VS_CC depanEstimateStage1GetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    DepanEstimateData *d = reinterpret_cast<DepanEstimateData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->clip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->clip, frameCtx);
        VSFrame *dst = vsapi->copyFrame(src, core);
        vsapi->freeFrame(src);

        const uint8_t *dstp = vsapi->getReadPtr(dst, 0);
        ptrdiff_t stride = vsapi->getStride(dst, 0);

        fftwf_complex *fft = (fftwf_complex *)fftwf_malloc(d->fftsize);

        int winleft = d->wleft; // left of fft window

        get_plane_fft(dstp, stride, fft, d->winx, d->winy, winleft, d->wtop, d->plan, d->vi->format.bytesPerSample);

        VSMap *dst_props = vsapi->getFramePropertiesRW(dst);

        vsapi->mapSetData(dst_props, prop_DepanEstimateFFT, (const char *)fft, static_cast<int>(d->fftsize), dtBinary, maReplace);
        fftwf_free(fft);

        if (d->zoommax != 1.0f) {
            fftwf_complex *fft2 = (fftwf_complex *)fftwf_malloc(d->fftsize);

            int winleft2 = d->wleft + d->vi->width / 2; // left edge of right (2)fft window

            get_plane_fft(dstp, stride, fft2, d->winx, d->winy, winleft2, d->wtop, d->plan, d->vi->format.bytesPerSample);

            vsapi->mapSetData(dst_props, prop_DepanEstimateFFT2, (const char *)fft2, static_cast<int>(d->fftsize), dtBinary, maReplace);
            fftwf_free(fft2);
        }

        return dst;
    }

    return nullptr;
}


static const VSFrame *VS_CC depanEstimateStage2GetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    DepanEstimateData *d = reinterpret_cast<DepanEstimateData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(std::max(0, n - 1), d->clip, frameCtx);
        vsapi->requestFrameFilter(n, d->clip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *prev = vsapi->getFrameFilter(std::max(0, n - 1), d->clip, frameCtx);
        const VSFrame *cur = vsapi->getFrameFilter(n, d->clip, frameCtx);

        VSFrame *dst = nullptr;

        try {
            const VSMap *prev_props = vsapi->getFramePropertiesRO(prev);
            const VSMap *cur_props = vsapi->getFramePropertiesRO(cur);
            int err;

            bool top_field = false;
            if (d->fields)
                top_field = GetTopField(cur, n, d->tff_exists, d->tff, true, vsapi);

            if (d->fftsize != (size_t)vsapi->mapGetDataSize(prev_props, prop_DepanEstimateFFT, 0, &err) ||
                d->fftsize != (size_t)vsapi->mapGetDataSize(cur_props, prop_DepanEstimateFFT, 0, &err))
                throw std::runtime_error("temporary property '" + std::string(prop_DepanEstimateFFT) + "' has the wrong size. This should never happen");

            if (d->zoommax != 1.0f) {
                if (d->fftsize != (size_t)vsapi->mapGetDataSize(prev_props, prop_DepanEstimateFFT2, 0, &err) ||
                    d->fftsize != (size_t)vsapi->mapGetDataSize(cur_props, prop_DepanEstimateFFT2, 0, &err))
                    throw std::runtime_error("temporary property '" + std::string(prop_DepanEstimateFFT2) + "' has the wrong size. This should never happen");
            }

            const fftwf_complex *fftprev = (const fftwf_complex *)vsapi->mapGetData(prev_props, prop_DepanEstimateFFT, 0, &err);
            const fftwf_complex *fftcur = (const fftwf_complex *)vsapi->mapGetData(cur_props, prop_DepanEstimateFFT, 0, &err);

            // memory for correlation matrice
            fftwf_complex *correl = (fftwf_complex *)fftwf_malloc(d->fftsize);

            MotionData md1;
            float trust1;

            // prepare correlation data = mult fftsrc* by fftprev
            mult_conj_data2d(fftcur, fftprev, correl, d->winx, d->winy);
            // make inverse fft of prepared correl data
            fftwf_execute_dft_c2r(d->planinv, correl, (float *)correl);
            // now correl is is true correlation surface
            // find global motion vector as maximum on correlation sufrace
            // save vector to motion table
            get_motion_vector((float *)correl, d->winx, d->winy, d->trust_limit, d->dxmax, d->dymax, d->stab, d->fields, top_field, d->pixaspect, md1, &trust1);


            int winleft = d->wleft;

            dst = vsapi->copyFrame(cur, core);
            uint8_t *dstp = nullptr;
            ptrdiff_t dst_stride = 0;

            if (d->show) { // show correlation sufrace
                dstp = vsapi->getWritePtr(dst, 0);
                dst_stride = vsapi->getStride(dst, 0);

                showcorrelation((float *)correl, d->winx, d->winy, dstp, dst_stride, winleft, d->wtop, d->pixel_max);
            }

            fftwf_free(correl);

            MotionData result;
            float trust;

            if (d->zoommax == 1.0f) { // NO ZOOM
                result.dx = md1.dx;
                result.dy = md1.dy;
                result.badMotion = md1.badMotion;
                trust = trust1;
            } else { // ZOOM, calculate 2 data sets (left and right)
                int winleft2 = d->wleft + d->vi->width / 2; // left edge of right (2)fft window

                const fftwf_complex *fftprev2 = (const fftwf_complex *)vsapi->mapGetData(prev_props, prop_DepanEstimateFFT2, 0, &err);
                const fftwf_complex *fftcur2 = (const fftwf_complex *)vsapi->mapGetData(cur_props, prop_DepanEstimateFFT2, 0, &err);

                fftwf_complex *correl2 = (fftwf_complex *)fftwf_malloc(d->fftsize);

                MotionData md2;
                float trust2;

                // right window
                // prepare correlation data = mult fftsrc* by fftprev
                mult_conj_data2d(fftcur2, fftprev2, correl2, d->winx, d->winy);
                // make inverse fft of prepared correl data
                fftwf_execute_dft_c2r(d->planinv, correl2, (float *)correl2);
                // now correl is is true correlation surface
                // find global motion vector as maximum on correlation sufrace
                // save vector to motion table
                get_motion_vector((float *)correl2, d->winx, d->winy, d->trust_limit, d->dxmax, d->dymax, d->stab, d->fields, top_field, d->pixaspect, md2, &trust2);

                // now we have 2 motion data sets for left and right windows
                // estimate zoom factor
                float zoom = 1.0f + (md2.dx - md1.dx) / (winleft2 - winleft);
                if (!md1.badMotion && !md2.badMotion && (fabsf(zoom - 1.0f) < (d->zoommax - 1.0f))) { // if motion data and zoom good
                    result.dx = (md1.dx + md2.dx) / 2.0f;
                    result.dy = (md1.dy + md2.dy) / 2.0f;
                    result.zoom = zoom;
                    trust = std::min(trust1, trust2);
                } else { // bad zoom,
                    result.badMotion = true;
                    trust = std::min(trust1, trust2);
                }

                if (d->show) // show correlation sufrace
                    showcorrelation((float *)correl2, d->winx, d->winy, dstp, dst_stride, winleft2, d->wtop, d->pixel_max);

                fftwf_free(correl2);
            }

            // the FFT data aliased into prev/cur has been consumed; release them now
            vsapi->freeFrame(prev);
            prev = nullptr;
            vsapi->freeFrame(cur);
            cur = nullptr;

            VSMap *dst_props = vsapi->getFramePropertiesRW(dst);

            vsapi->mapDeleteKey(dst_props, prop_DepanEstimateFFT);
            vsapi->mapDeleteKey(dst_props, prop_DepanEstimateFFT2);

            if (n == 0) {
                result.badMotion = true; // frame 0 has no predecessor: forced scene start
                trust = 0.0f;            // a zero trust disables frame 1's backward scene-change test in stage3 (preserved from the original)
            }

            vsapi->mapSetFloat(dst_props, prop_DepanEstimateX, result.dx, maReplace);
            vsapi->mapSetFloat(dst_props, prop_DepanEstimateY, result.dy, maReplace);
            vsapi->mapSetFloat(dst_props, prop_DepanEstimateZoom, result.zoom, maReplace);
            vsapi->mapSetInt(dst_props, prop_DepanEstimateGood, result.badMotion ? 0 : 1, maReplace);
            vsapi->mapSetFloat(dst_props, prop_DepanEstimateTrust, trust, maReplace);

            return dst;
        } catch (const std::exception &e) {
            vsapi->setFilterError(("DepanEstimate: " + std::string(e.what())).c_str(), frameCtx);
            vsapi->freeFrame(prev);
            vsapi->freeFrame(cur);
            vsapi->freeFrame(dst);
            return nullptr;
        }
    }

    return nullptr;
}


static const VSFrame *VS_CC depanEstimateStage3GetFrame(int n, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    DepanEstimateData *d = reinterpret_cast<DepanEstimateData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(std::max(0, n - 1), d->clip, frameCtx);
        vsapi->requestFrameFilter(n, d->clip, frameCtx);
        vsapi->requestFrameFilter(std::min(n + 1, d->vi->numFrames - 1), d->clip, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src[3];
        src[0] = vsapi->getFrameFilter(std::max(0, n - 1), d->clip, frameCtx);
        src[1] = vsapi->getFrameFilter(n, d->clip, frameCtx);
        src[2] = vsapi->getFrameFilter(std::min(n + 1, d->vi->numFrames - 1), d->clip, frameCtx);

        VSFrame *dst = nullptr;

        try {
            const VSMap *src_props[3];

            float trust[3];

            for (int i = 0; i < 3; i++) {
                src_props[i] = vsapi->getFramePropertiesRO(src[i]);

                int err;
                trust[i] = vsapi->mapGetFloatSaturated(src_props[i], prop_DepanEstimateTrust, 0, &err);
                if (err)
                    throw std::runtime_error("temporary property '" + std::string(prop_DepanEstimateTrust) + "' not found in input frame. This should never happen");
            }

            vsapi->freeFrame(src[0]);
            src[0] = nullptr;
            vsapi->freeFrame(src[2]);
            src[2] = nullptr;

            int err[4];
            MotionData m;
            m.dx = vsapi->mapGetFloatSaturated(src_props[1], prop_DepanEstimateX, 0, &err[0]);
            m.dy = vsapi->mapGetFloatSaturated(src_props[1], prop_DepanEstimateY, 0, &err[1]);
            m.zoom = vsapi->mapGetFloatSaturated(src_props[1], prop_DepanEstimateZoom, 0, &err[2]);
            bool good = vsapi->mapGetInt(src_props[1], prop_DepanEstimateGood, 0, &err[3]) != 0;

            if (err[0] || err[1] || err[2] || err[3])
                throw std::runtime_error("some temporary property was not found in input frame. This should never happen");

            m.badMotion = !good;

            // check scenechanges in range, as sharp decreasing of trust
            if (n - 1 >= 0 && n < d->vi->numFrames && trust[1] < d->trust_limit * 2.0f && trust[1] < 0.5f * trust[0])
                m.badMotion = true; // very sharp decrease of not very big trust, probably due to scenechange
            if (n >= 0 && n + 1 < d->vi->numFrames && trust[1] < d->trust_limit * 2.0f && trust[1] < 0.5f * trust[2])
                m.badMotion = true; // very sharp decrease of not very big trust, probably due to scenechange

            if (m.badMotion) { // keep the in-memory values finite for a bad frame
                m.dx = 0.0f;
                m.dy = 0.0f;
                m.zoom = 1.0f;
            }

            dst = vsapi->copyFrame(src[1], core);
            vsapi->freeFrame(src[1]);
            src[1] = nullptr;

            VSMap *dst_props = vsapi->getFramePropertiesRW(dst);

            vsapi->mapDeleteKey(dst_props, prop_DepanEstimateX);
            vsapi->mapDeleteKey(dst_props, prop_DepanEstimateY);
            vsapi->mapDeleteKey(dst_props, prop_DepanEstimateZoom);
            vsapi->mapDeleteKey(dst_props, prop_DepanEstimateGood);
            vsapi->mapDeleteKey(dst_props, prop_DepanEstimateTrust);

            mapSetMotion(dst_props, m, vsapi);

            if (d->info) {
#define INFO_SIZE 128
                char info[INFO_SIZE + 1] = { 0 };

                snprintf(info, INFO_SIZE, "fn=%d dx=%.2f dy=%.2f zoom=%.5f trust=%.2f bad=%d", n, m.dx, m.dy, m.zoom, trust[1], m.badMotion);
#undef INFO_SIZE

                vsapi->mapSetData(dst_props, prop_DepanEstimate_info, info, -1, dtUtf8, maReplace);
            }

            return dst;
        } catch (const std::exception &e) {
            vsapi->setFilterError(("DepanEstimate: " + std::string(e.what())).c_str(), frameCtx);
            vsapi->freeFrame(src[0]);
            vsapi->freeFrame(src[1]);
            vsapi->freeFrame(src[2]);
            vsapi->freeFrame(dst);
            return nullptr;
        }
    }

    return nullptr;
}


static void VS_CC depanEstimateCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<DepanEstimateData> data1 = std::make_unique<DepanEstimateData>(vsapi);

    int err;

    data1->trust_limit = vsapi->mapGetFloatSaturated(in, "trust", 0, &err);
    if (err)
        data1->trust_limit = 4.0f;

    data1->winx = vsapi->mapGetIntSaturated(in, "winx", 0, &err);

    data1->winy = vsapi->mapGetIntSaturated(in, "winy", 0, &err);

    data1->wleft = vsapi->mapGetIntSaturated(in, "wleft", 0, &err);
    if (err)
        data1->wleft = -1;

    data1->wtop = vsapi->mapGetIntSaturated(in, "wtop", 0, &err);
    if (err)
        data1->wtop = -1;

    data1->dxmax = vsapi->mapGetIntSaturated(in, "dxmax", 0, &err);
    if (err)
        data1->dxmax = -1;

    data1->dymax = vsapi->mapGetIntSaturated(in, "dymax", 0, &err);
    if (err)
        data1->dymax = -1;

    data1->zoommax = vsapi->mapGetFloatSaturated(in, "zoommax", 0, &err);
    if (err)
        data1->zoommax = 1.0f;

    data1->stab = vsapi->mapGetFloatSaturated(in, "stab", 0, &err);
    if (err)
        data1->stab = 1.0f;

    data1->pixaspect = vsapi->mapGetFloatSaturated(in, "pixaspect", 0, &err);
    if (err)
        data1->pixaspect = 1.0f;

    data1->info = !!vsapi->mapGetInt(in, "info", 0, &err);

    data1->show = !!vsapi->mapGetInt(in, "show", 0, &err);

    data1->fields = !!vsapi->mapGetInt(in, "fields", 0, &err);

    data1->tff = !!vsapi->mapGetInt(in, "tff", 0, &err);
    data1->tff_exists = !err;


    try {
        if (data1->trust_limit < 0.0f || data1->trust_limit > 100.0f)
            throw std::runtime_error("trust must be between 0.0 and 100.0 (inclusive)");

        if (data1->pixaspect <= 0.0f)
            throw std::runtime_error("pixaspect must be positive");

        data1->clip = vsapi->mapGetNode(in, "clip", 0, nullptr);
        data1->vi = vsapi->getVideoInfo(data1->clip);

        if (!vsh::isConstantVideoFormat(data1->vi) ||
            (data1->vi->format.colorFamily != cfYUV && data1->vi->format.colorFamily != cfGray) ||
            (data1->vi->format.sampleType == stInteger && data1->vi->format.bitsPerSample > 16) ||
            (data1->vi->format.sampleType == stFloat && data1->vi->format.bitsPerSample != 32))
            throw std::runtime_error("clip must have constant format and dimensions, it must be YUV or Gray, and it must be 8..16 bit integer or 32 bit float");

        // used only for luma
        if (data1->vi->format.sampleType == stFloat)
            data1->pixel_max = 1;
        else
            data1->pixel_max = (1 << data1->vi->format.bitsPerSample) - 1;


        int wleft0 = data1->wleft; // save
        if (data1->wleft < 0)
            data1->wleft = 0; // auto

        // check and set fft window x size

        if (data1->winx > data1->vi->width - data1->wleft)
            throw std::runtime_error("winx must not be greater than width-wleft");
        if (data1->winx < 0 || (data1->winx & 1)) // odd winx overreads the source by one column in frame_data2d's i+=2 loop
            throw std::runtime_error("winx must be a non-negative even number");
        if (data1->winx == 0) { // auto
            data1->winx = data1->vi->width - data1->wleft;
            // find max fft window size (power of 2)
            int wx = 1;
            for (int i = 0; i < 13; i++) {
                if (wx * 2 <= data1->winx)
                    wx = wx * 2;
            }
            data1->winx = wx;
            // The doubling loop lands on 1 (odd) when width-wleft == 1, which overreads the source by one
            // column in frame_data2d's i+=2 loop just like a user-supplied odd winx; reject it.
            if (data1->winx & 1)
                throw std::runtime_error("the automatic winx resolved to a 1-pixel window (width-wleft is too small); use a larger frame or smaller wleft");
        }

        if (data1->zoommax != 1.0f) {
            data1->winx = data1->winx / 2; // devide window x by 2 part (left and right)
            // The even check above ran on the pre-halved winx; halving an even winx yields an odd one
            // (e.g. 66 -> 33) unless the original was divisible by 4. An odd winx overreads the source
            // by one column in frame_data2d's i+=2 loop, so reject it here.
            if (data1->winx & 1)
                throw std::runtime_error("with zoommax != 1, winx must be divisible by 4 (it is halved for the left/right zoom windows)");
            if (wleft0 < 0)
                data1->wleft = (data1->vi->width - data1->winx * 2) / 4;
        } else if (wleft0 < 0)
            data1->wleft = (data1->vi->width - data1->winx) / 2;

        // when zooming, the right window starts at wleft + width/2 and spans winx
        // columns; reject a (user-supplied) wleft that would push it past the edge
        if (data1->zoommax != 1.0f && data1->wleft + data1->vi->width / 2 + data1->winx > data1->vi->width)
            throw std::runtime_error("wleft is too large: the right zoom window (wleft + width/2 + winx) exceeds the frame width");

        int wtop0 = data1->wtop; // save
        if (data1->wtop < 0)
            data1->wtop = 0; // auto

        // check and set fft window y size
        if (data1->winy > data1->vi->height - data1->wtop)
            throw std::runtime_error("winy must not be greater than height-wtop");
        if (data1->winy == 0) {
            data1->winy = data1->vi->height - data1->wtop; // start value
            // find max fft window size (power of 2)
            int wy = 1;
            for (int i = 0; i < 13; i++) {
                if (wy * 2 <= data1->winy)
                    wy = wy * 2;
            }
            data1->winy = wy;
        }

        if (wtop0 < 0)
            data1->wtop = (data1->vi->height - data1->winy) / 2; // auto

        // max dx shift must be less than winx/2
        if (data1->dxmax < 0)
            data1->dxmax = data1->winx / 4; // default
        if (data1->dymax < 0)
            data1->dymax = data1->winy / 4; // default

        if (data1->dxmax >= data1->winx / 2)
            throw std::runtime_error("dxmax must be less than winx/2");
        if (data1->dymax >= data1->winy / 2)
            throw std::runtime_error("dymax must be less than winy/2");


        int winxpadded = (data1->winx / 2 + 1) * 2;
        data1->fftsize = data1->winy * winxpadded / 2; //complex
        data1->fftsize *= sizeof(fftwf_complex);
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("DepanEstimate: " + std::string(e.what())).c_str());
        return;
    }


    std::unique_ptr<DepanEstimateData> data2 = std::make_unique<DepanEstimateData>(vsapi);
    std::unique_ptr<DepanEstimateData> data3 = std::make_unique<DepanEstimateData>(vsapi);

    *data2 = *data1;
    *data3 = *data1;
    data2->clip = nullptr;
    data3->clip = nullptr;

    fftwf_complex *unused_array = (fftwf_complex *)fftwf_malloc(data1->fftsize);
    {
        std::scoped_lock guard(g_fftw_plans_mutex);
        data1->plan = fftwf_plan_dft_r2c_2d(data1->winy, data1->winx, (float *)unused_array, unused_array, FFTW_ESTIMATE);    // forward fft (stage1)
        data2->planinv = fftwf_plan_dft_c2r_2d(data1->winy, data1->winx, unused_array, (float *)unused_array, FFTW_ESTIMATE); // inverse fft (stage2)
    }
    fftwf_free(unused_array);

    const bool info = data1->info;

    VSFilterDependency deps1[1] = { 
        {data1->clip, rpStrictSpatial},
    };

    vsapi->createVideoFilter(out, "DepanEstimateStage1", data1->vi, depanEstimateStage1GetFrame, filterFree<DepanEstimateData>, fmParallel, deps1, ARRAY_SIZE(deps1), data1.get(), core);
    data1.release();

    if (vsapi->mapGetError(out))
        return;

    data2->clip = vsapi->mapGetNode(out, "clip", 0, nullptr);
    vsapi->clearMap(out);

    VSFilterDependency deps2[1] = { 
        {data2->clip, rpGeneral},
    };

    vsapi->createVideoFilter(out, "DepanEstimateStage2", data2->vi, depanEstimateStage2GetFrame, filterFree<DepanEstimateData>, fmParallel, deps2, ARRAY_SIZE(deps2), data2.get(), core);
    data2.release();

    if (vsapi->mapGetError(out))
        return;

    data3->clip = vsapi->mapGetNode(out, "clip", 0, nullptr);
    vsapi->clearMap(out);

    VSFilterDependency deps3[1] = { 
        {data3->clip, rpGeneral},
    };

    vsapi->createVideoFilter(out, "DepanEstimateStage3", data3->vi, depanEstimateStage3GetFrame, filterFree<DepanEstimateData>, fmParallel, deps3, ARRAY_SIZE(deps3), data3.get(), core);
    data3.release();

    if (vsapi->mapGetError(out))
        return;

    if (info) {
        if (!invokeFrameProps(prop_DepanEstimate_info, out, core, vsapi)) {
            vsapi->mapSetError(out, std::string("DepanEstimate: failed to invoke text.FrameProps: ").append(vsapi->mapGetError(out)).c_str());
            return;
        }
    }
}


void depanEstimateRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->registerFunction("DepanEstimate",
                 "clip:vnode;"
                 "trust:float:opt;"
                 "winx:int:opt;"
                 "winy:int:opt;"
                 "wleft:int:opt;"
                 "wtop:int:opt;"
                 "dxmax:int:opt;"
                 "dymax:int:opt;"
                 "zoommax:float:opt;"
                 "stab:float:opt;"
                 "pixaspect:float:opt;"
                 "info:int:opt;"
                 "show:int:opt;"
                 "fields:int:opt;"
                 "tff:int:opt;",
                 "clip:vnode;",
                 depanEstimateCreate, nullptr, plugin);
}
