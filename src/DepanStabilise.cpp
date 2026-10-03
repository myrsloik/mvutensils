#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "Common.h"
#include "DepanShared.h"
#include "DepanCompensatePlane.h"

constexpr char prop_DepanStabilise_info[] = "DepanStabilise_info";


struct DepanStabiliseData {
    VSNode *clip = nullptr;
    VSNode *data = nullptr;
    float cutoff;
    float damping;
    float initzoom;
    bool addzoom;
    int prev;
    int next;
    int mirror;
    int blur;
    float dxmax;
    float dymax;
    float zoommax;
    float rotmax;
    int subpixel;
    float pixaspect;
    int fitlast;
    float tzoom;
    bool info;
    int method;
    bool fields;

    const VSVideoInfo *vi;

    int pixel_max;

    int nfields;

    std::vector<std::optional<MotionData>> motion;


    transform nonlinfactor;

    float fps;        // frame per second
    float freqnative; // native frequency
    int radius;       // stabilization radius

    std::vector<float> wint; // average window
    int wintsize;
    std::vector<float> winrz; // rize zoom window
    std::vector<float> winfz; // fall zoom window
    int winrzsize;
    int winfzsize;

    float xcenter; // center of frame
    float ycenter;

    std::mutex motion_mutex;
    std::atomic_flag warned = ATOMIC_FLAG_INIT; // WarnIfDiverged fires at most once per instance

    const VSAPI *vsapi;

    DepanStabiliseData(const VSAPI *vsapi) : vsapi(vsapi) {};

    ~DepanStabiliseData() {
        vsapi->freeNode(clip);
        vsapi->freeNode(data);
    }
};


// The inertial smoother is only conditionally stable. Its feedback is scaled by
// nonlinfactor = 5/dxmax times the tracking error, so setting dxmax/dymax far below the clip's
// actual motion makes the correction grow faster than the error and the recurrence runs away --
// measured going from 4e8 to 1.6e32 in a single frame. InertialLimit does not catch it: its "soft
// limit" sqrtf(dxdif*dxmax) is a geometric mean, which only halves the exponent (1e32 -> 1e16), and
// its only hard guard is the isfinite() check, which fires just when the blowup happens to reach
// infinity rather than a merely astronomical finite value.
//
// The compensate kernels now clamp the shift so this can no longer read outside the plane, but the
// frames are still wrong, so say so once rather than silently emitting broken output. Static clips
// never trigger this (the error stays zero); the threshold is content dependent, roughly governed by
// motion/dxmax rather than dxmax alone.
static void WarnIfDiverged(DepanStabiliseData *d, const transform &tr, int ndest, VSCore *core, const VSAPI *vsapi) {
    const float sane = (float)(d->vi->width + d->vi->height) * 8.0f;
    if (std::isfinite(tr.dxc) && std::isfinite(tr.dyc) && fabsf(tr.dxc) <= sane && fabsf(tr.dyc) <= sane)
        return;
    if (d->warned.test_and_set(std::memory_order_relaxed)) // once per filter instance
        return;

    char msg[320];
    snprintf(msg, sizeof(msg),
        "DepanStabilise: motion smoothing diverged at frame %d (dx=%.4g, dy=%.4g for a %dx%d frame). "
        "dxmax/dymax are likely far below the clip's actual motion; try larger values. "
        "Affected frames are stabilised incorrectly.",
        ndest, (double)tr.dxc, (double)tr.dyc, d->vi->width, d->vi->height);
    vsapi->logMessage(mtWarning, msg, core);
}


static void Inertial(DepanStabiliseData *d, transform *trcumul, transform *trsmoothed, float *azoom, float *azoomsmoothed, const int nbase, const int ndest, transform *ptrdif) {
    const transform nonlinfactor = d->nonlinfactor;
    const float damping = d->damping;
    const float fps = d->fps;
    const float freqnative = d->freqnative;
    const float pixaspect = d->pixaspect;
    const int nfields = d->nfields;
    const bool addzoom = d->addzoom;
    const float initzoom = d->initzoom;
    const float xcenter = d->xcenter;
    const float ycenter = d->ycenter;
    const int width = d->vi->width;
    const int height = d->vi->height;
    const float cutoff = d->cutoff;
    const float tzoom = d->tzoom;

    // set null as smoothed for base - v1.12
    // set null as smoothed for base+1 - v1.12
    trsmoothed[nbase].setNull();
    trsmoothed[nbase + 1].setNull();

    float cdamp = 12.56f * damping / fps;
    float cquad = 39.44f / (fps * fps);

    // recurrent calculation of smoothed cumulative transforms from base+2 to ndest frames
    for (int n = nbase + 2; n <= ndest; n++) {

        // dxc predictor:
        trsmoothed[n].dxc = 2 * trsmoothed[n - 1].dxc - trsmoothed[n - 2].dxc -
                            cdamp * freqnative * (trsmoothed[n - 1].dxc - trsmoothed[n - 2].dxc - trcumul[n - 1].dxc + trcumul[n - 2].dxc) *
                                (1 + 0.5f * nonlinfactor.dxc / freqnative * fabsf(trsmoothed[n - 1].dxc - trsmoothed[n - 2].dxc - trcumul[n - 1].dxc + trcumul[n - 2].dxc)) -
                            cquad * freqnative * freqnative * (trsmoothed[n - 1].dxc - trcumul[n - 1].dxc) *
                                (1 + nonlinfactor.dxc * fabsf(trsmoothed[n - 1].dxc - trcumul[n - 1].dxc)); // predictor
        // corrector, one iteration must be enough:
        trsmoothed[n].dxc = 2 * trsmoothed[n - 1].dxc - trsmoothed[n - 2].dxc -
                            cdamp * freqnative * 0.5f * (trsmoothed[n].dxc - trsmoothed[n - 2].dxc - trcumul[n].dxc + trcumul[n - 2].dxc) *
                                (1 + 0.5f * nonlinfactor.dxc / freqnative * 0.5f * fabsf(trsmoothed[n].dxc - trsmoothed[n - 2].dxc - trcumul[n].dxc + trcumul[n - 2].dxc)) -
                            cquad * freqnative * freqnative * (trsmoothed[n - 1].dxc - trcumul[n - 1].dxc) *
                                (1 + nonlinfactor.dxc * fabsf(trsmoothed[n - 1].dxc - trcumul[n - 1].dxc));

        // very light (2 frames interval) stabilization of zoom
        trsmoothed[n].dxx = 0.5f * (trcumul[n].dxx + trsmoothed[n - 1].dxx);

        // dxy predictor:
        // double cutoff frequency for rotation
        trsmoothed[n].dxy = 2 * trsmoothed[n - 1].dxy - trsmoothed[n - 2].dxy -
                            cdamp * 2 * freqnative * (trsmoothed[n - 1].dxy - trsmoothed[n - 2].dxy - trcumul[n - 1].dxy + trcumul[n - 2].dxy) *
                                (1 + 0.5f * nonlinfactor.dxy / freqnative * fabsf(trsmoothed[n - 1].dxy - trsmoothed[n - 2].dxy - trcumul[n - 1].dxy + trcumul[n - 2].dxy)) -
                            cquad * 4 * freqnative * freqnative * (trsmoothed[n - 1].dxy - trcumul[n - 1].dxy) *
                                (1 + nonlinfactor.dxy * fabsf(trsmoothed[n - 1].dxy - trcumul[n - 1].dxy)); // predictor
        // corrector, one iteration must be enough:
        trsmoothed[n].dxy = 2 * trsmoothed[n - 1].dxy - trsmoothed[n - 2].dxy -
                            cdamp * 2 * freqnative * 0.5f * (trsmoothed[n].dxy - trsmoothed[n - 2].dxy - trcumul[n].dxy + trcumul[n - 2].dxy) *
                                (1 + 0.5f * nonlinfactor.dxy / freqnative * 0.5f * fabsf(trsmoothed[n].dxy - trsmoothed[n - 2].dxy - trcumul[n].dxy + trcumul[n - 2].dxy)) -
                            cquad * 4 * freqnative * freqnative * (trsmoothed[n - 1].dxy - trcumul[n - 1].dxy) *
                                (1 + nonlinfactor.dxy * fabsf(trsmoothed[n - 1].dxy - trcumul[n - 1].dxy)); // corrector, one iteration must be enough

        // dyx predictor:
        trsmoothed[n].dyx = -trsmoothed[n].dxy * (pixaspect / nfields) * (pixaspect / nfields); // must be consistent

        // dyc predictor:
        trsmoothed[n].dyc = 2 * trsmoothed[n - 1].dyc - trsmoothed[n - 2].dyc -
                            cdamp * freqnative * (trsmoothed[n - 1].dyc - trsmoothed[n - 2].dyc - trcumul[n - 1].dyc + trcumul[n - 2].dyc) *
                                (1 + 0.5f * nonlinfactor.dyc / freqnative * fabsf(trsmoothed[n - 1].dyc - trsmoothed[n - 2].dyc - trcumul[n - 1].dyc + trcumul[n - 2].dyc)) -
                            cquad * freqnative * freqnative * (trsmoothed[n - 1].dyc - trcumul[n - 1].dyc) *
                                (1 + nonlinfactor.dyc * fabsf(trsmoothed[n - 1].dyc - trcumul[n - 1].dyc)); // predictor
        // corrector, one iteration must be enough:
        trsmoothed[n].dyc = 2 * trsmoothed[n - 1].dyc - trsmoothed[n - 2].dyc -
                            cdamp * freqnative * 0.5f * (trsmoothed[n].dyc - trsmoothed[n - 2].dyc - trcumul[n].dyc + trcumul[n - 2].dyc) *
                                (1 + 0.5f * nonlinfactor.dyc / freqnative * 0.5f * fabsf(trsmoothed[n].dyc - trsmoothed[n - 2].dyc - trcumul[n].dyc + trcumul[n - 2].dyc)) -
                            cquad * freqnative * freqnative * (trsmoothed[n - 1].dyc - trcumul[n - 1].dyc) *
                                (1 + nonlinfactor.dyc * fabsf(trsmoothed[n - 1].dyc - trcumul[n - 1].dyc)); // corrector, one iteration must be enough


        // dyy
        trsmoothed[n].dyy = trsmoothed[n].dxx; //must be equal to dxx
    }


    if (addzoom) { // calculate and add adaptive zoom factor to fill borders (for all frames from base to ndest)

        azoom[nbase] = initzoom;
        azoom[nbase + 1] = initzoom;
        azoomsmoothed[nbase] = initzoom;
        azoomsmoothed[nbase + 1] = initzoom;
        for (int n = nbase + 2; n <= ndest; n++) {
            transform trinv, trcur, trtemp;
            // get inverse transform
            inversetransform(&trcumul[n], &trinv);
            // calculate difference between smoothed and original non-smoothed cumulative transform
            sumtransform(&trinv, &trsmoothed[n], &trcur);
            // find adaptive zoom factor
            //                transform2motion (trcur, 1, xcenter, ycenter, pixaspect/nfields, &dxdif, &dydif, &rotdif, &zoomdif);
            azoom[n] = initzoom;
            float azoomtest = 1 + (trcur.dxc + trcur.dxy * ycenter) / xcenter; // xleft
            if (azoomtest < azoom[n])
                azoom[n] = azoomtest;
            azoomtest = 1 - (trcur.dxc + trcur.dxx * width + trcur.dxy * ycenter - width) / xcenter; //xright
            if (azoomtest < azoom[n])
                azoom[n] = azoomtest;
            azoomtest = 1 + (trcur.dyc + trcur.dyx * xcenter) / ycenter; // ytop
            if (azoomtest < azoom[n])
                azoom[n] = azoomtest;
            azoomtest = 1 - (trcur.dyc + trcur.dyx * xcenter + trcur.dyy * height - height) / ycenter; //ybottom
            if (azoomtest < azoom[n])
                azoom[n] = azoomtest;

            // limit zoom to max - added in v.1.4.0
            //                if (fabsf(azoom[n]-1) > fabsf(zoommax)-1)
            //                    azoom[n] =     2 - fabsf(zoommax) ;


            // smooth adaptive zoom
            // zoom time factor
            float zf = 1 / (cutoff * tzoom);
            // predictor
            azoomsmoothed[n] = 2 * azoomsmoothed[n - 1] - azoomsmoothed[n - 2] -
                               zf * cdamp * freqnative * (azoomsmoothed[n - 1] - azoomsmoothed[n - 2] - azoom[n - 1] + azoom[n - 2])
                               //                    *( 1 + 0.5f*nonlinfactor.dxx/freqnative*fabsf(azoomsmoothed[n-1] - azoomsmoothed[n-2] - azoom[n-1] + azoom[n-2])) // disabled in v.1.4.0 for more smooth
                               - zf * zf * cquad * freqnative * freqnative * (azoomsmoothed[n - 1] - azoom[n - 1])
                //                    *( 1 + nonlinfactor.dxx*fabsf(azoomsmoothed[n-1] - azoom[n-1]) )
                ;
            // corrector, one iteration must be enough:
            azoomsmoothed[n] = 2 * azoomsmoothed[n - 1] - azoomsmoothed[n - 2] -
                               zf * cdamp * freqnative * 0.5f * (azoomsmoothed[n] - azoomsmoothed[n - 2] - azoom[n] + azoom[n - 2])
                               //                    *( 1 + 0.5f*nonlinfactor.dxx/freqnative*0.5f*fabsf(azoomsmoothed[n] - azoomsmoothed[n-2] - azoom[n] + azoom[n-2]) )
                               - zf * zf * cquad * freqnative * freqnative * (azoomsmoothed[n - 1] - azoom[n - 1])
                //                    *( 1 + nonlinfactor.dxx*fabsf(azoomsmoothed[n-1] - azoom[n-1]) )
                ;
            zf = zf * 0.7f;                              // slower zoom decreasing
            if (azoomsmoothed[n] > azoomsmoothed[n - 1]) // added in v.1.4.0 for slower zoom decreasing
            {
                // predictor
                azoomsmoothed[n] = 2 * azoomsmoothed[n - 1] - azoomsmoothed[n - 2] -
                                   zf * cdamp * freqnative * (azoomsmoothed[n - 1] - azoomsmoothed[n - 2] - azoom[n - 1] + azoom[n - 2])
                                   //                    *( 1 + 0.5f*nonlinfactor.dxx/freqnative*fabsf(azoomsmoothed[n-1] - azoomsmoothed[n-2] - azoom[n-1] + azoom[n-2]))
                                   - zf * zf * cquad * freqnative * freqnative * (azoomsmoothed[n - 1] - azoom[n - 1])
                    //                    *( 1 + nonlinfactor.dxx*fabsf(azoomsmoothed[n-1] - azoom[n-1]) )
                    ;
                // corrector, one iteration must be enough:
                azoomsmoothed[n] = 2 * azoomsmoothed[n - 1] - azoomsmoothed[n - 2] -
                                   zf * cdamp * freqnative * 0.5f * (azoomsmoothed[n] - azoomsmoothed[n - 2] - azoom[n] + azoom[n - 2])
                                   //                    *( 1 + 0.5f*nonlinfactor.dxx/freqnative*0.5f*fabsf(azoomsmoothed[n] - azoomsmoothed[n-2] - azoom[n] + azoom[n-2]) )
                                   - zf * zf * cquad * freqnative * freqnative * (azoomsmoothed[n - 1] - azoom[n - 1])
                    //                    *( 1 + nonlinfactor.dxx*fabsf(azoomsmoothed[n-1] - azoom[n-1]) )
                    ;
            }
            //            azoomsmoothed[n] = azoomcumul[n]; // debug - no azoom smoothing
            if (azoomsmoothed[n] > 1)
                azoomsmoothed[n] = 1; // not decrease image size
            // make zoom transform
            motion2transform(0, 0, 0, azoomsmoothed[n], pixaspect / nfields, xcenter, ycenter, 1, 1.0, &trtemp); // added in v.1.5.0
            // get non-adaptive image zoom from transform
            //                transform2motion (trsmoothed[n], 1, xcenter, ycenter, pixaspect/nfields, &dxdif, &dydif, &rotdif, &zoomdif); // disabled in v.1.5.0
            // modify transform with adaptive zoom added
            //                motion2transform (dxdif, dydif, rotdif, zoomdif*azoomsmoothed[n], pixaspect/nfields,  xcenter,  ycenter, 1, 1.0, &trsmoothed[n]); // replaced in v.1.5.0 by:
            sumtransform(&trsmoothed[n], &trtemp, &trsmoothed[n]); // added v.1.5.0
        }
    } else {
        transform trtemp;
        motion2transform(0, 0, 0, initzoom, pixaspect / nfields, xcenter, ycenter, 1, 1.0, &trtemp); // added in v.1.7
        sumtransform(&trsmoothed[ndest], &trtemp, &trsmoothed[ndest]);                                 // added v.1.7
    }

    // calculate difference between smoothed and original non-smoothed cumulative tranform
    // it will be used as stabilization values

    transform trinv;
    inversetransform(&trcumul[ndest], &trinv);
    sumtransform(&trinv, &trsmoothed[ndest], ptrdif);
}


static void Average(DepanStabiliseData *d, transform *trcumul, float *azoom, const int nbase, const int ndest, const int nmax, transform *ptrdif) {
    transform trsmoothed;
    const float * const wint = d->wint.data();
    const float pixaspect = d->pixaspect;
    const int nfields = d->nfields;
    const bool addzoom = d->addzoom;
    const float initzoom = d->initzoom;
    const float xcenter = d->xcenter;
    const float ycenter = d->ycenter;
    const int width = d->vi->width;
    const int height = d->vi->height;
    const int winfzsize = d->winfzsize;
    const int winrzsize = d->winrzsize;
    const float * const winfz = d->winfz.data();
    const float * const winrz = d->winrz.data();


    float norm = 0;
    trsmoothed.dxc = 0;
    trsmoothed.dyc = 0;
    trsmoothed.dxy = 0;
    for (int n = nbase; n < ndest; n++) {
        trsmoothed.dxc += trcumul[n].dxc * wint[ndest - n];
        trsmoothed.dyc += trcumul[n].dyc * wint[ndest - n];
        trsmoothed.dxy += trcumul[n].dxy * wint[ndest - n];
        norm += wint[ndest - n];
    }
    for (int n = ndest; n <= nmax; n++) {
        trsmoothed.dxc += trcumul[n].dxc * wint[n - ndest];
        trsmoothed.dyc += trcumul[n].dyc * wint[n - ndest];
        trsmoothed.dxy += trcumul[n].dxy * wint[n - ndest];
        norm += wint[n - ndest];
    }
    trsmoothed.dxc /= norm;
    trsmoothed.dyc /= norm;
    trsmoothed.dxy /= norm;
    trsmoothed.dyx = -trsmoothed.dxy * (pixaspect / nfields) * (pixaspect / nfields); // must be consistent
    norm = 0;
    trsmoothed.dxx = 0;
    for (int n = std::max(nbase, ndest - 1); n < ndest; n++) { // very short interval
        trsmoothed.dxx += trcumul[n].dxx * wint[ndest - n];
        norm += wint[ndest - n];
    }
    for (int n = ndest; n <= std::min(nmax, ndest + 1); n++) {
        trsmoothed.dxx += trcumul[n].dxx * wint[n - ndest];
        norm += wint[n - ndest];
    }
    trsmoothed.dxx /= norm;
    trsmoothed.dyy = trsmoothed.dxx;

    //            motion2transform (0, 0, 0, initzoom, pixaspect/nfields, xcenter, ycenter, 1, 1.0, &trtemp); // added in v.1.7
    //            sumtransform (trsmoothed[ndest],trtemp,  &trsmoothed[ndest]); // added v.1.7

    if (addzoom) { // calculate and add adaptive zoom factor to fill borders (for frames in the window nbasez..nmaxz around ndest)

        int nbasez = std::max(nbase, ndest - winfzsize);
        int nmaxz = std::min(nmax, ndest + winrzsize);
        // symmetrical
        //               nmaxz = ndest + min(nmaxz-ndest, ndest-nbasez);
        //               nbasez = ndest - min(nmaxz-ndest, ndest-nbasez);

        azoom[nbasez] = initzoom;
        for (int n = nbasez + 1; n <= nmaxz; n++) {
            transform trinv, trcur;
            // get inverse transform
            inversetransform(&trcumul[n], &trinv);
            // difference of the cumulative transform with its own inverse (the smoothed variant just below is disabled)
            //                    sumtransform(trinv, trsmoothed[n], &trcur);
            sumtransform(&trinv, &trcumul[n], &trcur);
            // find adaptive zoom factor
            azoom[n] = initzoom;
            float azoomtest = 1 + (trcur.dxc + trcur.dxy * ycenter) / xcenter; // xleft
            if (azoomtest < azoom[n])
                azoom[n] = azoomtest;
            azoomtest = 1 - (trcur.dxc + trcur.dxx * width + trcur.dxy * ycenter - width) / xcenter; //xright
            if (azoomtest < azoom[n])
                azoom[n] = azoomtest;
            azoomtest = 1 + (trcur.dyc + trcur.dyx * xcenter) / ycenter; // ytop
            if (azoomtest < azoom[n])
                azoom[n] = azoomtest;
            azoomtest = 1 - (trcur.dyc + trcur.dyx * xcenter + trcur.dyy * height - height) / ycenter; //ybottom
            if (azoomtest < azoom[n])
                azoom[n] = azoomtest;
            //                    azoom[n] = initzoom;
        }

        // smooth adaptive zoom
        // zoom time factor
        //                    zf = 1/(cutoff*tzoom);

        norm = 0;
        float azoomsmoothed = 0.0;
        for (int n = nbasez; n < ndest; n++) {
            azoomsmoothed += azoom[n] * winfz[ndest - n]; // fall
            norm += winfz[ndest - n];
        }
        for (int n = ndest; n <= nmaxz; n++) {
            azoomsmoothed += azoom[n] * winrz[n - ndest]; // rize
            norm += winrz[n - ndest];
        }
        azoomsmoothed /= norm;
        //                    zf = zf*0.7f; // slower zoom decreasing
        //                    if (azoomsmoothed[n] > azoomsmoothed[n-1]) // added in v.1.4.0 for slower zoom decreasing
        //                    {
        //                    }

        //azoomsmoothed[ndest] = azoom[ndest]; // debug - no azoom smoothing

        if (azoomsmoothed > 1)
            azoomsmoothed = 1; // not decrease image size
        // make zoom transform
        transform trtemp;
        motion2transform(0, 0, 0, azoomsmoothed, pixaspect / nfields, xcenter, ycenter, 1, 1.0, &trtemp);
        sumtransform(&trsmoothed, &trtemp, &trsmoothed);

        //            }
    } else // no addzoom
    {
        transform trtemp;
        motion2transform(0, 0, 0, initzoom, pixaspect / nfields, xcenter, ycenter, 1, 1.0, &trtemp); // added in v.1.7
        sumtransform(&trsmoothed, &trtemp, &trsmoothed); // added v.1.7
    }
    // calculate difference between smoothed and original non-smoothed cumulative tranform
    // it will be used as stabilization values

    transform trinv;
    inversetransform(&trcumul[ndest], &trinv);
    sumtransform(&trinv, &trsmoothed, ptrdif);
}


static void InertialLimit(DepanStabiliseData *d, float *dxdif, float *dydif, float *zoomdif, float *rotdif, int ndest, int *nbase) {
    const float initzoom = d->initzoom;
    const float dxmax = d->dxmax;
    const float dymax = d->dymax;
    const float zoommax = d->zoommax;
    const float rotmax = d->rotmax;

    // limit max motion corrections
    if (!(isfinite(*dxdif))) // check added in v.1.1.3
    {                       // infinite or NAN
        *dxdif = 0;
        *dydif = 0;
        *zoomdif = initzoom;
        *rotdif = 0;
        *nbase = ndest;
    } else if (fabsf(*dxdif) > fabsf(dxmax)) {
        if (dxmax >= 0) {
            *dxdif = *dxdif >= 0 ? sqrtf(*dxdif * dxmax) : -sqrtf(-*dxdif * dxmax); // soft limit v.1.8.2
            // The "soft limit" is a geometric mean, not a clamp: it only halves the exponent, so a
            // diverged smoother (see WarnIfDiverged) still gets through at ~1e16. Second pass plus a
            // hard bound, ported from pinterf/mvtools DePan (its ADD_SAFETY block). Note that fork
            // writes `*dxdif >= 0 ? dxmax : dxmax` on the final line, which flips the sign of large
            // negative corrections; the negative branch is -dxmax here.
            if (fabsf(*dxdif) > fabsf(dxmax))
                *dxdif = *dxdif >= 0 ? sqrtf(*dxdif * dxmax) : -sqrtf(-*dxdif * dxmax);
            if (fabsf(*dxdif) > fabsf(dxmax * 1.5f))
                *dxdif = *dxdif >= 0 ? dxmax : -dxmax;
        } else {
            *dxdif = 0;
            *dydif = 0;
            *zoomdif = initzoom;
            *rotdif = 0;
            *nbase = ndest;
        }
    }

    if (!(isfinite(*dydif))) { // infinite or NAN
        *dxdif = 0;
        *dydif = 0;
        *zoomdif = initzoom;
        *rotdif = 0;
        *nbase = ndest;
    } else if (fabsf(*dydif) > fabsf(dymax)) {
        if (dymax >= 0) {
            *dydif = *dydif >= 0 ? sqrtf(*dydif * dymax) : -sqrtf(-*dydif * dymax); // soft limit v.1.8.2
            // see the dxdif branch above
            if (fabsf(*dydif) > fabsf(dymax))
                *dydif = *dydif >= 0 ? sqrtf(*dydif * dymax) : -sqrtf(-*dydif * dymax);
            if (fabsf(*dydif) > fabsf(dymax * 1.5f))
                *dydif = *dydif >= 0 ? dymax : -dymax;
        } else {
            *dxdif = 0;
            *dydif = 0;
            *zoomdif = initzoom;
            *rotdif = 0;
            *nbase = ndest;
        }
    }

    if (!(isfinite(*zoomdif))) { // infinite or NAN
        *dxdif = 0;
        *dydif = 0;
        *zoomdif = initzoom;
        *rotdif = 0;
        *nbase = ndest;
    } else if (fabsf(*zoomdif - 1) > fabsf(zoommax) - 1) {
        if (zoommax >= 0) {
            *zoomdif = *zoomdif >= 1 ? 1 + sqrtf(fabsf(*zoomdif - 1) * fabsf(zoommax - 1)) : 1 - sqrtf(fabsf(*zoomdif - 1) * fabsf(zoommax - 1)); // soft limit v.1.8.2
        } else {
            *dxdif = 0;
            *dydif = 0;
            *zoomdif = initzoom;
            *rotdif = 0;
            *nbase = ndest;
        }
    }

    if (!(isfinite(*rotdif))) { // infinite or NAN
        *dxdif = 0;
        *dydif = 0;
        *zoomdif = initzoom;
        *rotdif = 0;
        *nbase = ndest;
    } else if (fabsf(*rotdif) > fabsf(rotmax)) {
        if (rotmax >= 0) {
            *rotdif = *rotdif >= 0 ? sqrtf(*rotdif * rotmax) : -sqrtf(-*rotdif * rotmax); // soft limit v.1.8.2
        } else {
            *dxdif = 0;
            *dydif = 0;
            *zoomdif = initzoom;
            *rotdif = 0;
            *nbase = ndest;
        }
    }
}


static int getDepanProps(std::optional<MotionData> &slot, const VSFrame *frame, VSFrameContext *frameCtx, const VSAPI *vsapi) {
    MotionData m;
    if (!mapGetMotion(m, vsapi->getFramePropertiesRO(frame), vsapi)) {
        vsapi->setFilterError("DepanStabilise: required frame properties not found in data clip.", frameCtx);
        return 0;
    }
    slot = m;
    return 1;
}


static void compensateFrame(const VSFrame *src, VSFrame *dst, DepanStabiliseData *d, int notfilled, const transform *trdif, int *work2width4356, const VSAPI *vsapi) {
    int border[3] = { -1, -1, -1 };
    int blur[3] = { d->blur, d->blur, d->blur };
    transform tr[3];

    if (notfilled) {
        border[0] = 0;
        border[1] = border[2] = 1 << (d->vi->format.bitsPerSample - 1);
    }

    tr[0] = tr[1] = *trdif;
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

        // move src frame plane by vector to partially motion compensated position
        if (d->vi->format.bytesPerSample == 1) {
            if (d->subpixel == 0)
                compensate_plane_nearest<uint8_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror * notfilled, border[plane], work2width4356, blur[plane]);
            else if (d->subpixel == 1)
                compensate_plane_bilinear<uint8_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror * notfilled, border[plane], work2width4356, blur[plane]);
            else
                compensate_plane_bicubic<uint8_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror * notfilled, border[plane], work2width4356, blur[plane], d->pixel_max);
        } else {
            if (d->subpixel == 0)
                compensate_plane_nearest<uint16_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror * notfilled, border[plane], work2width4356, blur[plane]);
            else if (d->subpixel == 1)
                compensate_plane_bilinear<uint16_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror * notfilled, border[plane], work2width4356, blur[plane]);
            else
                compensate_plane_bicubic<uint16_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror * notfilled, border[plane], work2width4356, blur[plane], d->pixel_max);
        }
    }
}


static void fillBorderPrev(VSFrame *dst, DepanStabiliseData *d, int nbase, int ndest, const transform *trdif, int *work2width4356, int *notfilled, VSFrameContext *frameCtx, const VSAPI *vsapi) {
    int nprev = ndest - d->prev; // get prev frame number
    if (nprev < nbase)
        nprev = nbase; //  prev distance not exceed base

    int nprevbest = nprev;
    float dabsmin = 10000;

    transform tr[3];
    tr[0] = *trdif; // luma transform

    for (int n = ndest - 1; n >= nprev; n--) { // summary inverse transform
        transform trcur;
        motion2transform(d->motion[n + 1]->dx, d->motion[n + 1]->dy, d->motion[n + 1]->rot, d->motion[n + 1]->zoom, d->pixaspect / d->nfields, d->xcenter, d->ycenter, 1, 1.0f, &trcur);
        sumtransform(&tr[0], &trcur, &tr[0]);
        float dxt1, dyt1, rott1, zoomt1;
        transform2motion(&tr[0], 1, d->xcenter, d->ycenter, d->pixaspect / d->nfields, &dxt1, &dyt1, &rott1, &zoomt1);
        if ((fabsf(dxt1) + fabsf(dyt1) + ndest - n) < dabsmin) { // most centered and nearest
            dabsmin = fabsf(dxt1) + fabsf(dyt1) + ndest - n;
            nprevbest = n;
        }
    }

    // get original previous source frame
    const VSFrame *src = vsapi->getFrameFilter(nprevbest, d->clip, frameCtx);

    int border[3] = { 0, 1 << (d->vi->format.bitsPerSample - 1), border[1] };
    int blur[3] = { d->blur, d->blur, d->blur };

    // chroma planes use the same (accumulated) luma transform, scaled below for subsampling
    tr[1] = tr[0];

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

        if (d->vi->format.bytesPerSample == 1)
            compensate_plane_nearest<uint8_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror, border[plane], work2width4356, blur[plane]);
        else
            compensate_plane_nearest<uint16_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror, border[plane], work2width4356, blur[plane]);
    }

    *notfilled = 0; // mark as FILLED

    vsapi->freeFrame(src);
}


static int fillBorderNext(VSFrame *dst, DepanStabiliseData *d, int ndest, const transform *trdif, int *work2width4356, int *notfilled, VSFrameContext *frameCtx, const VSAPI *vsapi) {
    int nnext = ndest + d->next;
    if (nnext >= d->vi->numFrames)
        nnext = d->vi->numFrames - 1;
    int nnextbest = nnext;
    float dabsmin = 1000;

    transform tr[3];
    tr[0] = *trdif; // luma transform for current frame

    // get motion info about the next frames in interval from dest+1 to nnext (forward order)
    {
        std::scoped_lock guard(d->motion_mutex);

        for (int n = ndest + 1; n <= nnext; n++) {
            if (!d->motion[n].has_value()) {
                const VSFrame *dataframe = vsapi->getFrameFilter(n, d->data, frameCtx);
                if (!getDepanProps(d->motion[n], dataframe, frameCtx, vsapi)) {
                    vsapi->freeFrame(dataframe);
                    return 0;
                }
                vsapi->freeFrame(dataframe);
            }
        }
    }

    for (int n = ndest + 1; n <= nnext; n++) {
        if (!d->motion[n]->badMotion) { //if good
            transform trcur, trinv;
            motion2transform(d->motion[n]->dx, d->motion[n]->dy, d->motion[n]->rot, d->motion[n]->zoom, d->pixaspect / d->nfields, d->xcenter, d->ycenter, 1, 1.0f, &trcur);
            inversetransform(&trcur, &trinv);
            sumtransform(&trinv, &tr[0], &tr[0]);
            float dxt1, dyt1, rott1, zoomt1;
            transform2motion(&tr[0], 1, d->xcenter, d->ycenter, d->pixaspect / d->nfields, &dxt1, &dyt1, &rott1, &zoomt1);
            if ((fabsf(dxt1) + fabsf(dyt1) + n - ndest) < dabsmin) { // most centered and nearest
                dabsmin = fabsf(dxt1) + fabsf(dyt1) + n - ndest;
                nnextbest = n;
            }
        } else { // bad
            nnextbest = n - 1; // limit fill frame to last good
            break;
        }
    }

    // get original previous source frame
    const VSFrame *src = vsapi->getFrameFilter(nnextbest, d->clip, frameCtx);

    int border[3] = { -1, -1, -1 };
    int blur[3] = { d->blur, d->blur, d->blur };

    // chroma planes use the same (accumulated) luma transform, scaled below for subsampling
    tr[1] = tr[0];

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

    if (*notfilled) {
        border[0] = 0;
        border[1] = border[2] = 1 << (d->vi->format.bitsPerSample - 1);
    }

    for (int plane = 0; plane < d->vi->format.numPlanes; plane++) {
        const uint8_t *srcp = vsapi->getReadPtr(src, plane);
        int src_width = vsapi->getFrameWidth(src, plane);
        int src_height = vsapi->getFrameHeight(src, plane);
        ptrdiff_t src_pitch = vsapi->getStride(src, plane);

        uint8_t *dstp = vsapi->getWritePtr(dst, plane);

        // move src frame plane by vector to partially motion compensated position
        if (d->vi->format.bytesPerSample == 1)
            compensate_plane_nearest<uint8_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror * *notfilled, border[plane], work2width4356, blur[plane]);
        else
            compensate_plane_nearest<uint16_t>(dstp, srcp, src_pitch, src_width, src_height, &tr[plane], d->mirror * *notfilled, border[plane], work2width4356, blur[plane]);
    }

    *notfilled = 0; // mark as filled

    vsapi->freeFrame(src);

    return 1;
}


// show text info on frame image
static void attachInfo(VSFrame *dst, int nbase, int ndest, float dxdif, float dydif, float rotdif, float zoomdif, const VSAPI *vsapi) {
#define INFO_SIZE 128
    char info[INFO_SIZE + 1] = { 0 };

    snprintf(info, INFO_SIZE, "frame=%d %s=%d dx=%.2f dy=%.2f rot=%.3f zoom=%.5f", ndest, nbase == ndest ? "BASE!" : "base ", nbase, dxdif, dydif, rotdif, zoomdif);
#undef INFO_SIZE

    VSMap *dst_props = vsapi->getFramePropertiesRW(dst);
    vsapi->mapSetData(dst_props, prop_DepanStabilise_info, info, -1, dtUtf8, maReplace);
}


static const VSFrame *VS_CC depanStabiliseGetFrame0(int ndest, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    DepanStabiliseData *d = reinterpret_cast<DepanStabiliseData *>(instanceData);

    // How far back the "almost full stabilisation" window reaches. A small cutoff makes this
    // enormous (fps=50, cutoff=0.5 gives 1000 frames = 20 seconds), which is both slow and gives the
    // inertial recurrence a very long run in which to go unstable. Capped at 5 seconds, ported from
    // pinterf/mvtools DePan. Also keeps the float in int range for the cast below.
    float framesback = 10 * d->fps / d->cutoff;
    if (!(framesback < d->fps * 5.0f)) // NaN-safe: also catches non-finite from a tiny cutoff
        framesback = d->fps * 5.0f;

    int nbase = (int)(ndest - framesback);
    if (nbase < 0)
        nbase = 0;

    if (activationReason == arInitial) {
        int nprev = std::max(nbase, ndest - d->prev);

        std::scoped_lock guard(d->motion_mutex);

        for (int i = nbase; i <= ndest; i++) {
            if (!d->motion[i].has_value())
                vsapi->requestFrameFilter(i, d->data, frameCtx);
            if (d->prev && i >= nprev)
                vsapi->requestFrameFilter(i, d->clip, frameCtx);
        }

        vsapi->requestFrameFilter(ndest, d->clip, frameCtx);

        if (d->next) {
            for (int i = ndest + 1; i <= std::min(ndest + d->next, d->vi->numFrames - 1); i++) {
                if (!d->motion[i].has_value())
                    vsapi->requestFrameFilter(i, d->data, frameCtx);
                vsapi->requestFrameFilter(i, d->clip, frameCtx);
            }
        }
    } else if (activationReason == arAllFramesReady) {
        // get motion info about frames in interval from begin source to dest in reverse order
        {
            std::scoped_lock guard(d->motion_mutex);

            for (int n = ndest; n >= nbase; n--) {
                if (!d->motion[n].has_value()) {
                    const VSFrame *dataframe = vsapi->getFrameFilter(n, d->data, frameCtx);
                    if (!getDepanProps(d->motion[n], dataframe, frameCtx, vsapi)) {
                        vsapi->freeFrame(dataframe);
                        return nullptr;
                    }
                    vsapi->freeFrame(dataframe);
                }

                if (d->motion[n]->badMotion) {
                    if (n > nbase)
                        nbase = n;
                    break; // bad frame becomes the lower bound (excluded by the nbase+1 loop start)
                }
            }
        }


        const VSFrame *src = nullptr;
        VSFrame *dst = nullptr;

        try {
            float dxdif, dydif, zoomdif, rotdif;
            transform trdif;

            if (nbase == ndest) { // we are at new scene start,
                motion2transform(0, 0, 0, d->initzoom, d->pixaspect / d->nfields, d->xcenter, d->ycenter, 1, 1.0f, &trdif);
            } else { // prepare stabilization data by estimation and smoothing of cumulative motion

                // cumulative transform (position) for all sequence from base

                size_t elements = ndest - nbase + 1;

                std::vector<transform> trcumul(elements);
                std::vector<transform> trsmoothed(elements);
                std::vector<float> azoom(elements);
                std::vector<float> azoomsmoothed(elements);

                // base as null
                trcumul[0].setNull();

                // get cumulative transforms from base to ndest
                for (int n = nbase + 1; n <= ndest; n++) {
                    transform trcur;

                    motion2transform(d->motion[n]->dx, d->motion[n]->dy, d->motion[n]->rot, d->motion[n]->zoom, d->pixaspect / d->nfields, d->xcenter, d->ycenter, 1, 1.0f, &trcur);
                    sumtransform(&trcumul[n - nbase - 1], &trcur, &trcumul[n - nbase]);
                }

                Inertial(d, trcumul.data() - nbase, trsmoothed.data() - nbase, azoom.data() - nbase, azoomsmoothed.data() - nbase, nbase, ndest, &trdif);

                // summary motion from summary transform
                transform2motion(&trdif, 1, d->xcenter, d->ycenter, d->pixaspect / d->nfields, &dxdif, &dydif, &rotdif, &zoomdif);
                // fit last - decrease motion correction near end of clip - added in v.1.2.0

                if (d->vi->numFrames < d->fitlast + ndest + 1) {
                    float endFactor = ((float)(d->vi->numFrames - ndest - 1)) / d->fitlast; // decrease factor
                    dxdif *= endFactor;
                    dydif *= endFactor;
                    rotdif *= endFactor;
                    zoomdif = d->initzoom + (zoomdif - d->initzoom) * endFactor;
                }

                InertialLimit(d, &dxdif, &dydif, &zoomdif, &rotdif, ndest, &nbase);

                // summary motion from summary transform after max correction
                motion2transform(dxdif, dydif, rotdif, zoomdif, d->pixaspect / d->nfields, d->xcenter, d->ycenter, 1, 1.0f, &trdif);
            }

            // ---------------------------------------------------------------------------
            WarnIfDiverged(d, trdif, ndest, core, vsapi);

            src = vsapi->getFrameFilter(ndest, d->clip, frameCtx);
            dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core);

            //--------------------------------------------------------------------------
            // Ready to make motion stabilization,

            std::vector<int> work2width4356(2 * d->vi->width + 4356); // work

            // --------------------------------------------------------------------
            // use some previous frame to fill borders
            int notfilled = 1; // init as not filled (borders by neighbor frames)

            if (d->prev > 0)
                fillBorderPrev(dst, d, nbase, ndest, &trdif, work2width4356.data(), &notfilled, frameCtx, vsapi);

            // use next frame to fill borders
            if (d->next > 0) {
                if (!fillBorderNext(dst, d, ndest, &trdif, work2width4356.data(), &notfilled, frameCtx, vsapi)) {
                    vsapi->freeFrame(dst);
                    vsapi->freeFrame(src);
                    return nullptr;
                }
            }

            compensateFrame(src, dst, d, notfilled, &trdif, work2width4356.data(), vsapi);

            vsapi->freeFrame(src);
            src = nullptr;

            if (d->info) {
                transform2motion(&trdif, 1, d->xcenter, d->ycenter, d->pixaspect / d->nfields, &dxdif, &dydif, &rotdif, &zoomdif);

                attachInfo(dst, nbase, ndest, dxdif, dydif, rotdif, zoomdif, vsapi);
            }

            return dst;
        } catch (const std::exception &e) {
            vsapi->setFilterError(("DepanStabilise: " + std::string(e.what())).c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }
    }

    return nullptr;
}


static const VSFrame *VS_CC depanStabiliseGetFrame1(int ndest, int activationReason, void *instanceData, [[maybe_unused]] void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) noexcept {
    DepanStabiliseData *d = reinterpret_cast<DepanStabiliseData *>(instanceData);

    int nbase = std::max(0, ndest - d->radius);
    int nmax = std::min(ndest + d->radius, d->vi->numFrames - 1);

    if (activationReason == arInitial) {
        int nprev = std::max(nbase, ndest - d->prev);

        std::scoped_lock guard(d->motion_mutex);

        for (int i = nbase; i <= ndest; i++) {
            if (!d->motion[i].has_value())
                vsapi->requestFrameFilter(i, d->data, frameCtx);
            if (d->prev && i >= nprev)
                vsapi->requestFrameFilter(i, d->clip, frameCtx);
        }

        vsapi->requestFrameFilter(ndest, d->clip, frameCtx);

        int nnext = std::min(ndest + d->next, d->vi->numFrames - 1);

        if (nnext < nmax) {
            for (int i = ndest + 1; i <= nnext; i++) {
                if (!d->motion[i].has_value())
                    vsapi->requestFrameFilter(i, d->data, frameCtx);
                if (d->next)
                    vsapi->requestFrameFilter(i, d->clip, frameCtx);
            }

            for (int i = nnext + 1; i <= nmax; i++) {
                if (!d->motion[i].has_value())
                    vsapi->requestFrameFilter(i, d->data, frameCtx);
            }
        } else {
            for (int i = ndest + 1; i <= nmax; i++) {
                if (!d->motion[i].has_value())
                    vsapi->requestFrameFilter(i, d->data, frameCtx);
                if (d->next)
                    vsapi->requestFrameFilter(i, d->clip, frameCtx);
            }

            if (d->next) {
                for (int i = nmax + 1; i <= nnext; i++) {
                    if (!d->motion[i].has_value())
                        vsapi->requestFrameFilter(i, d->data, frameCtx);
                    vsapi->requestFrameFilter(i, d->clip, frameCtx);
                }
            }
        }
    } else if (activationReason == arAllFramesReady) {
        // get motion info about frames in interval from begin source to dest in reverse order
        {
            std::scoped_lock guard(d->motion_mutex);

            for (int n = ndest; n >= nbase; n--) {
                if (!d->motion[n].has_value()) {
                    const VSFrame *dataframe = vsapi->getFrameFilter(n, d->data, frameCtx);
                    if (!getDepanProps(d->motion[n], dataframe, frameCtx, vsapi)) {
                        vsapi->freeFrame(dataframe);
                        return nullptr;
                    }
                    vsapi->freeFrame(dataframe);
                }

                if (d->motion[n]->badMotion) {
                    if (n > nbase)
                        nbase = n;
                    break; // bad frame becomes the lower bound (excluded by the nbase+1 loop start)
                }
            }


            for (int n = ndest + 1; n <= nmax; n++) {
                if (!d->motion[n].has_value()) {
                    const VSFrame *dataframe = vsapi->getFrameFilter(n, d->data, frameCtx);
                    if (!getDepanProps(d->motion[n], dataframe, frameCtx, vsapi)) {
                        vsapi->freeFrame(dataframe);
                        return nullptr;
                    }
                    vsapi->freeFrame(dataframe);
                }

                if (d->motion[n]->badMotion) {
                    // exclude the bad frame from the upper bound unconditionally, so a bad
                    // frame landing exactly on nmax cannot leak a NaN into the cumulative sum
                    nmax = std::max(n - 1, ndest);
                    break;
                }
            }
        }


        const VSFrame *src = nullptr;
        VSFrame *dst = nullptr;

        try {
            int smaller_distance = std::min(nmax - ndest, ndest - nbase);
            nmax = ndest + smaller_distance;
            nbase = ndest - smaller_distance;


            // cumulative transform (position) for all sequence from base

            size_t elements = nmax - nbase + 1;

            std::vector<transform> trcumul(elements);
            std::vector<float> azoom(elements);

            // base as null
            trcumul[0].setNull();

            float dxdif, dydif, zoomdif, rotdif;
            transform trdif;

            // get cumulative transforms from base to ndest
            for (int n = nbase + 1; n <= nmax; n++) {
                transform trcur;

                motion2transform(d->motion[n]->dx, d->motion[n]->dy, d->motion[n]->rot, d->motion[n]->zoom, d->pixaspect / d->nfields, d->xcenter, d->ycenter, 1, 1.0f, &trcur);
                sumtransform(&trcumul[n - nbase - 1], &trcur, &trcumul[n - nbase]);
            }

            Average(d, trcumul.data() - nbase, azoom.data() - nbase, nbase, ndest, nmax, &trdif);

            // summary motion from summary transform
            transform2motion(&trdif, 1, d->xcenter, d->ycenter, d->pixaspect / d->nfields, &dxdif, &dydif, &rotdif, &zoomdif);

            // summary motion from summary transform after max correction
            motion2transform(dxdif, dydif, rotdif, zoomdif, d->pixaspect / d->nfields, d->xcenter, d->ycenter, 1, 1.0f, &trdif);


            // ---------------------------------------------------------------------------
            WarnIfDiverged(d, trdif, ndest, core, vsapi);

            src = vsapi->getFrameFilter(ndest, d->clip, frameCtx);
            dst = vsapi->newVideoFrame(&d->vi->format, d->vi->width, d->vi->height, src, core);

            //--------------------------------------------------------------------------
            // Ready to make motion stabilization,

            std::vector<int> work2width4356(2 * d->vi->width + 4356); // work

            // --------------------------------------------------------------------
            // use some previous frame to fill borders
            int notfilled = 1; // init as not filled (borders by neighbor frames)

            if (d->prev > 0)
                fillBorderPrev(dst, d, nbase, ndest, &trdif, work2width4356.data(), &notfilled, frameCtx, vsapi);

            // use next frame to fill borders
            if (d->next > 0) {
                if (!fillBorderNext(dst, d, ndest, &trdif, work2width4356.data(), &notfilled, frameCtx, vsapi)) {
                    vsapi->freeFrame(dst);
                    vsapi->freeFrame(src);
                    return nullptr;
                }
            }

            compensateFrame(src, dst, d, notfilled, &trdif, work2width4356.data(), vsapi);

            vsapi->freeFrame(src);
            src = nullptr;

            if (d->info) {
                transform2motion(&trdif, 1, d->xcenter, d->ycenter, d->pixaspect / d->nfields, &dxdif, &dydif, &rotdif, &zoomdif);

                attachInfo(dst, nbase, ndest, dxdif, dydif, rotdif, zoomdif, vsapi);
            }

            return dst;
        } catch (const std::exception &e) {
            vsapi->setFilterError(("DepanStabilise: " + std::string(e.what())).c_str(), frameCtx);
            vsapi->freeFrame(src);
            vsapi->freeFrame(dst);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC depanStabiliseCreate(const VSMap *in, VSMap *out, [[maybe_unused]] void *userData, VSCore *core, const VSAPI *vsapi) noexcept {
    std::unique_ptr<DepanStabiliseData> d = std::make_unique<DepanStabiliseData>(vsapi);

    int err;

    d->cutoff = vsapi->mapGetFloatSaturated(in, "cutoff", 0, &err);
    if (err)
        d->cutoff = 1.0f;

    d->damping = vsapi->mapGetFloatSaturated(in, "damping", 0, &err);
    if (err)
        d->damping = 0.9f;

    d->initzoom = vsapi->mapGetFloatSaturated(in, "initzoom", 0, &err);
    if (err)
        d->initzoom = 1.0f;

    d->addzoom = !!vsapi->mapGetInt(in, "addzoom", 0, &err);

    d->prev = vsapi->mapGetIntSaturated(in, "prev", 0, &err);

    d->next = vsapi->mapGetIntSaturated(in, "next", 0, &err);

    d->mirror = vsapi->mapGetIntSaturated(in, "mirror", 0, &err);

    d->blur = vsapi->mapGetIntSaturated(in, "blur", 0, &err);

    d->dxmax = vsapi->mapGetFloatSaturated(in, "dxmax", 0, &err);
    if (err)
        d->dxmax = 60.0f;

    d->dymax = vsapi->mapGetFloatSaturated(in, "dymax", 0, &err);
    if (err)
        d->dymax = 30.0f;

    d->zoommax = vsapi->mapGetFloatSaturated(in, "zoommax", 0, &err);
    if (err)
        d->zoommax = 1.05f;

    d->rotmax = vsapi->mapGetFloatSaturated(in, "rotmax", 0, &err);
    if (err)
        d->rotmax = 1.0f;

    d->subpixel = vsapi->mapGetIntSaturated(in, "subpixel", 0, &err);
    if (err)
        d->subpixel = 2;

    d->pixaspect = vsapi->mapGetFloatSaturated(in, "pixaspect", 0, &err);
    if (err)
        d->pixaspect = 1.0f;

    d->fitlast = vsapi->mapGetIntSaturated(in, "fitlast", 0, &err);

    d->tzoom = vsapi->mapGetFloatSaturated(in, "tzoom", 0, &err);
    if (err)
        d->tzoom = 3.0f;

    d->info = !!vsapi->mapGetInt(in, "info", 0, &err);

    d->method = vsapi->mapGetIntSaturated(in, "method", 0, &err);

    d->fields = !!vsapi->mapGetInt(in, "fields", 0, &err);


    try {
        // sanity checks
        if (!std::isfinite(d->cutoff) || d->cutoff <= 0.0f)
            throw std::runtime_error("cutoff must be a finite value greater than 0");

        if (d->prev < 0)
            throw std::runtime_error("prev must not be negative");

        if (d->next < 0)
            throw std::runtime_error("next must not be negative");

        if (d->subpixel < 0 || d->subpixel > 2)
            throw std::runtime_error("subpixel must be between 0 and 2 (inclusive)");

        if (!std::isfinite(d->pixaspect) || d->pixaspect <= 0.0f)
            throw std::runtime_error("pixaspect must be a finite value greater than 0");

        if (!std::isfinite(d->damping))
            throw std::runtime_error("damping must be a finite value");

        if (!std::isfinite(d->initzoom) || d->initzoom <= 0.0f)
            throw std::runtime_error("initzoom must be a finite value greater than 0");

        if (!std::isfinite(d->dxmax) || !std::isfinite(d->dymax) || !std::isfinite(d->zoommax) || !std::isfinite(d->rotmax))
            throw std::runtime_error("dxmax, dymax, zoommax and rotmax must be finite values");

        if (!std::isfinite(d->tzoom) || d->tzoom < 0.0f)
            throw std::runtime_error("tzoom must be a finite non-negative value");

        if (d->mirror < 0 || d->mirror > 15)
            throw std::runtime_error("mirror must be between 0 and 15 (inclusive)");

        if (d->blur < 0)
            throw std::runtime_error("blur must not be negative");

        if (d->method < 0 || d->method > 1)
            throw std::runtime_error("method must be between 0 and 1 (inclusive)");


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

        if (d->vi->fpsNum == 0 || d->vi->fpsDen == 0)
            throw std::runtime_error("clip must have known frame rate");

        // tzoom is in seconds and the adaptive zoom is smoothed over fps * tzoom / 4 frames. Less than one frame makes
        // method 1 average over an empty window (a NaN zoom) and method 0's zoom time factor 1 / (cutoff * tzoom) blow up.
        if (d->addzoom && !(static_cast<double>(d->tzoom) * d->vi->fpsNum / d->vi->fpsDen >= 4.0))
            throw std::runtime_error("with addzoom, tzoom must be at least 4 / fps seconds, so the zoom is smoothed over at least one frame");

        d->data = vsapi->mapGetNode(in, "data", 0, nullptr);

        CheckClipLength(d->data, "data", d->vi->numFrames, "clip", vsapi);

        d->zoommax = d->zoommax > 0 ? std::max(d->zoommax, d->initzoom) : -std::max(-d->zoommax, d->initzoom);

        // correction for fieldbased
        if (d->fields)
            d->nfields = 2;
        else
            d->nfields = 1;


        d->motion.resize(d->vi->numFrames);
    } catch (const std::exception &e) {
        vsapi->mapSetError(out, ("DepanStabilise: " + std::string(e.what())).c_str());
        return;
    }

    // frame 0 is a forced scene start: present (known) and bad; the rest stay nullopt (not fetched)
    d->motion[0] = MotionData{ 0.0f, 0.0f, 0.0f, 1.0f, true };


    // prepare coefficients for inertial motion smoothing filter

    //  relative frequency lambda at half height of response
    float lambda = sqrtf(1 + 6 * d->damping * d->damping + sqrtf((1 + 6 * d->damping * d->damping) * (1 + 6 * d->damping * d->damping) + 3));
    // native oscillation frequency
    d->freqnative = d->cutoff / lambda;
    // frames per secomd
    d->fps = (float)d->vi->fpsNum / d->vi->fpsDen;

    // approximate factor values for nonlinear members as half of max
    if (d->dxmax != 0.0f) {
        d->nonlinfactor.dxc = 5 / fabsf(d->dxmax);
    } else {
        d->nonlinfactor.dxc = 0;
    }
    if (fabsf(d->zoommax) != 1.0f) {
        d->nonlinfactor.dxx = 5 / (fabsf(d->zoommax) - 1);
        d->nonlinfactor.dyy = 5 / (fabsf(d->zoommax) - 1);
    } else {
        d->nonlinfactor.dxx = 0;
        d->nonlinfactor.dyy = 0;
    }
    if (d->dymax != 0.0f) {
        d->nonlinfactor.dyc = 5 / fabsf(d->dymax);
    } else {
        d->nonlinfactor.dyc = 0;
    }
    if (d->rotmax != 0.0f) {
        d->nonlinfactor.dxy = 5 / fabsf(d->rotmax);
        d->nonlinfactor.dyx = 5 / fabsf(d->rotmax);
    } else {
        d->nonlinfactor.dxy = 0;
        d->nonlinfactor.dyx = 0;
    }


    d->initzoom = 1 / d->initzoom;

    // FIXME, investigate wintsize more, maybe max isn't the right solution here
    d->wintsize = std::max(1, (int)(d->fps / (4 * d->cutoff)));
    d->radius = d->wintsize;
    d->wint.resize(d->wintsize + 1);

    constexpr float PI = std::numbers::pi_v<float>;
    for (int i = 0; i < d->wintsize; i++)
        d->wint[i] = cosf(i * 0.5f * PI / d->wintsize);
    d->wint[d->wintsize] = 0;

    d->winrz.resize(d->wintsize + 1);
    d->winfz.resize(d->wintsize + 1);
    // at least one frame even when float rounding puts fps * tzoom just under 4, and clamped before the conversion as a
    // huge tzoom would overflow it (tzoom is unused without addzoom)
    d->winrzsize = d->winfzsize = static_cast<int>(std::clamp(d->fps * d->tzoom / 4, 1.0f, static_cast<float>(d->wintsize)));
    for (int i = 0; i < d->winrzsize; i++)
        d->winrz[i] = cosf(i * 0.5f * PI / d->winrzsize);
    for (int i = d->winrzsize; i <= d->wintsize; i++)
        d->winrz[i] = 0;
    for (int i = 0; i < d->winfzsize; i++)
        d->winfz[i] = cosf(i * 0.5f * PI / d->winfzsize);
    for (int i = d->winfzsize; i <= d->wintsize; i++)
        d->winfz[i] = 0;

    d->xcenter = d->vi->width / 2.0f; // center of frame
    d->ycenter = d->vi->height / 2.0f;

    d->pixel_max = (1 << d->vi->format.bitsPerSample) - 1;


    VSFilterGetFrame getframe_functions[2] = {
        depanStabiliseGetFrame0,
        depanStabiliseGetFrame1,
    };

    VSFilterDependency deps[2] = {
        {d->clip, rpGeneral},
        {d->data, rpGeneral},
    };

    bool info = d->info;

    vsapi->createVideoFilter(out, "DepanStabilise", d->vi, getframe_functions[d->method], filterFree<DepanStabiliseData>, fmParallel, deps, ARRAY_SIZE(deps), d.get(), core);
    d.release();

    if (vsapi->mapGetError(out))
        return;

    if (info) {
        if (!invokeFrameProps(prop_DepanStabilise_info, out, core, vsapi)) {
            vsapi->mapSetError(out, std::string("DepanStabilise: failed to invoke text.FrameProps: ").append(vsapi->mapGetError(out)).c_str());
            return;
        }
    }
}


void depanStabiliseRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->registerFunction("DepanStabilise",
                 "clip:vnode;"
                 "data:vnode;"
                 "cutoff:float:opt;"
                 "damping:float:opt;"
                 "initzoom:float:opt;"
                 "addzoom:int:opt;"
                 "prev:int:opt;"
                 "next:int:opt;"
                 "mirror:int:opt;"
                 "blur:int:opt;"
                 "dxmax:float:opt;"
                 "dymax:float:opt;"
                 "zoommax:float:opt;"
                 "rotmax:float:opt;"
                 "subpixel:int:opt;"
                 "pixaspect:float:opt;"
                 "fitlast:int:opt;"
                 "tzoom:float:opt;"
                 "info:int:opt;"
                 "method:int:opt;"
                 "fields:int:opt;",
                 "clip:vnode;",
                 depanStabiliseCreate, nullptr, plugin);
}
