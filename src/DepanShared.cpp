#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>

#include <VapourSynth4.h>

#include "DepanShared.h"


void transform2motion(const transform *tr, bool forward, float xcenter, float ycenter, float pixaspect, float *dx, float *dy, float *rot, float *zoom) {
    constexpr float PI = std::numbers::pi_v<float>;

    float rotradian = -atanf(pixaspect * tr->dxy / tr->dxx);
    *rot = rotradian * 180 / PI;
    float sinus = sinf(rotradian);
    float cosinus = cosf(rotradian);
    *zoom = tr->dxx / cosinus;

    if (forward) { //  get motion for forward
        *dx = tr->dxc - xcenter - (-xcenter * cosinus + ycenter / pixaspect * sinus) * (*zoom);
        *dy = tr->dyc / pixaspect - ycenter / pixaspect - ((-ycenter) / pixaspect * cosinus + (-xcenter) * sinus) * (*zoom); // dyc

    } else { // coefficients for backward

        //        tr.dxc/(*zoom) = xcenter/(*zoom) + (-xcenter + dx)*cosinus - ((-ycenter)/pixaspect + dy)*sinus ;
        //        tr.dyc/(*zoom)/pixaspect = ycenter/(*zoom)/pixaspect +  ((-ycenter)/pixaspect +dy)*cosinus + (-xcenter + dx)*sinus ;
        // *cosinus:
        //        tr.dxc/(*zoom)*cosinus = xcenter/(*zoom)*cosinus + (-xcenter + dx)*cosinus*cosinus - ((-ycenter)/pixaspect + dy)*sinus*cosinus ;
        // *sinus:
        //        tr.dyc/(*zoom)/pixaspect*sinus = ycenter/(*zoom)/pixaspect*sinus +  ((-ycenter)/pixaspect +dy)*cosinus*sinus + (-xcenter + dx)*sinus*sinus ;
        // summa:
        //        tr.dxc/(*zoom)*cosinus + tr.dyc/(*zoom)/pixaspect*sinus = xcenter/(*zoom)*cosinus + (-xcenter + dx) + ycenter/(*zoom)/pixaspect*sinus   ;
        *dx = tr->dxc / (*zoom) * cosinus + tr->dyc / (*zoom) / pixaspect * sinus - xcenter / (*zoom) * cosinus + xcenter - ycenter / (*zoom) / pixaspect * sinus;

        // *sinus:
        //        tr.dxc/(*zoom)*sinus = xcenter/(*zoom)*sinus + (-xcenter + dx)*cosinus*sinus - ((-ycenter)/pixaspect + dy)*sinus*sinus ;
        // *cosinus:
        //        tr.dyc/(*zoom)/pixaspect*cosinus = ycenter/(*zoom)/pixaspect*cosinus +  ((-ycenter)/pixaspect +dy)*cosinus*cosinus + (-xcenter + dx)*sinus*cosinus ;
        // diff:
        //        tr.dxc/(*zoom)*sinus - tr.dyc/(*zoom)/pixaspect*cosinus = xcenter/(*zoom)*sinus - (-ycenter/pixaspect + dy) - ycenter/(*zoom)/pixaspect*cosinus   ;
        *dy = -tr->dxc / (*zoom) * sinus + tr->dyc / (*zoom) / pixaspect * cosinus + xcenter / (*zoom) * sinus - (-ycenter / pixaspect) - ycenter / (*zoom) / pixaspect * cosinus;
    }
}


//------------------------------------------------------------
//  get  coefficients for inverse coordinates transformation,
//  fransform_inv ( transform_A ) = null transform
void inversetransform(const transform *ta, transform *tinv) noexcept {
    float pixaspect = (ta->dxy != 0.0f) ? sqrtf(-ta->dyx / ta->dxy) : 1.0f;

    tinv->dxx = ta->dxx / ((ta->dxx) * ta->dxx + ta->dxy * ta->dxy * pixaspect * pixaspect);
    tinv->dyy = tinv->dxx;
    tinv->dxy = -tinv->dxx * ta->dxy / ta->dxx;
    tinv->dyx = -tinv->dxy * pixaspect * pixaspect;
    tinv->dxc = -tinv->dxx * ta->dxc - tinv->dxy * ta->dyc;
    tinv->dyc = -tinv->dyx * ta->dxc - tinv->dyy * ta->dyc;
}


//****************************************************************************
//  motion to transform
//  get  coefficients for coordinates transformation,
//  which defines source (xsrc, ysrc)  for current destination (x,y)
//
//  fractoffset is the fraction of deformation (from 0 to 1 for forward, from -1 to 0 for backward time direction)
//
//
//  output: fills the transform struct tr->{dxc, dxx, dxy, dyc, dyx, dyy}
//
//
//   xsrc = dxc + dxx*x + dxy*y
//   ysrc = dyc + dyx*x + dyy*y
//
//  if no rotation, then dxy, dyx = 0,
//  if no rotation and zoom, then also dxx, dyy = 1.
//
void motion2transform(float dx1, float dy1, float rot, float zoom1, float pixaspect, float xcenter, float ycenter, bool forward, float fractoffset, transform *tr) {
    const float PI = std::numbers::pi_v<float>;

    // fractoffset > 0 for forward, <0 for backward
    float dx = fractoffset * dx1;
    float dy = fractoffset * dy1;
    float rotradian = fractoffset * rot * PI / 180; // from degree to radian
    if (fabsf(rotradian) < 1e-6f)
        rotradian = 0.0f; // for some stability of rounding precision
    if (zoom1 <= 0.0f)
        zoom1 = 1.0f; // guard: a non-positive zoom (e.g. from an untrusted data clip) makes logf produce NaN/-inf and poisons the transform
    float zoom = expf(fractoffset * logf(zoom1)); // zoom**(fractoffset) = exp(fractoffset*ln(zoom))
    if (fabsf(zoom - 1.0f) < 1e-6f)
        zoom = 1.0f; // for some stability of rounding precision

    float sinus = sinf(rotradian);
    float cosinus = cosf(rotradian);

    //    xcenter = row_size_p*(1+uv)/2.0;      //  middle x
    //    ycenter = height_p*(1+uv)/2.0;       //  middle y

    if (forward) { //  get coefficients for forward
        tr->dxc = xcenter + (-xcenter * cosinus + ycenter / pixaspect * sinus) * zoom + dx; // dxc            /(1+uv);
        tr->dxx = cosinus * zoom;                                                           // dxx
        tr->dxy = -sinus / pixaspect * zoom;                                                // dxy

        tr->dyc = ycenter + (((-ycenter) / pixaspect * cosinus + (-xcenter) * sinus) * zoom + dy) * pixaspect; // dyc      /(1+uv);
        tr->dyx = sinus * zoom * pixaspect;                                                                    // dyx
        tr->dyy = cosinus * zoom;                                                                              // dyy
    } else { // coefficients for backward
        tr->dxc = xcenter + ((-xcenter + dx) * cosinus - ((-ycenter) / pixaspect + dy) * sinus) * zoom; //     /(1+uv);
        tr->dxx = cosinus * zoom;
        tr->dxy = -sinus / pixaspect * zoom;

        tr->dyc = ycenter + (((-ycenter) / pixaspect + dy) * cosinus + (-xcenter + dx) * sinus) * zoom * pixaspect; //      /(1+uv);
        tr->dyx = sinus * zoom * pixaspect;
        tr->dyy = cosinus * zoom;
    }
}


//****************************************************************************
//  get  summary coefficients for summary combined coordinates transformation,
//  transform_BA = fransform_B ( transform_A )
//  output: fills the transform struct tba->{dxc, dxx, dxy, dyc, dyx, dyy}
void sumtransform(const transform *ta, const transform *tb, transform *tba) {
    transform temp;

    temp.dxc = tb->dxc + tb->dxx * ta->dxc + tb->dxy * ta->dyc;

    temp.dxx = tb->dxx * ta->dxx + tb->dxy * ta->dyx;

    temp.dxy = tb->dxx * ta->dxy + tb->dxy * ta->dyy;

    temp.dyc = tb->dyc + tb->dyx * ta->dxc + tb->dyy * ta->dyc;

    temp.dyx = tb->dyx * ta->dxx + tb->dyy * ta->dyx;

    temp.dyy = tb->dyx * ta->dxy + tb->dyy * ta->dyy;

    *tba = temp;
}


bool mapGetMotion(MotionData &m, const VSMap *props, const VSAPI *vsapi) {
    int err[5];
    float dx = vsapi->mapGetFloatSaturated(props, prop_Depan_dx, 0, &err[0]);
    float dy = vsapi->mapGetFloatSaturated(props, prop_Depan_dy, 0, &err[1]);
    float rot = vsapi->mapGetFloatSaturated(props, prop_Depan_rot, 0, &err[2]);
    float zoom = vsapi->mapGetFloatSaturated(props, prop_Depan_zoom, 0, &err[3]);
    int64_t good = vsapi->mapGetInt(props, prop_Depan_goodmotion, 0, &err[4]);

    if (err[0] || err[1] || err[2] || err[3] || err[4])
        return false;

    // Reject non-finite motion from an untrusted/corrupt data clip: it would poison the affine transform
    // and reach an undefined float-to-int conversion in the compensate planes (pixel indexing there is
    // already clamp-guarded, so this is UB in the cast only). A well-formed DepanAnalyse/DepanEstimate
    // clip never produces these.
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(rot) || !std::isfinite(zoom))
        return false;

    // Clamp finite magnitudes to values far beyond any real global motion, so the transformed source
    // coordinates stay well within int range for the (int)floorf casts regardless of frame size.
    m.dx = std::clamp(dx, -1.0e6f, 1.0e6f);
    m.dy = std::clamp(dy, -1.0e6f, 1.0e6f);
    m.rot = std::clamp(rot, -3.6e5f, 3.6e5f);   // sin/cos already bound rot's effect; keep it sane
    m.zoom = std::clamp(zoom, 1.0e-2f, 1.0e2f); // also keeps zoom strictly positive for logf
    m.badMotion = (good == 0);
    return true;
}

void mapSetMotion(VSMap *props, const MotionData &m, const VSAPI *vsapi) {
    vsapi->mapSetFloat(props, prop_Depan_dx, m.dx, maReplace);
    vsapi->mapSetFloat(props, prop_Depan_dy, m.dy, maReplace);
    vsapi->mapSetFloat(props, prop_Depan_rot, m.rot, maReplace);
    vsapi->mapSetFloat(props, prop_Depan_zoom, m.zoom, maReplace);
    vsapi->mapSetInt(props, prop_Depan_goodmotion, m.badMotion ? 0 : 1, maReplace);
}


bool invokeFrameProps(const char *prop, VSMap *out, VSCore *core, const VSAPI *vsapi) noexcept {
    VSPlugin *text_plugin = vsapi->getPluginByID("com.vapoursynth.text", core);

    VSNode *node = vsapi->mapGetNode(out, "clip", 0, nullptr);
    VSMap *args = vsapi->createMap();
    vsapi->mapSetNode(args, "clip", node, maReplace);
    vsapi->freeNode(node);
    vsapi->mapSetData(args, "props", prop, -1, dtUtf8, maReplace);

    VSMap *ret = vsapi->invoke(text_plugin, "FrameProps", args);
    vsapi->freeMap(args);

    if (vsapi->mapGetError(ret)) {
        vsapi->mapSetError(out, vsapi->mapGetError(ret));
        vsapi->freeMap(ret);
        return false;
    }

    node = vsapi->mapGetNode(ret, "clip", 0, nullptr);
    vsapi->freeMap(ret);
    vsapi->mapSetNode(out, "clip", node, maReplace);
    vsapi->freeNode(node);

    return true;
}


void depanRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    depanAnalyseRegister(plugin, vspapi);
    depanEstimateRegister(plugin, vspapi);
    depanCompensateRegister(plugin, vspapi);
    depanStabiliseRegister(plugin, vspapi);
}
