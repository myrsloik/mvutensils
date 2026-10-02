#include <VapourSynth4.h>
#include <cstdlib>


// Extra indirection to keep the parameter lists with the respective filters.


void superRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi);
void analyseRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi);
void degrainRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi);
void flowRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi);
void flowFetchRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi);
void compensateRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi);
void recalculateRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi);
void maskRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi);
void scdetectionRegister(VSPlugin *plugin, const VSPLUGINAPI *vspapi);


VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    const int packageVersion = atoi(PACKAGE_VERSION);

    vspapi->configPlugin("com.vapoursynth.mvgputensils", "mvgpu", "MVGPUtensils v" PACKAGE_VERSION, VS_MAKE_VERSION(packageVersion, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    superRegister(plugin, vspapi);
    analyseRegister(plugin, vspapi);
    degrainRegister(plugin, vspapi);
    flowRegister(plugin, vspapi);
    flowFetchRegister(plugin, vspapi);
    compensateRegister(plugin, vspapi);
    recalculateRegister(plugin, vspapi);
    maskRegister(plugin, vspapi);
    scdetectionRegister(plugin, vspapi);
}
