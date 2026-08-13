// 32-bit FLOAT pixel SAD, baseline / cross-platform path. THIS TU MUST BE BUILT WITH FAST-MATH
// (/fp:fast on MSVC/clang-cl, -ffast-math on gcc/clang) so the scalar reduction reassociates and
// auto-vectorizes; without it clang keeps a single strict-order FP-add chain (~3-10x slower). It is
// the only float-SAD path on non-x86; on x86 the AVX2/AVX-512 hand kernels override it for speed.
#include <cstdint>

#include "FunctionTable.h"
#include "SADFunctions.h"
#include "SADFunctions_Float.h"

template <unsigned W, unsigned H>
static unsigned int sad_f32_base(const uint8_t *s, intptr_t sp, const uint8_t *r, intptr_t rp) noexcept {
    return scale_f32_sad(sad_f32_c_raw<W, H>(s, sp, r, rp));
}

#define KEY(width, height, bits) (unsigned)(width) << 24 | (height) << 16 | (bits) << 8
#define SAD_F32(width, height) { KEY(width, height, 32), sad_f32_base<width, height> },

// Same block-size set as the integer SAD map.
static constexpr auto sad_f32_functions = std::to_array<FunctionTableEntry<SADFunction>>({
    SAD_F32(2, 2) SAD_F32(2, 4) SAD_F32(4, 2) SAD_F32(4, 4) SAD_F32(4, 8)
    SAD_F32(8, 1) SAD_F32(8, 2) SAD_F32(8, 4) SAD_F32(8, 8) SAD_F32(8, 16)
    SAD_F32(16, 1) SAD_F32(16, 2) SAD_F32(16, 4) SAD_F32(16, 8) SAD_F32(16, 16) SAD_F32(16, 32)
    SAD_F32(32, 8) SAD_F32(32, 16) SAD_F32(32, 32) SAD_F32(32, 64)
    SAD_F32(64, 16) SAD_F32(64, 32) SAD_F32(64, 64) SAD_F32(64, 128)
    SAD_F32(128, 32) SAD_F32(128, 64) SAD_F32(128, 128)
});

void selectSADFunctionFloat(unsigned width, unsigned height, SADFunction &sad) {
    if (SADFunction fn = findFunction(sad_f32_functions, KEY(width, height, 32)))
        sad = fn;
}


// ===================== Float SATD (pure scalar, auto-vectorized) =====================
// Kernel is satd_f32_c<W,H> in SADFunctions_Float.h; this fast-math TU instantiates it at the SSE2
// baseline. There are no per-ISA (AVX2/AVX-512) float SATD overrides -- this is the only float SATD path.
#define SATD_F32(width, height) { KEY(width, height, 32), satd_f32_c<width, height> },

// Same block-size set as the integer SATD map.
static constexpr auto satd_f32_functions = std::to_array<FunctionTableEntry<SADFunction>>({
    SATD_F32(4, 4) SATD_F32(8, 4) SATD_F32(8, 8) SATD_F32(16, 8) SATD_F32(16, 16)
    SATD_F32(32, 16) SATD_F32(32, 32) SATD_F32(64, 32) SATD_F32(64, 64) SATD_F32(128, 64) SATD_F32(128, 128)
});

void selectSATDFunctionFloat(unsigned width, unsigned height, SADFunction &satd) {
    if (SADFunction fn = findFunction(satd_f32_functions, KEY(width, height, 32)))
        satd = fn;
}
