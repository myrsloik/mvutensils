#include <cstdint>
#include <cstring>

#include "CopyCode.h"
#include "FunctionTable.h"

template <unsigned width, unsigned height>
void copyBlock(uint8_t * __restrict pDst, ptrdiff_t nDstPitch, const uint8_t * __restrict pSrc, ptrdiff_t nSrcPitch) {
    for (unsigned j = 0; j < height; j++)
        memcpy(pDst + j * nDstPitch, pSrc + j * nSrcPitch, width);
}


#define KEY(width, height, bits) ((width) << 16 | (height) << 8 | (bits))
#define COPY(width, height) \
    { KEY(width, height, 8), copyBlock<width * sizeof(uint8_t), height> }, \
    { KEY(width, height, 16), copyBlock<width * sizeof(uint16_t), height> }, \
    { KEY(width, height, 32), copyBlock<width * sizeof(float), height> },

static constexpr auto copy_functions = std::to_array<FunctionTableEntry<COPYFunction>>({
    COPY(2, 2)
    COPY(2, 4)
    COPY(4, 2)
    COPY(4, 4)
    COPY(4, 8)
    COPY(8, 1)
    COPY(8, 2)
    COPY(8, 4)
    COPY(8, 8)
    COPY(8, 16)
    COPY(16, 1)
    COPY(16, 2)
    COPY(16, 4)
    COPY(16, 8)
    COPY(16, 16)
    COPY(16, 32)
    COPY(32, 8)
    COPY(32, 16)
    COPY(32, 32)
    COPY(32, 64)
    COPY(64, 16)
    COPY(64, 32)
    COPY(64, 64)
    COPY(64, 128)
    COPY(128, 32)
    COPY(128, 64)
    COPY(128, 128)
});

COPYFunction selectCopyFunction(unsigned width, unsigned height, unsigned bits) {
    return findFunctionOrThrow(copy_functions, KEY(width, height, bits));
}

#undef COPY
#undef KEY

