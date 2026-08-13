#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>


// Block-size -> kernel lookup tables.
//
// These tables must be *constant* initialised. A file-scope std::unordered_map needs a dynamic
// initialiser, which the CRT runs when the plugin is loaded -- long before any CPU dispatch has
// happened. In a translation unit compiled for a higher ISA (SADFunctions_AVX512.cpp and friends)
// clang builds that table with whatever the target allows, so it emitted 64-byte `vmovups %zmm0`
// just to copy the initialiser_list onto the stack. Merely loading the plugin then executed
// AVX-512 on machines that do not have it. A constexpr array has no initialiser to run at all: it
// is baked into the image and only needs load-time base relocation for the function pointers.
//
// A linear scan replaces the hash lookup. The tables are consulted when a filter picks its kernels
// (InitMotionEstimationFields, selectFunctions), once per level at construction -- never per block,
// where the already-resolved function pointer is used -- so the scan costs nothing measurable.
//
// Declare the tables `static constexpr`: on a variable that already *requires* constant
// initialisation (it is a hard error otherwise), which is the whole guarantee we need. `consteval`
// is not an option here -- it applies only to functions, not variables. The lookups below stay
// `constexpr` rather than `consteval` because they are called with runtime block sizes.

template <typename F>
struct FunctionTableEntry {
    uint32_t key;
    F fn;
};

// Returns nullptr when the size is not in the table, for the ISA override selectors that only
// replace the entries they actually implement.
template <typename F, size_t N>
constexpr F findFunction(const std::array<FunctionTableEntry<F>, N> &table, uint32_t key) noexcept {
    for (const auto &entry : table)
        if (entry.key == key)
            return entry.fn;
    return nullptr;
}

// Matches the std::unordered_map::at() this replaced: throws when the block size is unsupported,
// which the filter creation functions turn into an error message.
template <typename F, size_t N>
constexpr F findFunctionOrThrow(const std::array<FunctionTableEntry<F>, N> &table, uint32_t key) {
    const F fn = findFunction(table, key);
    if (!fn)
        throw std::out_of_range("unsupported block size");
    return fn;
}
