#include "simd/simd.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define SYMFT_GNU_CPU_SUPPORTS 1
#else
#define SYMFT_GNU_CPU_SUPPORTS 0
#endif

namespace symft::simd {

#if defined(SYMFT_COMPILED_AVX2)
const KernelTable& avx2_table();
#endif

#if defined(SYMFT_COMPILED_AVX512)
const KernelTable& avx512_table();
#endif

namespace {

const KernelTable& choose_table() {
#if SYMFT_GNU_CPU_SUPPORTS && defined(SYMFT_COMPILED_AVX512)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")) {
        return avx512_table();
    }
#endif
#if SYMFT_GNU_CPU_SUPPORTS && defined(SYMFT_COMPILED_AVX2)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return avx2_table();
    }
#endif
    return scalar_table();
}

} // namespace

const KernelTable& dispatch_table() {
    static const KernelTable& table = choose_table();
    return table;
}

const char* dispatch_name() {
    return dispatch_table().name;
}

} // namespace symft::simd
