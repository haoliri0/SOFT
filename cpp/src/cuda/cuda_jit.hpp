#pragma once
#include "cuda/cuda_program.hpp"
#include <memory>
#include <cstdlib>
#include <string_view>

namespace symft::cuda {
inline bool cuda_env_enabled(const char* name) {
    const char* value=std::getenv(name);
    return value && std::string_view(value)!="0" && std::string_view(value)!="false";
}
// Circuit-specialized warp sampler. Compilation belongs to preparation.
class CudaJitSampler {
public:
    explicit CudaJitSampler(const CudaProgramData& program);
    ~CudaJitSampler();
    bool packed_expressions() const;
    void launch(const std::uint64_t* expressions, std::size_t shot_words,
                int shots, std::uint64_t seed, bool postselect,
                std::uint8_t* discarded, std::uint8_t* logical) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
