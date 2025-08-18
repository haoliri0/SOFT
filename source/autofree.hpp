#ifndef STN_CUDA_AUTOFREE_HPP
#define STN_CUDA_AUTOFREE_HPP
#include <functional>

class AutoFree {
    std::function<void()> const free;
    bool canceled;
public:
    explicit
    AutoFree(std::function<void()> const &free) : free(free), canceled(false) {}
    ~AutoFree() { if (!this->canceled) this->free(); }
    void cancel() { canceled = true; }
};

#endif
