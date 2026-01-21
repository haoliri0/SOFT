#ifndef STN_CUDA_CLEANER_HPP
#define STN_CUDA_CLEANER_HPP

#include <functional>

class Cleaner {
    std::function<void()> const free;
    bool canceled;
public:
    explicit
    Cleaner(std::function<void()> const &free) : free(free), canceled(false) {}
    ~Cleaner() { if (!this->canceled) this->free(); }
    void cancel() { canceled = true; }
};

#endif
