#ifndef SYMFT_JTHREAD_COMPAT_HPP
#define SYMFT_JTHREAD_COMPAT_HPP

#include <thread>

namespace symft {

#if __cplusplus >= 202002L && !defined(__APPLE__)
    // Use std::jthread on C++20 compliant compilers (excluding Apple)
    using std::jthread;
#else
    // Simplified jthread substitute for compatibility with Apple clang
    class jthread {
    public:
        template <typename Function, typename... Args>
        explicit jthread(Function&& f, Args&&... args) 
            : thread_(std::forward<Function>(f), std::forward<Args>(args)...) {}
        
        ~jthread() {
            if (thread_.joinable()) {
                thread_.join();
            }
        }
        
        jthread(const jthread&) = delete;
        jthread& operator=(const jthread&) = delete;
        
        jthread(jthread&&) noexcept = default;
        jthread& operator=(jthread&&) noexcept = default;
        
        void join() { thread_.join(); }
        bool joinable() const { return thread_.joinable(); }
        
    private:
        std::thread thread_;
    };
#endif

} // namespace symft

#endif // SYMFT_JTHREAD_COMPAT_HPP
