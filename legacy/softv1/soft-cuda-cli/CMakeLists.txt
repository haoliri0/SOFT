add_executable(
    soft-cuda-cli
    "source/exec.cpp")

target_link_libraries(
    soft-cuda-cli
    PRIVATE soft-cuda-lib)
