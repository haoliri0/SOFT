add_executable(
    soft-cuda-cli
    "sources/exec.cpp")

target_link_libraries(
    soft-cuda-cli
    PRIVATE soft-cuda-lib)
