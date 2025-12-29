cmake_minimum_required(VERSION 3.22)
project(soft-cuda
    LANGUAGES CXX CUDA
    VERSION 0.1.17)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
include_directories("${CUDAToolkit_INCLUDE_DIRS}")

add_library(soft_cuda
    "source/simulator.cu"
    "source/decompose.cu"
    "source/classical.cu"
    "source/measure.cu"
    "source/noises.cu"
    "source/gates_t.cu"
    "source/gates.cu")

add_executable(soft_cuda_exec "source/exec.cpp")
target_link_libraries(soft_cuda_exec PRIVATE soft_cuda)
