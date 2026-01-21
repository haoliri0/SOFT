cmake_minimum_required(VERSION 3.22)
project(soft-cuda
    LANGUAGES CXX CUDA
    VERSION 0.1.17)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
include_directories("${CUDAToolkit_INCLUDE_DIRS}")

add_subdirectory(soft-cuda-lib)
add_subdirectory(soft-cuda-cli)
