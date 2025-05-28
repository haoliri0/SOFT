cmake_minimum_required(VERSION 3.22)
project(stn-cuda CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
include_directories("${CUDAToolkit_INCLUDE_DIRS}")

add_library(stn_cuda "source/simulator.cu")

add_executable(stn_cuda_tests "tests/main.cu")
target_link_libraries(stn_cuda_tests PRIVATE stn_cuda)
