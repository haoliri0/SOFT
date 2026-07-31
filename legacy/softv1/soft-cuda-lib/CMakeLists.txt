add_library(
    soft-cuda-lib
    "source/simulator.cu"
    "source/decompose.cu"
    "source/classical.cu"
    "source/measure.cu"
    "source/noises.cu"
    "source/gates_t.cu"
    "source/gates.cu")

target_include_directories(
    soft-cuda-lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

set_target_properties(
    soft-cuda-lib PROPERTIES
    OUTPUT_NAME "soft-cuda")
