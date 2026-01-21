add_library(
    soft-cuda-lib
    "sources/simulator.cu"
    "sources/decompose.cu"
    "sources/classical.cu"
    "sources/measure.cu"
    "sources/noises.cu"
    "sources/gates_t.cu"
    "sources/gates.cu")

target_include_directories(
    soft-cuda-lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/includes>
    $<INSTALL_INTERFACE:include>)

set_target_properties(
    soft-cuda-lib PROPERTIES
    OUTPUT_NAME "soft-cuda")
