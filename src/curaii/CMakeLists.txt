
add_library(curaii STATIC
  v2/cuda.cc
  v2/cufft.cc
  v2/cublas.cc
  v2/cusolver_common.cc
  v2/cusolver_dn.cc
  v2/logger.cc
)

set_common_target_properties(curaii)
set_common_compile_options(curaii)

target_include_directories(curaii PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    $ENV{CUDA_PATH}/include
)

target_link_directories(curaii PUBLIC
    $ENV{CUDA_PATH}/lib/x64
)

target_link_libraries(curaii PUBLIC
    cudart
    cufft
    cublas
    cusolver
    tl::expected
    spdlog::spdlog
)