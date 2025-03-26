
add_library(curaii STATIC
  logger.cc
  library_types.cc
  cuda_runtime.cc
  cufft.cc
  cublas.cc
  cusolver_dn.cc
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