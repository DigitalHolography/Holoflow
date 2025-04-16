add_library(holoflow STATIC
  tensor.cc
  task.cc
  accumulator.cc
  source.cc
  sink.cc
  v3/model/compiler.cc
  v3/model/model.cc
  v3/model/descriptor.cc
  v3/model/runner.cc
)

set_common_target_properties(holoflow)
set_common_compile_options(holoflow)

target_include_directories(holoflow PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    $ENV{CUDA_PATH}/include
)

target_link_directories(holoflow PUBLIC
    $ENV{CUDA_PATH}/lib/x64
)

target_link_libraries(holoflow PUBLIC
    nlohmann_json::nlohmann_json
    cudart
    tl::expected
    spdlog::spdlog
    Boost::graph
)