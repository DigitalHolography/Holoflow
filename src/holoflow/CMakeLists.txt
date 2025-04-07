add_library(holoflow STATIC
  tensor.cc
  task.cc
  accumulator.cc
  model_descriptor.cc
  model.cc
  model_builder.cc
  source.cc
  sink.cc
  model2.cc
  v2/error.cc
  v2/model.cc
  v2/model_transaction.cc
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
)