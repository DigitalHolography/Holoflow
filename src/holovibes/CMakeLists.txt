qt_add_executable(holovibes
  main.cc
  ui/tensor_display_widget.cc
  ui/main_window.cc
  pipeline/worker.cc
  accumulators/batched_spsc_accumulator.cc
  accumulators/gate_accumulator.cc
  accumulators/sliding_average_accumulator.cu
  sources/holofile_source.cc
  sinks/qt_display_sink.cc
  sinks/holofile_record_sink.cc
  tasks/convert_task.cu
  tasks/identity_task.cc
  tasks/fresnel_diffraction_task.cu
  tasks/pca_task.cu
  tasks/average_task.cu
  tasks/percentile_clip_task.cu
  tasks/fft_shift_task.cu
  tasks/stft_task.cc
  tasks/angular_spectrum_task.cu
  tasks/memcpy_task.cc

  # Must add header files with Q_OBJECT for qt AUTMOC to run on them
  ${PROJECT_SOURCE_DIR}/include/holovibes/sinks/qt_display_sink.hh
  ${PROJECT_SOURCE_DIR}/include/holovibes/ui/tensor_display_widget.hh
  ${PROJECT_SOURCE_DIR}/include/holovibes/ui/main_window.hh
  ${PROJECT_SOURCE_DIR}/include/holovibes/pipeline/worker.hh
)

set_common_target_properties(holovibes)
set_common_compile_options(holovibes)
# target_compile_definitions(holovibes PRIVATE
#     CUDA_ARHC=86
#     CUDA_PATH=$ENV{CUDA_PATH}
#     -DSOURCE_PATH=${CMAKE_SOURCE_DIR}/src/holovibes
# )

target_include_directories(holovibes PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(holovibes PRIVATE
    tl::expected
    spdlog::spdlog
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    holoflow
    holofile
    cufft
    curaii
    nvJitLink
    nvrtc
)