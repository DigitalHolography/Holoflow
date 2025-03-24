qt_add_executable(holovibes
  main.cc
  accumulators/batched_spsc_accumulator.cc
  sources/holofile_source.cc
  sinks/qt_display_sink.cc
  ui/tensor_display_widget.cc

  # Must add header files with Q_OBJECT for qt AUTMOC to run on them
  ${PROJECT_SOURCE_DIR}/include/holovibes/sinks/qt_display_sink.hh
  ${PROJECT_SOURCE_DIR}/include/holovibes/ui/tensor_display_widget.hh
)

set_common_target_properties(holovibes)
set_common_compile_options(holovibes)

target_include_directories(holovibes PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(holovibes PRIVATE
    tl::expected
    glog::glog
    gflags::gflags
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    holoflow
    holofile
)