qt_add_executable(holovibes
  main.cc
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
)