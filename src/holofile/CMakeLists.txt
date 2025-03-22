add_library(holofile STATIC holofile.cc)

set_common_target_properties(holofile)
set_common_compile_options(holofile)

target_include_directories(holofile PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(holofile PUBLIC
    tl::expected
)