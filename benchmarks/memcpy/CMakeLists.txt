add_executable(memcpy_bench memcpy.cc)

set_common_target_properties(memcpy_bench)
set_common_compile_options(memcpy_bench)

target_link_libraries(memcpy_bench PRIVATE 
    benchmark::benchmark
)