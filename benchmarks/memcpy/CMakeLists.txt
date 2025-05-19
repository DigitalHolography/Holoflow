add_executable(memcpy_bench memcpy.cc)

set_common_target_properties(memcpy_bench)
set_common_compile_options(memcpy_bench)

target_link_libraries(memcpy_bench PRIVATE 
    benchmark::benchmark
    TBB::tbb
)

# target_compile_options(memcpy_bench PRIVATE
#     $<$<COMPILE_LANGUAGE:CXX>:
#         $<$<CXX_COMPILER_ID:MSVC>:
#             /FAs;
#             /Oi-
#         >
#     >
# )