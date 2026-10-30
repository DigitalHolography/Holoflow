find_package(cufftdx REQUIRED CONFIG PATHS "${mathdx_DIR}/../cufftdx")
add_executable(fresnel_propagation_bench main.cu)
target_link_libraries(fresnel_propagation_bench PRIVATE holotask curaii cufftdx::cufftdx)
set_target_properties(fresnel_propagation_bench PROPERTIES CUDA_ARCHITECTURES 86)
