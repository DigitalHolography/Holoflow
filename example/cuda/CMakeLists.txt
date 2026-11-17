find_package(CUDAToolkit REQUIRED)

add_executable(cub_reduce_workspace_repro cub_reduce_workspace_repro.cu)
target_link_libraries(cub_reduce_workspace_repro PRIVATE project_options CUDA::cudart)
target_include_directories(cub_reduce_workspace_repro PRIVATE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
set_target_properties(cub_reduce_workspace_repro PROPERTIES
    AUTOMOC OFF
    AUTOUIC OFF
    AUTORCC OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)
