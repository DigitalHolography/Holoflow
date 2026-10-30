# Copyright 2025 Digital Holography Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(TARGET holotask)

file(GLOB_RECURSE HOLOTASK_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holotask/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holotask/*.hxx
)

file(GLOB_RECURSE HOLOTASK_PRIVATE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx
)

file(GLOB_RECURSE HOLOTASK_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(${TARGET}
    STATIC
    ${HOLOTASK_SOURCES}
)

target_include_directories(${TARGET}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_sources(${TARGET}
    PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${HOLOTASK_PUBLIC_HEADERS}
)

target_sources(${TARGET}
    PRIVATE
    FILE_SET private_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES ${HOLOTASK_PRIVATE_HEADERS}
)

target_link_libraries(${TARGET}
    PUBLIC
    project_options
    holoflow_event
    curaii
    holofile
    holoflow
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Boost::graph
)

target_link_libraries(${TARGET}
    PRIVATE
    holoflow_cusolverdx_nvrtc
    cuda
    nvJitLink
    nvrtc
)

if(EGrabber_FOUND)
    target_compile_definitions(${TARGET} PRIVATE HOLOTASK_HAS_EGRABBER)

    target_link_libraries(${TARGET}
        PUBLIC
        EGrabber
    )
endif()

# NVRTC sources are formatted CUDA files, embedded rather than compiled by MSVC.
set(PCA_RUNTIME_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/syncs/pca/runtime/pca_heev_kernel.cu")
set_source_files_properties("${PCA_RUNTIME_SOURCE}" PROPERTIES HEADER_FILE_ONLY TRUE)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/embedded")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/embedded/pca_heev_source.hh"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${PCA_RUNTIME_SOURCE}"
            "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/embedded/pca_heev_source.hh"
            "-DSYMBOL=pca_heev_source" -P "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaSource.cmake"
    DEPENDS "${PCA_RUNTIME_SOURCE}" "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaSource.cmake"
    VERBATIM
)
target_sources(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/embedded/pca_heev_source.hh")
target_include_directories(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/embedded")

set(FRESNEL_RUNTIME_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/syncs/fresnel_diffraction/runtime/apply_lens_callback2.cu")
set_source_files_properties("${FRESNEL_RUNTIME_SOURCE}" PROPERTIES HEADER_FILE_ONLY TRUE)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/embedded/fresnel_callback_source.hh"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${FRESNEL_RUNTIME_SOURCE}"
            "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/embedded/fresnel_callback_source.hh"
            "-DSYMBOL=fresnel_callback_source" -P "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaSource.cmake"
    DEPENDS "${FRESNEL_RUNTIME_SOURCE}" "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaSource.cmake"
    VERBATIM
)
target_sources(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/embedded/fresnel_callback_source.hh")

get_filename_component(HOLOTASK_MATHDX_ROOT "${mathdx_DIR}/../../.." ABSOLUTE)
target_include_directories(${TARGET} SYSTEM PRIVATE "${HOLOTASK_MATHDX_ROOT}/external/cutlass/include")

set(FRESNEL_F16_RUNTIME_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/syncs/fresnel_diffraction/runtime/fused_f16.cu")
set_source_files_properties("${FRESNEL_F16_RUNTIME_SOURCE}" PROPERTIES HEADER_FILE_ONLY TRUE)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/embedded/fresnel_fused_f16_source.hh"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${FRESNEL_F16_RUNTIME_SOURCE}"
            "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/embedded/fresnel_fused_f16_source.hh"
            "-DSYMBOL=fresnel_fused_f16_source" -P "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaSource.cmake"
    DEPENDS "${FRESNEL_F16_RUNTIME_SOURCE}" "${PROJECT_SOURCE_DIR}/cmake/EmbedCudaSource.cmake"
    VERBATIM
)

target_sources(${TARGET} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/embedded/fresnel_fused_f16_source.hh")
