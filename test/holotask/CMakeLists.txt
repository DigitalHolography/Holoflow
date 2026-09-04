# Copyright 2025 Digital Holography Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(TARGET holotask_test)

set(SOURCES
    angular_spectrum_test.cu
    conversion_test.cu
    ffmpeg_sink_test.cc
    flatfield_test.cu
    filter2d_test.cu
    fresnel_diffraction_test.cu
    holofile_sink_test.cc
    npyfile_sink_test.cc
    normalize_test.cu
    pca_test.cu
    pct_clip_test.cu
    shack_hartmann_test.cu
    sliding_average_test.cu
    zernike_defocus_z_prop_test.cc
    zernike_phase_test.cu
    ../holonp/support/tensor_test_buffer.cc
    ../holonp/support/sync_task_runner.cc
)

add_executable(${TARGET} ${SOURCES})

target_include_directories(${TARGET}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../holonp/support
)

target_link_libraries(${TARGET}
    PRIVATE
    holotask
    GTest::gtest_main
)

if(TARGET holoflow_ffmpeg)
    target_link_libraries(${TARGET} PRIVATE holoflow_ffmpeg)
endif()

if(TARGET holoflow_ffmpeg)
    get_target_property(FFMPEG_RUNTIME_DLLS holoflow_ffmpeg HOLOFLOW_FFMPEG_RUNTIME_DLLS)
    foreach(FFMPEG_RUNTIME_DLL IN LISTS FFMPEG_RUNTIME_DLLS)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFMPEG_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:${TARGET}>"
            COMMENT "Copying FFmpeg runtime ${FFMPEG_RUNTIME_DLL}"
        )
    endforeach()
endif()

holoflow_discover_tests(${TARGET}
    LABELS gpu
    TIMEOUT 120
    RESOURCE_LOCK holoflow_gpu
    FILTER "-*Performance*"
)

holoflow_discover_tests(${TARGET}
    LABELS performance
    TIMEOUT 180
    RESOURCE_LOCK holoflow_gpu
    FILTER "*Performance*"
    PREFIX "performance::"
)
