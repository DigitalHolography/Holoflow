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

set(TARGET holovibes)
set(HOLOVIBES_VERSION "${PROJECT_VERSION_FULL}")
add_definitions(-DHOLOVIBES_VERSION_SEMVER2="${HOLOVIBES_VERSION}")

set(APP_VERSION_RESOURCE_WINDOWS "${CMAKE_CURRENT_BINARY_DIR}/holovibes_version.rc")
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/holovibes_version.rc.in"
    "${APP_VERSION_RESOURCE_WINDOWS}"
    @ONLY
)

file(GLOB_RECURSE HOLOVIBES_PRIVATE_HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx
)

file(GLOB_RECURSE HOLOVIBES_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

file(GLOB_RECURSE APP_RESOURCES
    ${CMAKE_SOURCE_DIR}/resources/holovibes/*
)

qt_add_executable(holovibes
    ${HOLOVIBES_SOURCES} ${APP_VERSION_RESOURCE_WINDOWS}
)

file(RELATIVE_PATH APP_RESOURCES_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${APP_RESOURCES}")

qt_add_resources(holovibes "app_resources"
    PREFIX "/"
    FILES ${APP_RESOURCES_REL}
    BASE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(holovibes 
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_sources(holovibes 
PRIVATE
    FILE_SET private_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES ${HOLOVIBES_PRIVATE_HEADERS}
)

target_link_libraries(holovibes PRIVATE
    project_options
    holoflow
    holotask
    holonp
    curaii
    holofile
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Boost::graph
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    Qt6::Network
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    Qt6::Graphs
    ads::qtadvanceddocking-qt6
    # EGrabber
)

if(ENABLE_COVERAGE)
    target_compile_definitions(holovibes PRIVATE HOLOFLOW_COVERAGE_PROBE)

    # Keep every object from the production libraries in the coverage image so
    # untouched host code is represented as uncovered instead of disappearing.
    target_link_options(holovibes PRIVATE
        "/WHOLEARCHIVE:$<TARGET_FILE:holoflow>"
        "/WHOLEARCHIVE:$<TARGET_FILE:holoflow_event>"
        "/WHOLEARCHIVE:$<TARGET_FILE:holotask>"
        "/WHOLEARCHIVE:$<TARGET_FILE:holonp>"
        "/WHOLEARCHIVE:$<TARGET_FILE:curaii>"
        "/WHOLEARCHIVE:$<TARGET_FILE:holofile>"
        "/OPT:NOREF"
        "/OPT:NOICF"
    )
endif()

add_custom_command(TARGET holovibes POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:ads::qtadvanceddocking-qt6>
        $<TARGET_FILE_DIR:holovibes>
    COMMENT "Copying Qt Advanced Docking System runtime"
)

if(TARGET holoflow_ffmpeg)
    get_target_property(FFMPEG_RUNTIME_DLLS holoflow_ffmpeg HOLOFLOW_FFMPEG_RUNTIME_DLLS)
    foreach(FFMPEG_RUNTIME_DLL IN LISTS FFMPEG_RUNTIME_DLLS)
        add_custom_command(TARGET holovibes POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFMPEG_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:holovibes>"
            COMMENT "Copying FFmpeg runtime ${FFMPEG_RUNTIME_DLL}"
        )
    endforeach()
endif()
