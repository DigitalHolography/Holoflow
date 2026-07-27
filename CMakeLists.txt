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

cmake_minimum_required(VERSION 3.24)

file(READ "${CMAKE_SOURCE_DIR}/VERSION" PROJECT_VERSION_RAW)
string(STRIP "${PROJECT_VERSION_RAW}" PROJECT_VERSION_FULL)

message(STATUS "Holoflow version ${PROJECT_VERSION_FULL}")

string(REGEX MATCH "^([0-9]+)\.([0-9]+)\.([0-9]+)(-[0-9A-Za-z.-]+)?$" _ "${PROJECT_VERSION_FULL}")

if(NOT CMAKE_MATCH_0 STREQUAL PROJECT_VERSION_FULL)
    message(FATAL_ERROR
        "VERSION must be a semantic version such as 0.2.0 or 0.2.0-dev.1; "
        "got '${PROJECT_VERSION_FULL}'"
    )
endif()

set(PROJECT_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(PROJECT_VERSION_MINOR "${CMAKE_MATCH_2}")
set(PROJECT_VERSION_PATCH "${CMAKE_MATCH_3}")

set(PROJECT_VERSION_NUMERIC
    "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
)

if(CMAKE_MATCH_4)
    string(SUBSTRING "${CMAKE_MATCH_4}" 1 -1 PROJECT_VERSION_PRERELEASE)
else()
    set(PROJECT_VERSION_PRERELEASE "")
endif()

set(CPACK_PACKAGE_FILE_NAME "Holoflow-${PROJECT_VERSION_FULL}-win64")

project(holoflow VERSION "${PROJECT_VERSION_NUMERIC}" LANGUAGES CXX CUDA)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "Only MSVC compiler is supported.")
endif()

# Required to silent boost warnings on Windows
add_compile_definitions(
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:_WIN32_WINNT=0x0A00>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:_WIN32_WINNT=0x0A00>
)

# NVCC does not support utf-8 correctly
# https://github.com/fmtlib/fmt/issues/4389
add_definitions(-DFMT_UNICODE=0)

option(ENABLE_IPO "IPO/LTO" ON)
option(ENABLE_TESTING "Build tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_DOCUMENTATION "Build documentation" ON)
option(ENABLE_EXAMPLES "Build examples" ON)
option(ENABLE_FETCHCONTENT "Use FetchContent for dependencies" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)
option(ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

# GPU arch list editable by users: -DPROJECT_CUDA_ARCHS=86;89;90
set(PROJECT_CUDA_ARCHS "75;86;89;120-real;120-virtual" CACHE STRING "CUDA archs for CUDA targets")
set(CMAKE_CUDA_ARCHITECTURES "${PROJECT_CUDA_ARCHS}" CACHE STRING "CUDA archs for CUDA targets" FORCE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/ProjectOptions.cmake)
project_options(
    ENABLE_IPO ${ENABLE_IPO}
    ENABLE_WARNINGS ${ENABLE_WARNINGS}
    ENABLE_WARNINGS_AS_ERRORS ${ENABLE_WARNINGS_AS_ERRORS}
)

find_package(EGrabber)
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui OpenGL OpenGLWidgets Graphs)
qt_standard_project_setup()

add_subdirectory(external)
add_subdirectory(src)

if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

if(ENABLE_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(ENABLE_EXAMPLES)
    add_subdirectory(example)
endif()

if(ENABLE_DOCUMENTATION)
    add_subdirectory(doc)
endif()

# CPack configuration

if(WIN32 AND MSVC)
    install(TARGETS holovibes
        RUNTIME DESTINATION bin
        COMPONENT binaries
    )

    install(FILES
        "$<TARGET_FILE:ads::qtadvanceddocking-qt6>"
        DESTINATION bin
        COMPONENT dependencies
    )

    qt_generate_deploy_app_script(
        TARGET holovibes
        OUTPUT_SCRIPT QT_DEPLOY_SCRIPT
        NO_UNSUPPORTED_PLATFORM_ERROR
        NO_COMPILER_RUNTIME
        NO_TRANSLATIONS
        PRE_EXCLUDE_REGEXES
            "api-ms-"
            "ext-ms-"
        POST_EXCLUDE_REGEXES
            ".*[\\\\/]system32[\\\\/].*"
    )
    install(SCRIPT "${QT_DEPLOY_SCRIPT}" COMPONENT dependencies)

    file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/bin/x64" CUDA_BIN_DIR)
    if(NOT IS_DIRECTORY "${CUDA_BIN_DIR}")
        message(FATAL_ERROR "CUDA runtime directory does not exist: ${CUDA_BIN_DIR}")
    endif()

    function(find_single_cuda_runtime OUTPUT_VARIABLE GLOB_PATTERN)
        file(GLOB MATCHES "${CUDA_BIN_DIR}/${GLOB_PATTERN}")
        list(LENGTH MATCHES MATCH_COUNT)
        if(NOT MATCH_COUNT EQUAL 1)
            message(FATAL_ERROR
                "Expected exactly one CUDA runtime matching "
                "'${CUDA_BIN_DIR}/${GLOB_PATTERN}', found ${MATCH_COUNT}: ${MATCHES}"
            )
        endif()
        list(GET MATCHES 0 MATCH)
        set("${OUTPUT_VARIABLE}" "${MATCH}" PARENT_SCOPE)
    endfunction()

    find_single_cuda_runtime(CUDA_CUBLAS "cublas64_*.dll")
    find_single_cuda_runtime(CUDA_CUBLASLT "cublasLt64_*.dll")
    find_single_cuda_runtime(CUDA_CUFFT "cufft64_*.dll")
    find_single_cuda_runtime(CUDA_CUSOLVER "cusolver64_*.dll")
    find_single_cuda_runtime(CUDA_CUSPARSE "cusparse64_*.dll")
    find_single_cuda_runtime(CUDA_NVJITLINK "nvJitLink_*.dll")
    find_single_cuda_runtime(CUDA_NVRTC "nvrtc64_*.dll")
    find_single_cuda_runtime(CUDA_NVRTC_BUILTINS "nvrtc-builtins64_*.dll")

    install(FILES
        "${CUDA_CUBLAS}"
        "${CUDA_CUBLASLT}"
        "${CUDA_CUFFT}"
        "${CUDA_CUSOLVER}"
        "${CUDA_CUSPARSE}"
        "${CUDA_NVJITLINK}"
        "${CUDA_NVRTC}"
        "${CUDA_NVRTC_BUILTINS}"
        DESTINATION bin
        COMPONENT dependencies
    )
endif()

include(CPackConfig)
