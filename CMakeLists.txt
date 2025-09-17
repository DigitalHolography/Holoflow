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
project(holoflow VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "Only MSVC compiler is supported.")
endif()

add_compile_definitions(
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:_WIN32_WINNT=0x0A00>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:_WIN32_WINNT=0x0A00>
)

option(ENABLE_IPO "IPO/LTO" ON)
option(ENABLE_TESTING "Build tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_DOCUMENTATION "Build documentation" ON)
option(ENABLE_EXAMPLES "Build examples" OFF)
option(ENABLE_FETCHCONTENT "Use FetchContent for dependencies" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)
option(ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

# GPU arch list editable by users: -DPROJECT_CUDA_ARCHS=86;89;90
set(PROJECT_CUDA_ARCHS "75;86;89;90" CACHE STRING "CUDA archs for CUDA targets")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/ProjectOptions.cmake)
project_options(
    ENABLE_IPO ${ENABLE_IPO}
    ENABLE_WARNINGS ${ENABLE_WARNINGS}
    ENABLE_WARNINGS_AS_ERRORS ${ENABLE_WARNINGS_AS_ERRORS}
)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui)
qt_standard_project_setup()

add_subdirectory(external)
add_subdirectory(src)

if(ENABLE_TESTING)
    add_subdirectory(test)
endif()

if(ENABLE_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(ENABLE_DOCUMENTATION)
    add_subdirectory(doc)
endif()