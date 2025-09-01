cmake_minimum_required(VERSION 3.24)
project(holoflow VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "Only MSVC compiler is supported.")
endif()

option(ENABLE_IPO "IPO/LTO" ON)
option(ENABLE_TESTING "Build tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_DOCUMENTATION "Build documentation" OFF)
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

add_subdirectory(external)
add_subdirectory(src)

if(ENABLE_TESTING)
    add_subdirectory(test)
endif()

if(ENABLE_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(ENABLE_DOCUMENTATION)
    add_subdirectory(docs)
endif()