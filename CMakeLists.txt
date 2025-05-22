cmake_minimum_required(VERSION 3.21)
project(holoflow LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 86-real 86-virtual)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# NVCC does not support utf-8 correctly
# https://github.com/fmtlib/fmt/issues/4389
add_definitions(-DFMT_UNICODE=0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CommonCMakeFunctions)

find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(Doxygen REQUIRED)
find_package(tl-expected CONFIG REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)
find_package(TBB REQUIRED)
find_package(MKL CONFIG REQUIRED)

qt_standard_project_setup()

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(docs)
add_subdirectory(examples)