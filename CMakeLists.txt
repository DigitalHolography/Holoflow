cmake_minimum_required(VERSION 3.18)
project(holoflow LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CommonCMakeFunctions)

find_package(GTest CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(Doxygen REQUIRED)
find_package(tl-expected CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)

qt_standard_project_setup()

enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(docs)
add_subdirectory(examples)