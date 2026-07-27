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

if(ENABLE_FETCHCONTENT)
    message(STATUS "Using FetchContent to acquire external dependencies...")
    include(FetchContent)

    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.fc" CACHE PATH "Persistent storage for FetchContent" FORCE)
    file(MAKE_DIRECTORY "${FETCHCONTENT_BASE_DIR}")
    message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

    macro(declare_persistent_dependency NAME REPO TAG)
        FetchContent_Declare(
            ${NAME}
            GIT_REPOSITORY ${REPO}
            GIT_TAG        ${TAG}
            GIT_SHALLOW    ON
            SOURCE_DIR     "${FETCHCONTENT_BASE_DIR}/${NAME}-src"
            BINARY_DIR     "${FETCHCONTENT_BASE_DIR}/${NAME}-build"
            SUBBUILD_DIR   "${FETCHCONTENT_BASE_DIR}/${NAME}-subbuild"
        )
    endmacro()

    macro(declare_persistent_archive NAME URL HASH)
        FetchContent_Declare(
            ${NAME}
            URL         ${URL}
            URL_HASH    ${HASH}
            SOURCE_DIR  "${FETCHCONTENT_BASE_DIR}/${NAME}-src"
            BINARY_DIR  "${FETCHCONTENT_BASE_DIR}/${NAME}-build"
            SUBBUILD_DIR "${FETCHCONTENT_BASE_DIR}/${NAME}-subbuild"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
    endmacro()

    declare_persistent_dependency(spdlog "https://github.com/gabime/spdlog.git" "v1.15.3")
    FetchContent_MakeAvailable(spdlog)

    if(ENABLE_TESTING)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        declare_persistent_dependency(googletest "https://github.com/google/googletest.git" "v1.17.0")
        FetchContent_MakeAvailable(googletest)
    endif()

    declare_persistent_dependency(boost "https://github.com/boostorg/boost.git" "boost-1.89.0")
    FetchContent_MakeAvailable(boost)

    declare_persistent_dependency(nlohmann_json "https://github.com/nlohmann/json.git" "v3.12.0")
    FetchContent_MakeAvailable(nlohmann_json)

    if(ENABLE_BENCHMARKS)
        declare_persistent_dependency(benchmark "https://github.com/google/benchmark.git" "v1.9.4")
        FetchContent_MakeAvailable(benchmark)
    endif()

    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable Qt ADS examples" FORCE)
    set(BUILD_STATIC OFF CACHE BOOL "Disable Qt ADS static library" FORCE)
    set(ADS_VERSION "5.0.0" CACHE STRING "Qt ADS version" FORCE)

    declare_persistent_dependency(qtads "https://github.com/githubuser0xFFFF/Qt-Advanced-Docking-System.git" "5.0.0")
    FetchContent_MakeAvailable(qtads)

    set(MATHDX_VERSION "26.06.1")
    set(MATHDX_CUDA_MAJOR "13")

    declare_persistent_archive(
        mathdx
        "https://developer.nvidia.com/downloads/compute/cuSOLVERDx/redist/cuSOLVERDx/cuda13/nvidia-mathdx-${MATHDX_VERSION}-cuda13.zip"
        "SHA256=c691252f1c7dc8bce2f96f4166604efe11eb11735998f492ffcf9bc2835e424f"
    )

    FetchContent_MakeAvailable(mathdx)

    set(MATHDX_ROOT
        "${mathdx_SOURCE_DIR}/nvidia/mathdx/26.06"
    )

    find_package(
        mathdx
        26.06
        REQUIRED
        COMPONENTS cusolverdx
        CONFIG
        PATHS "${MATHDX_ROOT}"
        NO_DEFAULT_PATH
    )

    if(NOT mathdx_FOUND)
        message(FATAL_ERROR "MathDx package was not found")
    endif()

    if(NOT cusolverdx_FOUND)
        message(FATAL_ERROR "cuSolverDx component was not found")
    endif()

    if(NOT EXISTS "${cusolverdx_FATBIN}")
        message(FATAL_ERROR
            "cuSolverDx fatbin does not exist: ${cusolverdx_FATBIN}"
        )
    endif()

    message(STATUS "MathDx version: ${mathdx_VERSION}")
    message(STATUS "cuSolverDx version: ${cusolverdx_VERSION}")
    message(STATUS "MathDx includes: ${mathdx_INCLUDE_DIRS}")
    message(STATUS "cuSolverDx includes: ${cusolverdx_INCLUDE_DIRS}")
    message(STATUS "cuSolverDx fatbin: ${cusolverdx_FATBIN}")

    file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/include" HOLOFLOW_CUDA_INCLUDE_DIR)
    file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/EULA.txt" HOLOFLOW_CUDA_LICENSE)
    file(TO_CMAKE_PATH "${cusolverdx_INCLUDE_DIR}" HOLOFLOW_CUSOLVERDX_INCLUDE_DIR)
    file(TO_CMAKE_PATH "${cusolverdx_cutlass_INCLUDE_DIR}" HOLOFLOW_CUSOLVERDX_CUTLASS_INCLUDE_DIR)
    file(TO_CMAKE_PATH "${cusolverdx_FATBIN}" HOLOFLOW_CUSOLVERDX_FATBIN)

    add_library(holoflow_cusolverdx_nvrtc INTERFACE)
    target_compile_definitions(holoflow_cusolverdx_nvrtc
        INTERFACE
        HOLOFLOW_CUDA_INCLUDE_DIR="${HOLOFLOW_CUDA_INCLUDE_DIR}"
        HOLOFLOW_CUSOLVERDX_INCLUDE_DIR="${HOLOFLOW_CUSOLVERDX_INCLUDE_DIR}"
        HOLOFLOW_CUSOLVERDX_CUTLASS_INCLUDE_DIR="${HOLOFLOW_CUSOLVERDX_CUTLASS_INCLUDE_DIR}"
        HOLOFLOW_CUSOLVERDX_FATBIN="${HOLOFLOW_CUSOLVERDX_FATBIN}"
        HOLOFLOW_NVRTC_ASSET_RELATIVE_DIR="../share/holovibes/nvrtc"
    )

    if(WIN32)
        install(
            DIRECTORY "${HOLOFLOW_CUDA_INCLUDE_DIR}/"
            DESTINATION share/holovibes/nvrtc/cuda/include
            COMPONENT dependencies
        )
        install(
            FILES
                "${MATHDX_ROOT}/include/cusolverdx.hpp"
                "${MATHDX_ROOT}/include/cusolverdx_io.hpp"
            DESTINATION share/holovibes/nvrtc/mathdx/include
            COMPONENT dependencies
        )
        install(
            DIRECTORY
                "${MATHDX_ROOT}/include/cusolverdx"
                "${MATHDX_ROOT}/include/commondx"
            DESTINATION share/holovibes/nvrtc/mathdx/include
            COMPONENT dependencies
        )
        install(
            DIRECTORY "${cusolverdx_cutlass_INCLUDE_DIR}/cute"
            DESTINATION share/holovibes/nvrtc/mathdx/cutlass/include
            COMPONENT dependencies
        )
        install(
            FILES
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/array_subbyte.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/array.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/bfloat16.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/complex.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/coord.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/cuda_host_adapter.hpp"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/cutlass.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/exmy_base.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/fast_math.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/float_subbyte.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/float8.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/floating_point_nvrtc.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/functional.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/half.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/integer_subbyte.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/numeric_size.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/numeric_types.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/real.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/tfloat32.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/trace.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/uint128.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/uint256.h"
            DESTINATION share/holovibes/nvrtc/mathdx/cutlass/include/cutlass
            COMPONENT dependencies
        )
        install(
            FILES
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/arch/config.h"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/arch/synclog.hpp"
            DESTINATION share/holovibes/nvrtc/mathdx/cutlass/include/cutlass/arch
            COMPONENT dependencies
        )
        install(
            FILES
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/detail/dependent_false.hpp"
                "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/detail/helper_macros.hpp"
            DESTINATION share/holovibes/nvrtc/mathdx/cutlass/include/cutlass/detail
            COMPONENT dependencies
        )
        install(
            FILES "${cusolverdx_cutlass_INCLUDE_DIR}/cutlass/platform/platform.h"
            DESTINATION share/holovibes/nvrtc/mathdx/cutlass/include/cutlass/platform
            COMPONENT dependencies
        )
        install(
            FILES "${cusolverdx_FATBIN}"
            DESTINATION share/holovibes/nvrtc/mathdx/lib
            COMPONENT dependencies
        )
        install(
            FILES "${HOLOFLOW_CUDA_LICENSE}"
            DESTINATION share/licenses
            RENAME NVIDIA-CUDA-EULA.txt
            COMPONENT dependencies
        )
        install(
            FILES "${MATHDX_ROOT}/LICENSE.txt"
            DESTINATION share/licenses
            RENAME NVIDIA-MathDx-LICENSE.txt
            COMPONENT dependencies
        )
    endif()

    message(STATUS "FetchContent dependencies acquired.")

else()
    find_package(spdlog REQUIRED)
    find_package(Boost REQUIRED COMPONENTS system filesystem thread)
    find_package(nlohmann_json REQUIRED)
    find_package(qtadvanceddockingsystem 5.0.0 EXACT REQUIRED)

    if(ENABLE_TESTING)
        find_package(GTest REQUIRED)
    endif()

    if(ENABLE_BENCHMARKS)
        find_package(benchmark REQUIRED)
    endif()
endif()

if (ENABLE_DOCS)
    find_package(Doxygen REQUIRED)
endif()
