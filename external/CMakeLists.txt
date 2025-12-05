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
    include(FetchContent)

if(NOT DEFINED FETCHCONTENT_BASE_DIR)
    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.fc")
endif()

    # spdlog
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.15.3
    )
    FetchContent_MakeAvailable(spdlog)

    # gtest
    if(ENABLE_TESTING)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.17.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    # boost
    FetchContent_Declare(
        boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.89.0
    )
    FetchContent_MakeAvailable(boost)

    # nlohmann-json
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.12.0
    )
    FetchContent_MakeAvailable(nlohmann_json)

    # benchmark
    if(ENABLE_BENCHMARKS)
        FetchContent_Declare(
            benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.4
        )
        FetchContent_MakeAvailable(benchmark)
    endif()
else()
    find_package(spdlog REQUIRED)

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

find_package(TBB REQUIRED)
find_package(MKL CONFIG REQUIRED)
find_program(MKL_INTEL_THREAD_DLL NAMES mkl_intel_thread.2.dll REQUIRED)
find_program(MKL_AVX512_DLL NAMES mkl_avx512.2.dll REQUIRED)
find_program(MKL_DEF_DLL NAMES mkl_def.2.dll REQUIRED)
find_program(MKL_CORE_DLL NAMES mkl_core.2.dll REQUIRED)
find_program(LIBIOMP5MD_DLL NAMES libiomp5md.dll REQUIRED)