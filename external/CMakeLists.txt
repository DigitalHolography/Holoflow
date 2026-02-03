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

    message(STATUS "FetchContent dependencies acquired.")

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