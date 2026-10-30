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

find_package(CUDAToolkit REQUIRED)
get_filename_component(PCA_MATHDX_ROOT "${mathdx_DIR}/../../.." ABSOLUTE)
set(cusolverdx_INCLUDE_DIR "${PCA_MATHDX_ROOT}/include")
set(cusolverdx_cutlass_INCLUDE_DIR "${PCA_MATHDX_ROOT}/external/cutlass/include")
set(cusolverdx_FATBIN "${PCA_MATHDX_ROOT}/lib/libcusolverdx.fatbin")

set(PCA_BENCH_CUDA_ARCH "86" CACHE STRING "PCA benchmark CUDA architecture (86 or 120)")
if(NOT PCA_BENCH_CUDA_ARCH MATCHES "^(86|120)$")
    message(FATAL_ERROR "PCA_BENCH_CUDA_ARCH must be 86 or 120")
endif()
math(EXPR PCA_BENCH_SM "${PCA_BENCH_CUDA_ARCH} * 10")
add_executable(pca_bench main.cu)
target_link_libraries(pca_bench PRIVATE holotask curaii benchmark::benchmark CUDA::cuda_driver CUDA::nvrtc CUDA::nvJitLink)
target_compile_definitions(pca_bench PRIVATE PCA_BENCH_SM=${PCA_BENCH_SM})
set_target_properties(pca_bench PROPERTIES CUDA_ARCHITECTURES ${PCA_BENCH_CUDA_ARCH})

target_include_directories(pca_bench SYSTEM PRIVATE "${cusolverdx_cutlass_INCLUDE_DIR}")

configure_file(assets.hh.in assets.hh @ONLY)
target_include_directories(pca_bench PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

if(ENABLE_TESTING)
    add_executable(pca_bench_validation validation.cu)
    target_link_libraries(pca_bench_validation PRIVATE holotask curaii benchmark::benchmark GTest::gtest_main CUDA::cuda_driver CUDA::nvrtc CUDA::nvJitLink)
    target_compile_definitions(pca_bench_validation PRIVATE PCA_BENCH_SM=${PCA_BENCH_SM})
    target_include_directories(pca_bench_validation PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_include_directories(pca_bench_validation SYSTEM PRIVATE "${cusolverdx_cutlass_INCLUDE_DIR}")
    set_target_properties(pca_bench_validation PROPERTIES CUDA_ARCHITECTURES ${PCA_BENCH_CUDA_ARCH})
    add_test(NAME PcaBenchmarkValidation COMMAND pca_bench_validation)
    set_tests_properties(PcaBenchmarkValidation PROPERTIES LABELS gpu RESOURCE_LOCK holoflow_gpu TIMEOUT 600)
endif()
