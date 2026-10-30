# Copyright 2026 Digital Holography Foundation
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

find_package(cufftdx REQUIRED CONFIG PATHS "${mathdx_DIR}/../cufftdx")
add_executable(fresnel_propagation_bench main.cu)
target_link_libraries(fresnel_propagation_bench PRIVATE holotask curaii cufftdx::cufftdx benchmark::benchmark)
set(FRESNEL_BENCH_CUDA_ARCH "86" CACHE STRING "Fresnel benchmark CUDA architecture (86 or 120)")
if(NOT FRESNEL_BENCH_CUDA_ARCH MATCHES "^(86|120)$")
    message(FATAL_ERROR "FRESNEL_BENCH_CUDA_ARCH must be 86 or 120")
endif()
math(EXPR FRESNEL_BENCH_SM "${FRESNEL_BENCH_CUDA_ARCH} * 10")
target_compile_definitions(fresnel_propagation_bench PRIVATE FRESNEL_BENCH_SM=${FRESNEL_BENCH_SM})
set_target_properties(fresnel_propagation_bench PROPERTIES CUDA_ARCHITECTURES ${FRESNEL_BENCH_CUDA_ARCH})

if(ENABLE_TESTING)
    add_executable(fresnel_propagation_validation validation.cu)
    target_link_libraries(fresnel_propagation_validation PRIVATE holotask curaii cufftdx::cufftdx benchmark::benchmark GTest::gtest_main)
    target_compile_definitions(fresnel_propagation_validation PRIVATE FRESNEL_BENCH_SM=${FRESNEL_BENCH_SM})
    set_target_properties(fresnel_propagation_validation PROPERTIES CUDA_ARCHITECTURES ${FRESNEL_BENCH_CUDA_ARCH})
    add_test(NAME FresnelBenchmarkValidation COMMAND fresnel_propagation_validation)
    set_tests_properties(FresnelBenchmarkValidation PROPERTIES LABELS gpu RESOURCE_LOCK holoflow_gpu TIMEOUT 600)
endif()
