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

set(TARGET holonp_test)

# abs_test.cu is compiled by NVCC (host code only, no kernels) so that the
# linker correctly resolves device code from the holonp static library.
set(SOURCES
    abs_test.cu
    add_test.cu
    arange_test.cu
    argmax_test.cu
    asarray_test.cu
    ascontiguousarray_test.cu
    support/tensor_test_buffer.cc
    support/sync_task_runner.cc
    support/python_oracle.cc
)

add_executable(${TARGET} ${SOURCES})

target_include_directories(${TARGET}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/support
)

target_link_libraries(${TARGET}
    PRIVATE
    holonp
    GTest::gtest_main
)

# Bake the absolute path to the oracle script into the binary at configure time.
# The escaped inner quotes produce a C string literal: #define HOLONP_TEST_ORACLE_SCRIPT "..."
target_compile_definitions(${TARGET}
    PRIVATE
    "HOLONP_TEST_ORACLE_SCRIPT=\"${CMAKE_CURRENT_SOURCE_DIR}/python/oracle.py\""
)

include(GoogleTest)
gtest_discover_tests(${TARGET})
