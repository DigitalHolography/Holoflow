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

add_executable(holoflow_core_test
    unit/core/graph_spec_test.cc
    unit/core/tensor_test.cc
    unit/core/task_test.cc
    unit/core/registry_test.cc
)

target_include_directories(holoflow_core_test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(holoflow_core_test
    PRIVATE
    holoflow
    holoflow_event
    GTest::gtest_main
)

holoflow_discover_tests(holoflow_core_test
    LABELS fast
    TIMEOUT 10
)

add_executable(holoflow_runtime_test
    unit/runtime/compiler_additional_test.cc
    unit/runtime/compiler_scheduler_edge_test.cc
    unit/runtime/graph_display_test.cc
    unit/runtime/compiler_test.cc
    functional/functional_device_test.cu
    functional/functional_test.cc
    unit/runtime/scheduler_additional_test.cc
    unit/runtime/scheduler_functional_test.cc
)

target_include_directories(holoflow_runtime_test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(holoflow_runtime_test
    PRIVATE
    holoflow
    GTest::gtest_main
)

holoflow_discover_tests(holoflow_runtime_test
    LABELS gpu
    TIMEOUT 60
    RESOURCE_LOCK holoflow_gpu
)
