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
    core_test.cc
    event_router_test.cc
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
    compiler_test.cc
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
