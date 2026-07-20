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

set(TARGET holovibes_test)

set(SOURCES
    signal_history_test.cc
    settings_test.cc
    ../../src/holovibes/src/pipeline/validation.cc
    ../../src/holovibes/src/settings_loader.cc
    ../../src/holovibes/src/ui/widgets/signal_history.cc
)

add_executable(${TARGET} ${SOURCES})

target_include_directories(${TARGET}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/holovibes/src
)

target_link_libraries(${TARGET}
    PRIVATE
    project_options
    nlohmann_json::nlohmann_json
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(${TARGET})

