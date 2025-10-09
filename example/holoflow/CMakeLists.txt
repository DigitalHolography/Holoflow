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

set(TARGET holoflow_examples)

file(GLOB_RECURSE HOLOFLOW_EXAMPLES_PUBLIC_HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holoflow/examples/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holoflow/examples/*.hxx
)

file(GLOB_RECURSE HOLOFLOW_EXAMPLES_PRIVATE_HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx
)

file(GLOB_RECURSE HOLOFLOW_EXAMPLES_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(${TARGET}
STATIC
    ${HOLOFLOW_EXAMPLES_SOURCES}
)

target_include_directories(${TARGET} 
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_sources(${TARGET}
PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${HOLOFLOW_EXAMPLES_PUBLIC_HEADERS}
)

target_sources(holoflow_examples 
PRIVATE
    FILE_SET private_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES ${HOLOFLOW_EXAMPLES_PRIVATE_HEADERS}
)

target_link_libraries(holoflow_examples
PUBLIC
    project_options
    holoflow
    curaii
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Boost::graph
)
