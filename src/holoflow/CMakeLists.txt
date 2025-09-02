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

set(TARGET holoflow)

file(GLOB_RECURSE HOLOFLOW_PUBLIC_HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holoflow/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holoflow/*.hxx
)

file(GLOB_RECURSE HOLOFLOW_PRIVATE_HEADERS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx
)

file(GLOB_RECURSE HOLOFLOW_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
)

add_library(${TARGET} 
STATIC
    ${HOLOFLOW_SOURCES}
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
    FILES ${HOLOFLOW_PUBLIC_HEADERS}
)

target_sources(${TARGET} 
PRIVATE
    FILE_SET private_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES ${HOLOFLOW_PRIVATE_HEADERS}
)

target_link_libraries(${TARGET} 
PUBLIC
    project_options
    spdlog::spdlog
    curaii
)