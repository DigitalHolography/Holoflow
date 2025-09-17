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

set(TARGET curaii)

file(GLOB_RECURSE CURAII_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/curaii/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/curaii/*.hxx
)

file(GLOB_RECURSE CURAII_PRIVATE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hxx
)

file(GLOB_RECURSE CURAII_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
)

add_library(${TARGET}
STATIC
    ${CURAII_SOURCES}
)

target_include_directories(${TARGET}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{CUDA_PATH}/include
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_sources(${TARGET}
PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES ${CURAII_PUBLIC_HEADERS}
)

target_sources(${TARGET}
PRIVATE
    FILE_SET private_headers
    TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    FILES ${CURAII_PRIVATE_HEADERS}
)

target_link_directories(${TARGET}
PUBLIC
    $ENV{CUDA_PATH}/lib/x64
)

target_link_libraries(${TARGET}
PUBLIC
    project_options
    cudart
    cublas
    spdlog::spdlog
)