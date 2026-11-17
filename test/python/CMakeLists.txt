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

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(NAME python::tracing
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${CMAKE_BINARY_DIR}/python/$<CONFIG>"
        ${Python3_EXECUTABLE} -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}" -p test_tracing.py
)
set_tests_properties(python::tracing PROPERTIES LABELS fast TIMEOUT 60)

add_test(NAME python::doppler
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${CMAKE_BINARY_DIR}/python/$<CONFIG>"
        ${Python3_EXECUTABLE} -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}" -p test_execution.py
)
set_tests_properties(python::doppler PROPERTIES
    LABELS gpu TIMEOUT 180 RESOURCE_LOCK holoflow_gpu
)

add_test(NAME python::finite
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${CMAKE_BINARY_DIR}/python/$<CONFIG>"
        ${Python3_EXECUTABLE} -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}" -p test_finite.py
)
set_tests_properties(python::finite PROPERTIES LABELS gpu TIMEOUT 120 RESOURCE_LOCK holoflow_gpu)

add_test(NAME python::fresnel_viewer
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${CMAKE_BINARY_DIR}/python/$<CONFIG>"
        ${Python3_EXECUTABLE} -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}" -p test_fresnel_viewer.py
)
set_tests_properties(python::fresnel_viewer PROPERTIES LABELS gpu TIMEOUT 180 RESOURCE_LOCK holoflow_gpu)

add_test(NAME python::aberration
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${CMAKE_BINARY_DIR}/python/$<CONFIG>"
        ${Python3_EXECUTABLE} -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}" -p test_aberration.py
)
set_tests_properties(python::aberration PROPERTIES LABELS gpu TIMEOUT 180 RESOURCE_LOCK holoflow_gpu)
