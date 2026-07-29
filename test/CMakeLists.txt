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

enable_testing()

include(Testing)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
    RESULT_VARIABLE HOLOFLOW_NUMPY_IMPORT_RESULT
    OUTPUT_QUIET
    ERROR_VARIABLE HOLOFLOW_NUMPY_IMPORT_ERROR
)
if(NOT HOLOFLOW_NUMPY_IMPORT_RESULT EQUAL 0)
    message(FATAL_ERROR
        "The test suite requires NumPy for ${Python3_EXECUTABLE}: "
        "${HOLOFLOW_NUMPY_IMPORT_ERROR}"
    )
endif()

file(TO_CMAKE_PATH "${Python3_EXECUTABLE}" HOLOFLOW_TEST_PYTHON_EXECUTABLE)

add_subdirectory(holoflow)
add_subdirectory(holonp)
add_subdirectory(holotask)
add_subdirectory(holovibes)

if(ENABLE_COVERAGE)
    add_test(
        NAME coverage::production_image
        COMMAND holovibes --coverage-probe
    )
    set_tests_properties(coverage::production_image PROPERTIES
        LABELS coverage
        TIMEOUT 10
    )
endif()
