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

add_subdirectory(holoflow)
add_subdirectory(holonp)
add_subdirectory(holotask)
add_subdirectory(holovibes)
add_subdirectory(holoflow_event)

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
