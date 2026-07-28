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

include_guard(GLOBAL)
include(GoogleTest)

function(holoflow_discover_tests TARGET)
    cmake_parse_arguments(
        TEST
        ""
        "TIMEOUT;RESOURCE_LOCK;FILTER;PREFIX"
        "LABELS"
        ${ARGN}
    )

    if(NOT TEST_TIMEOUT)
        set(TEST_TIMEOUT 60)
    endif()

    set(DISCOVERY_ARGUMENTS
        DISCOVERY_TIMEOUT 30
    )

    if(TEST_FILTER)
        list(APPEND DISCOVERY_ARGUMENTS TEST_FILTER "${TEST_FILTER}")
    endif()

    if(TEST_PREFIX)
        list(APPEND DISCOVERY_ARGUMENTS TEST_PREFIX "${TEST_PREFIX}")
    endif()

    if(TEST_RESOURCE_LOCK)
        gtest_discover_tests(${TARGET}
            ${DISCOVERY_ARGUMENTS}
            PROPERTIES
                TIMEOUT "${TEST_TIMEOUT}"
                LABELS "${TEST_LABELS}"
                RESOURCE_LOCK "${TEST_RESOURCE_LOCK}"
        )
    else()
        gtest_discover_tests(${TARGET}
            ${DISCOVERY_ARGUMENTS}
            PROPERTIES
                TIMEOUT "${TEST_TIMEOUT}"
                LABELS "${TEST_LABELS}"
        )
    endif()
endfunction()
