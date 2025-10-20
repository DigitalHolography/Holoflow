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

find_path(EGRABBER_INCLUDE_DIR
        NAMES EGrabber.h
        PATHS
        ENV EGRABBER_ROOT
        "C:/Program Files/Euresys/eGrabber/include"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EGrabber
        REQUIRED_VARS EGRABBER_INCLUDE_DIR
)

if (EGrabber_FOUND)
    set(EGRABBER_INCLUDE_DIRS ${EGRABBER_INCLUDE_DIR})

    add_library(EGrabber INTERFACE)
    target_include_directories(EGrabber INTERFACE
            SYSTEM
            $<BUILD_INTERFACE:${EGRABBER_INCLUDE_DIRS}>
    )
    target_compile_options(EGrabber INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:
            $<$<CXX_COMPILER_ID:MSVC>:
            /external:W0;
            /external:anglebrackets;
            >
            >
    )
    mark_as_advanced(EGRABBER_INCLUDE_DIR)
endif ()