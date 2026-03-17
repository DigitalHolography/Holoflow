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

function(project_options)
    cmake_parse_arguments(OP "" "ENABLE_IPO;ENABLE_WARNINGS;ENABLE_WARNINGS_AS_ERRORS" "" ${ARGN})

    add_library(project_options INTERFACE)

    target_compile_features(project_options INTERFACE
        cxx_std_20
        cuda_std_20
    )

    set_target_properties(project_options PROPERTIES
        INTERFACE_CXX_STANDARD 20
        INTERFACE_CXX_STANDARD_REQUIRED ON
        INTERFACE_CXX_EXTENSIONS OFF
        INTERFACE_CUDA_STANDARD 20
        INTERFACE_CUDA_STANDARD_REQUIRED ON
        INTERFACE_CUDA_EXTENSIONS OFF
    )

    target_compile_definitions(project_options INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:_WIN32_WINNT=0x0A00>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:_WIN32_WINNT=0x0A00>
    )

    set(_MSVC_COMMON
        /utf-8
        /Zc:preprocessor
        /openmp
        /external:anglebrackets
        /external:W0
        /wd4211
    )
    set(_NVCC_COMMON
        --generate-line-info
        --extended-lambda
        -Wno-deprecated-gpu-targets
        --expt-relaxed-constexpr
    )
    string(JOIN "," _MSVC_COMMON_CSV ${_MSVC_COMMON})
    target_compile_options(project_options INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${_MSVC_COMMON}>
        $<$<COMPILE_LANGUAGE:CUDA>:${_NVCC_COMMON}>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=${_MSVC_COMMON_CSV}>
    )

    if(OP_ENABLE_IPO)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT ok)

        if(ok)
            set_target_properties(project_options PROPERTIES
                INTERFACE_INTERPROCEDURAL_OPTIMIZATION ON
            )
        else()
            message(FATAL_ERROR "Interprocedural optimization is not supported. Disable it
            with -DOP_ENABLE_IPO=OFF flag.")
        endif()
    endif()

    if(OP_ENABLE_WARNINGS)
        set(_MSVC_WARN
            /W4
            /permissive-
            /wd4456
            /wd4505
            /wd5046
            /wd4702
            /wd4251
            /wd4275
            /wd4324
        )
        set(_NVCC_WARN
            -Xcudafe="--diag_suppress=1394"
            -Xcudafe="--diag_suppress=27"
            -Xcudafe="--diag_suppress=1388"
        )
        string(JOIN "," _MSVC_WARN_CSV ${_MSVC_WARN})
        target_compile_options(project_options INTERFACE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${_MSVC_WARN}>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=${_MSVC_WARN_CSV}>
            $<$<COMPILE_LANGUAGE:CUDA>:${_NVCC_WARN}>
        )
    endif()

    if(OP_ENABLE_WARNINGS_AS_ERRORS)
        set(_MSVC_WERROR
            /WX
        )
        set(_NVCC_WERROR
        )
        string(JOIN "," _MSVC_WERROR_CSV ${_MSVC_WERROR})
        target_compile_options(project_options INTERFACE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:${_MSVC_WERROR}>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=${_MSVC_WERROR_CSV}>
            $<$<COMPILE_LANGUAGE:CUDA>:${_NVCC_WERROR}>
        )
    endif()
endfunction()
