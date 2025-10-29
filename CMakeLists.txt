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

cmake_minimum_required(VERSION 3.24)
project(holoflow VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "Only MSVC compiler is supported.")
endif()

# Required to silent boost warnings on Windows
add_compile_definitions(
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:_WIN32_WINNT=0x0A00>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:_WIN32_WINNT=0x0A00>
)

# NVCC does not support utf-8 correctly
# https://github.com/fmtlib/fmt/issues/4389
add_definitions(-DFMT_UNICODE=0)

option(ENABLE_IPO "IPO/LTO" ON)
option(ENABLE_TESTING "Build tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_DOCUMENTATION "Build documentation" ON)
option(ENABLE_EXAMPLES "Build examples" ON)
option(ENABLE_FETCHCONTENT "Use FetchContent for dependencies" ON)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)
option(ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

# GPU arch list editable by users: -DPROJECT_CUDA_ARCHS=86;89;90
set(PROJECT_CUDA_ARCHS "75;86;89;120-real;120-virtual" CACHE STRING "CUDA archs for CUDA targets")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/ProjectOptions.cmake)
project_options(
    ENABLE_IPO ${ENABLE_IPO}
    ENABLE_WARNINGS ${ENABLE_WARNINGS}
    ENABLE_WARNINGS_AS_ERRORS ${ENABLE_WARNINGS_AS_ERRORS}
)

find_package(EGrabber REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui OpenGL OpenGLWidgets)
qt_standard_project_setup()

add_subdirectory(external)
add_subdirectory(src)

if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

if(ENABLE_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(ENABLE_EXAMPLES)
    add_subdirectory(example)
endif()

if(ENABLE_DOCUMENTATION)
    add_subdirectory(doc)
endif()

# CPack configuration

if(WIN32 AND MSVC)
    install(TARGETS holovibes
        RUNTIME DESTINATION bin COMPONENT binaries
    )
    
    get_target_property(QT_QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(QT_BIN_DIR "${QT_QMAKE_EXECUTABLE}" DIRECTORY)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${QT_BIN_DIR}")
    
    if(NOT WINDEPLOYQT_EXECUTABLE)
        message(WARNING "windeployqt not found. Qt dependencies might be missing in the installer.")
    else()
        message(STATUS "Found windeployqt: ${WINDEPLOYQT_EXECUTABLE}")
        
        install(CODE "
            set(_qt_deps_dir \"\${CMAKE_INSTALL_PREFIX}/_qt_deps\")
            execute_process(
                COMMAND \"${CMAKE_COMMAND}\" -E make_directory \"\${_qt_deps_dir}\"
            )
            
            # Run windeployqt to gather Qt dependencies
            execute_process(
                COMMAND \"${WINDEPLOYQT_EXECUTABLE}\" 
                    --verbose 1
                    --no-compiler-runtime
                    --no-opengl-sw
                    --dir \"\${_qt_deps_dir}\"
                    \"\${CMAKE_INSTALL_PREFIX}/../binaries/bin/holovibes.exe\"
                RESULT_VARIABLE _result
            )
            
            if(NOT _result EQUAL 0)
                message(WARNING \"windeployqt failed: \${_result}\")
            else()
                # Copy Qt dependencies to bin directory
                execute_process(
                    COMMAND \"${CMAKE_COMMAND}\" -E copy_directory \"\${_qt_deps_dir}\" \"\${CMAKE_INSTALL_PREFIX}/bin\"
                )
            endif()
            
            # Clean up temporary directory
            execute_process(
                COMMAND \"${CMAKE_COMMAND}\" -E remove_directory \"\${_qt_deps_dir}\"
            )
        " COMPONENT dependencies)
    endif()

    install(FILES
        "C:/Program Files (x86)/Intel/oneAPI/compiler/2025.2/bin/libiomp5md.dll"
        "C:/Program Files (x86)/Intel/oneAPI/mkl/2025.2/bin/mkl_intel_thread.2.dll"
        DESTINATION bin
        COMPONENT dependencies
    )

    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
    ReadEnvStr \\$0 CUDA_PATH
    StrCmp \\$0 \\\"\\\" 0 cuda_ok
    MessageBox MB_OK|MB_ICONEXCLAMATION \
        'CUDA Toolkit not detected!\\$\\\\n\\$\\\\nPlease install CUDA Toolkit from:\\$\\\\nhttps://developer.nvidia.com/cuda-downloads\\$\\\\n\\$\\\\nNVRTC features will not work without CUDA.'
    cuda_ok:
    ")
endif()

include (CPackConfig)