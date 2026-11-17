find_package(Python3 3.14 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
if(ENABLE_FETCHCONTENT)
    declare_persistent_dependency(pybind11 "https://github.com/pybind/pybind11.git" "v3.1.0")
    FetchContent_MakeAvailable(pybind11)
else()
    find_package(pybind11 3.1.0 EXACT CONFIG REQUIRED)
endif()

pybind11_add_module(holoflow_python MODULE bindings.cc)
target_link_libraries(holoflow_python PRIVATE holoflow holonp holotask)
set_target_properties(holoflow_python PROPERTIES
    OUTPUT_NAME _native
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/$<CONFIG>/holoflow"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/$<CONFIG>/holoflow"
)
target_compile_definitions(holoflow_python PRIVATE
    HOLOFLOW_PYTHON_VERSION="${PROJECT_VERSION_FULL}"
)
add_custom_command(TARGET holoflow_python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/python/holoflow"
        "$<TARGET_FILE_DIR:holoflow_python>"
)
# Pure Python edits must be staged even when the extension does not need relinking.
add_custom_target(holoflow_python_package ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/python/holoflow"
        "${CMAKE_BINARY_DIR}/python/$<CONFIG>/holoflow"
    DEPENDS holoflow_python
)

file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/bin" PYTHON_CUDA_BIN)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/python/$<CONFIG>/holoflow/_build_config.py"
    CONTENT "CUDA_BIN = r\"${PYTHON_CUDA_BIN}\"\n"
)
file(GLOB PYTHON_CUDA_DLLS
    "${PYTHON_CUDA_BIN}/cudart64*.dll" "${PYTHON_CUDA_BIN}/cublas*.dll"
    "${PYTHON_CUDA_BIN}/cufft*.dll" "${PYTHON_CUDA_BIN}/cusolver*.dll"
    "${PYTHON_CUDA_BIN}/cusparse*.dll" "${PYTHON_CUDA_BIN}/nvrtc*.dll"
    "${PYTHON_CUDA_BIN}/nvJitLink*.dll"
    "${PYTHON_CUDA_BIN}/x64/cudart64*.dll" "${PYTHON_CUDA_BIN}/x64/cublas*.dll"
    "${PYTHON_CUDA_BIN}/x64/cufft*.dll" "${PYTHON_CUDA_BIN}/x64/cusolver*.dll"
    "${PYTHON_CUDA_BIN}/x64/cusparse*.dll" "${PYTHON_CUDA_BIN}/x64/nvrtc*.dll"
    "${PYTHON_CUDA_BIN}/x64/nvJitLink*.dll"
)
install(FILES ${PYTHON_CUDA_DLLS} DESTINATION python/holoflow/_dlls COMPONENT python)
install(TARGETS holoflow_python
    LIBRARY DESTINATION python/holoflow COMPONENT python
    RUNTIME DESTINATION python/holoflow COMPONENT python
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/python/holoflow/"
    DESTINATION python/holoflow COMPONENT python
    PATTERN __pycache__ EXCLUDE
)
