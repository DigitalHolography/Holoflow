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
