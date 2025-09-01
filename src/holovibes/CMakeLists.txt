add_executable(holovibes
    src/main.cc
)

target_link_libraries(holovibes PRIVATE
    project_options
)