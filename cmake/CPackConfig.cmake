include(InstallRequiredSystemLibraries)

# installer rules. 
set(CPACK_PACKAGE_NAME "Holoflow")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Holoflow - Digital Holography Software")
set(CPACK_PACKAGE_VENDOR "Holoflow Developers")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
# set(CPACK_NSIS_MODIFY_PATH ON)

set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/holovibes/assets/holovibes_logo.ico")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Holoflow ${PROJECT_VERSION}")
set(CPACK_NSIS_INSTALLED_ICON_NAME "${CMAKE_CURRENT_SOURCE_DIR}/resources/holovibes/assets/holovibes_logo.ico")
set(CPACK_NSIS_DISPLAY_NAME "Holoflow ${PROJECT_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

set(CPACK_GENERATOR "NSIS")

include(CPackComponent)

cpack_add_install_type(Full DISPLAY_NAME "Install Everything")

cpack_add_component(binaries
    DISPLAY_NAME "Holoflow Binaries"
    DESCRIPTION "This will install the main application."
    REQUIRED
    INSTALL_TYPES Full 
)

cpack_add_component(dependencies
    DISPLAY_NAME "Holoflow Dependencies"
    DESCRIPTION "This will install the required dependencies."
    REQUIRED
    INSTALL_TYPES Full 
)

set(CPACK_COMPONENTS_ALL binaries dependencies)

if (CMAKE_CL_64)
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
else (CMAKE_CL_64)
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
endif (CMAKE_CL_64)

include(CPack)