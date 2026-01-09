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

include(InstallRequiredSystemLibraries)

file(READ "${CMAKE_SOURCE_DIR}/VERSION" PROJECT_VERSION_RAW)
string(STRIP "${PROJECT_VERSION_RAW}" PROJECT_VERSION_FULL)

# installer rules. 
set(CPACK_PACKAGE_NAME "Holovibes")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Holoflow - Digital Holography Software")
set(CPACK_PACKAGE_VENDOR "Holoflow Developers")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION_FULL})
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)

set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/holovibes/assets/holovibes_logo.ico")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_VERSION_FULL}")
set(CPACK_NSIS_INSTALLED_ICON_NAME "${CMAKE_CURRENT_SOURCE_DIR}/resources/holovibes/assets/holovibes_logo.ico")
set(CPACK_NSIS_DISPLAY_NAME "Holovibes ${PROJECT_VERSION_FULL}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

set(CPACK_GENERATOR "NSIS")

include(CPackComponent)

cpack_add_install_type(Full DISPLAY_NAME "Install Everything")

cpack_add_component(binaries
    DISPLAY_NAME "Holovibes Binaries"
    DESCRIPTION "This will install the main application."
    REQUIRED
    INSTALL_TYPES Full 
)

cpack_add_component(dependencies
    DISPLAY_NAME "Holovibes Dependencies"
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

set(CPACK_NSIS_CREATE_ICONS_EXTRA "
    CreateShortCut '$DESKTOP\\\\Holovibes ${PROJECT_VERSION_FULL}.lnk' '$INSTDIR\\\\bin\\\\holovibes.exe'
")

set(CPACK_NSIS_DELETE_ICONS_EXTRA "
    Delete '$DESKTOP\\\\Holovibes ${PROJECT_VERSION_FULL}.lnk'
")

include(CPack)