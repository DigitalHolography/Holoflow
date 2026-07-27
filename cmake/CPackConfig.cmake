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

set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION bin)
set(CMAKE_INSTALL_SYSTEM_RUNTIME_COMPONENT dependencies)
set(CMAKE_INSTALL_OPENMP_LIBRARIES ON)
include(InstallRequiredSystemLibraries)

file(READ "${CMAKE_SOURCE_DIR}/VERSION" PROJECT_VERSION_RAW)
string(STRIP "${PROJECT_VERSION_RAW}" PROJECT_VERSION_FULL)

# installer rules. 
set(CPACK_PACKAGE_NAME "Holovibes")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Holoflow - Digital Holography Software")
set(CPACK_PACKAGE_VENDOR "Digital Holography Foundation")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION_FULL})
set(CPACK_PACKAGE_FILE_NAME "Holoflow-${PROJECT_VERSION_FULL}-win64")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Holovibes\\\\${PROJECT_VERSION_FULL}")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "Holovibes ${PROJECT_VERSION_FULL}")
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL OFF)

set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/holovibes/assets/holovibes_logo.ico")
set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\holovibes.exe")
set(CPACK_NSIS_DISPLAY_NAME "Holovibes ${PROJECT_VERSION_FULL}")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall Holovibes ${PROJECT_VERSION_FULL}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

set(CPACK_GENERATOR "NSIS")

include(CPackComponent)

cpack_add_component(binaries
    DISPLAY_NAME "Holovibes"
    DESCRIPTION "Holovibes application binaries."
    REQUIRED
)

cpack_add_component(dependencies
    DISPLAY_NAME "Runtime dependencies"
    DESCRIPTION "Runtime libraries required by Holovibes."
    REQUIRED
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
