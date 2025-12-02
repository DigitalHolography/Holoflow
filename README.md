# Holoflow

Ultra high throughput image processing for laser doppler holography applied
to retinal imaging.

## Features
> TODO

## Project layout
```
holoflow
|   .clang-format
|   CMakeLists.txt
|   CMakePresets.json
|
+---bench/
|   |   CMakeLists.txt
|   +---curaii/      CMakeLists.txt  ...
|   +---holofile/    CMakeLists.txt  ...
|   +---holoflow/    CMakeLists.txt  ...
|   \---holovibes/   CMakeLists.txt  ...
|
+---cmake/
|       ProjectOptions.cmake
|
+---doc/
|       CMakeLists.txt
|
+---external/
|       CMakeLists.txt
|
+---src/
|   |   CMakeLists.txt
|   +---curaii/
|   |   |   CMakeLists.txt
|   |   +---include/curaii/*.hh
|   |   \---src/*.cc *.cu
|   +---holofile/
|   |   |   CMakeLists.txt
|   |   +---include/holofile/*.hh
|   |   \---src/*.cc
|   +---holoflow/
|   |   |   CMakeLists.txt
|   |   +---include/holoflow/*.hh
|   |   \---src/*.cc *.cu
|   \---holovibes/
|       |   CMakeLists.txt
|       \---src/
|           main.cc  ...
|
\---test/
    |   CMakeLists.txt
    +---curaii/      CMakeLists.txt  ...
    +---holofile/    CMakeLists.txt  ...
    +---holoflow/    CMakeLists.txt  ...
    \---holovibes/   CMakeLists.txt  ...
```

## Prerequisites
- **Windows 10/11 x64**
- **[MSVC](https://visualstudio.microsoft.com/)** (Visual Studio Build Tools 2022) and SDK
- **CMake >= 3.24** usually comes with VS
- **Ninja** (Multi-Config) usually comes with VS
- **[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)** (13.0+ recommended)
- **x64 Native Tools Command Prompt** (for running all commands)
- **[Python](https://www.python.org/downloads/) >= 3.10**
- **[Git](https://git-scm.com/downloads/win)**
- **[Qt >= 6.5](https://www.qt.io/download-qt-installer-oss)** set `Qt6_DIR` env var to your Qt installation path, e.g. `C:\Qt\6.5.2\msvc2019_64\lib\cmake\Qt6`, add `C:\Qt\6.5.2\msvc2019_64\bin` to your `PATH`
- **[Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)** add `C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin` to your `PATH`
- **[Node.js](https://nodejs.org/en/download/)** (both `node` and `npm` in your `PATH`)
- **[quicktype](https://www.npmjs.com/package/quicktype)** install via `npm install -g quicktype`

## Python dependencies
The python dependencies are listed in `requirements.txt`. The recommended way to set up the Python environment is by using a virtual environment. You can create and activate a virtual environment using the following commands:

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\activate.bat
```

After activating the virtual environment, you can install the required Python packages using:

```powershell
pip install -r requirements.txt
```

## Configure
Ninja Multi-Config is used. One configure step, many build configs.
```powershell
cmake --preset msvc-multi
```

### Common cache variables
- `PROJECT_CUDA_ARCHS` (string list, default `75;86;89;90`)
- `ENABLE_TESTING` (ON/OFF)
- `ENABLE_BENCHMARKS` (ON/OFF)
- `ENABLE_DOCUMENTATION` (ON/OFF)
- `ENABLE_FETCHCONTENT` (ON/OFF)
- `ENABLE_WARNINGS` (ON/OFF)
- `ENABLE_WARNINGS_AS_ERRORS` (ON/OFF)
- `ENABLE_IPO` (ON/OFF)

Set at configure time:
```powershell
cmake --preset msvc-multi -DPROJECT_CUDA_ARCHS="86;89" -DENABLE_DOCUMENTATION=ON
```

## Build
```powershell
cmake --build --preset build-Debug   -j
cmake --build --preset build-Release -j
```
Artifacts appear under `build/msvc-multi/`.

## Run
The main application is `holovibes`.
```powershell
# Example
build\msvc-multi\Release\holovibes.exe --help
```

## Test
The individual tests can be run from cmd:
```powershell
build\msvc-multi\Debug\holoflow_test.exe
```

## Benchmarks
> TODO

## Documentation
This project’s documentation is built using **MkDocs** and organized into a structured set of Markdown (`.md`) files.

Link to the [**documentation**](https://www.google.com). <!-- #TODO : Change the link to the real documentation link when it will be up -->

### Build and Run
> Before building or serving the documentation, ensure that all **required Python dependencies** are installed. If not, you can refer to the Python dependencies section of the README.

Make sure you are working inside the project’s virtual environment. If it is not already activated, you can activate it with:
```cmd
.\.env\Scripts\activate
```
Navigate to the documentation directory, which contains the `mkdocs.yml` configuration file:
```cmd
cd doc\mkdocs
```
> You must run all MkDocs commands from the directory where the mkdocs.yml file is located.

You can now start the MkDocs development server:
```cmd
mkdocs serve
```
This command will launch a local web server where you can preview the documentation during development.

### Generate JSON schema
The JSON schemas are generated using the `json-schema-for-humans` dependency. They are used to display the settings associated of each node.

To generate a new schema, you must first create the corresponding `xxx_settings.json` file inside the `src/holovibes/schemas/tasks/` directory.

Once the file is created, run the following command:
```cmd
generate-schema-doc src\holovibes\schemas\tasks\node_type\xxx_settings.json doc\mkdocs\docs\schemas\node_type\xxx_settings.md --config template_name=md
```
> You must run this command in Python Virtual Environment

## Dependency management
- **FetchContent**: enabled by `ENABLE_FETCHCONTENT=ON`. Dependencies declared in `external/CMakeLists.txt`.

## Git
This project uses Git for version control. Make sure to have Git installed and configured on your system.
This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification.
This project comes with pre-commit hooks to enforce coding standards.
You can enable these hooks by running:
```powershell
pre-commit install
```

## License
This project is licensed under the Apache License 2.0.
See [LICENSE](LICENSE) for details.
