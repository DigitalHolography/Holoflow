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
- Windows 10/11 x64
- **MSVC** (Visual Studio Build Tools 2022) and SDK
- **CMake >= 3.24**
- **Ninja** (Multi-Config)
- **CUDA Toolkit** (13.0+ recommended)
- **x64 Native Tools Command Prompt** (for running all commands)
- **Python >= 3.10**
- **Git**

## Python dependencies
The python dependencies are listed in `requirements.txt`. The recommended way to set up the Python environment is by using a virtual environment. You can create and activate a virtual environment using the following commands:

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\venv\Scripts\activate.bat
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
build\msvc-multi\src\holovibes\Release\Release\holovibes.exe --help
```

## Test
> TODO

## Benchmarks
> TODO

## Documentation
> TODO

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
