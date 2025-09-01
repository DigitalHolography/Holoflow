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
- **CMake ≥ 3.24**
- **Ninja** (Multi-Config)
- **CUDA Toolkit** (13.0+ recommended)
- **x64 Native Tools Command Prompt** (for running all commands)

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

## License
This project is licensed under the Apache License 2.0.
See [LICENSE](LICENSE) for details.
