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
- **[Qt >= 6.5](https://www.qt.io/download-qt-installer-oss)** with `Qt Graphs` package enabled during a custom installation. Set `Qt6_DIR` env var to your Qt installation path, e.g. `C:\Qt\6.5.2\msvc2019_64\lib\cmake\Qt6`, add `C:\Qt\6.5.2\msvc2019_64\bin` to your `PATH`
- **[Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)** add `C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin` and `C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin` to your `PATH`
- **[LLVM](https://github.com/llvm/llvm-project/releases/tag/llvmorg-22.1.0) (22.2.0+ recommended)

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
- `PROJECT_CUDA_ARCHS` (string list, default `75;86;89;120-real;120-virtual`)
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
Build the desired configuration first, then use the CTest presets:

```powershell
ctest --preset test-fast
ctest --preset test-gpu
ctest --preset test-Release
```

`test-fast` contains deterministic host tests. `test-gpu` contains CUDA tests and serializes
access to the GPU. `test-Release` runs every CTest-registered Release test except tests labelled
`performance`.

### Coverage

Coverage collection is supported on Windows and requires:

- Visual Studio with **Desktop development with C++** and the **Code coverage tools** component
- the .NET SDK (`dotnet`)
- PowerShell 7 (`pwsh`)
- the CUDA toolkit and other normal project build prerequisites

Verify that the two tools used by the coverage script are available:

```powershell
Microsoft.CodeCoverage.Console --help
dotnet --info
```

From the repository root, configure and build the dedicated instrumented tree:

```powershell
cmake --preset msvc-coverage
cmake --build --preset build-Coverage -j 4
```

The first build can take several minutes because it builds the complete production image and all
CUDA tests. Do not run another configure or build against the same tree concurrently.

Run every non-performance test under the native collector and generate the reports:

```powershell
pwsh -NoProfile -File tools/run_coverage.ps1
```

The script also restores the repository-pinned ReportGenerator .NET tool. On success, open the
interactive report:

```powershell
Start-Process build\coverage\html\index.html
```

Generated artifacts are written under `build/coverage`:

- `coverage.cobertura.xml`: source-line coverage grouped by production module
- `coverage.raw.cobertura.xml`: ungrouped collector output
- `test-results.junit.xml`: results from the instrumented CTest run
- `html/index.html`: interactive HTML report

To print the files with the largest Holoflow gaps:

```powershell
pwsh -NoProfile -File tools/report_coverage_gaps.ps1 -Module holoflow -Limit 50
```

To inspect another production module, replace `holoflow` with `curaii`, `holofile`,
`holoflow_event`, `holonp`, `holotask`, or `holovibes`.

If sources or tests changed after the last coverage build, rebuild `build-Coverage` before
collecting again. If CTest reports missing executables, the coverage build did not finish
successfully. If `Microsoft.CodeCoverage.Console` is not found, add the Visual Studio coverage
component and run the commands from a fresh terminal.

CUDA device instructions are validated by numerical and regression tests but are not represented by
native host line coverage.

The report is grouped by production module and merges repeated source lines from every test
executable. A coverage-only launch of the complete application image retains otherwise unreferenced
host code so that untouched functions are reported as uncovered. The coverable-line count therefore
represents MSVC-instrumentable source lines, not physical repository LOC: comments, blank lines,
declarations, CMake/JSON files, and code emitted only for CUDA devices are not part of that
denominator.

## Package

This project uses **CMake**, **CPack**, and the **NSIS installer generator** to produce a Windows installer (`.exe`) that contains the compiled application and all required Qt runtime libraries.

### Prerequisites

Before packaging, make sure these following tools are installed :

- **[NSIS >= 3.03](https://nsis.sourceforge.io/Download)**

### Command

To generate the installer, run:

```powershell
cmake --build --preset build-Release --target package -j
```

It will generate the installer inside the `build\msvc-multi\` folder.

Each version installs independently under
`Program Files\Holovibes\<version>` and creates a versioned desktop shortcut.
Installing or uninstalling one version does not remove other installed versions.

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
This project uses [Commitizen](https://commitizen-tools.github.io/commitizen/)
for version bumps, changelog generation, and release tags. Install the pinned
release tool and enable both hook types:

```powershell
pipx install commitizen==4.16.3
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

To prepare a release, first run the manually dispatched Windows Verification
workflow against `main`. Once it succeeds, create the version commit and tag
locally. Development versions use an explicit version to preserve the
`dev.<number>` spelling:

```powershell
cz bump 0.2.0-dev.1 --check-consistency
git push origin main --follow-tags
```

Only protected `v*` tags publish installers. Internal pull requests run Release
builds and tests on the persistent self-hosted Windows runner. Pull request
workflows from forks must never be approved; review their changes first, then
move trusted commits to a branch in this repository before running Windows CI.

Native coverage runs on `main` after merge and is published with the
documentation. Packaging is additionally checked on pull requests that change
build, dependency, runtime asset, or packaging inputs.

## License
This project is licensed under the Apache License 2.0.
See [LICENSE](LICENSE) for details.
