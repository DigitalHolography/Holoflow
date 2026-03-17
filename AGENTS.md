# Holoflow Agent Guide

## Scope
- Use this file when working in `Holoflow`.
- Treat the repository as a Windows-only MSVC/CUDA project. Do not propose Linux or Clang-first workflows unless the user explicitly asks for them.

## Command Environment
- Run build, configure, test, packaging, Doxygen, and other toolchain-dependent commands only from a `vcvars64`-activated environment.
- The README already requires the Visual Studio x64 native tools environment. Preserve that assumption in every command you run or suggest.
- If you need to execute a one-off command from PowerShell, invoke it through a shell that activates `vcvars64` first so MSVC, Ninja, CMake, and CUDA tool discovery match the expected setup.
- Assume Qt, CUDA, Intel oneAPI MKL, Node.js, and Python are installed on the machine only if the environment or repo setup indicates they are present.

## Repository Shape
- `src/` contains the main libraries and app:
  - `holovibes`: Qt desktop application and main executable.
  - `holoflow`: graph/runtime core library.
  - `holotask`: task implementations.
  - `holonp`: CUDA numeric operators.
  - `curaii`: CUDA wrapper utilities.
  - `holofile`, `holoflow_event`: support libraries.
- `test/` contains GoogleTest-based test executables.
- `bench/` contains benchmark targets.
- `example/` contains example targets.
- `doc/` contains Doxygen setup and MkDocs content.
- `resources/` contains app resources used by `holovibes`.

## Build Workflow
- Use the checked-in CMake presets. The primary configure preset is `msvc-multi`.
- Configure with:
  - `cmake --preset msvc-multi`
- Build with:
  - `cmake --build --preset build-Debug -j`
  - `cmake --build --preset build-Release -j`
- Artifacts are produced under `build/msvc-multi/`.
- Prefer changing CMake options at configure time rather than editing presets unless the task is specifically about project configuration.

## Test Workflow
- Prefer building first, then running the produced test binaries.
- The canonical unit test target currently exposed in the repo is `holoflow_test`.
- Run either:
  - `build\msvc-multi\Debug\holoflow_test.exe`
  - `build\msvc-multi\Release\holoflow_test.exe`
- `CMakePresets.json` also defines `test-Debug` and `test-Release` presets for `ctest`, but the project CI currently runs the executable directly. If `ctest` behaves differently, verify against the direct executable.

## Documentation
- MkDocs content lives in `doc\mkdocs`.
- Run MkDocs commands from `doc\mkdocs`, not the repo root.
- Schema docs are generated from JSON schemas under `src\holovibes\schemas\tasks\...`.
- The helper command documented by the repo is:
  - `generate-schema-doc src\holovibes\schemas\tasks\node_type\xxx_settings.json doc\mkdocs\docs\schemas\node_type\xxx_settings.md --config template_name=md`
- Python documentation commands should run inside the project virtual environment when required by the tool.
- Doxygen is configured through `doc/CMakeLists.txt` and builds the `doc_doxygen` target.

## Packaging
- Packaging is done through CPack from the Release build:
  - `cmake --build --preset build-Release --target package -j`
- Packaging depends on the Windows/MSVC environment, Qt deployment tools, CUDA runtime DLLs, and Intel oneAPI MKL runtime DLLs being discoverable.

## Style And Validation
- Respect `.clang-format` for C++ and CUDA files.
- Respect `.pre-commit-config.yaml`; the repo uses:
  - `addlicense`
  - `clang-format`
- Keep Apache 2.0 license headers intact on files that already use them.
- Follow Conventional Commits if asked to prepare commit messages.

## Practical Editing Guidance
- Prefer minimal, localized changes in the relevant module instead of cross-cutting refactors.
- When touching runtime or task code, inspect adjacent tests under `test/holoflow` or `test/holotask` and add or update coverage when practical.
- When touching schemas or task settings, update the corresponding generated docs if the task requires documentation parity.
- When touching `holovibes`, remember it is a Qt application linked against `Qt6::Core`, `Widgets`, `Gui`, `OpenGL`, `OpenGLWidgets`, and `Graphs`.

## Useful References
- `README.md` for developer setup and canonical commands.
- `CMakeLists.txt` and `CMakePresets.json` for supported build configuration.
- `.github/workflows/main.yml` for the CI build, test, package, and docs flow.
