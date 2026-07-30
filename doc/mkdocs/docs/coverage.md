# Test coverage

The coverage report represents the latest successful test run from the `main` branch.

<a href="coverage/index.html">Open the latest interactive coverage report</a>

The native collector measures MSVC-instrumentable host C++ code. Results from the test executables
and a complete production image are merged by source line, then grouped under the `curaii`,
`holofile`, `holoflow`, `holoflow_event`, `holonp`, `holotask`, and `holovibes` modules. Host
functions that are retained in the production image but never executed are reported as uncovered.

The coverable-line total is not repository LOC. It excludes comments, blank lines, declarations,
CMake and JSON files, and code emitted only for CUDA devices. CUDA kernels are exercised by GPU
numerical and regression tests, but they are not represented by native host line coverage.

Pull requests run the Release build and test suite without collecting native coverage. Coverage is
collected from `main` after merge so this report always represents the protected branch.
