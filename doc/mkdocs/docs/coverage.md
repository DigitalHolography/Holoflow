# Test coverage

The coverage report represents the latest successful test run from the `main` branch.

<a href="coverage/index.html">Open the latest interactive coverage report</a>

The report measures native host C++ source lines. CUDA device instructions are validated by the
GPU numerical and regression tests, but they are not represented by native host line coverage.

Pull requests run the Release build and test suite without collecting native coverage. Coverage is
collected from `main` after merge so this report always represents the protected branch.
