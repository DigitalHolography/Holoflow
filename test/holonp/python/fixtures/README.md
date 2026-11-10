# holonp oracle fixtures

The checked-in binary files here contain the expected outputs captured from the
NumPy oracle. `python_oracle.cc` fingerprints each operation, settings, input
descriptor, and input payload to select the corresponding files. Normal test
runs read only these files and do not launch Python or import NumPy.

To refresh the corpus deliberately, set `HOLOFLOW_RECORD_ORACLE=1` while
running `holonp_test.exe`. That one-time maintenance path invokes
`../oracle.py` and overwrites the matching golden outputs. Review fixture
changes alongside the related test changes.
