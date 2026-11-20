# Copyright 2026 Digital Holography Foundation
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

"""Register a CMake Python build in the current virtual environment."""

import argparse
import importlib.machinery
from pathlib import Path
import sys
import sysconfig
import tomllib


def install_example_launchers():
    """Expose configured examples as commands available through uv run."""
    root = Path(__file__).resolve().parents[1]
    settings = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = Path(sysconfig.get_path("scripts"))
    for alias, filename in settings["tool"]["holoflow"]["examples"].items():
        if not alias.startswith("hf-") or not alias.replace("-", "").isalnum():
            raise ValueError(f"Invalid example command: {alias}")
        example = (root / "example/python" / filename).resolve()
        if not example.is_relative_to(root / "example/python") or not example.is_file():
            raise ValueError(f"Missing or invalid example: {filename}")
        # Use the launcher's own interpreter, preserving the chosen CMake ABI.
        script_path = str(example).replace("%", "%%")
        (scripts / f"{alias}.cmd").write_text(
            f'@"%~dp0python.exe" "{script_path}" %*\n', encoding="utf-8")
        print(f"  uv run {alias}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build/msvc-multi")
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Run this helper with uv run python tools/setup_python.py")
    package_root = (args.build_dir / "python" / args.config).resolve()
    package = package_root / "holoflow"
    if not any((package / ("_native" + suffix)).is_file()
               for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        parser.error(f"No compatible native extension in {package}; build with this interpreter first")
    # Import before registering so missing runtime DLLs also fail during setup.
    sys.path.insert(0, str(package_root))
    import holoflow
    destination = Path(sysconfig.get_path("purelib"))
    if not destination.is_relative_to(Path(sys.prefix)):
        parser.error("Python's package directory is outside this virtual environment")
    destination.mkdir(parents=True, exist_ok=True)
    registration = destination / "holoflow-cmake.pth"
    registration.write_text(str(package_root) + "\n", encoding="utf-8")
    print(f"Registered Holoflow {holoflow.__version__}: {package_root}")
    install_example_launchers()


if __name__ == "__main__":
    main()
