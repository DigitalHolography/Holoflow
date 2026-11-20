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

"""Locations of opt-in diagnostic artifacts."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class Artifacts:
    directory: Path

    @classmethod
    def create(cls, parent):
        parent = Path(parent).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S-")
        return cls(Path(tempfile.mkdtemp(prefix=prefix, dir=parent)))

    @property
    def descriptor_json(self):
        return self.directory / "descriptor.json"

    @property
    def descriptor_dot(self):
        return self.directory / "descriptor.dot"

    @property
    def compiled_dot(self):
        return self.directory / "compilation_success.dot"

    @property
    def failure_dot(self):
        return self.directory / "compilation_failure.dot"

    @property
    def compiler_log(self):
        return self.directory / "compiler.log"

    @property
    def perfetto(self):
        return self.directory / "trace_events.perfetto-trace"

    @property
    def report(self):
        return self.directory / "benchmark.json"

    def files(self):
        """Return existing files, including per-section CUDA graph diagnostics."""
        return tuple(sorted(p for p in self.directory.rglob("*") if p.is_file()))
