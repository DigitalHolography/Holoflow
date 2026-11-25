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

"""End-to-end measurements around native graph execution."""

from dataclasses import dataclass, field
import json
import math
from threading import Event, Thread
from time import perf_counter
from typing import Any

from . import _native
from ._diagnostics import Artifacts
from ._graph import trace


@dataclass(frozen=True)
class BenchmarkReport:
    mode: str
    source_node_id: str
    input_frames: int
    steady_input_frames: int
    steady_seconds: float
    input_frames_per_second: float | None
    makespan_seconds: float
    timings: dict[str, float]
    diagnostics_enabled: bool
    perfetto_available: bool
    artifacts: Artifacts | None
    results: Any = field(default=None, repr=False, compare=False)

    def to_json(self):
        """Return JSON-compatible measurements, without collected arrays."""
        return {name: (str(value.directory) if isinstance(value, Artifacts) else value)
                for name, value in vars(self).items() if name != "results"}


def benchmark(pipeline, *, args=(), kwargs=None, mode="finite", source_node_id=None,
              warmup_seconds=2.0, duration_seconds=10.0, log_dir=None, fuse_tasks=True):
    """Measure input frames/s and trace-to-completion wall time.

    Finite mode returns the pipeline's collected results. Looping mode stops after
    warmup plus duration_seconds. Diagnostics are opt-in and finalized on return.
    Source progress includes work admitted to queues, so use a long steady window.
    """
    if mode not in ("finite", "looping"):
        raise ValueError("mode must be 'finite' or 'looping'")
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("warmup_seconds must be finite and nonnegative")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    artifacts = None if log_dir is None else Artifacts.create(log_dir)
    capture = None
    run = None
    timings = {}
    primary_error = None
    try:
        if artifacts is not None:
            capture = _native.ProfileCapture(str(artifacts.directory))
            capture.begin_trace()
        started = perf_counter()
        try:
            graph = trace(pipeline, *args, **({} if kwargs is None else kwargs))
        finally:
            if capture is not None:
                capture.end_trace()
        traced = perf_counter()
        timings["trace_seconds"] = traced - started
        nodes = graph.to_json()["nodes"]
        sources = {key: node for key, node in nodes.items() if node["type"] == "Holofile"}
        if source_node_id is None:
            if len(sources) != 1:
                raise ValueError("Select source_node_id when the graph does not have exactly one Holofile source")
            source_node_id = next(iter(sources))
        if source_node_id not in sources:
            raise ValueError("source_node_id must identify a Holofile source")
        for node in sources.values():
            if node["params"]["loop"] != (mode == "looping"):
                raise ValueError("All Holofile sources must match the benchmark mode")
        batch_size = sources[source_node_id]["params"]["batch_size"]
        compile_started = perf_counter()
        run = graph._compile(artifacts, capture=False, fuse_tasks=fuse_tasks)
        compiled = perf_counter()
        timings["compile_seconds"] = compiled - compile_started
        run.start()
        running = perf_counter()
        timings["startup_seconds"] = running - compiled
        initial = run.progress()
        baseline = initial if warmup_seconds == 0 else None
        done = Event()
        outcome = {}

        def finish():
            try:
                outcome["results"] = run.result() if mode == "finite" else run.wait()
                outcome["end"] = perf_counter()
                outcome["progress"] = run.progress()
            except BaseException as error:
                outcome["error"] = error
            finally:
                done.set()

        if mode == "finite":
            worker = Thread(target=finish, name="Holoflow benchmark completion")
            worker.start()
            try:
                if baseline is None and not done.wait(warmup_seconds):
                    baseline = run.progress()
                done.wait()
            finally:
                if not done.is_set():
                    run.stop()
                worker.join()
        else:
            # Event polling surfaces native failures without consuming task events.
            deadline = running + warmup_seconds
            while perf_counter() < deadline:
                run._get().check_failure()
                done.wait(min(0.01, max(0, deadline - perf_counter())))
            baseline = run.progress()
            deadline = perf_counter() + duration_seconds
            while perf_counter() < deadline:
                run._get().check_failure()
                done.wait(min(0.01, max(0, deadline - perf_counter())))
            outcome["steady_end"] = run.progress()
            stop_started = perf_counter()
            run.stop()
            finish()
            timings["shutdown_seconds"] = perf_counter() - stop_started
        if "error" in outcome:
            raise outcome["error"]
        ended = outcome["end"]
        final = outcome["progress"]
        steady_end = outcome.get("steady_end", final)
        seconds = 0.0 if baseline is None else max(
            0.0, steady_end["timestamp_seconds"] - baseline["timestamp_seconds"])

        def count(snapshot):
            return snapshot["successful_emissions"].get(source_node_id, 0) * batch_size

        steady_frames = 0 if baseline is None else count(steady_end) - count(baseline)
        timings["execution_seconds"] = ended - running
        timings["cleanup_seconds"] = 0.0
        timings["artifact_finalization_seconds"] = 0.0
        report = BenchmarkReport(
            mode, source_node_id, count(final), steady_frames, seconds,
            steady_frames / seconds if seconds > 0 else None,
            ended - started, timings, artifacts is not None,
            _native.perfetto_available(), artifacts, outcome["results"])
    except BaseException as error:
        primary_error = error
        if artifacts is not None:
            error.add_note(f"Holoflow diagnostics: {artifacts.directory}")
        raise
    finally:
        cleanup_started = perf_counter()
        try:
            if run is not None:
                run.close(_finalize_diagnostics=False)
        except BaseException as cleanup_error:
            if primary_error is None:
                primary_error = cleanup_error
                raise
            primary_error.add_note(f"Execution cleanup failed: {cleanup_error}")
        finally:
            timings["cleanup_seconds"] = perf_counter() - cleanup_started
            export_started = perf_counter()
            try:
                if capture is not None:
                    capture.close()
            except BaseException as export_error:
                if primary_error is None:
                    primary_error = export_error
                    raise
                primary_error.add_note(f"Artifact finalization failed: {export_error}")
            finally:
                timings["artifact_finalization_seconds"] = perf_counter() - export_started
                if artifacts is not None and primary_error is not None:
                    try:
                        artifacts.report.write_text(json.dumps({
                            "status": "failed", "error": str(primary_error),
                            "mode": mode, "source_node_id": source_node_id,
                            "timings": timings,
                        }, indent=2), encoding="utf-8")
                    except Exception as report_error:
                        primary_error.add_note(f"Could not write benchmark failure report: {report_error}")
    if artifacts is not None:
        artifacts.report.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    return report
