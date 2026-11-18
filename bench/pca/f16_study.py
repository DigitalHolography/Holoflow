"""Run the production FP16 covariance study from the repository root."""
import argparse
import csv
import io
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess

release = Path("build/msvc-multi/Release")
artifacts = Path("build/pca-f16-study")
artifacts.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, HOLOFLOW_PCA_F16_STUDY="1", HOLOFLOW_PCA_BENCHMARK_MS="3000")
test_filter = "PcaF16Production.*:PcaPipeline.F16QueueWraparoundAndGraphReplay"


def run(*args, log=None):
    if log is None:
        subprocess.run(args, check=True, env=env)
        return ""
    result = subprocess.run(args, check=False, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    (artifacts / log).write_text(result.stdout, encoding="utf-8")
    result.check_returncode()
    return result.stdout


def build(mode):
    run("cmake", "--preset", "msvc-multi", "-DPCA_BENCH_CUDA_ARCH=120",
        f"-DPCA_F16_LOWER_TRIANGLE={mode}")
    run("cmake", "--build", "--preset", "build-Release", "--target", "pca_bench", "holotask_test")
    for name in ("pca_bench", "holotask_test"):
        shutil.copy2(release / f"{name}.exe", executable(name, mode))


def executable(name, mode):
    return release / f"{name}-f16-{mode.lower()}.exe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys", type=Path, help="Path to the Nsight Systems CLI")
    args = parser.parse_args()
    checks = {}
    try:
        for mode in ("OFF", "ON"):
            build(mode)
            bench = executable("pca_bench", mode)
            validation = run(bench, "--validate-f16-study", log=f"{mode}-validation.log")
            checks[mode] = [line for line in validation.splitlines() if line.startswith("CHECK,")]
            run(executable("holotask_test", mode), f"--gtest_filter={test_filter}",
                log=f"{mode}-tests.log")
        if len(checks["OFF"]) != 20 or checks["OFF"] != checks["ON"]:
            raise RuntimeError("Gram or sign-aligned projection signatures differ")

        sanitizer = shutil.which("compute-sanitizer")
        if not sanitizer:
            raise RuntimeError("Compute Sanitizer is required on PATH")
        sanitizer = Path(sanitizer)
        if sanitizer.suffix.lower() == ".bat":
            sanitizer = sanitizer.parent.parent / "compute-sanitizer" / "compute-sanitizer.exe"
        run(sanitizer, "--tool", "initcheck", "--error-exitcode", "1",
            executable("pca_bench", "ON"), "--validate-f16-study", log="ON-initcheck.log")
        samples = {}
        with (artifacts / "timings.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("pair", "variant", "stage", "windows", "components", "value"))
            for phase in ("covariance", "pca"):
                for pair in range(10):
                    modes = ("OFF", "ON") if pair % 2 == 0 else ("ON", "OFF")
                    for mode in modes:
                        timing = run(executable("pca_bench", mode), f"--{phase}-f16-study",
                                     log=f"{phase}-{pair}-{mode}.log")
                        rows = [row[1:] for row in csv.reader(io.StringIO(timing))
                                if row and row[0] == "TIME"]
                        expected = 80
                        if len(rows) != expected:
                            raise RuntimeError(f"Expected {expected} timing samples for {phase}")
                        if phase == "pca":
                            queue = run(executable("holotask_test", mode),
                                        "--gtest_filter=PcaPipeline.F16QueueThroughputStudy",
                                        log=f"{pair}-{mode}-queue.log")
                            rate, = [line.split("=")[-1] for line in queue.splitlines()
                                     if line.startswith("FP16 PCA windows/s=")]
                            rows.append(["queue", "4", "16", rate])
                        for stage, windows, components, value in rows:
                            writer.writerow((pair, mode, stage, windows, components, value))
                            samples.setdefault((mode, stage, windows, components), []).append(
                                float(value))
                        output.flush()
        with (artifacts / "summary.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("variant", "stage", "windows", "components", "median", "p95",
                             "minimum", "maximum"))
            for key, values in sorted(samples.items()):
                ordered = sorted(values)
                writer.writerow((*key, statistics.median(ordered),
                                 ordered[math.ceil(.95 * len(ordered)) - 1],
                                 ordered[0], ordered[-1]))
        if args.nsys:
            for mode in ("OFF", "ON"):
                run(args.nsys, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
                    "--force-overwrite=true", f"--output={artifacts / (mode + '-queue')}",
                    executable("holotask_test", mode),
                    "--gtest_filter=PcaPipeline.F16QueueThroughputStudy",
                    log=f"{mode}-nsys.log")
    finally:
        run("cmake", "--preset", "msvc-multi", "-DPCA_F16_LOWER_TRIANGLE=OFF")


if __name__ == "__main__":
    main()
