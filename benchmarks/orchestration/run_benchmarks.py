import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PURE_PYTHON_DIRECTORY = PROJECT_ROOT / "pure-python"
HYBRID_DIRECTORY = PROJECT_ROOT / "hybrid-python-cpp"
CPP_DIRECTORY = PROJECT_ROOT / "cpp-opencv-core"

HYBRID_RUNNER = (
    PROJECT_ROOT
    / "benchmarks"
    / "runners"
    / "hybrid_benchmark.py"
)

ANALYSIS_SCRIPT = (
    PROJECT_ROOT
    / "benchmarks"
    / "analysis"
    / "analyze_results.py"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
)

RESOURCE_OUTPUT_PATH = (
    RESULTS_DIRECTORY
    / "benchmark_resource_summary.csv"
)

ANALYSIS_SCRIPT = (
    PROJECT_ROOT
    / "benchmarks"
    / "analysis"
    / "analyze_results.py"
)

SAMPLING_INTERVAL_SECONDS = 0.1

@dataclass(frozen=True)
class BenchmarkCommand:
    architecture: str
    command: tuple[str, ...]

@dataclass(frozen=True)
class ResourceMeasurement:
    architecture: str
    wall_time_seconds: float
    cpu_time_seconds: float
    average_cpu_percent: float
    peak_rss_mib: float
    sample_count: int
    sampling_interval_seconds: float
    exit_code: int


def get_virtual_environment_python(
    project_directory: Path,
) -> Path:
    if os.name == "nt":
        interpreter = (
            project_directory
            / ".venv"
            / "Scripts"
            / "python.exe"
        )
    else:
        interpreter = (
            project_directory
            / ".venv"
            / "bin"
            / "python"
        )

    if not interpreter.is_file():
        raise FileNotFoundError(
            "Virtual-environment interpreter was not "
            f"found: {interpreter}"
        )

    return interpreter


def cmake_cache_is_release(
    build_directory: Path,
) -> bool:
    cache_path = build_directory / "CMakeCache.txt"

    if not cache_path.is_file():
        return False

    with cache_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as cache_file:
        for line in cache_file:
            if line.startswith("CMAKE_BUILD_TYPE:"):
                _, separator, value = line.partition("=")

                return (
                    bool(separator)
                    and value.strip().lower() == "release"
                )

    return False


def is_release_executable(
    executable_path: Path,
) -> bool:
    if "release" in str(
        executable_path.parent
    ).lower():
        return True

    return (
        cmake_cache_is_release(
            executable_path.parent
        )
        or cmake_cache_is_release(
            executable_path.parent.parent
        )
    )


def find_cpp_benchmark_executable() -> Path:
    configured_build_directory = os.environ.get(
        "VISIONLAB_CPP_BUILD_DIR"
    )

    if configured_build_directory:
        search_directory = Path(
            configured_build_directory
        ).resolve()
    else:
        search_directory = CPP_DIRECTORY / "build"

    if not search_directory.is_dir():
        raise FileNotFoundError(
            "Pure C++ build directory was not found: "
            f"{search_directory}"
        )

    executable_names = (
        "VisionLabCppBenchmark.exe",
        "VisionLabCppBenchmark",
    )

    candidates = []

    for executable_name in executable_names:
        candidates.extend(
            path
            for path in search_directory.rglob(
                executable_name
            )
            if path.is_file()
            and is_release_executable(path)
        )

    unique_candidates = sorted(
        set(candidates)
    )

    if not unique_candidates:
        raise FileNotFoundError(
            "Pure C++ Release benchmark executable "
            f"was not found under: {search_directory}"
        )

    if len(unique_candidates) > 1:
        candidate_list = "\n".join(
            f"  {candidate}"
            for candidate in unique_candidates
        )

        raise RuntimeError(
            "Multiple Pure C++ Release executables "
            "were found. Set VISIONLAB_CPP_BUILD_DIR "
            "to the exact build directory:\n"
            f"{candidate_list}"
        )

    return unique_candidates[0]


def create_benchmark_commands(
) -> list[BenchmarkCommand]:
    pure_python_interpreter = (
        get_virtual_environment_python(
            PURE_PYTHON_DIRECTORY
        )
    )

    hybrid_interpreter = (
        get_virtual_environment_python(
            HYBRID_DIRECTORY
        )
    )

    for runner_path in (
        PURE_PYTHON_RUNNER,
        HYBRID_RUNNER,
    ):
        if not runner_path.is_file():
            raise FileNotFoundError(
                f"Benchmark runner was not found: "
                f"{runner_path}"
            )

    cpp_executable = (
        find_cpp_benchmark_executable()
    )

    return [
        BenchmarkCommand(
            architecture="pure_python",
            command=(
                str(pure_python_interpreter),
                str(PURE_PYTHON_RUNNER),
            ),
        ),
        BenchmarkCommand(
            architecture="hybrid",
            command=(
                str(hybrid_interpreter),
                str(HYBRID_RUNNER),
            ),
        ),
        BenchmarkCommand(
            architecture="pure_cpp",
            command=(str(cpp_executable),),
        ),
    ]

def discover_process_tree(
    root_process: psutil.Process,
    tracked_processes: dict[int, psutil.Process],
) -> list[psutil.Process]:
    try:
        discovered_processes = [
            root_process,
            *root_process.children(recursive=True),
        ]
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        discovered_processes = [root_process]

    for process in discovered_processes:
        if process.pid in tracked_processes:
            continue

        try:
            process.cpu_percent(interval=None)
            tracked_processes[process.pid] = process
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            continue

    return list(tracked_processes.values())


def sample_process_tree(
    root_process: psutil.Process,
    tracked_processes: dict[int, psutil.Process],
    cpu_times_by_process: dict[int, float],
) -> tuple[float, int]:
    processes = discover_process_tree(
        root_process,
        tracked_processes,
    )

    total_cpu_percent = 0.0
    total_rss_bytes = 0

    for process in processes:
        try:
            total_cpu_percent += process.cpu_percent(
                interval=None
            )
            total_rss_bytes += process.memory_info().rss

            cpu_times = process.cpu_times()
            cpu_times_by_process[process.pid] = (
                cpu_times.user + cpu_times.system
            )
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        ):
            continue

    return total_cpu_percent, total_rss_bytes


def terminate_process_tree(
    root_process: psutil.Process,
) -> None:
    try:
        processes = [
            *root_process.children(recursive=True),
            root_process,
        ]
    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        processes = [root_process]

    for process in reversed(processes):
        try:
            process.terminate()
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            continue

    _, alive_processes = psutil.wait_procs(
        processes,
        timeout=3.0,
    )

    for process in alive_processes:
        try:
            process.kill()
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            continue


def run_monitored_benchmark(
    benchmark_command: BenchmarkCommand,
) -> ResourceMeasurement:
    printable_command = subprocess.list2cmdline(
        benchmark_command.command
    )

    print(
        f"\nStarting {benchmark_command.architecture}:\n"
        f"{printable_command}"
    )

    start_time = time.perf_counter()

    child_process = subprocess.Popen(
        benchmark_command.command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
    )

    root_process = psutil.Process(
        child_process.pid
    )

    tracked_processes = {
        root_process.pid: root_process
    }
    cpu_times_by_process: dict[int, float] = {}

    root_process.cpu_percent(interval=None)

    cpu_percent_samples = []
    peak_rss_bytes = 0

    try:
        while child_process.poll() is None:
            time.sleep(
                SAMPLING_INTERVAL_SECONDS
            )

            (
                cpu_percent,
                rss_bytes,
            ) = sample_process_tree(
                root_process,
                tracked_processes,
                cpu_times_by_process,
            )

            cpu_percent_samples.append(
                cpu_percent
            )

            peak_rss_bytes = max(
                peak_rss_bytes,
                rss_bytes,
            )
    except KeyboardInterrupt:
        terminate_process_tree(root_process)
        raise

    exit_code = child_process.wait()

    wall_time_seconds = (
        time.perf_counter() - start_time
    )

    average_cpu_percent = (
        fmean(cpu_percent_samples)
        if cpu_percent_samples
        else 0.0
    )

    measurement = ResourceMeasurement(
        architecture=benchmark_command.architecture,
        wall_time_seconds=wall_time_seconds,
        cpu_time_seconds=sum(
            cpu_times_by_process.values()
        ),
        average_cpu_percent=average_cpu_percent,
        peak_rss_mib=(
            peak_rss_bytes / (1024 ** 2)
        ),
        sample_count=len(cpu_percent_samples),
        sampling_interval_seconds=(
            SAMPLING_INTERVAL_SECONDS
        ),
        exit_code=exit_code,
    )

    print(
        f"Completed {measurement.architecture}: "
        f"exit={measurement.exit_code}, "
        f"wall={measurement.wall_time_seconds:.3f}s, "
        f"CPU={measurement.average_cpu_percent:.2f}%, "
        f"peak RSS={measurement.peak_rss_mib:.2f} MiB"
    )

    return measurement


def write_resource_summary(
    measurements: list[ResourceMeasurement],
) -> Path:
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        RESOURCE_OUTPUT_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    fieldnames = [
        "architecture",
        "wall_time_seconds",
        "cpu_time_seconds",
        "average_cpu_percent",
        "peak_rss_mib",
        "sample_count",
        "sampling_interval_seconds",
        "exit_code",
    ]

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for measurement in measurements:
            writer.writerow(
                {
                    "architecture": (
                        measurement.architecture
                    ),
                    "wall_time_seconds": (
                        f"{measurement.wall_time_seconds:.6f}"
                    ),
                    "cpu_time_seconds": (
                        f"{measurement.cpu_time_seconds:.6f}"
                    ),
                    "average_cpu_percent": (
                        f"{measurement.average_cpu_percent:.6f}"
                    ),
                    "peak_rss_mib": (
                        f"{measurement.peak_rss_mib:.6f}"
                    ),
                    "sample_count": (
                        measurement.sample_count
                    ),
                    "sampling_interval_seconds": (
                        f"{measurement.sampling_interval_seconds:.3f}"
                    ),
                    "exit_code": measurement.exit_code,
                }
            )

    temporary_path.replace(
        RESOURCE_OUTPUT_PATH
    )

    return RESOURCE_OUTPUT_PATH

def run_result_analysis() -> None:
    if not ANALYSIS_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Result analyzer was not found: "
            f"{ANALYSIS_SCRIPT}"
        )

    print("\nValidating benchmark results...")

    subprocess.run(
        [
            sys.executable,
            str(ANALYSIS_SCRIPT),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print("Benchmark result validation passed.")

def main() -> None:
    commands = create_benchmark_commands()
    measurements = []

    for benchmark_command in commands:
        measurement = run_monitored_benchmark(
            benchmark_command
        )

        if measurement.exit_code != 0:
            raise RuntimeError(
                f"{measurement.architecture} benchmark "
                f"failed with exit code "
                f"{measurement.exit_code}."
            )

        measurements.append(measurement)

    run_result_analysis()

    output_path = write_resource_summary(
        measurements
    )

    print(
        "\nResource summary created: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
