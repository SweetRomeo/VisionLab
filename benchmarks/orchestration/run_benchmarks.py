import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PURE_PYTHON_DIRECTORY = PROJECT_ROOT / "pure-python"
HYBRID_DIRECTORY = PROJECT_ROOT / "hybrid-python-cpp"
CPP_DIRECTORY = PROJECT_ROOT / "cpp-opencv-core"

PURE_PYTHON_RUNNER = (
    PROJECT_ROOT
    / "benchmarks"
    / "runners"
    / "pure_python_benchmark.py"
)

HYBRID_RUNNER = (
    PROJECT_ROOT
    / "benchmarks"
    / "runners"
    / "hybrid_benchmark.py"
)


@dataclass(frozen=True)
class BenchmarkCommand:
    architecture: str
    command: tuple[str, ...]


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


def main() -> None:
    commands = create_benchmark_commands()

    print("Resolved benchmark commands:")

    for benchmark_command in commands:
        printable_command = subprocess.list2cmdline(
            benchmark_command.command
        )

        print(
            f"{benchmark_command.architecture}: "
            f"{printable_command}"
        )


if __name__ == "__main__":
    main()