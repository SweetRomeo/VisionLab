import ctypes
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

CPP_PROJECT_DIRECTORY = PROJECT_ROOT / "cpp-opencv-core"
HYBRID_PROJECT_DIRECTORY = PROJECT_ROOT / "hybrid-python-cpp"
PURE_PYTHON_DIRECTORY = PROJECT_ROOT / "pure-python"


def calculate_sha256(file_path: Path) -> str:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Required file was not found: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def run_command(
    command: list[str],
    working_directory: Path = PROJECT_ROOT,
) -> str:
    try:
        process = subprocess.run(
            command,
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Required command was not found: {command[0]}"
        ) from error
    except subprocess.CalledProcessError as error:
        error_output = (
            error.stderr or error.stdout or str(error)
        ).strip()

        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"{error_output}"
        ) from error

    return (
        process.stdout
        or process.stderr
    ).strip()


def get_total_memory_bytes() -> Optional[int]:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                (
                    "total_physical",
                    ctypes.c_ulonglong,
                ),
                (
                    "available_physical",
                    ctypes.c_ulonglong,
                ),
                (
                    "total_page_file",
                    ctypes.c_ulonglong,
                ),
                (
                    "available_page_file",
                    ctypes.c_ulonglong,
                ),
                (
                    "total_virtual",
                    ctypes.c_ulonglong,
                ),
                (
                    "available_virtual",
                    ctypes.c_ulonglong,
                ),
                (
                    "available_extended_virtual",
                    ctypes.c_ulonglong,
                ),
            ]

        memory_status = MemoryStatus()
        memory_status.length = ctypes.sizeof(
            MemoryStatus
        )

        succeeded = ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(memory_status)
        )

        if succeeded:
            return int(memory_status.total_physical)

        return None

    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            page_count = os.sysconf(
                "SC_PHYS_PAGES"
            )
            return int(page_size * page_count)
        except (ValueError, OSError):
            pass

    if platform.system() == "Darwin":
        try:
            return int(
                run_command(
                    ["sysctl", "-n", "hw.memsize"]
                )
            )
        except RuntimeError:
            pass

    return None


def get_python_interpreter(
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


def collect_python_environment(
    project_directory: Path,
) -> dict:
    interpreter = get_python_interpreter(
        project_directory
    )

    probe_code = (
        "import json, platform, struct, sys; "
        "import cv2; "
        "import numpy as np; "
        "print(json.dumps({"
        "'python_version': sys.version.split()[0], "
        "'python_implementation': "
        "platform.python_implementation(), "
        "'architecture_bits': struct.calcsize('P') * 8, "
        "'numpy_version': np.__version__, "
        "'opencv_version': cv2.__version__"
        "}))"
    )

    output = run_command(
        [str(interpreter), "-c", probe_code]
    )

    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Could not parse Python environment "
            f"information: {output}"
        ) from error


def parse_cmake_cache(
    cache_path: Path,
) -> dict[str, str]:
    values = {}

    with cache_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as cache_file:
        for line in cache_file:
            stripped_line = line.strip()

            if (
                not stripped_line
                or stripped_line.startswith(("//", "#"))
                or "=" not in stripped_line
                or ":" not in stripped_line.split(
                    "=",
                    maxsplit=1,
                )[0]
            ):
                continue

            key_with_type, value = (
                stripped_line.split("=", maxsplit=1)
            )
            key, _ = key_with_type.split(
                ":",
                maxsplit=1,
            )
            values[key] = value

    return values


def find_release_cmake_cache(
    project_directory: Path,
) -> Path:
    build_directory = project_directory / "build"

    if not build_directory.is_dir():
        raise FileNotFoundError(
            f"Build directory was not found: "
            f"{build_directory}"
        )

    release_caches = []

    for cache_path in build_directory.rglob(
        "CMakeCache.txt"
    ):
        cache_values = parse_cmake_cache(
            cache_path
        )
        build_type = cache_values.get(
            "CMAKE_BUILD_TYPE",
            "",
        )

        if (
            build_type.lower() == "release"
            or "release"
            in cache_path.parent.name.lower()
        ):
            release_caches.append(cache_path)

    if not release_caches:
        raise FileNotFoundError(
            "A Release CMake cache was not found for: "
            f"{project_directory.name}"
        )

    return max(
        release_caches,
        key=lambda path: path.stat().st_mtime,
    )


def read_cmake_set_value(
    file_content: str,
    variable_name: str,
) -> Optional[str]:
    pattern = re.compile(
        rf'set\(\s*{re.escape(variable_name)}'
        rf'\s+"?([^"\s\)]+)"?\s*\)',
        re.IGNORECASE,
    )
    match = pattern.search(file_content)

    return match.group(1) if match else None


def detect_qt_version(
    cache_values: dict[str, str],
) -> Optional[str]:
    candidate_values = [
        cache_values.get("Qt6_DIR", ""),
        cache_values.get("Qt6Core_DIR", ""),
        cache_values.get("CMAKE_PREFIX_PATH", ""),
    ]

    version_pattern = re.compile(
        r"[\\/]Qt[\\/](\d+\.\d+\.\d+)[\\/]",
        re.IGNORECASE,
    )

    for candidate_value in candidate_values:
        match = version_pattern.search(
            candidate_value
        )

        if match:
            return match.group(1)

    return None

def detect_opencv_version(
    cache_values: dict[str, str],
    project_directory: Path,
) -> Optional[str]:
    cached_version = cache_values.get(
        "OpenCV_VERSION"
    )

    if cached_version:
        return cached_version

    opencv_directory = cache_values.get(
        "OpenCV_DIR"
    )

    if not opencv_directory:
        cmake_path = (
            project_directory / "CMakeLists.txt"
        )

        if cmake_path.is_file():
            cmake_content = cmake_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            directory_match = re.search(
                (
                    r"set\s*\(\s*OpenCV_DIR\s+"
                    r'"?([^"\s\)]+)"?'
                ),
                cmake_content,
                re.IGNORECASE,
            )

            if directory_match:
                opencv_directory = (
                    directory_match.group(1)
                )

    if not opencv_directory:
        return None

    opencv_path = Path(opencv_directory)

    if not opencv_path.is_absolute():
        opencv_path = (
            project_directory / opencv_path
        ).resolve()

    version_file = (
        opencv_path
        / "OpenCVConfig-version.cmake"
    )

    if not version_file.is_file():
        return None

    content = version_file.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return read_cmake_set_value(
        content,
        "OpenCV_VERSION",
    )

def collect_compiler_information(
    cache_path: Path,
) -> dict:
    compiler_files = list(
        cache_path.parent.rglob(
            "CMakeCXXCompiler.cmake"
        )
    )

    if not compiler_files:
        return {
            "id": None,
            "version": None,
        }

    compiler_file = max(
        compiler_files,
        key=lambda path: path.stat().st_mtime,
    )
    content = compiler_file.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return {
        "id": read_cmake_set_value(
            content,
            "CMAKE_CXX_COMPILER_ID",
        ),
        "version": read_cmake_set_value(
            content,
            "CMAKE_CXX_COMPILER_VERSION",
        ),
    }


def collect_cmake_build(
    project_directory: Path,
    include_qt: bool,
) -> dict:
    cache_path = find_release_cmake_cache(
        project_directory
    )
    cache_values = parse_cmake_cache(cache_path)

    version_parts = [
        cache_values.get(
            "CMAKE_CACHE_MAJOR_VERSION"
        ),
        cache_values.get(
            "CMAKE_CACHE_MINOR_VERSION"
        ),
        cache_values.get(
            "CMAKE_CACHE_PATCH_VERSION"
        ),
    ]

    cmake_version = (
        ".".join(version_parts)
        if all(version_parts)
        else None
    )

    return {
        "build_type": cache_values.get(
            "CMAKE_BUILD_TYPE",
            "Release",
        ),
        "generator": cache_values.get(
            "CMAKE_GENERATOR"
        ),
        "cmake_version": cmake_version,
        "compiler": collect_compiler_information(
            cache_path
        ),
        "opencv_version": detect_opencv_version(
            cache_values,
            project_directory,
        ),
        "qt_version": (
            detect_qt_version(cache_values)
            if include_qt
            else None
        ),
    }


def collect_git_information() -> dict:
    commit_sha = run_command(
        ["git", "rev-parse", "HEAD"]
    )
    branch_name = run_command(
        ["git", "branch", "--show-current"]
    )
    status_output = run_command(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ]
    )

    return {
        "commit_sha": commit_sha,
        "branch": (
            branch_name
            if branch_name
            else "detached"
        ),
        "working_tree_clean": not bool(
            status_output
        ),
    }

def get_processor_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            registry_path = (
                r"HARDWARE\DESCRIPTION\System"
                r"\CentralProcessor\0"
            )

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                registry_path,
            ) as registry_key:
                processor_name, _ = (
                    winreg.QueryValueEx(
                        registry_key,
                        "ProcessorNameString",
                    )
                )

            normalized_name = " ".join(
                str(processor_name).split()
            )

            if normalized_name:
                return normalized_name

        except OSError:
            pass

    return (
        platform.processor().strip()
        or os.environ.get(
            "PROCESSOR_IDENTIFIER",
            "",
        ).strip()
        or "unknown"
    )


def collect_system_information() -> dict:
    processor_name = get_processor_name()
    total_memory_bytes = get_total_memory_bytes()

    return {
        "operating_system": platform.system(),
        "operating_system_release": (
            platform.release()
        ),
        "operating_system_version": (
            platform.version()
        ),
        "architecture": platform.machine(),
        "processor": processor_name,
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory_bytes,
        "total_memory_gib": (
            round(
                total_memory_bytes
                / (1024 ** 3),
                2,
            )
            if total_memory_bytes is not None
            else None
        ),
    }


def load_benchmark_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Benchmark configuration was not found: "
            f"{CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return json.load(config_file)


def main() -> None:
    config = load_benchmark_config()

    video_relative_path = Path(
        config["input"]["video_path"]
    )
    video_path = (
        PROJECT_ROOT / video_relative_path
    ).resolve()

    output_directory = (
        PROJECT_ROOT
        / config["output"]["directory"]
    ).resolve()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "environment_metadata.json"
    )

    metadata = {
        "schema_version": 1,
        "collected_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "system": collect_system_information(),
        "python_environments": {
            "collector": {
                "python_version": (
                    sys.version.split()[0]
                ),
                "python_implementation": (
                    platform.python_implementation()
                ),
                "architecture_bits": (
                    struct.calcsize("P") * 8
                ),
            },
            "pure_python": (
                collect_python_environment(
                    PURE_PYTHON_DIRECTORY
                )
            ),
            "hybrid_python_cpp": (
                collect_python_environment(
                    HYBRID_PROJECT_DIRECTORY
                )
            ),
        },
        "builds": {
            "pure_cpp": collect_cmake_build(
                CPP_PROJECT_DIRECTORY,
                include_qt=True,
            ),
            "hybrid_python_cpp": collect_cmake_build(
                HYBRID_PROJECT_DIRECTORY,
                include_qt=False,
            ),
        },
        "repository": collect_git_information(),
        "experiment_files": {
            "benchmark_config": {
                "relative_path": str(
                    CONFIG_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ).replace("\\", "/"),
                "sha256": calculate_sha256(
                    CONFIG_PATH
                ),
            },
            "input_video": {
                "relative_path": str(
                    video_relative_path
                ).replace("\\", "/"),
                "sha256": calculate_sha256(
                    video_path
                ),
                "size_bytes": video_path.stat().st_size,
            },
        },
    }

    temporary_path = output_path.with_suffix(
        ".json.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            metadata,
            output_file,
            indent=2,
            ensure_ascii=False,
        )
        output_file.write("\n")

    temporary_path.replace(output_path)

    relative_output_path = output_path.relative_to(
        PROJECT_ROOT
    )

    print(
        "Environment metadata created: "
        f"{relative_output_path}"
    )

    if not metadata["repository"][
        "working_tree_clean"
    ]:
        print(
            "Warning: the Git working tree is not clean."
        )


if __name__ == "__main__":
    main()