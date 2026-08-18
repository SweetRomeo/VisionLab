import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import importlib
import os
import sys
import csv
import math
import subprocess

from skimage.metrics import structural_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

VALIDATION_CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "output_validation_config.json"
)

PURE_PYTHON_DIRECTORY = (
    PROJECT_ROOT / "pure-python"
)

HYBRID_PROJECT_DIRECTORY = (
    PROJECT_ROOT / "hybrid-python-cpp"
)

DLL_DIRECTORY_HANDLES = []

def get_pure_python_interpreter_path() -> Path:
    interpreter_path = (
        PURE_PYTHON_DIRECTORY / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else PURE_PYTHON_DIRECTORY / ".venv" / "bin" / "python"
    )

    if not interpreter_path.is_file():
        raise FileNotFoundError(
            "Pure Python virtual environment interpreter was "
            "not found. Expected: "
            f"{interpreter_path}"
        )

    return interpreter_path

def read_runtime_dependency_versions(
    python_executable: Path,
) -> dict[str, str]:
    command = [
        str(python_executable),
        "-c",
        (
            "import cv2, json, numpy as np, sys; "
            "print(json.dumps({"
            "'python': sys.version.split()[0], "
            "'numpy': np.__version__, "
            "'opencv': cv2.__version__"
            "}))"
        ),
    ]
    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        stderr_output = (error.stderr or "").strip()
        raise RuntimeError(
            "Could not read runtime dependencies from "
            f"{python_executable}: {stderr_output or error}"
        ) from error

    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Could not parse runtime dependencies from "
            f"{python_executable}: {process.stdout.strip()}"
        ) from error

def validate_runtime_dependencies_match() -> None:
    current_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    pure_python_versions = read_runtime_dependency_versions(
        get_pure_python_interpreter_path()
    )
    mismatches = [
        dependency_name
        for dependency_name in current_versions
        if current_versions[dependency_name]
        != pure_python_versions.get(dependency_name)
    ]

    if mismatches:
        mismatch_message = ", ".join(
            (
                f"{dependency_name} "
                f"(hybrid={current_versions[dependency_name]}, "
                "pure_python="
                f"{pure_python_versions.get(dependency_name)})"
            )
            for dependency_name in mismatches
        )
        raise RuntimeError(
            "Output equivalence validation requires matching "
            "runtime dependencies between the active environment "
            "and pure-python/.venv. Mismatches: "
            f"{mismatch_message}"
        )

def find_cpp_validation_executable() -> Path:
    configured_executable = os.getenv(
        "VISIONLAB_CPP_VALIDATION_EXE"
    )

    if configured_executable:
        executable_path = Path(
            configured_executable
        ).resolve()

        if not executable_path.is_file():
            raise FileNotFoundError(
                "VISIONLAB_CPP_VALIDATION_EXE "
                "is invalid: "
                f"{executable_path}"
            )

        return executable_path

    build_directory = (
        PROJECT_ROOT
        / "cpp-opencv-core"
        / "build"
    )

    candidates = []

    for executable_name in (
        "VisionLabCppValidation.exe",
        "VisionLabCppValidation",
    ):
        candidates.extend(
            candidate
            for candidate
            in build_directory.rglob(
                executable_name
            )
            if candidate.is_file()
        )

    release_candidates = [
        candidate
        for candidate in candidates
        if is_release_candidate(
            candidate,
            build_directory,
        )
    ]

    if not release_candidates:
        raise FileNotFoundError(
            "The Release build of "
            "VisionLabCppValidation could not be "
            "found. Build the target in Release "
            "mode or set "
            "VISIONLAB_CPP_VALIDATION_EXE."
        )

    return max(
        release_candidates,
        key=lambda path: path.stat().st_mtime,
    )

def is_release_candidate(
    candidate: Path,
    build_directory: Path,
) -> bool:
    if "release" in str(candidate.parent).lower():
        return True

    cache_path = build_directory / "CMakeCache.txt"

    if not cache_path.is_file():
        return False

    with cache_path.open(
        "r",
        encoding="utf-8",
    ) as cache_file:
        for line in cache_file:
            if line.startswith("CMAKE_BUILD_TYPE:STRING="):
                build_type = line.removeprefix(
                    "CMAKE_BUILD_TYPE:STRING="
                ).strip()

                return build_type.lower() == "release"

    return False

def find_hybrid_module_directory() -> Path:
    configured_directory = os.getenv(
        "VISIONLAB_CPP_MODULE_DIR"
    )

    if configured_directory:
        module_directory = Path(
            configured_directory
        ).resolve()

        if not module_directory.is_dir():
            raise FileNotFoundError(
                "VISIONLAB_CPP_MODULE_DIR is invalid: "
                f"{module_directory}"
            )

        return module_directory


    build_directory = (
        HYBRID_PROJECT_DIRECTORY / "build"
    )

    candidates = []

    for pattern in (
        "visionlab_cpp*.pyd",
        "visionlab_cpp*.so",
    ):
        candidates.extend(
            build_directory.rglob(pattern)
        )

    release_candidates = [
        candidate
        for candidate in candidates
        if is_release_candidate(
            candidate,
            build_directory,
        )
    ]

    if not release_candidates:
        raise FileNotFoundError(
            "The Release build of visionlab_cpp "
            "could not be found. Build the Hybrid "
            "project in Release mode or set "
            "VISIONLAB_CPP_MODULE_DIR."
        )

    newest_module = max(
        release_candidates,
        key=lambda path: path.stat().st_mtime,
    )

    return newest_module.parent


def load_implementations():
    if str(PURE_PYTHON_DIRECTORY) not in sys.path:
        sys.path.insert(
            0,
            str(PURE_PYTHON_DIRECTORY),
        )

    image_process_module = importlib.import_module(
        "image_process"
    )

    hybrid_module_directory = (
        find_hybrid_module_directory()
    )

    if str(hybrid_module_directory) not in sys.path:
        sys.path.insert(
            0,
            str(hybrid_module_directory),
        )

    if (
        os.name == "nt"
        and hasattr(os, "add_dll_directory")
    ):
        DLL_DIRECTORY_HANDLES.append(
            os.add_dll_directory(
                str(hybrid_module_directory)
            )
        )

    hybrid_module = importlib.import_module(
        "visionlab_cpp"
    )

    return image_process_module, hybrid_module


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration must contain a JSON object: {path}"
        )

    return data


def validate_frame_indices(
    validation_config: dict[str, Any],
) -> list[int]:
    frame_indices = validation_config.get(
        "frame_indices"
    )

    if (
        not isinstance(frame_indices, list)
        or not frame_indices
    ):
        raise ValueError(
            "frame_indices must be a non-empty list."
        )

    if any(
        not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
        or frame_index <= 0
        for frame_index in frame_indices
    ):
        raise ValueError(
            "Frame indices must be positive integers."
        )

    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError(
            "frame_indices must not contain duplicates."
        )

    return sorted(frame_indices)


def read_selected_frames(
    video_path: Path,
    frame_indices: list[int],
) -> dict[int, np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Input video was not found: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Input video could not be opened: {video_path}"
        )

    requested_indices = set(frame_indices)
    selected_frames: dict[int, np.ndarray] = {}

    current_frame_index = 0
    final_frame_index = max(frame_indices)

    try:
        while current_frame_index < final_frame_index:
            frame_received, frame = capture.read()

            if not frame_received:
                break

            # Validation configuration uses 1-based indices.
            current_frame_index += 1

            if current_frame_index in requested_indices:
                selected_frames[current_frame_index] = (
                    frame.copy()
                )
    finally:
        capture.release()

    missing_indices = (
        requested_indices - selected_frames.keys()
    )

    if missing_indices:
        raise RuntimeError(
            "The following frames could not be read: "
            f"{sorted(missing_indices)}"
        )

    return selected_frames


def save_validation_inputs(
    frames: dict[int, np.ndarray],
    resolutions: list[dict[str, Any]],
    output_directory: Path,
) -> int:
    input_directory = output_directory / "inputs"
    saved_image_count = 0

    for resolution in resolutions:
        width = resolution.get("width")
        height = resolution.get("height")

        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ValueError(
                f"Invalid resolution: {resolution}"
            )

        resolution_name = f"{width}x{height}"
        resolution_directory = (
            input_directory / resolution_name
        )
        resolution_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for frame_index, frame in frames.items():
            resized_frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            output_path = (
                resolution_directory
                / f"frame_{frame_index:06d}.png"
            )

            image_written = cv2.imwrite(
                str(output_path),
                resized_frame,
            )

            if not image_written:
                raise RuntimeError(
                    f"Image could not be written: "
                    f"{output_path}"
                )

            saved_image_count += 1

    return saved_image_count

def save_image(
    output_path: Path,
    image: np.ndarray,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(
            f"Image could not be written: {output_path}"
        )


def calculate_image_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    difference = np.abs(
        reference.astype(np.int16)
        - candidate.astype(np.int16)
    )

    mean_absolute_error = float(
        np.mean(difference)
    )
    maximum_absolute_error = int(
        np.max(difference)
    )

    squared_difference = np.square(
        reference.astype(np.float64)
        - candidate.astype(np.float64)
    )
    mean_squared_error = float(
        np.mean(squared_difference)
    )

    psnr_db = (
        float("inf")
        if mean_squared_error == 0.0
        else 10.0
        * math.log10(
            (255.0 ** 2) / mean_squared_error
        )
    )

    ssim_value = float(
        structural_similarity(
            reference,
            candidate,
            channel_axis=2,
            data_range=255,
        )
    )

    return {
        "mean_absolute_error": (
            mean_absolute_error
        ),
        "maximum_absolute_error": (
            maximum_absolute_error
        ),
        "mean_squared_error": (
            mean_squared_error
        ),
        "psnr_db": (
            psnr_db
        ),
        "ssim" : ssim_value,
    }


def compare_python_and_hybrid(
    image_process_module,
    hybrid_module,
    frames: dict[int, np.ndarray],
    resolutions: list[dict[str, Any]],
    algorithms: list[dict[str, Any]],
    output_directory: Path,
) -> Path:
    pure_python_processor = (
        image_process_module.ImageProcess()
    )

    python_algorithm_map = {
        "original": (
            image_process_module
            .ProcessingAlgorithm.ORIGINAL
        ),
        "gamma_correction": (
            image_process_module
            .ProcessingAlgorithm.GAMMA
        ),
        "histogram_equalization": (
            image_process_module
            .ProcessingAlgorithm.HISTOGRAM
        ),
        "clahe": (
            image_process_module
            .ProcessingAlgorithm.CLAHE
        ),
    }

    hybrid_algorithm_map = {
        "original": (
            hybrid_module
            .ProcessingAlgorithm.ORIGINAL
        ),
        "gamma_correction": (
            hybrid_module
            .ProcessingAlgorithm.GAMMA
        ),
        "histogram_equalization": (
            hybrid_module
            .ProcessingAlgorithm.HISTOGRAM
        ),
        "clahe": (
            hybrid_module
            .ProcessingAlgorithm.CLAHE
        ),
    }

    comparison_path = (
        output_directory
        / "python_hybrid_comparison.csv"
    )

    fieldnames = [
        "reference_architecture",
        "candidate_architecture",
        "algorithm",
        "resolution",
        "frame_index",
        "mean_absolute_error",
        "maximum_absolute_error",
        "mean_squared_error",
        "psnr_db",
        "ssim",
    ]

    with comparison_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for resolution in resolutions:
            width = int(resolution["width"])
            height = int(resolution["height"])
            resolution_name = f"{width}x{height}"

            for algorithm_config in algorithms:
                algorithm_name = (
                    algorithm_config["name"]
                )

                if (
                    algorithm_name
                    not in python_algorithm_map
                    or algorithm_name
                    not in hybrid_algorithm_map
                ):
                    raise ValueError(
                        "Unsupported algorithm: "
                        f"{algorithm_name}"
                    )

                parameter_config = (
                    algorithm_config.get(
                        "parameters",
                        {},
                    )
                )

                gamma_value = float(
                    parameter_config.get(
                        "gamma_value",
                        0.6,
                    )
                )
                clahe_clip_limit = float(
                    parameter_config.get(
                        "clip_limit",
                        4.0,
                    )
                )
                clahe_grid_size = int(
                    parameter_config.get(
                        "grid_size",
                        8,
                    )
                )

                python_parameters = (
                    image_process_module
                    .ProcessingParameters(
                        gamma_value=gamma_value,
                        clahe_clip_limit=(
                            clahe_clip_limit
                        ),
                        clahe_grid_size=(
                            clahe_grid_size
                        ),
                    )
                )

                for frame_index, frame in frames.items():
                    source = cv2.resize(
                        frame,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    source_before_processing = (
                        source.copy()
                    )

                    pure_python_output = (
                        pure_python_processor.process(
                            source,
                            python_algorithm_map[
                                algorithm_name
                            ],
                            python_parameters,
                        )
                    )

                    hybrid_output = (
                        hybrid_module.process_frame(
                            source,
                            hybrid_algorithm_map[
                                algorithm_name
                            ],
                            gamma_value=gamma_value,
                            clahe_clip_limit=(
                                clahe_clip_limit
                            ),
                            clahe_grid_size=(
                                clahe_grid_size
                            ),
                        )
                    )

                    if not np.array_equal(
                        source,
                        source_before_processing,
                    ):
                        raise RuntimeError(
                            "An implementation modified "
                            "the source image."
                        )

                    if (
                        pure_python_output.shape
                        != hybrid_output.shape
                    ):
                        raise RuntimeError(
                            "Output shape mismatch for "
                            f"{algorithm_name}, "
                            f"{resolution_name}, "
                            f"frame {frame_index}."
                        )

                    if (
                        pure_python_output.dtype
                        != hybrid_output.dtype
                    ):
                        raise RuntimeError(
                            "Output dtype mismatch for "
                            f"{algorithm_name}, "
                            f"{resolution_name}, "
                            f"frame {frame_index}."
                        )

                    if algorithm_name == "original":
                        if np.shares_memory(
                            source,
                            pure_python_output,
                        ):
                            raise RuntimeError(
                                "Pure Python Original "
                                "returned shared memory."
                            )

                        if np.shares_memory(
                            source,
                            hybrid_output,
                        ):
                            raise RuntimeError(
                                "Hybrid Original returned "
                                "shared memory."
                            )

                    metrics = calculate_image_metrics(
                        pure_python_output,
                        hybrid_output,
                    )

                    writer.writerow(
                        {
                            "reference_architecture": (
                                "pure_python"
                            ),
                            "candidate_architecture": (
                                "hybrid"
                            ),
                            "algorithm": algorithm_name,
                            "resolution": (
                                resolution_name
                            ),
                            "frame_index": frame_index,
                            **{
                                name: f"{value:.6f}"
                                for name, value
                                in metrics.items()
                            },
                        }
                    )

                    file_name = (
                        f"frame_{frame_index:06d}.png"
                    )

                    save_image(
                        output_directory
                        / "outputs"
                        / "pure_python"
                        / algorithm_name
                        / resolution_name
                        / file_name,
                        pure_python_output,
                    )

                    save_image(
                        output_directory
                        / "outputs"
                        / "hybrid"
                        / algorithm_name
                        / resolution_name
                        / file_name,
                        hybrid_output,
                    )

    return comparison_path
def generate_cpp_outputs(
    executable_path: Path,
    frames: dict[int, np.ndarray],
    resolutions: list[dict[str, Any]],
    algorithms: list[dict[str, Any]],
    output_directory: Path,
) -> int:
    generated_output_count = 0

    for resolution in resolutions:
        width = int(resolution["width"])
        height = int(resolution["height"])
        resolution_name = f"{width}x{height}"

        for algorithm_config in algorithms:
            algorithm_name = (
                algorithm_config["name"]
            )

            parameter_config = (
                algorithm_config.get(
                    "parameters",
                    {},
                )
            )

            gamma_value = float(
                parameter_config.get(
                    "gamma_value",
                    0.6,
                )
            )
            clahe_clip_limit = float(
                parameter_config.get(
                    "clip_limit",
                    4.0,
                )
            )
            clahe_grid_size = int(
                parameter_config.get(
                    "grid_size",
                    8,
                )
            )

            for frame_index in frames:
                file_name = (
                    f"frame_{frame_index:06d}.png"
                )

                input_path = (
                    output_directory
                    / "inputs"
                    / resolution_name
                    / file_name
                )

                output_path = (
                    output_directory
                    / "outputs"
                    / "pure_cpp"
                    / algorithm_name
                    / resolution_name
                    / file_name
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                completed_process = subprocess.run(
                    [
                        str(executable_path),
                        str(input_path),
                        str(output_path),
                        algorithm_name,
                        str(gamma_value),
                        str(clahe_clip_limit),
                        str(clahe_grid_size),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if completed_process.returncode != 0:
                    raise RuntimeError(
                        "Pure C++ validation failed.\n"
                        f"Command output:\n"
                        f"{completed_process.stdout}\n"
                        f"Command error:\n"
                        f"{completed_process.stderr}"
                    )

                if not output_path.is_file():
                    raise RuntimeError(
                        "Pure C++ output was not "
                        f"created: {output_path}"
                    )

                generated_output_count += 1

    return generated_output_count


def compare_python_and_cpp(
    frames: dict[int, np.ndarray],
    resolutions: list[dict[str, Any]],
    algorithms: list[dict[str, Any]],
    output_directory: Path,
) -> Path:
    comparison_path = (
        output_directory
        / "python_cpp_comparison.csv"
    )

    fieldnames = [
        "reference_architecture",
        "candidate_architecture",
        "algorithm",
        "resolution",
        "frame_index",
        "mean_absolute_error",
        "maximum_absolute_error",
        "mean_squared_error",
        "psnr_db",
        "ssim",
    ]

    with comparison_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for resolution in resolutions:
            width = int(resolution["width"])
            height = int(resolution["height"])
            resolution_name = f"{width}x{height}"

            for algorithm_config in algorithms:
                algorithm_name = (
                    algorithm_config["name"]
                )

                for frame_index in frames:
                    file_name = (
                        f"frame_{frame_index:06d}.png"
                    )

                    pure_python_path = (
                        output_directory
                        / "outputs"
                        / "pure_python"
                        / algorithm_name
                        / resolution_name
                        / file_name
                    )

                    pure_cpp_path = (
                        output_directory
                        / "outputs"
                        / "pure_cpp"
                        / algorithm_name
                        / resolution_name
                        / file_name
                    )

                    pure_python_output = cv2.imread(
                        str(pure_python_path),
                        cv2.IMREAD_COLOR,
                    )
                    pure_cpp_output = cv2.imread(
                        str(pure_cpp_path),
                        cv2.IMREAD_COLOR,
                    )

                    if (
                        pure_python_output is None
                        or pure_cpp_output is None
                    ):
                        raise RuntimeError(
                            "A validation output could "
                            "not be read."
                        )

                    if (
                        pure_python_output.shape
                        != pure_cpp_output.shape
                    ):
                        raise RuntimeError(
                            "Pure Python and Pure C++ "
                            "output shapes do not match."
                        )

                    if (
                        pure_python_output.dtype
                        != pure_cpp_output.dtype
                    ):
                        raise RuntimeError(
                            "Pure Python and Pure C++ "
                            "output types do not match."
                        )

                    metrics = calculate_image_metrics(
                        pure_python_output,
                        pure_cpp_output,
                    )

                    writer.writerow(
                        {
                            "reference_architecture": (
                                "pure_python"
                            ),
                            "candidate_architecture": (
                                "pure_cpp"
                            ),
                            "algorithm": algorithm_name,
                            "resolution": (
                                resolution_name
                            ),
                            "frame_index": frame_index,
                            **{
                                name: f"{value:.6f}"
                                for name, value
                                in metrics.items()
                            },
                        }
                    )

    return comparison_path

def load_validation_rules(
    validation_config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, bool]]:
    thresholds_config = validation_config.get(
        "thresholds"
    )
    exact_match_config = validation_config.get(
        "exact_match_rules"
    )

    if not isinstance(thresholds_config, dict):
        raise ValueError(
            "Validation thresholds are missing."
        )

    if not isinstance(exact_match_config, dict):
        raise ValueError(
            "Exact-match rules are missing."
        )

    threshold_names = (
        "maximum_mean_absolute_error",
        "maximum_absolute_error",
        "minimum_psnr_db",
        "minimum_ssim",
    )

    thresholds = {}

    for threshold_name in threshold_names:
        threshold_value = thresholds_config.get(
            threshold_name
        )

        if (
            isinstance(threshold_value, bool)
            or not isinstance(
                threshold_value,
                (int, float),
            )
            or not math.isfinite(
                float(threshold_value)
            )
        ):
            raise ValueError(
                "Validation threshold must be "
                "a finite number: "
                f"{threshold_name}"
            )

        thresholds[threshold_name] = float(
            threshold_value
        )

    if (
        thresholds[
            "maximum_mean_absolute_error"
        ] < 0.0
        or thresholds[
            "maximum_absolute_error"
        ] < 0.0
        or thresholds["minimum_psnr_db"] < 0.0
        or not (
            0.0
            <= thresholds["minimum_ssim"]
            <= 1.0
        )
    ):
        raise ValueError(
            "Validation thresholds are invalid."
        )

    exact_rule_names = (
        "original",
        "hybrid_vs_pure_cpp",
    )

    exact_match_rules = {}

    for rule_name in exact_rule_names:
        rule_value = exact_match_config.get(
            rule_name
        )

        if not isinstance(rule_value, bool):
            raise ValueError(
                "Exact-match rule must be "
                f"a boolean: {rule_name}"
            )

        exact_match_rules[rule_name] = rule_value

    return thresholds, exact_match_rules


def evaluate_metrics(
    metrics: dict[str, float],
    reference_architecture: str,
    candidate_architecture: str,
    algorithm_name: str,
    thresholds: dict[str, float],
    exact_match_rules: dict[str, bool],
) -> tuple[bool, list[str]]:
    architecture_pair = {
        reference_architecture,
        candidate_architecture,
    }

    exact_match_required = (
        (
            algorithm_name == "original"
            and exact_match_rules["original"]
        )
        or (
            architecture_pair
            == {"hybrid", "pure_cpp"}
            and exact_match_rules[
                "hybrid_vs_pure_cpp"
            ]
        )
    )

    failure_reasons = []

    if exact_match_required:
        if metrics["maximum_absolute_error"] != 0:
            failure_reasons.append(
                "exact pixel match required"
            )

        return (
            exact_match_required,
            failure_reasons,
        )

    if (
        metrics["mean_absolute_error"]
        > thresholds[
            "maximum_mean_absolute_error"
        ]
    ):
        failure_reasons.append(
            "mean absolute error exceeded"
        )

    if (
        metrics["maximum_absolute_error"]
        > thresholds["maximum_absolute_error"]
    ):
        failure_reasons.append(
            "maximum absolute error exceeded"
        )

    if (
        metrics["psnr_db"]
        < thresholds["minimum_psnr_db"]
    ):
        failure_reasons.append(
            "PSNR below minimum"
        )

    if (
        metrics["ssim"]
        < thresholds["minimum_ssim"]
    ):
        failure_reasons.append(
            "SSIM below minimum"
        )

    return exact_match_required, failure_reasons


def write_equivalence_report(
    frames: dict[int, np.ndarray],
    resolutions: list[dict[str, Any]],
    algorithms: list[dict[str, Any]],
    output_directory: Path,
    thresholds: dict[str, float],
    exact_match_rules: dict[str, bool],
) -> tuple[Path, int, int]:
    architecture_pairs = [
        ("pure_python", "hybrid"),
        ("pure_python", "pure_cpp"),
        ("hybrid", "pure_cpp"),
    ]

    report_path = (
        output_directory
        / "output_equivalence_report.csv"
    )

    fieldnames = [
        "reference_architecture",
        "candidate_architecture",
        "algorithm",
        "resolution",
        "frame_index",
        "shape_match",
        "dtype_match",
        "mean_absolute_error",
        "maximum_absolute_error",
        "mean_squared_error",
        "psnr_db",
        "ssim",
        "exact_match_required",
        "passed",
        "failure_reasons",
    ]

    comparison_count = 0
    failure_count = 0

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for resolution in resolutions:
            width = int(resolution["width"])
            height = int(resolution["height"])
            resolution_name = f"{width}x{height}"

            for algorithm_config in algorithms:
                algorithm_name = (
                    algorithm_config["name"]
                )

                for frame_index in frames:
                    file_name = (
                        f"frame_{frame_index:06d}.png"
                    )

                    for (
                        reference_architecture,
                        candidate_architecture,
                    ) in architecture_pairs:
                        reference_path = (
                            output_directory
                            / "outputs"
                            / reference_architecture
                            / algorithm_name
                            / resolution_name
                            / file_name
                        )

                        candidate_path = (
                            output_directory
                            / "outputs"
                            / candidate_architecture
                            / algorithm_name
                            / resolution_name
                            / file_name
                        )

                        reference = cv2.imread(
                            str(reference_path),
                            cv2.IMREAD_COLOR,
                        )
                        candidate = cv2.imread(
                            str(candidate_path),
                            cv2.IMREAD_COLOR,
                        )

                        if (
                            reference is None
                            or candidate is None
                        ):
                            raise RuntimeError(
                                "A validation image "
                                "could not be read."
                            )

                        shape_match = (
                            reference.shape
                            == candidate.shape
                        )
                        dtype_match = (
                            reference.dtype
                            == candidate.dtype
                        )

                        failure_reasons = []

                        if not shape_match:
                            failure_reasons.append(
                                "shape mismatch"
                            )

                        if not dtype_match:
                            failure_reasons.append(
                                "dtype mismatch"
                            )

                        metrics = {
                            "mean_absolute_error": (
                                float("nan")
                            ),
                            "maximum_absolute_error": (
                                float("nan")
                            ),
                            "mean_squared_error": (
                                float("nan")
                            ),
                            "psnr_db": float("nan"),
                            "ssim": float("nan"),
                        }

                        exact_match_required = False

                        if shape_match and dtype_match:
                            metrics = (
                                calculate_image_metrics(
                                    reference,
                                    candidate,
                                )
                            )

                            (
                                exact_match_required,
                                metric_failures,
                            ) = evaluate_metrics(
                                metrics=metrics,
                                reference_architecture=(
                                    reference_architecture
                                ),
                                candidate_architecture=(
                                    candidate_architecture
                                ),
                                algorithm_name=(
                                    algorithm_name
                                ),
                                thresholds=thresholds,
                                exact_match_rules=(
                                    exact_match_rules
                                ),
                            )

                            failure_reasons.extend(
                                metric_failures
                            )

                        passed = not failure_reasons

                        comparison_count += 1

                        if not passed:
                            failure_count += 1

                        writer.writerow(
                            {
                                "reference_architecture": (
                                    reference_architecture
                                ),
                                "candidate_architecture": (
                                    candidate_architecture
                                ),
                                "algorithm": (
                                    algorithm_name
                                ),
                                "resolution": (
                                    resolution_name
                                ),
                                "frame_index": (
                                    frame_index
                                ),
                                "shape_match": (
                                    str(shape_match).lower()
                                ),
                                "dtype_match": (
                                    str(dtype_match).lower()
                                ),
                                **{
                                    name: f"{value:.6f}"
                                    for name, value
                                    in metrics.items()
                                },
                                "exact_match_required": (
                                    str(
                                        exact_match_required
                                    ).lower()
                                ),
                                "passed": (
                                    str(passed).lower()
                                ),
                                "failure_reasons": "; ".join(
                                    failure_reasons
                                ),
                            }
                        )

    return (
        report_path,
        comparison_count,
        failure_count,
    )

def main() -> None:
    benchmark_config = load_json(
        BENCHMARK_CONFIG_PATH
    )
    validation_config = load_json(
        VALIDATION_CONFIG_PATH
    )

    validate_runtime_dependencies_match()

    frame_indices = validate_frame_indices(
        validation_config
    )

    video_relative_path = benchmark_config["input"][
        "video_path"
    ]
    video_path = PROJECT_ROOT / video_relative_path

    resolutions = benchmark_config["resolutions"]

    output_relative_path = validation_config[
        "output_directory"
    ]
    output_directory = (
        PROJECT_ROOT / output_relative_path
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_frames = read_selected_frames(
        video_path,
        frame_indices,
    )

    saved_image_count = save_validation_inputs(
        selected_frames,
        resolutions,
        output_directory,
    )

    image_process_module, hybrid_module = (
        load_implementations()
    )

    comparison_path = compare_python_and_hybrid(
        image_process_module=image_process_module,
        hybrid_module=hybrid_module,
        frames=selected_frames,
        resolutions=resolutions,
        algorithms=benchmark_config["algorithms"],
        output_directory=output_directory,
    )

    cpp_validation_executable = (
        find_cpp_validation_executable()
    )

    cpp_output_count = generate_cpp_outputs(
        executable_path=cpp_validation_executable,
        frames=selected_frames,
        resolutions=resolutions,
        algorithms=benchmark_config["algorithms"],
        output_directory=output_directory,
    )

    cpp_comparison_path = (
        compare_python_and_cpp(
            frames=selected_frames,
            resolutions=resolutions,
            algorithms=benchmark_config[
                "algorithms"
            ],
            output_directory=output_directory,
        )
    )

    thresholds, exact_match_rules = (
        load_validation_rules(
            validation_config
        )
    )

    (
        equivalence_report_path,
        equivalence_comparison_count,
        equivalence_failure_count,
    ) = write_equivalence_report(
        frames=selected_frames,
        resolutions=resolutions,
        algorithms=benchmark_config["algorithms"],
        output_directory=output_directory,
        thresholds=thresholds,
        exact_match_rules=exact_match_rules,
    )

    print(
        f"Selected deterministic frames: "
        f"{len(selected_frames)}"
    )
    print(
        f"Prepared validation input images: "
        f"{saved_image_count}"
    )
    print(
        f"Output directory: {output_directory}"
    )
    print(
        f"Python-Hybrid comparisons: "
        f"{len(selected_frames) * len(resolutions) * len(benchmark_config['algorithms'])}"
    )
    print(
        f"Comparison report: {comparison_path}"
    )
    print(
        f"Pure C++ outputs created: "
        f"{cpp_output_count}"
    )
    print(
        f"Python-C++ comparison report: "
        f"{cpp_comparison_path}"
    )

    print(
        f"Equivalence comparisons: "
        f"{equivalence_comparison_count}"
    )
    print(
        f"Failed comparisons: "
        f"{equivalence_failure_count}"
    )
    print(
        f"Equivalence report: "
        f"{equivalence_report_path}"
    )

    if equivalence_failure_count:
        raise RuntimeError(
            "Output equivalence validation failed."
        )

    print(
        "Output equivalence validation passed."
    )

if __name__ == "__main__":
    main()