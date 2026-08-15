import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PURE_PYTHON_DIRECTORY = PROJECT_ROOT / "pure-python"
HYBRID_PROJECT_DIRECTORY = PROJECT_ROOT / "hybrid-python-cpp"
CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

DLL_DIRECTORY_HANDLES = []

sys.path.insert(0, str(PURE_PYTHON_DIRECTORY))

from image_process import (  # noqa: E402
    ImageProcess,
    ProcessingAlgorithm,
    ProcessingParameters,
)


PURE_PYTHON_ALGORITHMS = {
    "original": ProcessingAlgorithm.ORIGINAL,
    "gamma_correction": ProcessingAlgorithm.GAMMA,
    "histogram_equalization": ProcessingAlgorithm.HISTOGRAM,
    "clahe": ProcessingAlgorithm.CLAHE,
}


@dataclass(frozen=True)
class EquivalenceMetrics:
    mean_absolute_error: float
    maximum_absolute_error: int
    psnr: float
    exact_match: bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pixel-level output equivalence between "
            "the Pure Python implementation and the shared "
            "C++ processing core."
        )
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help=(
            "Number of deterministic video frames to validate "
            "across the full video. Default: 5."
        ),
    )
    parser.add_argument(
        "--max-mae",
        type=float,
        default=0.0,
        help=(
            "Maximum accepted mean absolute error. "
            "Default: 0.0."
        ),
    )
    parser.add_argument(
        "--max-absolute-error",
        type=int,
        default=0,
        help=(
            "Maximum accepted per-channel absolute error. "
            "Default: 0."
        ),
    )

    arguments = parser.parse_args()

    if arguments.sample_count <= 0:
        parser.error("--sample-count must be greater than zero.")

    if not math.isfinite(arguments.max_mae):
        parser.error("--max-mae must be finite.")

    if arguments.max_mae < 0.0:
        parser.error("--max-mae cannot be negative.")

    if arguments.max_absolute_error < 0:
        parser.error(
            "--max-absolute-error cannot be negative."
        )

    return arguments


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return json.load(config_file)


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


def find_cpp_module_directory() -> Path:
    configured_directory = os.getenv(
        "VISIONLAB_CPP_MODULE_DIR"
    )

    if configured_directory:
        module_directory = Path(
            configured_directory
        ).resolve()

        if not module_directory.is_dir():
            raise FileNotFoundError(
                "VISIONLAB_CPP_MODULE_DIR is not a directory: "
                f"{module_directory}"
            )

        configured_candidates = [
            candidate
            for pattern in (
                "visionlab_cpp*.pyd",
                "visionlab_cpp*.so",
            )
            for candidate in module_directory.glob(pattern)
        ]

        if not configured_candidates:
            raise FileNotFoundError(
                "The configured directory does not contain "
                "a visionlab_cpp module: "
                f"{module_directory}"
            )

        return module_directory

    build_directory = HYBRID_PROJECT_DIRECTORY / "build"
    candidates = []

    for pattern in (
        "visionlab_cpp*.pyd",
        "visionlab_cpp*.so",
    ):
        candidates.extend(build_directory.rglob(pattern))

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
            "A Release build of visionlab_cpp was not found. "
            "Build the Hybrid project in Release mode or set "
            "VISIONLAB_CPP_MODULE_DIR."
        )

    newest_module = max(
        release_candidates,
        key=lambda path: path.stat().st_mtime,
    )

    return newest_module.parent


def import_cpp_module():
    module_directory = find_cpp_module_directory()
    sys.path.insert(0, str(module_directory))

    if os.name == "nt" and hasattr(
        os,
        "add_dll_directory",
    ):
        DLL_DIRECTORY_HANDLES.append(
            os.add_dll_directory(str(module_directory))
        )

    import visionlab_cpp  # noqa: PLC0415

    return visionlab_cpp


def choose_frame_indices(
    frame_count: int,
    sample_count: int,
) -> list[int]:
    if frame_count <= 0:
        raise ValueError(
            "The benchmark video reports no readable frames."
        )

    selected_count = min(frame_count, sample_count)

    if selected_count == 1:
        return [0]

    indices = {
        round(
            sample_index
            * (frame_count - 1)
            / (selected_count - 1)
        )
        for sample_index in range(selected_count)
    }

    return sorted(indices)


def read_frame_at(
    capture: cv2.VideoCapture,
    frame_index: int,
) -> np.ndarray:
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
        raise RuntimeError(
            f"Could not seek to video frame {frame_index}."
        )
    frame_received, frame = capture.read()

    if not frame_received or frame is None or frame.size == 0:
        raise RuntimeError(
            f"Could not read video frame {frame_index}."
        )

    if (
        frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError(
            f"Frame {frame_index} is not an H x W x 3 "
            "uint8 BGR image."
        )

    return frame


def create_python_parameters(
    algorithm_config: dict,
) -> ProcessingParameters:
    parameters = algorithm_config.get("parameters", {})

    return ProcessingParameters(
        gamma_value=float(
            parameters.get("gamma_value", 0.6)
        ),
        clahe_clip_limit=float(
            parameters.get("clip_limit", 4.0)
        ),
        clahe_grid_size=int(
            parameters.get("grid_size", 8)
        ),
    )


def create_cpp_parameters(
    algorithm_config: dict,
) -> tuple[float, float, int]:
    parameters = algorithm_config.get("parameters", {})

    return (
        float(parameters.get("gamma_value", 0.6)),
        float(parameters.get("clip_limit", 4.0)),
        int(parameters.get("grid_size", 8)),
    )


def validate_output_structure(
    reference: np.ndarray,
    comparison: np.ndarray,
    context: str,
) -> None:
    if reference.size == 0 or comparison.size == 0:
        raise ValueError(
            f"An empty output was produced for {context}."
        )

    if reference.shape != comparison.shape:
        raise ValueError(
            f"Output shape mismatch for {context}: "
            f"{reference.shape} != {comparison.shape}"
        )

    if reference.dtype != comparison.dtype:
        raise ValueError(
            f"Output data type mismatch for {context}: "
            f"{reference.dtype} != {comparison.dtype}"
        )

    if reference.dtype != np.uint8:
        raise ValueError(
            f"Output data type must be uint8 for {context}."
        )


def calculate_metrics(
    reference: np.ndarray,
    comparison: np.ndarray,
) -> EquivalenceMetrics:
    signed_difference = (
        reference.astype(np.int16)
        - comparison.astype(np.int16)
    )
    absolute_difference = np.abs(signed_difference)

    mean_absolute_error = float(
        absolute_difference.mean(dtype=np.float64)
    )
    maximum_absolute_error = int(
        absolute_difference.max()
    )

    squared_difference = np.square(
        signed_difference.astype(np.float64)
    )
    mean_squared_error = float(
        squared_difference.mean(dtype=np.float64)
    )

    psnr = (
        math.inf
        if mean_squared_error == 0.0
        else 10.0
        * math.log10(
            (255.0 * 255.0) / mean_squared_error
        )
    )

    return EquivalenceMetrics(
        mean_absolute_error=mean_absolute_error,
        maximum_absolute_error=maximum_absolute_error,
        psnr=psnr,
        exact_match=bool(
            np.array_equal(reference, comparison)
        ),
    )


def format_psnr(psnr: float) -> str:
    return "inf" if math.isinf(psnr) else f"{psnr:.6f}"


def main() -> int:
    arguments = parse_arguments()
    config = load_config()
    visionlab_cpp = import_cpp_module()

    cpp_algorithms = {
        "original": (
            visionlab_cpp.ProcessingAlgorithm.ORIGINAL
        ),
        "gamma_correction": (
            visionlab_cpp.ProcessingAlgorithm.GAMMA
        ),
        "histogram_equalization": (
            visionlab_cpp.ProcessingAlgorithm.HISTOGRAM
        ),
        "clahe": visionlab_cpp.ProcessingAlgorithm.CLAHE,
    }

    video_path = (
        PROJECT_ROOT / config["input"]["video_path"]
    ).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Benchmark video was not found: {video_path}"
        )

    output_directory = (
        PROJECT_ROOT / config["output"]["directory"]
    ).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory
        / "output_equivalence_results.csv"
    )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Benchmark video could not be opened: {video_path}"
        )

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    frame_indices = choose_frame_indices(
        frame_count,
        arguments.sample_count,
    )

    processor = ImageProcess()
    rows = []
    failure_count = 0

    try:
        for frame_index in frame_indices:
            frame = read_frame_at(capture, frame_index)

            for resolution in config["resolutions"]:
                width = int(resolution["width"])
                height = int(resolution["height"])
                resolution_name = f"{width}x{height}"

                resized_frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

                for algorithm_config in config["algorithms"]:
                    algorithm_name = algorithm_config["name"]

                    if algorithm_name not in PURE_PYTHON_ALGORITHMS:
                        raise ValueError(
                            "Unsupported Pure Python algorithm: "
                            f"{algorithm_name}"
                        )

                    if algorithm_name not in cpp_algorithms:
                        raise ValueError(
                            "Unsupported C++ algorithm: "
                            f"{algorithm_name}"
                        )

                    python_output = processor.process(
                        resized_frame,
                        PURE_PYTHON_ALGORITHMS[
                            algorithm_name
                        ],
                        create_python_parameters(
                            algorithm_config
                        ),
                    )

                    (
                        gamma_value,
                        clahe_clip_limit,
                        clahe_grid_size,
                    ) = create_cpp_parameters(
                        algorithm_config
                    )

                    cpp_output = visionlab_cpp.process_frame(
                        resized_frame,
                        cpp_algorithms[algorithm_name],
                        gamma_value=gamma_value,
                        clahe_clip_limit=clahe_clip_limit,
                        clahe_grid_size=clahe_grid_size,
                    )

                    context = (
                        f"algorithm={algorithm_name}, "
                        f"resolution={resolution_name}, "
                        f"frame={frame_index}"
                    )
                    validate_output_structure(
                        python_output,
                        cpp_output,
                        context,
                    )

                    metrics = calculate_metrics(
                        python_output,
                        cpp_output,
                    )
                    validation_passed = (
                        metrics.mean_absolute_error
                        <= arguments.max_mae
                        and metrics.maximum_absolute_error
                        <= arguments.max_absolute_error
                    )

                    if not validation_passed:
                        failure_count += 1

                    rows.append(
                        {
                            "algorithm": algorithm_name,
                            "resolution": resolution_name,
                            "frame_index": frame_index,
                            "reference_architecture": (
                                "pure_python"
                            ),
                            "comparison_architecture": (
                                "hybrid_and_pure_cpp_shared_core"
                            ),
                            "mean_absolute_error": (
                                repr(metrics.mean_absolute_error)
                            ),
                            "maximum_absolute_error": (
                                metrics.maximum_absolute_error
                            ),
                            "psnr": format_psnr(metrics.psnr),
                            "exact_match": str(
                                metrics.exact_match
                            ).lower(),
                            "validation_passed": str(
                                validation_passed
                            ).lower(),
                        }
                    )

                    status = (
                        "PASS" if validation_passed else "FAIL"
                    )
                    print(
                        f"{status} | {algorithm_name} | "
                        f"{resolution_name} | "
                        f"frame {frame_index} | "
                        f"MAE {metrics.mean_absolute_error:.6f} | "
                        "max error "
                        f"{metrics.maximum_absolute_error}"
                    )
    finally:
        capture.release()

    fieldnames = [
        "algorithm",
        "resolution",
        "frame_index",
        "reference_architecture",
        "comparison_architecture",
        "mean_absolute_error",
        "maximum_absolute_error",
        "psnr",
        "exact_match",
        "validation_passed",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Validation results created: {output_path}")

    if failure_count > 0:
        print(
            f"Output equivalence failed for {failure_count} "
            "comparison(s)."
        )
        return 1

    print(
        f"Output equivalence passed for {len(rows)} "
        "comparison(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())