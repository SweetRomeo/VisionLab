import json
import math
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PURE_PYTHON_DIRECTORY = (
    PROJECT_ROOT / "pure-python"
)

BENCHMARK_CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

for import_directory in (
    PROJECT_ROOT,
    PURE_PYTHON_DIRECTORY,
):
    import_path = str(import_directory)

    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from benchmarks.realtime.realtime_config import (  # noqa: E402
    RealtimeConfig,
    load_realtime_config,
)
from benchmarks.realtime.realtime_pipeline import (  # noqa: E402
    iter_video_frames,
    run_realtime_trial,
)
from benchmarks.realtime.realtime_records import (  # noqa: E402
    RealtimeFrameRecord,
    write_frame_records,
)
from image_process import (  # noqa: E402
    ImageProcess,
    ProcessingAlgorithm,
    ProcessingParameters,
)


FrameProcessor = Callable[
    [np.ndarray],
    np.ndarray,
]


ALGORITHM_MAP = {
    "original": ProcessingAlgorithm.ORIGINAL,
    "gamma_correction": (
        ProcessingAlgorithm.GAMMA
    ),
    "histogram_equalization": (
        ProcessingAlgorithm.HISTOGRAM
    ),
    "clahe": ProcessingAlgorithm.CLAHE,
}


def load_benchmark_config() -> dict[str, Any]:
    if not BENCHMARK_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            "Benchmark configuration was not "
            f"found: {BENCHMARK_CONFIG_PATH}"
        )

    try:
        with BENCHMARK_CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            config = json.load(config_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Benchmark configuration contains "
            "invalid JSON."
        ) from error

    if not isinstance(config, dict):
        raise ValueError(
            "Benchmark configuration must "
            "contain a JSON object."
        )

    required_sections = {
        "input",
        "benchmark",
        "resolutions",
        "algorithms",
    }

    missing_sections = (
        required_sections - set(config)
    )

    if missing_sections:
        raise ValueError(
            "Missing benchmark configuration "
            f"sections: {sorted(missing_sections)}"
        )

    return config


def validate_shared_execution_counts(
    benchmark_config: dict[str, Any],
    realtime_config: RealtimeConfig,
) -> None:
    benchmark_settings = (
        benchmark_config.get("benchmark")
    )

    if not isinstance(
        benchmark_settings,
        dict,
    ):
        raise ValueError(
            "benchmark must be a JSON object."
        )

    expected_values = {
        "warmup_frames": (
            realtime_config.warmup_frames
        ),
        "measured_frames": (
            realtime_config.measured_frames
        ),
        "trials": (
            realtime_config.trial_count
        ),
    }

    for field_name, expected_value in (
        expected_values.items()
    ):
        actual_value = benchmark_settings.get(
            field_name
        )

        if (
            isinstance(actual_value, bool)
            or not isinstance(
                actual_value,
                int,
            )
        ):
            raise ValueError(
                f"benchmark.{field_name} must "
                "be an integer."
            )

        if actual_value != expected_value:
            raise ValueError(
                "Offline and real-time experiment "
                "counts must match. "
                f"benchmark.{field_name}="
                f"{actual_value}, realtime="
                f"{expected_value}."
            )


def resolve_video_path(
    benchmark_config: dict[str, Any],
) -> Path:
    input_config = benchmark_config.get(
        "input"
    )

    if not isinstance(input_config, dict):
        raise ValueError(
            "input must be a JSON object."
        )

    video_path_value = input_config.get(
        "video_path"
    )

    if (
        not isinstance(video_path_value, str)
        or not video_path_value.strip()
    ):
        raise ValueError(
            "input.video_path must be a "
            "non-empty string."
        )

    relative_path = Path(
        video_path_value
    )

    if relative_path.is_absolute():
        raise ValueError(
            "input.video_path must be relative "
            "to the repository root."
        )

    video_path = (
        PROJECT_ROOT / relative_path
    ).resolve()

    try:
        video_path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(
            "input.video_path cannot point "
            "outside the repository."
        ) from error

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Benchmark video was not found: "
            f"{video_path}"
        )

    return video_path


def load_resolutions(
    benchmark_config: dict[str, Any],
) -> list[tuple[int, int]]:
    resolution_config = (
        benchmark_config.get("resolutions")
    )

    if (
        not isinstance(
            resolution_config,
            list,
        )
        or not resolution_config
    ):
        raise ValueError(
            "resolutions must be a "
            "non-empty list."
        )

    resolutions = []

    for resolution in resolution_config:
        if not isinstance(resolution, dict):
            raise ValueError(
                "Each resolution must be "
                "a JSON object."
            )

        width = resolution.get("width")
        height = resolution.get("height")

        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise ValueError(
                f"Invalid resolution: {resolution}"
            )

        resolutions.append(
            (width, height)
        )

    if len(resolutions) != len(
        set(resolutions)
    ):
        raise ValueError(
            "Duplicate resolutions were found."
        )

    return resolutions


def load_algorithms(
    benchmark_config: dict[str, Any],
) -> list[dict[str, Any]]:
    algorithm_config = (
        benchmark_config.get("algorithms")
    )

    if (
        not isinstance(algorithm_config, list)
        or not algorithm_config
    ):
        raise ValueError(
            "algorithms must be a "
            "non-empty list."
        )

    algorithm_names = []

    for algorithm in algorithm_config:
        if not isinstance(algorithm, dict):
            raise ValueError(
                "Each algorithm must be "
                "a JSON object."
            )

        algorithm_name = algorithm.get("name")
        parameters = algorithm.get(
            "parameters",
            {},
        )

        if (
            not isinstance(algorithm_name, str)
            or algorithm_name
            not in ALGORITHM_MAP
        ):
            raise ValueError(
                "Unsupported algorithm: "
                f"{algorithm_name}"
            )

        if not isinstance(parameters, dict):
            raise ValueError(
                "Algorithm parameters must "
                "be a JSON object."
            )

        algorithm_names.append(
            algorithm_name
        )

    if len(algorithm_names) != len(
        set(algorithm_names)
    ):
        raise ValueError(
            "Duplicate algorithms were found."
        )

    return algorithm_config


def require_positive_number(
    parameters: dict[str, Any],
    field_name: str,
    default_value: float,
) -> float:
    value = parameters.get(
        field_name,
        default_value,
    )

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(
            f"{field_name} must be a finite "
            "number greater than zero."
        )

    return float(value)


def require_positive_integer(
    parameters: dict[str, Any],
    field_name: str,
    default_value: int,
) -> int:
    value = parameters.get(
        field_name,
        default_value,
    )

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be a "
            "positive integer."
        )

    return value


def create_frame_processor(
    algorithm_config: dict[str, Any],
) -> FrameProcessor:
    algorithm_name = algorithm_config["name"]
    parameters = algorithm_config.get(
        "parameters",
        {},
    )

    gamma_value = require_positive_number(
        parameters,
        "gamma_value",
        0.6,
    )
    clahe_clip_limit = (
        require_positive_number(
            parameters,
            "clip_limit",
            4.0,
        )
    )
    clahe_grid_size = (
        require_positive_integer(
            parameters,
            "grid_size",
            8,
        )
    )

    processing_parameters = (
        ProcessingParameters(
            gamma_value=gamma_value,
            clahe_clip_limit=(
                clahe_clip_limit
            ),
            clahe_grid_size=(
                clahe_grid_size
            ),
        )
    )
    processing_algorithm = ALGORITHM_MAP[
        algorithm_name
    ]
    image_processor = ImageProcess()

    def process_frame(
        frame: np.ndarray,
    ) -> np.ndarray:
        return image_processor.process(
            frame,
            processing_algorithm,
            processing_parameters,
        )

    return process_frame


def run_all_experiments(
    benchmark_config: dict[str, Any],
    realtime_config: RealtimeConfig,
) -> list[RealtimeFrameRecord]:
    validate_shared_execution_counts(
        benchmark_config,
        realtime_config,
    )

    video_path = resolve_video_path(
        benchmark_config
    )
    resolutions = load_resolutions(
        benchmark_config
    )
    algorithms = load_algorithms(
        benchmark_config
    )

    all_records = []

    for width, height in resolutions:
        resolution_name = (
            f"{width}x{height}"
        )

        for algorithm_config in algorithms:
            algorithm_name = (
                algorithm_config["name"]
            )

            for trial in range(
                1,
                realtime_config.trial_count + 1,
            ):
                print(
                    "Pure Python real-time | "
                    f"{algorithm_name} | "
                    f"{resolution_name} | "
                    f"trial {trial}/"
                    f"{realtime_config.trial_count}"
                )

                processor = (
                    create_frame_processor(
                        algorithm_config
                    )
                )

                trial_records = (
                    run_realtime_trial(
                        frame_source=(
                            iter_video_frames(
                                video_path
                            )
                        ),
                        processor=processor,
                        config=realtime_config,
                        architecture=(
                            "pure_python"
                        ),
                        algorithm=(
                            algorithm_name
                        ),
                        width=width,
                        height=height,
                        trial=trial,
                    )
                )

                all_records.extend(
                    trial_records
                )

    return all_records


def main() -> None:
    benchmark_config = (
        load_benchmark_config()
    )
    realtime_config = (
        load_realtime_config()
    )

    records = run_all_experiments(
        benchmark_config,
        realtime_config,
    )

    output_path = (
        realtime_config.output_directory
        / "pure_python"
        / realtime_config.frame_results_file
    )

    written_record_count = (
        write_frame_records(
            records,
            output_path,
        )
    )

    status_counts = Counter(
        record.frame_status.value
        for record in records
    )

    print(
        "Pure Python real-time evaluation "
        "completed."
    )
    print(
        f"Frame records: "
        f"{written_record_count}"
    )
    print(
        f"Processed: "
        f"{status_counts['processed']}"
    )
    print(
        f"Dropped: "
        f"{status_counts['dropped']}"
    )
    print(
        f"Skipped: "
        f"{status_counts['skipped']}"
    )
    print(
        f"Results: {output_path}"
    )


if __name__ == "__main__":
    main()