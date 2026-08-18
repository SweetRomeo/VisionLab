import json
import math
from pathlib import Path
from typing import Any

from benchmarks.realtime.realtime_config import (
    RealtimeConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

SUPPORTED_ALGORITHM_NAMES = {
    "original",
    "gamma_correction",
    "histogram_equalization",
    "clahe",
}


def load_benchmark_config(
    config_path: Path = BENCHMARK_CONFIG_PATH,
) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(
            "Benchmark configuration was not "
            f"found: {config_path}"
        )

    try:
        with config_path.open(
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
    *,
    project_root: Path = PROJECT_ROOT,
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

    resolved_project_root = (
        project_root.resolve()
    )
    video_path = (
        resolved_project_root / relative_path
    ).resolve()

    try:
        video_path.relative_to(
            resolved_project_root
        )
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
            not in SUPPORTED_ALGORITHM_NAMES
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