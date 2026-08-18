import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "realtime_config.json"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
).resolve()

REQUIRED_FIELDS = {
    "schema_version",
    "target_fps",
    "queue_capacity",
    "warmup_frames",
    "measured_frames",
    "trial_count",
    "drop_policy",
    "deadline_multiplier",
    "output_directory",
    "frame_results_file",
    "summary_file",
}

SUPPORTED_DROP_POLICIES = {
    "latest_frame",
}


@dataclass(frozen=True)
class RealtimeConfig:
    schema_version: int
    target_fps: float
    queue_capacity: int
    warmup_frames: int
    measured_frames: int
    trial_count: int
    drop_policy: str
    deadline_multiplier: float
    output_directory: Path
    frame_results_file: str
    summary_file: str

    @property
    def frame_period_ms(self) -> float:
        return 1000.0 / self.target_fps

    @property
    def deadline_ms(self) -> float:
        return (
            self.frame_period_ms
            * self.deadline_multiplier
        )

    @property
    def frames_per_trial(self) -> int:
        return (
            self.warmup_frames
            + self.measured_frames
        )

    @property
    def frame_results_path(self) -> Path:
        return (
            self.output_directory
            / self.frame_results_file
        )

    @property
    def summary_path(self) -> Path:
        return (
            self.output_directory
            / self.summary_file
        )


def require_integer(
    data: dict[str, Any],
    field_name: str,
    *,
    minimum: int,
) -> int:
    value = data.get(field_name)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"{field_name} must be an integer "
            f"greater than or equal to {minimum}."
        )

    return value


def require_positive_number(
    data: dict[str, Any],
    field_name: str,
) -> float:
    value = data.get(field_name)

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


def require_string(
    data: dict[str, Any],
    field_name: str,
) -> str:
    value = data.get(field_name)

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def validate_csv_file_name(
    file_name: str,
    field_name: str,
) -> str:
    file_path = Path(file_name)

    if (
        file_path.name != file_name
        or file_path.suffix.lower() != ".csv"
    ):
        raise ValueError(
            f"{field_name} must be a CSV file name "
            "without a directory component."
        )

    return file_name


def resolve_output_directory(
    relative_path: str,
) -> Path:
    path = Path(relative_path)

    if path.is_absolute():
        raise ValueError(
            "output_directory must be relative "
            "to the repository root."
        )

    resolved_path = (
        PROJECT_ROOT / path
    ).resolve()

    try:
        resolved_path.relative_to(RESULTS_ROOT)
    except ValueError as error:
        raise ValueError(
            "output_directory must be located under "
            "benchmarks/results."
        ) from error

    return resolved_path


def load_realtime_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> RealtimeConfig:
    if not config_path.is_file():
        raise FileNotFoundError(
            "Real-time configuration file was "
            f"not found: {config_path}"
        )

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            data = json.load(config_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Real-time configuration contains "
            f"invalid JSON: {config_path}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "Real-time configuration must "
            "contain a JSON object."
        )

    field_names = set(data)

    missing_fields = (
        REQUIRED_FIELDS - field_names
    )
    unexpected_fields = (
        field_names - REQUIRED_FIELDS
    )

    if missing_fields:
        raise ValueError(
            "Missing real-time configuration fields: "
            f"{sorted(missing_fields)}"
        )

    if unexpected_fields:
        raise ValueError(
            "Unexpected real-time configuration fields: "
            f"{sorted(unexpected_fields)}"
        )

    schema_version = require_integer(
        data,
        "schema_version",
        minimum=1,
    )

    if schema_version != 1:
        raise ValueError(
            "Unsupported real-time configuration "
            f"schema version: {schema_version}"
        )

    target_fps = require_positive_number(
        data,
        "target_fps",
    )
    queue_capacity = require_integer(
        data,
        "queue_capacity",
        minimum=1,
    )
    warmup_frames = require_integer(
        data,
        "warmup_frames",
        minimum=0,
    )
    measured_frames = require_integer(
        data,
        "measured_frames",
        minimum=1,
    )
    trial_count = require_integer(
        data,
        "trial_count",
        minimum=1,
    )
    deadline_multiplier = require_positive_number(
        data,
        "deadline_multiplier",
    )

    drop_policy = require_string(
        data,
        "drop_policy",
    )

    if drop_policy not in SUPPORTED_DROP_POLICIES:
        raise ValueError(
            f"Unsupported drop_policy: {drop_policy}"
        )

    if (
        drop_policy == "latest_frame"
        and queue_capacity != 1
    ):
        raise ValueError(
            "latest_frame currently requires "
            "queue_capacity to be 1."
        )

    output_directory = resolve_output_directory(
        require_string(
            data,
            "output_directory",
        )
    )

    frame_results_file = validate_csv_file_name(
        require_string(
            data,
            "frame_results_file",
        ),
        "frame_results_file",
    )
    summary_file = validate_csv_file_name(
        require_string(
            data,
            "summary_file",
        ),
        "summary_file",
    )

    if frame_results_file == summary_file:
        raise ValueError(
            "frame_results_file and summary_file "
            "must be different."
        )

    return RealtimeConfig(
        schema_version=schema_version,
        target_fps=target_fps,
        queue_capacity=queue_capacity,
        warmup_frames=warmup_frames,
        measured_frames=measured_frames,
        trial_count=trial_count,
        drop_policy=drop_policy,
        deadline_multiplier=deadline_multiplier,
        output_directory=output_directory,
        frame_results_file=frame_results_file,
        summary_file=summary_file,
    )


def main() -> None:
    config = load_realtime_config()

    print(
        "Real-time configuration validated."
    )
    print(
        f"Target FPS: {config.target_fps:.3f}"
    )
    print(
        f"Frame period: "
        f"{config.frame_period_ms:.3f} ms"
    )
    print(
        f"Deadline: {config.deadline_ms:.3f} ms"
    )
    print(
        f"Frames per trial: "
        f"{config.frames_per_trial}"
    )


if __name__ == "__main__":
    main()