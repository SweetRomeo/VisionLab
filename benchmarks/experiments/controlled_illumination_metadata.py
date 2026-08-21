from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "experiments"
    / "config"
    / "controlled_illumination_config.json"
)

REQUIRED_PHASES = {
    "constant_lux",
    "constant_source",
}

REQUIRED_ILLUMINANCE_POSITIONS = {
    "centre",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
}

REQUIRED_RUN_METADATA_FIELDS = {
    "experiment_id",
    "run_id",
    "collected_at_utc",
    "git_commit_sha",
    "phase",
    "platform",
    "architecture",
    "algorithm",
    "resolution",
    "trial_number",
    "measured_illuminance",
    "incidence_angle_degrees",
    "camera_settings",
}


class ControlledIlluminationConfigError(ValueError):
    """Raised when the experiment configuration is invalid."""


def require_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlledIlluminationConfigError(
            f"{field_name} must be an object."
        )

    return value


def require_list(
    value: Any,
    field_name: str,
) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ControlledIlluminationConfigError(
            f"{field_name} must be a non-empty list."
        )

    return value


def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlledIlluminationConfigError(
            f"{field_name} must be a non-empty string."
        )

    return value


def require_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ControlledIlluminationConfigError(
            f"{field_name} must be a positive integer."
        )

    return value


def require_non_negative_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ControlledIlluminationConfigError(
            f"{field_name} must be a non-negative integer."
        )

    return value


def require_positive_number(
    value: Any,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ControlledIlluminationConfigError(
            f"{field_name} must be a positive finite number."
        )

    return float(value)


def validate_unique_strings(
    values: Any,
    field_name: str,
) -> list[str]:
    items = require_list(values, field_name)

    for item in items:
        require_non_empty_string(item, field_name)

    if len(items) != len(set(items)):
        raise ControlledIlluminationConfigError(
            f"{field_name} contains duplicate values."
        )

    return items


def validate_experiment_matrix(
    matrix: dict[str, Any],
) -> None:
    illuminance_levels = require_list(
        matrix.get("target_illuminance_levels_lux"),
        "experiment_matrix.target_illuminance_levels_lux",
    )

    for index, value in enumerate(illuminance_levels):
        require_positive_number(
            value,
            (
                "experiment_matrix."
                f"target_illuminance_levels_lux[{index}]"
            ),
        )

    if len(illuminance_levels) != len(
        set(illuminance_levels)
    ):
        raise ControlledIlluminationConfigError(
            "Illuminance levels must be unique."
        )

    incidence_angles = require_list(
        matrix.get("incidence_angles_degrees"),
        "experiment_matrix.incidence_angles_degrees",
    )

    for index, angle in enumerate(incidence_angles):
        if (
            isinstance(angle, bool)
            or not isinstance(angle, (int, float))
            or not math.isfinite(angle)
            or angle < 0
            or angle > 90
        ):
            raise ControlledIlluminationConfigError(
                "experiment_matrix."
                f"incidence_angles_degrees[{index}] "
                "must be between 0 and 90 degrees."
            )

    if len(incidence_angles) != len(
        set(incidence_angles)
    ):
        raise ControlledIlluminationConfigError(
            "Incidence angles must be unique."
        )

    validate_unique_strings(
        matrix.get("algorithms"),
        "experiment_matrix.algorithms",
    )
    validate_unique_strings(
        matrix.get("architectures"),
        "experiment_matrix.architectures",
    )
    validate_unique_strings(
        matrix.get("platforms"),
        "experiment_matrix.platforms",
    )

    resolutions = require_list(
        matrix.get("resolutions"),
        "experiment_matrix.resolutions",
    )

    resolution_pairs: set[tuple[int, int]] = set()

    for index, resolution_value in enumerate(resolutions):
        resolution = require_mapping(
            resolution_value,
            f"experiment_matrix.resolutions[{index}]",
        )

        width = require_positive_integer(
            resolution.get("width"),
            (
                "experiment_matrix."
                f"resolutions[{index}].width"
            ),
        )
        height = require_positive_integer(
            resolution.get("height"),
            (
                "experiment_matrix."
                f"resolutions[{index}].height"
            ),
        )

        resolution_pair = (width, height)

        if resolution_pair in resolution_pairs:
            raise ControlledIlluminationConfigError(
                "Experiment resolutions must be unique."
            )

        resolution_pairs.add(resolution_pair)

    require_positive_integer(
        matrix.get("trial_count"),
        "experiment_matrix.trial_count",
    )


def validate_execution_config(
    execution: dict[str, Any],
) -> None:
    require_positive_number(
        execution.get("target_fps"),
        "execution.target_fps",
    )
    require_positive_integer(
        execution.get("queue_capacity"),
        "execution.queue_capacity",
    )
    require_non_negative_integer(
        execution.get("warmup_frames"),
        "execution.warmup_frames",
    )
    require_positive_integer(
        execution.get("measured_frames"),
        "execution.measured_frames",
    )
    require_positive_number(
        execution.get("deadline_multiplier"),
        "execution.deadline_multiplier",
    )
    require_non_empty_string(
        execution.get("drop_policy"),
        "execution.drop_policy",
    )

    if not isinstance(
        execution.get("randomize_run_order"),
        bool,
    ):
        raise ControlledIlluminationConfigError(
            "execution.randomize_run_order must be boolean."
        )

    require_non_negative_integer(
        execution.get("randomization_seed"),
        "execution.randomization_seed",
    )


def validate_controlled_illumination_config(
    config: dict[str, Any],
) -> None:
    if config.get("schema_version") != 1:
        raise ControlledIlluminationConfigError(
            "schema_version must be 1."
        )

    experiment = require_mapping(
        config.get("experiment"),
        "experiment",
    )
    require_non_empty_string(
        experiment.get("name"),
        "experiment.name",
    )
    require_non_empty_string(
        experiment.get("protocol_path"),
        "experiment.protocol_path",
    )

    phases = require_list(
        config.get("phases"),
        "phases",
    )

    phase_names: set[str] = set()

    for index, phase_value in enumerate(phases):
        phase = require_mapping(
            phase_value,
            f"phases[{index}]",
        )
        phase_name = require_non_empty_string(
            phase.get("name"),
            f"phases[{index}].name",
        )

        if phase_name in phase_names:
            raise ControlledIlluminationConfigError(
                f"Duplicate experiment phase: {phase_name}"
            )

        phase_names.add(phase_name)

    missing_phases = REQUIRED_PHASES - phase_names

    if missing_phases:
        raise ControlledIlluminationConfigError(
            "Missing experiment phases: "
            f"{sorted(missing_phases)}"
        )

    validate_experiment_matrix(
        require_mapping(
            config.get("experiment_matrix"),
            "experiment_matrix",
        )
    )

    validate_execution_config(
        require_mapping(
            config.get("execution"),
            "execution",
        )
    )

    illuminance_measurement = require_mapping(
        config.get("illuminance_measurement"),
        "illuminance_measurement",
    )

    measurement_positions = set(
        validate_unique_strings(
            illuminance_measurement.get("positions"),
            "illuminance_measurement.positions",
        )
    )

    missing_positions = (
        REQUIRED_ILLUMINANCE_POSITIONS
        - measurement_positions
    )

    if missing_positions:
        raise ControlledIlluminationConfigError(
            "Missing illuminance measurement positions: "
            f"{sorted(missing_positions)}"
        )

    camera_modes = require_mapping(
        config.get("camera_modes"),
        "camera_modes",
    )

    for required_mode in ("controlled", "operational"):
        require_mapping(
            camera_modes.get(required_mode),
            f"camera_modes.{required_mode}",
        )

    required_metadata_fields = set(
        validate_unique_strings(
            config.get("required_run_metadata"),
            "required_run_metadata",
        )
    )

    missing_metadata_fields = (
        REQUIRED_RUN_METADATA_FIELDS
        - required_metadata_fields
    )

    if missing_metadata_fields:
        raise ControlledIlluminationConfigError(
            "Missing required run metadata fields: "
            f"{sorted(missing_metadata_fields)}"
        )

    output = require_mapping(
        config.get("output"),
        "output",
    )
    require_non_empty_string(
        output.get("directory"),
        "output.directory",
    )
    require_non_empty_string(
        output.get("metadata_file_name"),
        "output.metadata_file_name",
    )


def load_controlled_illumination_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    path = (
        Path(config_path)
        if config_path is not None
        else DEFAULT_CONFIG_PATH
    )

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Controlled-illumination config not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            loaded_config = json.load(config_file)
    except json.JSONDecodeError as error:
        raise ControlledIlluminationConfigError(
            f"Invalid JSON in {path}: {error}"
        ) from error

    config = require_mapping(
        loaded_config,
        "configuration root",
    )

    validate_controlled_illumination_config(config)

    return config


def main() -> None:
    load_controlled_illumination_config()

    print(
        "Controlled-illumination configuration is valid."
    )


if __name__ == "__main__":
    main()