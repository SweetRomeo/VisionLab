from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


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

class ControlledIlluminationMetadataError(ValueError):
    """Raised when controlled-illumination metadata is invalid."""


def create_utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def create_unique_identifier(
    prefix: str,
) -> str:
    normalized_prefix = prefix.strip().lower().replace(
        " ",
        "-",
    )

    if not normalized_prefix:
        raise ControlledIlluminationMetadataError(
            "Identifier prefix must not be empty."
        )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    random_suffix = uuid4().hex[:8]

    return (
        f"{normalized_prefix}-"
        f"{timestamp}-"
        f"{random_suffix}"
    )


@dataclass(frozen=True)
class ResolutionMetadata:
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or self.width <= 0
        ):
            raise ControlledIlluminationMetadataError(
                "Resolution width must be a positive integer."
            )

        if (
            isinstance(self.height, bool)
            or not isinstance(self.height, int)
            or self.height <= 0
        ):
            raise ControlledIlluminationMetadataError(
                "Resolution height must be a positive integer."
            )


@dataclass(frozen=True)
class IlluminanceMeasurements:
    centre_lux: float
    top_left_lux: float
    top_right_lux: float
    bottom_left_lux: float
    bottom_right_lux: float
    measured_at_utc: str
    lux_meter_model: str
    lux_meter_range: str
    lux_meter_resolution: str

    def __post_init__(self) -> None:
        for field_name, value in self.position_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ControlledIlluminationMetadataError(
                    f"{field_name} must be a non-negative "
                    "finite number."
                )

        for field_name, value in (
            ("measured_at_utc", self.measured_at_utc),
            ("lux_meter_model", self.lux_meter_model),
            ("lux_meter_range", self.lux_meter_range),
            (
                "lux_meter_resolution",
                self.lux_meter_resolution,
            ),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ControlledIlluminationMetadataError(
                    f"{field_name} must be a non-empty string."
                )

    @property
    def position_values(self) -> dict[str, float]:
        return {
            "centre_lux": float(self.centre_lux),
            "top_left_lux": float(self.top_left_lux),
            "top_right_lux": float(self.top_right_lux),
            "bottom_left_lux": float(
                self.bottom_left_lux
            ),
            "bottom_right_lux": float(
                self.bottom_right_lux
            ),
        }

    @property
    def mean_lux(self) -> float:
        values = list(self.position_values.values())
        return sum(values) / len(values)

    @property
    def minimum_lux(self) -> float:
        return min(self.position_values.values())

    @property
    def maximum_lux(self) -> float:
        return max(self.position_values.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.position_values,
            "mean_lux": self.mean_lux,
            "minimum_lux": self.minimum_lux,
            "maximum_lux": self.maximum_lux,
            "measured_at_utc": self.measured_at_utc,
            "lux_meter_model": self.lux_meter_model,
            "lux_meter_range": self.lux_meter_range,
            "lux_meter_resolution": (
                self.lux_meter_resolution
            ),
        }


@dataclass(frozen=True)
class ControlledIlluminationRunMetadata:
    experiment_id: str
    run_id: str
    collected_at_utc: str
    git_commit_sha: str
    phase: str
    device_id: str
    operating_system: str
    platform: str
    architecture: str
    algorithm: str
    algorithm_parameters: dict[str, Any]
    resolution: ResolutionMetadata
    trial_number: int
    target_fps: float
    frame_deadline_ms: float
    target_illuminance_lux: float | None
    measured_illuminance: IlluminanceMeasurements
    incidence_angle_degrees: float
    source_output_setting: str | float | int
    camera_mode: str
    camera_settings: dict[str, Any]
    camera_to_target_distance_metres: float
    light_to_target_distance_metres: float
    input_scene_id: str
    power_mode: str
    clock_configuration: str
    starting_temperature_celsius: float
    ending_temperature_celsius: float
    maximum_temperature_celsius: float
    thermal_throttling_detected: bool
    software_versions: dict[str, str]
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        metadata = asdict(self)

        metadata["resolution"] = asdict(
            self.resolution
        )
        metadata["measured_illuminance"] = (
            self.measured_illuminance.to_dict()
        )

        return metadata


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

def validate_utc_timestamp(
    value: str,
    field_name: str,
) -> None:
    require_non_empty_string(value, field_name)

    try:
        parsed_timestamp = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ControlledIlluminationMetadataError(
            f"{field_name} must be a valid ISO 8601 timestamp."
        ) from error

    if (
        parsed_timestamp.tzinfo is None
        or parsed_timestamp.utcoffset()
        != timezone.utc.utcoffset(parsed_timestamp)
    ):
        raise ControlledIlluminationMetadataError(
            f"{field_name} must use UTC."
        )


def validate_safe_identifier(
    value: str,
    field_name: str,
) -> None:
    require_non_empty_string(value, field_name)

    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ControlledIlluminationMetadataError(
            f"{field_name} contains invalid path characters."
        )


def validate_run_metadata(
    metadata: ControlledIlluminationRunMetadata,
    config: dict[str, Any],
) -> None:
    validate_controlled_illumination_config(config)

    validate_safe_identifier(
        metadata.experiment_id,
        "experiment_id",
    )
    validate_safe_identifier(
        metadata.run_id,
        "run_id",
    )
    validate_utc_timestamp(
        metadata.collected_at_utc,
        "collected_at_utc",
    )
    validate_utc_timestamp(
        metadata.measured_illuminance.measured_at_utc,
        "measured_illuminance.measured_at_utc",
    )

    required_string_fields = (
        "device_id",
        "operating_system",
        "platform",
        "architecture",
        "algorithm",
        "camera_mode",
        "input_scene_id",
        "power_mode",
        "clock_configuration",
    )

    for field_name in required_string_fields:
        require_non_empty_string(
            getattr(metadata, field_name),
            field_name,
        )

    git_commit_sha = metadata.git_commit_sha.lower()

    if not isinstance(metadata.git_commit_sha, str):
        raise ControlledIlluminationMetadataError(
            "git_commit_sha must be a string."
        )

    if (
        len(git_commit_sha) != 40
        or any(
            character not in "0123456789abcdef"
            for character in git_commit_sha
        )
    ):
        raise ControlledIlluminationMetadataError(
            "git_commit_sha must contain a full "
            "40-character hexadecimal commit SHA."
        )

    phase_names = {
        phase["name"]
        for phase in config["phases"]
    }

    if metadata.phase not in phase_names:
        raise ControlledIlluminationMetadataError(
            f"Unsupported experiment phase: {metadata.phase}"
        )

    matrix = config["experiment_matrix"]

    for field_name, value, allowed_values in (
        (
            "platform",
            metadata.platform,
            matrix["platforms"],
        ),
        (
            "architecture",
            metadata.architecture,
            matrix["architectures"],
        ),
        (
            "algorithm",
            metadata.algorithm,
            matrix["algorithms"],
        ),
    ):
        if value not in allowed_values:
            raise ControlledIlluminationMetadataError(
                f"Unsupported {field_name}: {value}"
            )

    resolution_pair = (
        metadata.resolution.width,
        metadata.resolution.height,
    )

    allowed_resolutions = {
        (
            resolution["width"],
            resolution["height"],
        )
        for resolution in matrix["resolutions"]
    }

    if resolution_pair not in allowed_resolutions:
        raise ControlledIlluminationMetadataError(
            "Unsupported resolution: "
            f"{resolution_pair[0]}x{resolution_pair[1]}"
        )

    trial_count = matrix["trial_count"]

    if (
        isinstance(metadata.trial_number, bool)
        or not isinstance(metadata.trial_number, int)
        or metadata.trial_number < 1
        or metadata.trial_number > trial_count
    ):
        raise ControlledIlluminationMetadataError(
            "trial_number must be between "
            f"1 and {trial_count}."
        )

    allowed_angles = matrix[
        "incidence_angles_degrees"
    ]

    if not any(
        math.isclose(
            metadata.incidence_angle_degrees,
            allowed_angle,
            abs_tol=1e-9,
        )
        for allowed_angle in allowed_angles
    ):
        raise ControlledIlluminationMetadataError(
            "Unsupported incidence angle: "
            f"{metadata.incidence_angle_degrees}"
        )

    target_levels = matrix[
        "target_illuminance_levels_lux"
    ]

    if metadata.phase == "constant_lux":
        if metadata.target_illuminance_lux is None:
            raise ControlledIlluminationMetadataError(
                "constant_lux runs require "
                "target_illuminance_lux."
            )

        if not any(
            math.isclose(
                metadata.target_illuminance_lux,
                target_level,
                abs_tol=1e-9,
            )
            for target_level in target_levels
        ):
            raise ControlledIlluminationMetadataError(
                "Unsupported target illuminance: "
                f"{metadata.target_illuminance_lux}"
            )

    if (
        metadata.phase == "constant_source"
        and metadata.target_illuminance_lux is not None
    ):
        raise ControlledIlluminationMetadataError(
            "constant_source runs must not define "
            "target_illuminance_lux."
        )

    execution = config["execution"]

    if not math.isclose(
        metadata.target_fps,
        execution["target_fps"],
        abs_tol=1e-9,
    ):
        raise ControlledIlluminationMetadataError(
            "target_fps does not match the experiment config."
        )

    expected_deadline_ms = (
        1000.0
        / execution["target_fps"]
        * execution["deadline_multiplier"]
    )

    if not math.isclose(
        metadata.frame_deadline_ms,
        expected_deadline_ms,
        abs_tol=0.01,
    ):
        raise ControlledIlluminationMetadataError(
            "frame_deadline_ms does not match "
            "the configured frame deadline."
        )

    if not isinstance(
        metadata.algorithm_parameters,
        dict,
    ):
        raise ControlledIlluminationMetadataError(
            "algorithm_parameters must be an object."
        )

    camera_modes = config["camera_modes"]

    if metadata.camera_mode not in camera_modes:
        raise ControlledIlluminationMetadataError(
            f"Unsupported camera mode: {metadata.camera_mode}"
        )

    if (
        not isinstance(metadata.camera_settings, dict)
        or not metadata.camera_settings
    ):
        raise ControlledIlluminationMetadataError(
            "camera_settings must be a non-empty object."
        )

    required_camera_settings = set(
        camera_modes[metadata.camera_mode].get(
            "required_settings",
            [],
        )
    )

    missing_camera_settings = (
        required_camera_settings
        - set(metadata.camera_settings)
    )

    if missing_camera_settings:
        raise ControlledIlluminationMetadataError(
            "Missing camera settings: "
            f"{sorted(missing_camera_settings)}"
        )

    for field_name, distance in (
        (
            "camera_to_target_distance_metres",
            metadata.camera_to_target_distance_metres,
        ),
        (
            "light_to_target_distance_metres",
            metadata.light_to_target_distance_metres,
        ),
    ):
        require_positive_number(
            distance,
            field_name,
        )

    temperatures = {
        "starting_temperature_celsius": (
            metadata.starting_temperature_celsius
        ),
        "ending_temperature_celsius": (
            metadata.ending_temperature_celsius
        ),
        "maximum_temperature_celsius": (
            metadata.maximum_temperature_celsius
        ),
    }

    for field_name, temperature in temperatures.items():
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
        ):
            raise ControlledIlluminationMetadataError(
                f"{field_name} must be a finite number."
            )

    if metadata.maximum_temperature_celsius < max(
        metadata.starting_temperature_celsius,
        metadata.ending_temperature_celsius,
    ):
        raise ControlledIlluminationMetadataError(
            "maximum_temperature_celsius cannot be lower "
            "than the starting or ending temperature."
        )

    if not isinstance(
        metadata.thermal_throttling_detected,
        bool,
    ):
        raise ControlledIlluminationMetadataError(
            "thermal_throttling_detected must be boolean."
        )

    source_output = metadata.source_output_setting

    if isinstance(source_output, str):
        require_non_empty_string(
            source_output,
            "source_output_setting",
        )
    elif (
        isinstance(source_output, bool)
        or not isinstance(source_output, (int, float))
        or not math.isfinite(source_output)
        or source_output < 0
    ):
        raise ControlledIlluminationMetadataError(
            "source_output_setting must be a non-negative "
            "number or a non-empty string."
        )

    if (
        not isinstance(metadata.software_versions, dict)
        or not metadata.software_versions
    ):
        raise ControlledIlluminationMetadataError(
            "software_versions must be a non-empty object."
        )


def save_run_metadata_atomic(
    metadata: ControlledIlluminationRunMetadata,
    config: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    active_config = (
        config
        if config is not None
        else load_controlled_illumination_config()
    )

    validate_run_metadata(
        metadata,
        active_config,
    )

    if output_path is None:
        metadata_path = (
            PROJECT_ROOT
            / active_config["output"]["directory"]
            / metadata.platform
            / metadata.experiment_id
            / metadata.run_id
            / active_config["output"][
                "metadata_file_name"
            ]
        )
    else:
        metadata_path = Path(output_path)

        if not metadata_path.is_absolute():
            metadata_path = (
                PROJECT_ROOT / metadata_path
            )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = metadata_path.with_name(
        f".{metadata_path.name}."
        f"{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                metadata.to_dict(),
                output_file,
                indent=2,
                ensure_ascii=False,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(
            temporary_path,
            metadata_path,
        )
    finally:
        temporary_path.unlink(
            missing_ok=True,
        )

    return metadata_path

def require_metadata_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlledIlluminationMetadataError(
            f"{field_name} must be an object."
        )

    return value


def require_metadata_value(
    mapping: dict[str, Any],
    field_name: str,
    parent_name: str,
) -> Any:
    if field_name not in mapping:
        raise ControlledIlluminationMetadataError(
            f"Missing metadata field: "
            f"{parent_name}.{field_name}"
        )

    return mapping[field_name]


def run_metadata_from_dict(
    metadata_value: Any,
    config: dict[str, Any] | None = None,
) -> ControlledIlluminationRunMetadata:
    metadata_data = require_metadata_mapping(
        metadata_value,
        "metadata root",
    )

    resolution_data = require_metadata_mapping(
        require_metadata_value(
            metadata_data,
            "resolution",
            "metadata",
        ),
        "metadata.resolution",
    )

    illuminance_data = require_metadata_mapping(
        require_metadata_value(
            metadata_data,
            "measured_illuminance",
            "metadata",
        ),
        "metadata.measured_illuminance",
    )

    resolution = ResolutionMetadata(
        width=require_metadata_value(
            resolution_data,
            "width",
            "metadata.resolution",
        ),
        height=require_metadata_value(
            resolution_data,
            "height",
            "metadata.resolution",
        ),
    )

    illuminance = IlluminanceMeasurements(
        centre_lux=require_metadata_value(
            illuminance_data,
            "centre_lux",
            "metadata.measured_illuminance",
        ),
        top_left_lux=require_metadata_value(
            illuminance_data,
            "top_left_lux",
            "metadata.measured_illuminance",
        ),
        top_right_lux=require_metadata_value(
            illuminance_data,
            "top_right_lux",
            "metadata.measured_illuminance",
        ),
        bottom_left_lux=require_metadata_value(
            illuminance_data,
            "bottom_left_lux",
            "metadata.measured_illuminance",
        ),
        bottom_right_lux=require_metadata_value(
            illuminance_data,
            "bottom_right_lux",
            "metadata.measured_illuminance",
        ),
        measured_at_utc=require_metadata_value(
            illuminance_data,
            "measured_at_utc",
            "metadata.measured_illuminance",
        ),
        lux_meter_model=require_metadata_value(
            illuminance_data,
            "lux_meter_model",
            "metadata.measured_illuminance",
        ),
        lux_meter_range=require_metadata_value(
            illuminance_data,
            "lux_meter_range",
            "metadata.measured_illuminance",
        ),
        lux_meter_resolution=require_metadata_value(
            illuminance_data,
            "lux_meter_resolution",
            "metadata.measured_illuminance",
        ),
    )

    expected_summaries = {
        "mean_lux": illuminance.mean_lux,
        "minimum_lux": illuminance.minimum_lux,
        "maximum_lux": illuminance.maximum_lux,
    }

    for field_name, expected_value in (
        expected_summaries.items()
    ):
        stored_value = require_metadata_value(
            illuminance_data,
            field_name,
            "metadata.measured_illuminance",
        )

        if (
            isinstance(stored_value, bool)
            or not isinstance(
                stored_value,
                (int, float),
            )
            or not math.isfinite(stored_value)
            or not math.isclose(
                stored_value,
                expected_value,
                abs_tol=1e-9,
            )
        ):
            raise ControlledIlluminationMetadataError(
                "Stored illuminance summary does not "
                f"match measured values: {field_name}"
            )

    metadata_arguments = dict(metadata_data)
    metadata_arguments["resolution"] = resolution
    metadata_arguments[
        "measured_illuminance"
    ] = illuminance

    supported_fields = {
        metadata_field.name
        for metadata_field in fields(
            ControlledIlluminationRunMetadata
        )
    }

    unexpected_fields = (
        set(metadata_arguments)
        - supported_fields
    )

    if unexpected_fields:
        raise ControlledIlluminationMetadataError(
            "Unexpected metadata fields: "
            f"{sorted(unexpected_fields)}"
        )

    try:
        metadata = ControlledIlluminationRunMetadata(
            **metadata_arguments
        )
    except TypeError as error:
        raise ControlledIlluminationMetadataError(
            f"Invalid run metadata structure: {error}"
        ) from error

    active_config = (
        config
        if config is not None
        else load_controlled_illumination_config()
    )

    validate_run_metadata(
        metadata,
        active_config,
    )

    return metadata


def load_run_metadata(
    metadata_path: str | Path,
    config: dict[str, Any] | None = None,
) -> ControlledIlluminationRunMetadata:
    path = Path(metadata_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Run metadata file not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            metadata_value = json.load(
                metadata_file
            )
    except json.JSONDecodeError as error:
        raise ControlledIlluminationMetadataError(
            f"Invalid metadata JSON in {path}: {error}"
        ) from error

    return run_metadata_from_dict(
        metadata_value,
        config=config,
    )


def main() -> None:
    load_controlled_illumination_config()

    print(
        "Controlled-illumination configuration is valid."
    )


if __name__ == "__main__":
    main()