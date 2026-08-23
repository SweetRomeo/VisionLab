from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import csv
import json
import os
import re
from pathlib import Path
from uuid import uuid4
from typing import Any
from itertools import product
from random import Random

from benchmarks.experiments.controlled_illumination_metadata import (
    ControlledIlluminationConfigError,
    ResolutionMetadata,
    validate_controlled_illumination_config,
    validate_utc_timestamp,
)

RUN_PLAN_JSON_FILE_NAME = "run_plan.json"
RUN_PLAN_CSV_FILE_NAME = "run_plan.csv"

PLANNED_RUN_STATUS = "planned"

SUPPORTED_PHASES = {
    "constant_lux",
    "constant_source",
}

PLAN_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
)



class ControlledIlluminationRunPlanError(ValueError):
    """Raised when an experiment run plan is invalid."""

def require_safe_plan_identifier(
    value: Any,
    field_name: str,
) -> str:
    identifier = require_non_empty_plan_string(
        value,
        field_name,
    )

    if PLAN_IDENTIFIER_PATTERN.fullmatch(
        identifier
    ) is None:
        raise ControlledIlluminationRunPlanError(
            f"{field_name} contains unsafe characters."
        )

    return identifier


def require_non_empty_plan_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlledIlluminationRunPlanError(
            f"{field_name} must be a non-empty string."
        )

    return value


def require_positive_plan_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ControlledIlluminationRunPlanError(
            f"{field_name} must be a positive integer."
        )

    return value


def require_finite_plan_number(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ControlledIlluminationRunPlanError(
            f"{field_name} must be a finite number."
        )

    numeric_value = float(value)

    if positive and numeric_value <= 0.0:
        raise ControlledIlluminationRunPlanError(
            f"{field_name} must be positive."
        )

    if non_negative and numeric_value < 0.0:
        raise ControlledIlluminationRunPlanError(
            f"{field_name} must be non-negative."
        )

    return numeric_value


def validate_source_output_setting(
    value: Any,
) -> None:
    if isinstance(value, str):
        require_non_empty_plan_string(
            value,
            "source_output_setting",
        )
        return

    require_finite_plan_number(
        value,
        "source_output_setting",
        non_negative=True,
    )

@dataclass(frozen=True)
class RunCondition:
    phase: str
    platform: str
    architecture: str
    algorithm: str
    resolution: ResolutionMetadata
    trial_number: int
    incidence_angle_degrees: float
    target_illuminance_lux: float | None
    source_output_setting: str | int | float | None
    target_fps: float
    frame_deadline_ms: float


@dataclass(frozen=True)
class PlannedRun:
    execution_order: int
    experiment_id: str
    run_id: str
    phase: str
    platform: str
    architecture: str
    algorithm: str
    resolution: ResolutionMetadata
    trial_number: int
    incidence_angle_degrees: float
    target_illuminance_lux: float | None
    source_output_setting: str | int | float | None
    target_fps: float
    frame_deadline_ms: float
    status: str = PLANNED_RUN_STATUS

    @property
    def condition_key(self) -> tuple[Any, ...]:
        return (
            self.phase,
            self.platform,
            self.architecture,
            self.algorithm,
            self.resolution.width,
            self.resolution.height,
            self.trial_number,
            self.incidence_angle_degrees,
            self.target_illuminance_lux,
            self.source_output_setting,
            self.target_fps,
            self.frame_deadline_ms,
        )

    def __post_init__(self) -> None:
        if not isinstance(
                self.resolution,
                ResolutionMetadata,
        ):
            raise ControlledIlluminationRunPlanError(
                "resolution must be ResolutionMetadata."
            )
        require_positive_plan_integer(
            self.execution_order,
            "execution_order",
        )
        require_positive_plan_integer(
            self.trial_number,
            "trial_number",
        )

        require_safe_plan_identifier(
            self.experiment_id,
            "experiment_id",
        )
        require_safe_plan_identifier(
            self.run_id,
            "run_id",
        )

        for field_name in (
                "phase",
                "platform",
                "architecture",
                "algorithm",
                "status",
        ):
            require_non_empty_plan_string(
                getattr(self, field_name),
                field_name,
            )

        if self.phase not in SUPPORTED_PHASES:
            raise ControlledIlluminationRunPlanError(
                f"Unsupported phase: {self.phase}"
            )

        if self.status != PLANNED_RUN_STATUS:
            raise ControlledIlluminationRunPlanError(
                "New planned runs must use planned status."
            )

        require_finite_plan_number(
            self.incidence_angle_degrees,
            "incidence_angle_degrees",
            non_negative=True,
        )
        require_finite_plan_number(
            self.target_fps,
            "target_fps",
            positive=True,
        )
        require_finite_plan_number(
            self.frame_deadline_ms,
            "frame_deadline_ms",
            positive=True,
        )

        if self.phase == "constant_lux":
            if self.target_illuminance_lux is None:
                raise ControlledIlluminationRunPlanError(
                    "constant_lux runs require "
                    "target_illuminance_lux."
                )

            require_finite_plan_number(
                self.target_illuminance_lux,
                "target_illuminance_lux",
                positive=True,
            )

            if self.source_output_setting is not None:
                raise ControlledIlluminationRunPlanError(
                    "constant_lux plans must not define "
                    "source_output_setting."
                )

        if self.phase == "constant_source":
            if self.target_illuminance_lux is not None:
                raise ControlledIlluminationRunPlanError(
                    "constant_source runs must not define "
                    "target_illuminance_lux."
                )

            if self.source_output_setting is None:
                raise ControlledIlluminationRunPlanError(
                    "constant_source runs require "
                    "source_output_setting."
                )

            validate_source_output_setting(
                self.source_output_setting
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledIlluminationRunPlan:
    schema_version: int
    generated_at_utc: str
    experiment_id: str
    randomized: bool
    randomization_seed: int
    runs: tuple[PlannedRun, ...]

    def __post_init__(self) -> None:
        if (
                isinstance(self.schema_version, bool)
                or not isinstance(self.schema_version, int)
                or self.schema_version != 1
        ):
            raise ControlledIlluminationRunPlanError(
                "schema_version must be 1."
            )

        try:
            validate_utc_timestamp(
                self.generated_at_utc,
                "generated_at_utc",
            )
        except ValueError as error:
            raise ControlledIlluminationRunPlanError(
                str(error)
            ) from error

        require_safe_plan_identifier(
            self.experiment_id,
            "experiment_id",
        )

        if not isinstance(self.randomized, bool):
            raise ControlledIlluminationRunPlanError(
                "randomized must be boolean."
            )

        if (
            isinstance(self.randomization_seed, bool)
            or not isinstance(
                self.randomization_seed,
                int,
            )
            or self.randomization_seed < 0
        ):
            raise ControlledIlluminationRunPlanError(
                "randomization_seed must be a "
                "non-negative integer."
            )

        if not isinstance(self.runs, tuple) or not self.runs:
            raise ControlledIlluminationRunPlanError(
                "runs must be a non-empty tuple."
            )
        if not all(
                isinstance(planned_run, PlannedRun)
                for planned_run in self.runs
        ):
            raise ControlledIlluminationRunPlanError(
                "Every runs item must be PlannedRun."
            )

        expected_orders = list(
            range(1, len(self.runs) + 1)
        )
        actual_orders = [
            planned_run.execution_order
            for planned_run in self.runs
        ]

        if actual_orders != expected_orders:
            raise ControlledIlluminationRunPlanError(
                "execution_order values must be "
                "sequential and start at 1."
            )

        run_ids = [
            planned_run.run_id
            for planned_run in self.runs
        ]

        if len(run_ids) != len(set(run_ids)):
            raise ControlledIlluminationRunPlanError(
                "Run IDs must be unique."
            )

        condition_keys = [
            planned_run.condition_key
            for planned_run in self.runs
        ]

        if len(condition_keys) != len(
                set(condition_keys)
        ):
            raise ControlledIlluminationRunPlanError(
                "Duplicate experimental conditions "
                "are not allowed."
            )

        for planned_run in self.runs:
            if (
                planned_run.experiment_id
                != self.experiment_id
            ):
                raise ControlledIlluminationRunPlanError(
                    "Every run must use the plan's "
                    "experiment_id."
                )

    @property
    def run_count(self) -> int:
        return len(self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "experiment_id": self.experiment_id,
            "randomized": self.randomized,
            "randomization_seed": (
                self.randomization_seed
            ),
            "run_count": self.run_count,
            "runs": [
                planned_run.to_dict()
                for planned_run in self.runs
            ],
        }

def build_run_conditions(
        config: dict[str, Any],
) -> tuple[RunCondition, ...]:
    try:
        validate_controlled_illumination_config(
            config
        )
    except ControlledIlluminationConfigError as error:
        raise ControlledIlluminationRunPlanError(
            str(error)
        ) from error

    matrix = config.get("experiment_matrix")
    execution = config.get("execution")
    phases = config.get("phases")

    if not isinstance(matrix, dict):
        raise ControlledIlluminationRunPlanError(
            "experiment_matrix must be an object."
        )

    if not isinstance(execution, dict):
        raise ControlledIlluminationRunPlanError(
            "execution must be an object."
        )

    if not isinstance(phases, list) or not phases:
        raise ControlledIlluminationRunPlanError(
            "phases must be a non-empty list."
        )

    phase_names = []

    for phase in phases:
        if not isinstance(phase, dict):
            raise ControlledIlluminationRunPlanError(
                "Every phase must be an object."
            )

        phase_name = require_non_empty_plan_string(
            phase.get("name"),
            "phases.name",
        )

        if phase_name not in SUPPORTED_PHASES:
            raise ControlledIlluminationRunPlanError(
                f"Unsupported phase: {phase_name}"
            )

        phase_names.append(phase_name)

    if len(phase_names) != len(set(phase_names)):
        raise ControlledIlluminationRunPlanError(
            "Phase names must be unique."
        )

    target_fps = require_finite_plan_number(
        execution.get("target_fps"),
        "execution.target_fps",
        positive=True,
    )
    deadline_multiplier = require_finite_plan_number(
        execution.get("deadline_multiplier"),
        "execution.deadline_multiplier",
        positive=True,
    )
    frame_deadline_ms = (
                                1000.0 / target_fps
                        ) * deadline_multiplier

    trial_count = require_positive_plan_integer(
        matrix.get("trial_count"),
        "experiment_matrix.trial_count",
    )

    platforms = matrix.get("platforms")
    architectures = matrix.get("architectures")
    algorithms = matrix.get("algorithms")
    resolutions_data = matrix.get("resolutions")
    angles = matrix.get("incidence_angles_degrees")
    illuminance_levels = matrix.get(
        "target_illuminance_levels_lux"
    )

    named_lists = {
        "platforms": platforms,
        "architectures": architectures,
        "algorithms": algorithms,
    }

    for field_name, values in named_lists.items():
        if not isinstance(values, list) or not values:
            raise ControlledIlluminationRunPlanError(
                f"experiment_matrix.{field_name} "
                "must be a non-empty list."
            )

        for value in values:
            require_non_empty_plan_string(
                value,
                f"experiment_matrix.{field_name}",
            )

    if (
            not isinstance(resolutions_data, list)
            or not resolutions_data
    ):
        raise ControlledIlluminationRunPlanError(
            "experiment_matrix.resolutions must be "
            "a non-empty list."
        )

    resolutions = []

    for resolution_data in resolutions_data:
        if not isinstance(resolution_data, dict):
            raise ControlledIlluminationRunPlanError(
                "Every resolution must be an object."
            )

        width = require_positive_plan_integer(
            resolution_data.get("width"),
            "resolution.width",
        )
        height = require_positive_plan_integer(
            resolution_data.get("height"),
            "resolution.height",
        )

        resolutions.append(
            ResolutionMetadata(
                width=width,
                height=height,
            )
        )

    if not isinstance(angles, list) or not angles:
        raise ControlledIlluminationRunPlanError(
            "incidence_angles_degrees must be "
            "a non-empty list."
        )

    validated_angles = [
        require_finite_plan_number(
            angle,
            "incidence_angle_degrees",
            non_negative=True,
        )
        for angle in angles
    ]

    validated_illuminance_levels = []

    if "constant_lux" in phase_names:
        if (
                not isinstance(illuminance_levels, list)
                or not illuminance_levels
        ):
            raise ControlledIlluminationRunPlanError(
                "constant_lux requires "
                "target_illuminance_levels_lux."
            )

        validated_illuminance_levels = [
            require_finite_plan_number(
                level,
                "target_illuminance_levels_lux",
                positive=True,
            )
            for level in illuminance_levels
        ]

    source_output_settings = matrix.get(
        "source_output_settings"
    )

    if "constant_source" in phase_names:
        if (
                not isinstance(source_output_settings, list)
                or not source_output_settings
        ):
            raise ControlledIlluminationRunPlanError(
                "constant_source requires non-empty "
                "experiment_matrix.source_output_settings."
            )

        for setting in source_output_settings:
            validate_source_output_setting(setting)

    conditions: list[RunCondition] = []

    for phase_name in phase_names:
        control_values = (
            validated_illuminance_levels
            if phase_name == "constant_lux"
            else source_output_settings
        )

        for (
                platform,
                architecture,
                algorithm,
                resolution,
                incidence_angle,
                control_value,
                trial_number,
        ) in product(
            platforms,
            architectures,
            algorithms,
            resolutions,
            validated_angles,
            control_values,
            range(1, trial_count + 1),
        ):
            conditions.append(
                RunCondition(
                    phase=phase_name,
                    platform=platform,
                    architecture=architecture,
                    algorithm=algorithm,
                    resolution=resolution,
                    trial_number=trial_number,
                    incidence_angle_degrees=(
                        incidence_angle
                    ),
                    target_illuminance_lux=(
                        control_value
                        if phase_name == "constant_lux"
                        else None
                    ),
                    source_output_setting=(
                        control_value
                        if phase_name == "constant_source"
                        else None
                    ),
                    target_fps=target_fps,
                    frame_deadline_ms=(
                        frame_deadline_ms
                    ),
                )
            )

    return tuple(conditions)

def build_run_plan(
    config: dict[str, Any],
    *,
    experiment_id: str,
    generated_at_utc: str,
) -> ControlledIlluminationRunPlan:
    require_safe_plan_identifier(
        experiment_id,
        "experiment_id",
    )

    try:
        validate_utc_timestamp(
            generated_at_utc,
            "generated_at_utc",
        )
    except ValueError as error:
        raise ControlledIlluminationRunPlanError(
            str(error)
        ) from error

    execution = config.get("execution")

    if not isinstance(execution, dict):
        raise ControlledIlluminationRunPlanError(
            "execution must be an object."
        )

    randomized = execution.get(
        "randomize_run_order"
    )
    randomization_seed = execution.get(
        "randomization_seed"
    )

    if not isinstance(randomized, bool):
        raise ControlledIlluminationRunPlanError(
            "execution.randomize_run_order must "
            "be boolean."
        )

    if (
        isinstance(randomization_seed, bool)
        or not isinstance(randomization_seed, int)
        or randomization_seed < 0
    ):
        raise ControlledIlluminationRunPlanError(
            "execution.randomization_seed must be "
            "a non-negative integer."
        )

    conditions = list(
        build_run_conditions(config)
    )

    if randomized:
        Random(randomization_seed).shuffle(
            conditions
        )

    run_number_width = max(
        4,
        len(str(len(conditions))),
    )

    planned_runs = tuple(
        PlannedRun(
            execution_order=execution_order,
            experiment_id=experiment_id,
            run_id=(
                f"{experiment_id}-run-"
                f"{execution_order:0{run_number_width}d}"
            ),
            phase=condition.phase,
            platform=condition.platform,
            architecture=condition.architecture,
            algorithm=condition.algorithm,
            resolution=condition.resolution,
            trial_number=condition.trial_number,
            incidence_angle_degrees=(
                condition.incidence_angle_degrees
            ),
            target_illuminance_lux=(
                condition.target_illuminance_lux
            ),
            source_output_setting=(
                condition.source_output_setting
            ),
            target_fps=condition.target_fps,
            frame_deadline_ms=(
                condition.frame_deadline_ms
            ),
        )
        for execution_order, condition in enumerate(
            conditions,
            start=1,
        )
    )

    return ControlledIlluminationRunPlan(
        schema_version=1,
        generated_at_utc=generated_at_utc,
        experiment_id=experiment_id,
        randomized=randomized,
        randomization_seed=randomization_seed,
        runs=planned_runs,
    )

def write_run_plan_manifests_atomic(
    plan: ControlledIlluminationRunPlan,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    if not isinstance(
        plan,
        ControlledIlluminationRunPlan,
    ):
        raise ControlledIlluminationRunPlanError(
            "plan must be ControlledIlluminationRunPlan."
        )

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_path / RUN_PLAN_JSON_FILE_NAME
    )
    csv_path = (
        output_path / RUN_PLAN_CSV_FILE_NAME
    )

    json_temporary_path = json_path.with_name(
        f".{json_path.name}.{uuid4().hex}.tmp"
    )
    csv_temporary_path = csv_path.with_name(
        f".{csv_path.name}.{uuid4().hex}.tmp"
    )

    csv_fieldnames = [
        "schema_version",
        "generated_at_utc",
        "randomized",
        "randomization_seed",
        "execution_order",
        "experiment_id",
        "run_id",
        "status",
        "phase",
        "platform",
        "architecture",
        "algorithm",
        "resolution_width",
        "resolution_height",
        "trial_number",
        "incidence_angle_degrees",
        "target_illuminance_lux",
        "source_output_setting",
        "target_fps",
        "frame_deadline_ms",
    ]

    try:
        with json_temporary_path.open(
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(
                plan.to_dict(),
                json_file,
                indent=2,
                ensure_ascii=False,
            )
            json_file.write("\n")
            json_file.flush()
            os.fsync(json_file.fileno())

        with csv_temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=csv_fieldnames,
            )
            writer.writeheader()

            for planned_run in plan.runs:
                writer.writerow(
                    {
                        "schema_version": (
                            plan.schema_version
                        ),
                        "generated_at_utc": (
                            plan.generated_at_utc
                        ),
                        "randomized": plan.randomized,
                        "randomization_seed": (
                            plan.randomization_seed
                        ),
                        "execution_order": (
                            planned_run.execution_order
                        ),
                        "experiment_id": (
                            planned_run.experiment_id
                        ),
                        "run_id": planned_run.run_id,
                        "status": planned_run.status,
                        "phase": planned_run.phase,
                        "platform": (
                            planned_run.platform
                        ),
                        "architecture": (
                            planned_run.architecture
                        ),
                        "algorithm": (
                            planned_run.algorithm
                        ),
                        "resolution_width": (
                            planned_run.resolution.width
                        ),
                        "resolution_height": (
                            planned_run.resolution.height
                        ),
                        "trial_number": (
                            planned_run.trial_number
                        ),
                        "incidence_angle_degrees": (
                            planned_run
                            .incidence_angle_degrees
                        ),
                        "target_illuminance_lux": (
                            planned_run
                            .target_illuminance_lux
                            if planned_run
                            .target_illuminance_lux
                            is not None
                            else ""
                        ),
                        "source_output_setting": (
                            planned_run
                            .source_output_setting
                            if planned_run
                            .source_output_setting
                            is not None
                            else ""
                        ),
                        "target_fps": (
                            planned_run.target_fps
                        ),
                        "frame_deadline_ms": (
                            planned_run
                            .frame_deadline_ms
                        ),
                    }
                )

            csv_file.flush()
            os.fsync(csv_file.fileno())

        os.replace(
            json_temporary_path,
            json_path,
        )
        os.replace(
            csv_temporary_path,
            csv_path,
        )
    finally:
        json_temporary_path.unlink(
            missing_ok=True,
        )
        csv_temporary_path.unlink(
            missing_ok=True,
        )

    return json_path, csv_path

def require_plan_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlledIlluminationRunPlanError(
            f"{field_name} must be an object."
        )

    return value


def require_exact_plan_fields(
    value: dict[str, Any],
    required_fields: set[str],
    field_name: str,
) -> None:
    actual_fields = set(value)

    missing_fields = required_fields - actual_fields
    unexpected_fields = actual_fields - required_fields

    if missing_fields:
        raise ControlledIlluminationRunPlanError(
            f"Missing {field_name} fields: "
            f"{sorted(missing_fields)}"
        )

    if unexpected_fields:
        raise ControlledIlluminationRunPlanError(
            f"Unexpected {field_name} fields: "
            f"{sorted(unexpected_fields)}"
        )


def planned_run_from_dict(
    value: Any,
) -> PlannedRun:
    run_data = require_plan_mapping(
        value,
        "planned run",
    )

    required_fields = {
        "execution_order",
        "experiment_id",
        "run_id",
        "phase",
        "platform",
        "architecture",
        "algorithm",
        "resolution",
        "trial_number",
        "incidence_angle_degrees",
        "target_illuminance_lux",
        "source_output_setting",
        "target_fps",
        "frame_deadline_ms",
        "status",
    }

    require_exact_plan_fields(
        run_data,
        required_fields,
        "planned run",
    )

    resolution_data = require_plan_mapping(
        run_data["resolution"],
        "planned run resolution",
    )

    require_exact_plan_fields(
        resolution_data,
        {
            "width",
            "height",
        },
        "planned run resolution",
    )

    resolution = ResolutionMetadata(
        width=resolution_data["width"],
        height=resolution_data["height"],
    )

    return PlannedRun(
        execution_order=run_data[
            "execution_order"
        ],
        experiment_id=run_data[
            "experiment_id"
        ],
        run_id=run_data["run_id"],
        phase=run_data["phase"],
        platform=run_data["platform"],
        architecture=run_data["architecture"],
        algorithm=run_data["algorithm"],
        resolution=resolution,
        trial_number=run_data["trial_number"],
        incidence_angle_degrees=run_data[
            "incidence_angle_degrees"
        ],
        target_illuminance_lux=run_data[
            "target_illuminance_lux"
        ],
        source_output_setting=run_data[
            "source_output_setting"
        ],
        target_fps=run_data["target_fps"],
        frame_deadline_ms=run_data[
            "frame_deadline_ms"
        ],
        status=run_data["status"],
    )


def run_plan_from_dict(
    value: Any,
) -> ControlledIlluminationRunPlan:
    plan_data = require_plan_mapping(
        value,
        "run plan",
    )

    required_fields = {
        "schema_version",
        "generated_at_utc",
        "experiment_id",
        "randomized",
        "randomization_seed",
        "run_count",
        "runs",
    }

    require_exact_plan_fields(
        plan_data,
        required_fields,
        "run plan",
    )

    runs_data = plan_data["runs"]

    if not isinstance(runs_data, list) or not runs_data:
        raise ControlledIlluminationRunPlanError(
            "run plan runs must be a non-empty list."
        )

    plan = ControlledIlluminationRunPlan(
        schema_version=plan_data[
            "schema_version"
        ],
        generated_at_utc=plan_data[
            "generated_at_utc"
        ],
        experiment_id=plan_data[
            "experiment_id"
        ],
        randomized=plan_data["randomized"],
        randomization_seed=plan_data[
            "randomization_seed"
        ],
        runs=tuple(
            planned_run_from_dict(run_data)
            for run_data in runs_data
        ),
    )

    stored_run_count = plan_data["run_count"]

    if (
        isinstance(stored_run_count, bool)
        or not isinstance(stored_run_count, int)
        or stored_run_count != plan.run_count
    ):
        raise ControlledIlluminationRunPlanError(
            "Stored run_count does not match runs."
        )

    return plan


def load_run_plan_manifest(
    input_path: str | Path,
) -> ControlledIlluminationRunPlan:
    manifest_path = Path(input_path)

    if not manifest_path.is_file():
        raise ControlledIlluminationRunPlanError(
            "Run-plan manifest was not found: "
            f"{manifest_path}"
        )

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as manifest_file:
            manifest_data = json.load(
                manifest_file
            )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise ControlledIlluminationRunPlanError(
            "Run-plan manifest could not be loaded: "
            f"{manifest_path}: {error}"
        ) from error

    return run_plan_from_dict(
        manifest_data
    )
