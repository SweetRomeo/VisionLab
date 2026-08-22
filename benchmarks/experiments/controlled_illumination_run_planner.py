from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
    validate_utc_timestamp,
)


PLANNED_RUN_STATUS = "planned"

SUPPORTED_PHASES = {
    "constant_lux",
    "constant_source",
}


class ControlledIlluminationRunPlanError(ValueError):
    """Raised when an experiment run plan is invalid."""


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

        for field_name in (
            "experiment_id",
            "run_id",
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

        require_non_empty_plan_string(
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