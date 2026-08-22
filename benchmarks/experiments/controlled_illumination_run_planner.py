from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)


PLANNED_RUN_STATUS = "planned"


class ControlledIlluminationRunPlanError(ValueError):
    """Raised when an experiment run plan is invalid."""


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
    source_output_setting: str | float | None
    target_fps: float
    frame_deadline_ms: float
    status: str = PLANNED_RUN_STATUS

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