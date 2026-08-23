from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from benchmarks.experiments.controlled_illumination_metadata import (
    validate_utc_timestamp,
    ResolutionMetadata,
)

from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlan,
    PlannedRun,
)


RUN_PROGRESS_SCHEMA_VERSION = 1
RUN_PROGRESS_FILE_NAME = "run_progress.json"


class RunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


ALLOWED_STATUS_TRANSITIONS = {
    RunStatus.PLANNED: frozenset(
        {
            RunStatus.RUNNING,
            RunStatus.SKIPPED,
        }
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(
        {
            RunStatus.PLANNED,
        }
    ),
    RunStatus.SKIPPED: frozenset(
        {
            RunStatus.PLANNED,
        }
    ),
}


class ControlledIlluminationRunStateError(
    ValueError
):
    """Raised when experiment run state is invalid."""


def require_non_empty_state_string(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlledIlluminationRunStateError(
            f"{field_name} must be a non-empty string."
        )

    return value


def require_positive_state_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ControlledIlluminationRunStateError(
            f"{field_name} must be a positive integer."
        )

    return value


def require_non_negative_state_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ControlledIlluminationRunStateError(
            f"{field_name} must be a "
            "non-negative integer."
        )

    return value


def normalize_run_status(
    value: Any,
) -> RunStatus:
    if isinstance(value, RunStatus):
        return value

    if not isinstance(value, str):
        raise ControlledIlluminationRunStateError(
            "status must be a supported string."
        )

    try:
        return RunStatus(value)
    except ValueError as error:
        raise ControlledIlluminationRunStateError(
            f"Unsupported run status: {value}"
        ) from error


def validate_optional_timestamp(
    value: Any,
    field_name: str,
) -> None:
    if value is None:
        return

    try:
        validate_utc_timestamp(
            value,
            field_name,
        )
    except ValueError as error:
        raise ControlledIlluminationRunStateError(
            str(error)
        ) from error

def parse_validated_state_timestamp(
    value: str,
) -> datetime:
    normalized_value = (
        f"{value[:-1]}+00:00"
        if value.endswith("Z")
        else value
    )

    return datetime.fromisoformat(
        normalized_value
    )

def validate_optional_reason(
    value: Any,
    field_name: str,
) -> None:
    if value is None:
        return

    require_non_empty_state_string(
        value,
        field_name,
    )


@dataclass(frozen=True)
class RunState:
    run_id: str
    execution_order: int
    status: RunStatus = RunStatus.PLANNED
    attempt_count: int = 0
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    failure_reason: str | None = None
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        require_non_empty_state_string(
            self.run_id,
            "run_id",
        )
        require_positive_state_integer(
            self.execution_order,
            "execution_order",
        )
        require_non_negative_state_integer(
            self.attempt_count,
            "attempt_count",
        )

        normalized_status = normalize_run_status(
            self.status
        )

        object.__setattr__(
            self,
            "status",
            normalized_status,
        )

        validate_optional_timestamp(
            self.started_at_utc,
            "started_at_utc",
        )
        validate_optional_timestamp(
            self.finished_at_utc,
            "finished_at_utc",
        )

        if (
                self.started_at_utc is not None
                and self.finished_at_utc is not None
                and parse_validated_state_timestamp(
            self.finished_at_utc
        )
                < parse_validated_state_timestamp(
            self.started_at_utc
        )
        ):
            raise ControlledIlluminationRunStateError(
                "finished_at_utc must not be earlier "
                "than started_at_utc."
            )

        validate_optional_reason(
            self.failure_reason,
            "failure_reason",
        )
        validate_optional_reason(
            self.skip_reason,
            "skip_reason",
        )

        self.validate_status_fields()

    def validate_status_fields(self) -> None:
        if self.status == RunStatus.PLANNED:
            if any(
                value is not None
                for value in (
                    self.started_at_utc,
                    self.finished_at_utc,
                    self.failure_reason,
                    self.skip_reason,
                )
            ):
                raise ControlledIlluminationRunStateError(
                    "A planned run must not contain "
                    "execution timestamps or reasons."
                )

        if self.status == RunStatus.RUNNING:
            if self.started_at_utc is None:
                raise ControlledIlluminationRunStateError(
                    "A running run requires started_at_utc."
                )

            if (
                self.attempt_count <= 0
                or self.finished_at_utc is not None
                or self.failure_reason is not None
                or self.skip_reason is not None
            ):
                raise ControlledIlluminationRunStateError(
                    "Running run fields are inconsistent."
                )

        if self.status == RunStatus.COMPLETED:
            if (
                self.attempt_count <= 0
                or self.started_at_utc is None
                or self.finished_at_utc is None
                or self.failure_reason is not None
                or self.skip_reason is not None
            ):
                raise ControlledIlluminationRunStateError(
                    "Completed run fields are inconsistent."
                )

        if self.status == RunStatus.FAILED:
            if (
                self.attempt_count <= 0
                or self.started_at_utc is None
                or self.finished_at_utc is None
                or self.failure_reason is None
                or self.skip_reason is not None
            ):
                raise ControlledIlluminationRunStateError(
                    "Failed run fields are inconsistent."
                )

        if self.status == RunStatus.SKIPPED:
            if (
                self.started_at_utc is not None
                or self.finished_at_utc is None
                or self.failure_reason is not None
                or self.skip_reason is None
            ):
                raise ControlledIlluminationRunStateError(
                    "Skipped run fields are inconsistent."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "execution_order": self.execution_order,
            "status": self.status.value,
            "attempt_count": self.attempt_count,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "failure_reason": self.failure_reason,
            "skip_reason": self.skip_reason,
        }

def validate_required_state_timestamp(
        value: Any,
        field_name: str,
) -> str:
    timestamp = require_non_empty_state_string(
        value,
        field_name,
    )

    validate_optional_timestamp(
        timestamp,
        field_name,
    )

    return timestamp

def transition_run_state(
        state: RunState,
        target_status: RunStatus | str,
        transitioned_at_utc: str,
        *,
        reason: str | None = None,
) -> RunState:
    if not isinstance(state, RunState):
        raise ControlledIlluminationRunStateError(
            "state must be RunState."
        )

    normalized_target_status = normalize_run_status(
        target_status
    )
    transition_timestamp = (
        validate_required_state_timestamp(
            transitioned_at_utc,
            "transitioned_at_utc",
        )
    )

    allowed_targets = ALLOWED_STATUS_TRANSITIONS[
        state.status
    ]

    if normalized_target_status not in allowed_targets:
        raise ControlledIlluminationRunStateError(
            "Invalid run-state transition: "
            f"{state.status.value} -> "
            f"{normalized_target_status.value}"
        )

    reason_required = normalized_target_status in {
        RunStatus.FAILED,
        RunStatus.SKIPPED,
    }

    if reason_required:
        require_non_empty_state_string(
            reason,
            "reason",
        )
    elif reason is not None:
        raise ControlledIlluminationRunStateError(
            "A reason is only allowed for failed "
            "or skipped transitions."
        )

    if normalized_target_status == RunStatus.RUNNING:
        return replace(
            state,
            status=RunStatus.RUNNING,
            attempt_count=state.attempt_count + 1,
            started_at_utc=transition_timestamp,
            finished_at_utc=None,
            failure_reason=None,
            skip_reason=None,
        )

    if normalized_target_status == RunStatus.COMPLETED:
        return replace(
            state,
            status=RunStatus.COMPLETED,
            finished_at_utc=transition_timestamp,
            failure_reason=None,
            skip_reason=None,
        )

    if normalized_target_status == RunStatus.FAILED:
        return replace(
            state,
            status=RunStatus.FAILED,
            finished_at_utc=transition_timestamp,
            failure_reason=reason,
            skip_reason=None,
        )

    if normalized_target_status == RunStatus.SKIPPED:
        return replace(
            state,
            status=RunStatus.SKIPPED,
            started_at_utc=None,
            finished_at_utc=transition_timestamp,
            failure_reason=None,
            skip_reason=reason,
        )

    if normalized_target_status == RunStatus.PLANNED:
        return replace(
            state,
            status=RunStatus.PLANNED,
            started_at_utc=None,
            finished_at_utc=None,
            failure_reason=None,
            skip_reason=None,
        )

    raise ControlledIlluminationRunStateError(
        "Unsupported run-state transition."
    )

def calculate_run_plan_sha256(
    plan: ControlledIlluminationRunPlan,
) -> str:
    if not isinstance(
        plan,
        ControlledIlluminationRunPlan,
    ):
        raise ControlledIlluminationRunStateError(
            "plan must be ControlledIlluminationRunPlan."
        )

    canonical_plan = json.dumps(
        plan.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        canonical_plan
    ).hexdigest()


def validate_run_plan_sha256(
    value: Any,
) -> str:
    hash_value = require_non_empty_state_string(
        value,
        "run_plan_sha256",
    )

    if (
        len(hash_value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in hash_value
        )
    ):
        raise ControlledIlluminationRunStateError(
            "run_plan_sha256 must be a lowercase "
            "SHA-256 value."
        )

    return hash_value


@dataclass(frozen=True)
class ControlledIlluminationProgress:
    schema_version: int
    experiment_id: str
    run_plan_sha256: str
    created_at_utc: str
    updated_at_utc: str
    runs: tuple[RunState, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version
            != RUN_PROGRESS_SCHEMA_VERSION
        ):
            raise ControlledIlluminationRunStateError(
                "schema_version must be 1."
            )

        require_non_empty_state_string(
            self.experiment_id,
            "experiment_id",
        )
        validate_run_plan_sha256(
            self.run_plan_sha256
        )
        validate_required_state_timestamp(
            self.created_at_utc,
            "created_at_utc",
        )
        validate_required_state_timestamp(
            self.updated_at_utc,
            "updated_at_utc",
        )

        if parse_validated_state_timestamp(
                self.updated_at_utc
        ) < parse_validated_state_timestamp(
            self.created_at_utc
        ):
            raise ControlledIlluminationRunStateError(
                "updated_at_utc must not be earlier "
                "than created_at_utc."
            )

        if not isinstance(self.runs, tuple) or not self.runs:
            raise ControlledIlluminationRunStateError(
                "runs must be a non-empty tuple."
            )

        if not all(
            isinstance(run_state, RunState)
            for run_state in self.runs
        ):
            raise ControlledIlluminationRunStateError(
                "Every runs item must be RunState."
            )

        expected_orders = list(
            range(1, len(self.runs) + 1)
        )
        actual_orders = [
            run_state.execution_order
            for run_state in self.runs
        ]

        if actual_orders != expected_orders:
            raise ControlledIlluminationRunStateError(
                "execution_order values must be "
                "sequential and start at 1."
            )

        run_ids = [
            run_state.run_id
            for run_state in self.runs
        ]

        if len(run_ids) != len(set(run_ids)):
            raise ControlledIlluminationRunStateError(
                "Run IDs must be unique."
            )

        running_count = sum(
            run_state.status == RunStatus.RUNNING
            for run_state in self.runs
        )

        if running_count > 1:
            raise ControlledIlluminationRunStateError(
                "Only one run may be running."
            )

    @property
    def run_count(self) -> int:
        return len(self.runs)

    @property
    def status_counts(self) -> dict[str, int]:
        return {
            status.value: sum(
                run_state.status == status
                for run_state in self.runs
            )
            for status in RunStatus
        }

    @property
    def next_planned_run(
        self,
    ) -> RunState | None:
        return next(
            (
                run_state
                for run_state in self.runs
                if run_state.status
                == RunStatus.PLANNED
            ),
            None,
        )

    def get_run_state(
        self,
        run_id: str,
    ) -> RunState:
        require_non_empty_state_string(
            run_id,
            "run_id",
        )

        for run_state in self.runs:
            if run_state.run_id == run_id:
                return run_state

        raise ControlledIlluminationRunStateError(
            f"Unknown run ID: {run_id}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "run_plan_sha256": (
                self.run_plan_sha256
            ),
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "run_count": self.run_count,
            "status_counts": self.status_counts,
            "runs": [
                run_state.to_dict()
                for run_state in self.runs
            ],
        }


def initialize_run_progress(
    plan: ControlledIlluminationRunPlan,
    *,
    created_at_utc: str,
) -> ControlledIlluminationProgress:
    if not isinstance(
        plan,
        ControlledIlluminationRunPlan,
    ):
        raise ControlledIlluminationRunStateError(
            "plan must be ControlledIlluminationRunPlan."
        )

    creation_timestamp = (
        validate_required_state_timestamp(
            created_at_utc,
            "created_at_utc",
        )
    )

    run_states = tuple(
        RunState(
            run_id=planned_run.run_id,
            execution_order=(
                planned_run.execution_order
            ),
        )
        for planned_run in plan.runs
    )

    return ControlledIlluminationProgress(
        schema_version=RUN_PROGRESS_SCHEMA_VERSION,
        experiment_id=plan.experiment_id,
        run_plan_sha256=(
            calculate_run_plan_sha256(plan)
        ),
        created_at_utc=creation_timestamp,
        updated_at_utc=creation_timestamp,
        runs=run_states,
    )

def transition_progress_run(
    progress: ControlledIlluminationProgress,
    run_id: str,
    target_status: RunStatus | str,
    transitioned_at_utc: str,
    *,
    reason: str | None = None,
) -> ControlledIlluminationProgress:
    if not isinstance(
        progress,
        ControlledIlluminationProgress,
    ):
        raise ControlledIlluminationRunStateError(
            "progress must be "
            "ControlledIlluminationProgress."
        )

    current_state = progress.get_run_state(
        run_id
    )

    normalized_target_status = normalize_run_status(
        target_status
    )

    transition_timestamp = (
        validate_required_state_timestamp(
            transitioned_at_utc,
            "transitioned_at_utc",
        )
    )

    if parse_validated_state_timestamp(
        transition_timestamp
    ) < parse_validated_state_timestamp(
        progress.updated_at_utc
    ):
        raise ControlledIlluminationRunStateError(
            "transitioned_at_utc must not be earlier "
            "than progress.updated_at_utc."
        )

    if normalized_target_status == RunStatus.RUNNING:
        running_state = next(
            (
                run_state
                for run_state in progress.runs
                if (
                    run_state.status
                    == RunStatus.RUNNING
                    and run_state.run_id != run_id
                )
            ),
            None,
        )

        if running_state is not None:
            raise ControlledIlluminationRunStateError(
                "Another run is already running: "
                f"{running_state.run_id}"
            )

    updated_state = transition_run_state(
        current_state,
        normalized_target_status,
        transition_timestamp,
        reason=reason,
    )

    updated_runs = tuple(
        updated_state
        if run_state.run_id == run_id
        else run_state
        for run_state in progress.runs
    )

    return replace(
        progress,
        updated_at_utc=transition_timestamp,
        runs=updated_runs,
    )

def require_state_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlledIlluminationRunStateError(
            f"{field_name} must be an object."
        )

    return value


def require_exact_state_fields(
    value: dict[str, Any],
    required_fields: set[str],
    field_name: str,
) -> None:
    actual_fields = set(value)

    missing_fields = required_fields - actual_fields
    unexpected_fields = actual_fields - required_fields

    if missing_fields:
        raise ControlledIlluminationRunStateError(
            f"Missing {field_name} fields: "
            f"{sorted(missing_fields)}"
        )

    if unexpected_fields:
        raise ControlledIlluminationRunStateError(
            f"Unexpected {field_name} fields: "
            f"{sorted(unexpected_fields)}"
        )


def run_state_from_dict(
    value: Any,
) -> RunState:
    state_data = require_state_mapping(
        value,
        "run state",
    )

    required_fields = {
        "run_id",
        "execution_order",
        "status",
        "attempt_count",
        "started_at_utc",
        "finished_at_utc",
        "failure_reason",
        "skip_reason",
    }

    require_exact_state_fields(
        state_data,
        required_fields,
        "run state",
    )

    return RunState(
        run_id=state_data["run_id"],
        execution_order=(
            state_data["execution_order"]
        ),
        status=state_data["status"],
        attempt_count=state_data["attempt_count"],
        started_at_utc=(
            state_data["started_at_utc"]
        ),
        finished_at_utc=(
            state_data["finished_at_utc"]
        ),
        failure_reason=(
            state_data["failure_reason"]
        ),
        skip_reason=state_data["skip_reason"],
    )


def progress_from_dict(
    value: Any,
) -> ControlledIlluminationProgress:
    progress_data = require_state_mapping(
        value,
        "progress",
    )

    required_fields = {
        "schema_version",
        "experiment_id",
        "run_plan_sha256",
        "created_at_utc",
        "updated_at_utc",
        "run_count",
        "status_counts",
        "runs",
    }

    require_exact_state_fields(
        progress_data,
        required_fields,
        "progress",
    )

    runs_data = progress_data["runs"]

    if not isinstance(runs_data, list) or not runs_data:
        raise ControlledIlluminationRunStateError(
            "progress.runs must be a non-empty list."
        )

    progress = ControlledIlluminationProgress(
        schema_version=progress_data[
            "schema_version"
        ],
        experiment_id=progress_data[
            "experiment_id"
        ],
        run_plan_sha256=progress_data[
            "run_plan_sha256"
        ],
        created_at_utc=progress_data[
            "created_at_utc"
        ],
        updated_at_utc=progress_data[
            "updated_at_utc"
        ],
        runs=tuple(
            run_state_from_dict(run_data)
            for run_data in runs_data
        ),
    )

    stored_run_count = progress_data["run_count"]

    if (
        isinstance(stored_run_count, bool)
        or not isinstance(stored_run_count, int)
        or stored_run_count != progress.run_count
    ):
        raise ControlledIlluminationRunStateError(
            "Stored run_count does not match runs."
        )

    stored_status_counts = progress_data[
        "status_counts"
    ]

    if not isinstance(stored_status_counts, dict):
        raise ControlledIlluminationRunStateError(
            "status_counts must be an object."
        )

    if stored_status_counts != progress.status_counts:
        raise ControlledIlluminationRunStateError(
            "Stored status_counts do not match runs."
        )

    return progress


def validate_progress_matches_plan(
    progress: ControlledIlluminationProgress,
    plan: ControlledIlluminationRunPlan,
) -> None:
    if not isinstance(
        progress,
        ControlledIlluminationProgress,
    ):
        raise ControlledIlluminationRunStateError(
            "progress must be "
            "ControlledIlluminationProgress."
        )

    if not isinstance(
        plan,
        ControlledIlluminationRunPlan,
    ):
        raise ControlledIlluminationRunStateError(
            "plan must be ControlledIlluminationRunPlan."
        )

    if progress.experiment_id != plan.experiment_id:
        raise ControlledIlluminationRunStateError(
            "Progress experiment_id does not "
            "match the run plan."
        )

    expected_hash = calculate_run_plan_sha256(
        plan
    )

    if progress.run_plan_sha256 != expected_hash:
        raise ControlledIlluminationRunStateError(
            "Progress was created from a different "
            "run plan."
        )


def save_run_progress_atomic(
    progress: ControlledIlluminationProgress,
    output_path: str | Path,
) -> Path:
    if not isinstance(
        progress,
        ControlledIlluminationProgress,
    ):
        raise ControlledIlluminationRunStateError(
            "progress must be "
            "ControlledIlluminationProgress."
        )

    progress_path = Path(output_path)
    progress_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = progress_path.with_name(
        f".{progress_path.name}."
        f"{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as progress_file:
            json.dump(
                progress.to_dict(),
                progress_file,
                indent=2,
                ensure_ascii=False,
            )
            progress_file.write("\n")
            progress_file.flush()
            os.fsync(progress_file.fileno())

        os.replace(
            temporary_path,
            progress_path,
        )
    finally:
        temporary_path.unlink(
            missing_ok=True,
        )

    return progress_path


def load_run_progress(
    progress_path: str | Path,
    *,
    plan: ControlledIlluminationRunPlan | None = None,
) -> ControlledIlluminationProgress:
    input_path = Path(progress_path)

    if not input_path.is_file():
        raise ControlledIlluminationRunStateError(
            f"Progress file was not found: {input_path}"
        )

    try:
        with input_path.open(
            "r",
            encoding="utf-8",
        ) as progress_file:
            progress_data = json.load(
                progress_file
            )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise ControlledIlluminationRunStateError(
            f"Progress file could not be loaded: "
            f"{input_path}: {error}"
        ) from error

    progress = progress_from_dict(
        progress_data
    )

    if plan is not None:
        validate_progress_matches_plan(
            progress,
            plan,
        )

    return progress


def load_or_initialize_run_progress(
    plan: ControlledIlluminationRunPlan,
    progress_path: str | Path,
    *,
    created_at_utc: str,
) -> ControlledIlluminationProgress:
    output_path = Path(progress_path)

    if output_path.exists():
        return load_run_progress(
            output_path,
            plan=plan,
        )

    progress = initialize_run_progress(
        plan,
        created_at_utc=created_at_utc,
    )

    save_run_progress_atomic(
        progress,
        output_path,
    )

    return progress