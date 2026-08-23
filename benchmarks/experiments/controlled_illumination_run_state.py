from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from dataclasses import dataclass, replace

from benchmarks.experiments.controlled_illumination_metadata import (
    validate_utc_timestamp,
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