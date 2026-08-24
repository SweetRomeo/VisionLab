from __future__ import annotations

from collections.abc import Sequence
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any
from uuid import uuid4

from benchmarks.experiments.controlled_illumination_metadata import (
    validate_utc_timestamp,
)
from benchmarks.experiments.controlled_illumination_runner_context import (
    ControlledIlluminationRunnerContext,
)


FRAME_RESULTS_FILE_NAME = (
    "realtime_frame_results.csv"
)
EXECUTION_SUMMARY_FILE_NAME = (
    "execution_summary.json"
)

SUPPORTED_FRAME_OUTCOMES = frozenset(
    {
        "processed",
        "dropped",
        "skipped",
    }
)

FRAME_RESULT_FIELDS = [
    "frame_index",
    "outcome",
    "captured_at_ms",
    "processing_started_at_ms",
    "processing_finished_at_ms",
    "processing_time_ms",
    "end_to_end_latency_ms",
    "frame_deadline_ms",
    "deadline_met",
    "reason",
]


class ControlledIlluminationArtifactError(
    ValueError
):
    """Raised when run artifacts are invalid."""


def require_finite_artifact_number(
    value: Any,
    field_name: str,
    *,
    non_negative: bool = False,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ControlledIlluminationArtifactError(
            f"{field_name} must be a finite number."
        )

    numeric_value = float(value)

    if non_negative and numeric_value < 0.0:
        raise ControlledIlluminationArtifactError(
            f"{field_name} must be non-negative."
        )

    if positive and numeric_value <= 0.0:
        raise ControlledIlluminationArtifactError(
            f"{field_name} must be positive."
        )

    return numeric_value


def validate_optional_artifact_number(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None

    return require_finite_artifact_number(
        value,
        field_name,
        non_negative=True,
    )


def validate_optional_reason(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise ControlledIlluminationArtifactError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


@dataclass(frozen=True)
class ControlledIlluminationFrameRecord:
    frame_index: int
    outcome: str
    captured_at_ms: float
    processing_started_at_ms: float | None
    processing_finished_at_ms: float | None
    processing_time_ms: float | None
    end_to_end_latency_ms: float | None
    frame_deadline_ms: float
    deadline_met: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ControlledIlluminationArtifactError(
                "frame_index must be a "
                "non-negative integer."
            )

        if self.outcome not in (
            SUPPORTED_FRAME_OUTCOMES
        ):
            raise ControlledIlluminationArtifactError(
                f"Unsupported frame outcome: {self.outcome}"
            )

        captured_at_ms = (
            require_finite_artifact_number(
                self.captured_at_ms,
                "captured_at_ms",
                non_negative=True,
            )
        )
        frame_deadline_ms = (
            require_finite_artifact_number(
                self.frame_deadline_ms,
                "frame_deadline_ms",
                positive=True,
            )
        )

        processing_started_at_ms = (
            validate_optional_artifact_number(
                self.processing_started_at_ms,
                "processing_started_at_ms",
            )
        )
        processing_finished_at_ms = (
            validate_optional_artifact_number(
                self.processing_finished_at_ms,
                "processing_finished_at_ms",
            )
        )
        processing_time_ms = (
            validate_optional_artifact_number(
                self.processing_time_ms,
                "processing_time_ms",
            )
        )
        end_to_end_latency_ms = (
            validate_optional_artifact_number(
                self.end_to_end_latency_ms,
                "end_to_end_latency_ms",
            )
        )
        reason = validate_optional_reason(
            self.reason,
            "reason",
        )

        if not isinstance(self.deadline_met, bool):
            raise ControlledIlluminationArtifactError(
                "deadline_met must be boolean."
            )

        if self.outcome == "processed":
            required_timings = {
                "processing_started_at_ms": (
                    processing_started_at_ms
                ),
                "processing_finished_at_ms": (
                    processing_finished_at_ms
                ),
                "processing_time_ms": (
                    processing_time_ms
                ),
                "end_to_end_latency_ms": (
                    end_to_end_latency_ms
                ),
            }

            missing_timings = [
                field_name
                for field_name, field_value
                in required_timings.items()
                if field_value is None
            ]

            if missing_timings:
                raise ControlledIlluminationArtifactError(
                    "Processed frames require timing "
                    f"values: {missing_timings}"
                )

            assert processing_started_at_ms is not None
            assert processing_finished_at_ms is not None
            assert processing_time_ms is not None
            assert end_to_end_latency_ms is not None

            if (
                processing_started_at_ms
                < captured_at_ms
            ):
                raise ControlledIlluminationArtifactError(
                    "Processing cannot start before "
                    "frame capture."
                )

            if (
                processing_finished_at_ms
                < processing_started_at_ms
            ):
                raise ControlledIlluminationArtifactError(
                    "Processing cannot finish before "
                    "it starts."
                )

            measured_processing_time = (
                processing_finished_at_ms
                - processing_started_at_ms
            )

            if not math.isclose(
                processing_time_ms,
                measured_processing_time,
                rel_tol=1e-6,
                abs_tol=1e-3,
            ):
                raise ControlledIlluminationArtifactError(
                    "processing_time_ms does not match "
                    "the processing timestamps."
                )

            if (
                end_to_end_latency_ms
                + 1e-3
                < processing_time_ms
            ):
                raise ControlledIlluminationArtifactError(
                    "end_to_end_latency_ms cannot be "
                    "shorter than processing_time_ms."
                )

            expected_deadline_met = (
                end_to_end_latency_ms
                <= frame_deadline_ms
            )

            if self.deadline_met != expected_deadline_met:
                raise ControlledIlluminationArtifactError(
                    "deadline_met does not match the "
                    "recorded latency and deadline."
                )

            if reason is not None:
                raise ControlledIlluminationArtifactError(
                    "Processed frames must not define "
                    "a reason."
                )
        else:
            if any(
                value is not None
                for value in (
                    processing_started_at_ms,
                    processing_finished_at_ms,
                    processing_time_ms,
                    end_to_end_latency_ms,
                )
            ):
                raise ControlledIlluminationArtifactError(
                    "Dropped or skipped frames must not "
                    "contain processing timings."
                )

            if self.deadline_met:
                raise ControlledIlluminationArtifactError(
                    "Dropped or skipped frames cannot "
                    "meet the deadline."
                )

            if reason is None:
                raise ControlledIlluminationArtifactError(
                    "Dropped or skipped frames require "
                    "a reason."
                )

        object.__setattr__(
            self,
            "captured_at_ms",
            captured_at_ms,
        )
        object.__setattr__(
            self,
            "processing_started_at_ms",
            processing_started_at_ms,
        )
        object.__setattr__(
            self,
            "processing_finished_at_ms",
            processing_finished_at_ms,
        )
        object.__setattr__(
            self,
            "processing_time_ms",
            processing_time_ms,
        )
        object.__setattr__(
            self,
            "end_to_end_latency_ms",
            end_to_end_latency_ms,
        )
        object.__setattr__(
            self,
            "frame_deadline_ms",
            frame_deadline_ms,
        )
        object.__setattr__(
            self,
            "reason",
            reason,
        )

    def to_csv_row(self) -> dict[str, object]:
        def serialize_optional_number(
            value: float | None,
        ) -> str:
            return (
                ""
                if value is None
                else f"{value:.6f}"
            )

        return {
            "frame_index": self.frame_index,
            "outcome": self.outcome,
            "captured_at_ms": (
                f"{self.captured_at_ms:.6f}"
            ),
            "processing_started_at_ms": (
                serialize_optional_number(
                    self.processing_started_at_ms
                )
            ),
            "processing_finished_at_ms": (
                serialize_optional_number(
                    self.processing_finished_at_ms
                )
            ),
            "processing_time_ms": (
                serialize_optional_number(
                    self.processing_time_ms
                )
            ),
            "end_to_end_latency_ms": (
                serialize_optional_number(
                    self.end_to_end_latency_ms
                )
            ),
            "frame_deadline_ms": (
                f"{self.frame_deadline_ms:.6f}"
            ),
            "deadline_met": (
                "true"
                if self.deadline_met
                else "false"
            ),
            "reason": self.reason or "",
        }


def parse_artifact_timestamp(
    value: str,
    field_name: str,
) -> datetime:
    try:
        validate_utc_timestamp(
            value,
            field_name,
        )
    except ValueError as error:
        raise ControlledIlluminationArtifactError(
            str(error)
        ) from error

    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def validate_frame_records(
    records: Sequence[
        ControlledIlluminationFrameRecord
    ],
) -> tuple[ControlledIlluminationFrameRecord, ...]:
    if (
        isinstance(records, (str, bytes))
        or not isinstance(records, Sequence)
        or not records
    ):
        raise ControlledIlluminationArtifactError(
            "records must be a non-empty sequence."
        )

    normalized_records = tuple(records)

    if not all(
        isinstance(
            record,
            ControlledIlluminationFrameRecord,
        )
        for record in normalized_records
    ):
        raise ControlledIlluminationArtifactError(
            "Every record must be "
            "ControlledIlluminationFrameRecord."
        )

    actual_indices = [
        record.frame_index
        for record in normalized_records
    ]
    expected_indices = list(
        range(len(normalized_records))
    )

    if actual_indices != expected_indices:
        raise ControlledIlluminationArtifactError(
            "Frame indices must be sequential "
            "and start at zero."
        )

    return normalized_records


def calculate_file_sha256(
    file_path: Path,
) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as input_file:
        for chunk in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json_atomic(
    value: dict[str, Any],
    output_path: Path,
) -> Path:
    temporary_path = output_path.with_name(
        f".{output_path.name}."
        f"{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                value,
                output_file,
                indent=2,
                ensure_ascii=False,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(
            temporary_path,
            output_path,
        )
    finally:
        temporary_path.unlink(
            missing_ok=True
        )

    return output_path


def write_frame_results_atomic(
    records: tuple[
        ControlledIlluminationFrameRecord,
        ...,
    ],
    output_path: Path,
) -> Path:
    temporary_path = output_path.with_name(
        f".{output_path.name}."
        f"{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=FRAME_RESULT_FIELDS,
            )
            writer.writeheader()

            for record in records:
                writer.writerow(
                    record.to_csv_row()
                )

            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(
            temporary_path,
            output_path,
        )
    finally:
        temporary_path.unlink(
            missing_ok=True
        )

    return output_path


def write_completed_run_artifacts_atomic(
    context: ControlledIlluminationRunnerContext,
    records: Sequence[
        ControlledIlluminationFrameRecord
    ],
    *,
    started_at_utc: str,
    finished_at_utc: str,
    warmup_frame_count: int,
) -> tuple[Path, Path]:
    if not isinstance(
        context,
        ControlledIlluminationRunnerContext,
    ):
        raise ControlledIlluminationArtifactError(
            "context must be "
            "ControlledIlluminationRunnerContext."
        )

    normalized_records = validate_frame_records(
        records
    )

    if (
        isinstance(warmup_frame_count, bool)
        or not isinstance(warmup_frame_count, int)
        or warmup_frame_count < 0
    ):
        raise ControlledIlluminationArtifactError(
            "warmup_frame_count must be a "
            "non-negative integer."
        )

    started_at = parse_artifact_timestamp(
        started_at_utc,
        "started_at_utc",
    )
    finished_at = parse_artifact_timestamp(
        finished_at_utc,
        "finished_at_utc",
    )

    if finished_at < started_at:
        raise ControlledIlluminationArtifactError(
            "finished_at_utc cannot be earlier "
            "than started_at_utc."
        )

    output_directory = context.output_directory
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_results_path = (
        output_directory
        / FRAME_RESULTS_FILE_NAME
    )
    summary_path = (
        output_directory
        / EXECUTION_SUMMARY_FILE_NAME
    )

    if summary_path.exists():
        raise ControlledIlluminationArtifactError(
            "Completed run artifacts already exist: "
            f"{summary_path}"
        )

    write_frame_results_atomic(
        normalized_records,
        frame_results_path,
    )

    processed_records = [
        record
        for record in normalized_records
        if record.outcome == "processed"
    ]
    dropped_count = sum(
        record.outcome == "dropped"
        for record in normalized_records
    )
    skipped_count = sum(
        record.outcome == "skipped"
        for record in normalized_records
    )
    deadline_met_count = sum(
        record.outcome == "processed"
        and record.deadline_met
        for record in normalized_records
    )
    deadline_miss_count = sum(
        record.outcome == "processed"
        and not record.deadline_met
        for record in normalized_records
    )

    processing_times = [
        record.processing_time_ms
        for record in processed_records
        if record.processing_time_ms is not None
    ]
    end_to_end_latencies = [
        record.end_to_end_latency_ms
        for record in processed_records
        if record.end_to_end_latency_ms is not None
    ]

    summary = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": context.experiment_id,
        "run_id": context.run_id,
        "execution_order": (
            context.planned_run.execution_order
        ),
        "phase": context.planned_run.phase,
        "platform": (
            context.planned_run.platform
        ),
        "architecture": context.architecture,
        "algorithm": context.algorithm,
        "resolution": {
            "width": context.resolution.width,
            "height": context.resolution.height,
        },
        "trial_number": (
            context.planned_run.trial_number
        ),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "warmup_frame_count": (
            warmup_frame_count
        ),
        "measured_frame_count": len(
            normalized_records
        ),
        "processed_frame_count": len(
            processed_records
        ),
        "dropped_frame_count": dropped_count,
        "skipped_frame_count": skipped_count,
        "deadline_met_count": (
            deadline_met_count
        ),
        "deadline_miss_count": (
            deadline_miss_count
        ),
        "mean_processing_time_ms": (
            fmean(processing_times)
            if processing_times
            else None
        ),
        "mean_end_to_end_latency_ms": (
            fmean(end_to_end_latencies)
            if end_to_end_latencies
            else None
        ),
        "frame_results_file": (
            FRAME_RESULTS_FILE_NAME
        ),
        "frame_results_sha256": (
            calculate_file_sha256(
                frame_results_path
            )
        ),
    }

    # The summary is written last. Its presence indicates
    # that the complete artifact set was finalized.
    write_json_atomic(
        summary,
        summary_path,
    )

    return frame_results_path, summary_path