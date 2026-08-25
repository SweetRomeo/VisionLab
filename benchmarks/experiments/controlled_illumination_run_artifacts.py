from __future__ import annotations

from collections.abc import Sequence
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
from benchmarks.realtime.realtime_records import (
    FrameStatus,
    RealtimeFrameRecord,
    write_frame_records,
)


FRAME_RESULTS_FILE_NAME = (
    "realtime_frame_results.csv"
)
EXECUTION_SUMMARY_FILE_NAME = (
    "execution_summary.json"
)


class ControlledIlluminationArtifactError(
    ValueError
):
    """Raised when controlled-run artifacts are invalid."""


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
    context: ControlledIlluminationRunnerContext,
    records: Sequence[RealtimeFrameRecord],
) -> tuple[RealtimeFrameRecord, ...]:
    if not isinstance(
        context,
        ControlledIlluminationRunnerContext,
    ):
        raise ControlledIlluminationArtifactError(
            "context must be "
            "ControlledIlluminationRunnerContext."
        )

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
        isinstance(record, RealtimeFrameRecord)
        for record in normalized_records
    ):
        raise ControlledIlluminationArtifactError(
            "Every record must be RealtimeFrameRecord."
        )

    expected_indices = list(
        range(1, len(normalized_records) + 1)
    )
    actual_indices = [
        record.frame_index
        for record in normalized_records
    ]

    if actual_indices != expected_indices:
        raise ControlledIlluminationArtifactError(
            "Frame indices must be sequential "
            "and start at one."
        )

    expected_resolution = (
        f"{context.resolution.width}"
        f"x{context.resolution.height}"
    )
    expected_trial = (
        context.planned_run.trial_number
    )
    expected_deadline_ms = (
        context.planned_run.frame_deadline_ms
    )

    for record in normalized_records:
        if (
            record.architecture
            != context.architecture
        ):
            raise ControlledIlluminationArtifactError(
                "Frame architecture does not match "
                "the planned run."
            )

        if record.algorithm != context.algorithm:
            raise ControlledIlluminationArtifactError(
                "Frame algorithm does not match "
                "the planned run."
            )

        if record.resolution != expected_resolution:
            raise ControlledIlluminationArtifactError(
                "Frame resolution does not match "
                "the planned run."
            )

        if record.trial != expected_trial:
            raise ControlledIlluminationArtifactError(
                "Frame trial does not match "
                "the planned run."
            )

        if not math.isclose(
            record.deadline_ms,
            expected_deadline_ms,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ControlledIlluminationArtifactError(
                "Frame deadline does not match "
                "the planned run."
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
        RealtimeFrameRecord,
        ...,
    ],
    output_path: Path,
) -> Path:
    temporary_path = output_path.with_name(
        f".{output_path.name}."
        f"{uuid4().hex}.tmp"
        f"{output_path.suffix}"
    )

    try:
        written_record_count = write_frame_records(
            records,
            temporary_path,
        )

        if written_record_count != len(records):
            raise ControlledIlluminationArtifactError(
                "Not all frame records were written."
            )

        with temporary_path.open(
            "r+b"
        ) as output_file:
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
    records: Sequence[RealtimeFrameRecord],
    *,
    started_at_utc: str,
    finished_at_utc: str,
    warmup_frame_count: int,
) -> tuple[Path, Path]:
    normalized_records = validate_frame_records(
        context,
        records,
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
        if (
            record.frame_status
            == FrameStatus.PROCESSED
        )
    ]
    dropped_count = sum(
        record.frame_status == FrameStatus.DROPPED
        for record in normalized_records
    )
    skipped_count = sum(
        record.frame_status == FrameStatus.SKIPPED
        for record in normalized_records
    )
    deadline_met_count = sum(
        record.frame_status
        == FrameStatus.PROCESSED
        and record.deadline_missed is False
        for record in normalized_records
    )
    deadline_miss_count = sum(
        record.frame_status
        == FrameStatus.PROCESSED
        and record.deadline_missed is True
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

    # Written last: its presence marks a complete run.
    write_json_atomic(
        summary,
        summary_path,
    )

    return frame_results_path, summary_path