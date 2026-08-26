from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from pathlib import Path
from datetime import datetime
import json
import math
import os
from uuid import uuid4
from collections import Counter
import csv

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
    validate_safe_identifier,
    validate_utc_timestamp,
    ControlledIlluminationRunMetadata,
    load_run_metadata,
)
from benchmarks.experiments.controlled_illumination_run_state import (
    validate_run_plan_sha256,
)

from benchmarks.experiments.controlled_illumination_run_artifacts import (
    EXECUTION_SUMMARY_FILE_NAME,
    FRAME_RESULTS_FILE_NAME,
    calculate_file_sha256,
)

from benchmarks.experiments.controlled_illumination_run_planner import (
    PlannedRun,
)

from benchmarks.realtime.realtime_records import (
    FrameStatus,
    RealtimeFrameRecord,
)

RUN_BUNDLE_SCHEMA_VERSION = 1

RUN_METADATA_FILE_NAME = "run_metadata.json"

REQUIRED_BUNDLE_ARTIFACT_ORDER = (
    FRAME_RESULTS_FILE_NAME,
    EXECUTION_SUMMARY_FILE_NAME,
    RUN_METADATA_FILE_NAME,
)

RUN_BUNDLE_MANIFEST_FILE_NAME = (
    "run_bundle_manifest.json"
)

EXECUTION_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "experiment_id",
        "run_id",
        "execution_order",
        "phase",
        "platform",
        "architecture",
        "algorithm",
        "resolution",
        "trial_number",
        "started_at_utc",
        "finished_at_utc",
        "warmup_frame_count",
        "measured_frame_count",
        "processed_frame_count",
        "dropped_frame_count",
        "skipped_frame_count",
        "deadline_met_count",
        "deadline_miss_count",
        "mean_processing_time_ms",
        "mean_end_to_end_latency_ms",
        "frame_results_file",
        "frame_results_sha256",
    }
)

REQUIRED_BUNDLE_ARTIFACTS = frozenset(
    REQUIRED_BUNDLE_ARTIFACT_ORDER
)


class ControlledIlluminationRunBundleError(
    ValueError
):
    """Raised when a controlled run bundle is invalid."""


def validate_execution_summary_matches_run(
    summary: ControlledIlluminationExecutionSummary,
    planned_run: PlannedRun,
) -> None:
    expected_fields = {
        "experiment_id": planned_run.experiment_id,
        "run_id": planned_run.run_id,
        "execution_order": planned_run.execution_order,
        "phase": planned_run.phase,
        "platform": planned_run.platform,
        "architecture": planned_run.architecture,
        "algorithm": planned_run.algorithm,
        "trial_number": planned_run.trial_number,
    }

    for field_name, expected_value in (
        expected_fields.items()
    ):
        actual_value = getattr(
            summary,
            field_name,
        )

        if actual_value != expected_value:
            raise ControlledIlluminationRunBundleError(
                "Execution summary does not match "
                f"the planned run: {field_name}"
            )

    if (
        summary.resolution.width
        != planned_run.resolution.width
        or summary.resolution.height
        != planned_run.resolution.height
    ):
        raise ControlledIlluminationRunBundleError(
            "Execution summary does not match "
            "the planned run: resolution"
        )

def validate_metadata_matches_run(
    metadata: ControlledIlluminationRunMetadata,
    planned_run: PlannedRun,
) -> None:
    expected_fields = {
        "experiment_id": planned_run.experiment_id,
        "run_id": planned_run.run_id,
        "phase": planned_run.phase,
        "platform": planned_run.platform,
        "architecture": planned_run.architecture,
        "algorithm": planned_run.algorithm,
        "trial_number": planned_run.trial_number,
        "incidence_angle_degrees": (
            planned_run.incidence_angle_degrees
        ),
        "target_fps": planned_run.target_fps,
        "frame_deadline_ms": (
            planned_run.frame_deadline_ms
        ),
        "target_illuminance_lux": (
            planned_run.target_illuminance_lux
        ),
    }

    for field_name, expected_value in (
        expected_fields.items()
    ):
        actual_value = getattr(
            metadata,
            field_name,
        )

        if actual_value != expected_value:
            raise ControlledIlluminationRunBundleError(
                "Run metadata does not match "
                f"the planned run: {field_name}"
            )

    if (
        metadata.resolution.width
        != planned_run.resolution.width
        or metadata.resolution.height
        != planned_run.resolution.height
    ):
        raise ControlledIlluminationRunBundleError(
            "Run metadata does not match "
            "the planned run: resolution"
        )

    if (
        planned_run.phase == "constant_source"
        and metadata.source_output_setting
        != planned_run.source_output_setting
    ):
        raise ControlledIlluminationRunBundleError(
            "Run metadata source-output setting "
            "does not match the planned run."
        )

def validate_summary_counts_against_config(
    summary: ControlledIlluminationExecutionSummary,
    config: dict,
) -> None:
    execution_config = config.get("execution")

    if not isinstance(execution_config, dict):
        raise ControlledIlluminationRunBundleError(
            "Experiment execution configuration "
            "is missing or invalid."
        )

    expected_warmup_frames = (
        execution_config.get("warmup_frames")
    )
    expected_measured_frames = (
        execution_config.get("measured_frames")
    )

    if (
        isinstance(expected_warmup_frames, bool)
        or not isinstance(
            expected_warmup_frames,
            int,
        )
        or expected_warmup_frames < 0
    ):
        raise ControlledIlluminationRunBundleError(
            "Configured warm-up frame count "
            "is invalid."
        )

    if (
        isinstance(expected_measured_frames, bool)
        or not isinstance(
            expected_measured_frames,
            int,
        )
        or expected_measured_frames <= 0
    ):
        raise ControlledIlluminationRunBundleError(
            "Configured measured-frame count "
            "is invalid."
        )

    if (
        summary.warmup_frame_count
        != expected_warmup_frames
    ):
        raise ControlledIlluminationRunBundleError(
            "Execution-summary warm-up frame count "
            "does not match the configuration."
        )

    if (
        summary.measured_frame_count
        != expected_measured_frames
    ):
        raise ControlledIlluminationRunBundleError(
            "Execution-summary measured-frame count "
            "does not match the configuration."
        )

def validate_summary_frame_hash(
    summary: ControlledIlluminationExecutionSummary,
    artifacts: tuple[RunBundleArtifact, ...],
) -> None:
    if (
        summary.frame_results_file
        != FRAME_RESULTS_FILE_NAME
    ):
        raise ControlledIlluminationRunBundleError(
            "Execution summary references an "
            "unexpected frame-results file."
        )

    frame_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.file_name
        == summary.frame_results_file
    ]

    if len(frame_artifacts) != 1:
        raise ControlledIlluminationRunBundleError(
            "Exactly one frame-results artifact "
            "is required."
        )

    frame_artifact = frame_artifacts[0]

    if (
        frame_artifact.sha256
        != summary.frame_results_sha256
    ):
        raise ControlledIlluminationRunBundleError(
            "Frame-results SHA-256 does not match "
            "the execution summary."
        )

def validate_frame_results_against_run(
    frame_results_path: Path,
    planned_run: PlannedRun,
    summary: ControlledIlluminationExecutionSummary,
) -> None:
    expected_fields = [
        "architecture",
        "algorithm",
        "resolution",
        "trial",
        "frame_index",
        "scheduled_timestamp_ms",
        "enqueued_timestamp_ms",
        "processing_start_timestamp_ms",
        "processing_end_timestamp_ms",
        "drop_timestamp_ms",
        "source_delay_ms",
        "queue_wait_time_ms",
        "processing_time_ms",
        "end_to_end_latency_ms",
        "deadline_ms",
        "deadline_missed",
        "frame_status",
    ]

    if not isinstance(
        frame_results_path,
        Path,
    ):
        raise ControlledIlluminationRunBundleError(
            "frame_results_path must be Path."
        )

    if not frame_results_path.is_file():
        raise ControlledIlluminationRunBundleError(
            "Frame-results CSV was not found: "
            f"{frame_results_path}"
        )

    expected_resolution = (
        f"{planned_run.resolution.width}"
        f"x{planned_run.resolution.height}"
    )

    def parse_required_integer(
        value: str | None,
        field_name: str,
        line_number: int,
    ) -> int:
        if value is None or not value.strip():
            raise ControlledIlluminationRunBundleError(
                f"{field_name} is missing on CSV "
                f"line {line_number}."
            )

        try:
            parsed_value = int(value)
        except ValueError as error:
            raise ControlledIlluminationRunBundleError(
                f"{field_name} must be an integer "
                f"on CSV line {line_number}."
            ) from error

        return parsed_value

    def parse_required_number(
        value: str | None,
        field_name: str,
        line_number: int,
    ) -> float:
        if value is None or not value.strip():
            raise ControlledIlluminationRunBundleError(
                f"{field_name} is missing on CSV "
                f"line {line_number}."
            )

        try:
            parsed_value = float(value)
        except ValueError as error:
            raise ControlledIlluminationRunBundleError(
                f"{field_name} must be numeric "
                f"on CSV line {line_number}."
            ) from error

        if not math.isfinite(parsed_value):
            raise ControlledIlluminationRunBundleError(
                f"{field_name} must be finite "
                f"on CSV line {line_number}."
            )

        return parsed_value

    def parse_optional_number(
        value: str | None,
        field_name: str,
        line_number: int,
    ) -> float | None:
        if value is None or not value.strip():
            return None

        return parse_required_number(
            value,
            field_name,
            line_number,
        )

    def parse_optional_boolean(
        value: str | None,
        field_name: str,
        line_number: int,
    ) -> bool | None:
        if value is None or not value.strip():
            return None

        normalized_value = value.strip().lower()

        if normalized_value == "true":
            return True

        if normalized_value == "false":
            return False

        raise ControlledIlluminationRunBundleError(
            f"{field_name} must be true, false or "
            f"empty on CSV line {line_number}."
        )

    records: list[RealtimeFrameRecord] = []

    try:
        with frame_results_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as frame_results_file:
            reader = csv.DictReader(
                frame_results_file
            )

            if reader.fieldnames != expected_fields:
                raise ControlledIlluminationRunBundleError(
                    "Frame-results CSV fields do not "
                    "match the required schema. "
                    f"Expected: {expected_fields}; "
                    f"actual: {reader.fieldnames}"
                )

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                if None in row:
                    raise (
                        ControlledIlluminationRunBundleError(
                            "Frame-results CSV contains "
                            "unexpected columns on line "
                            f"{line_number}."
                        )
                    )

                try:
                    frame_status = FrameStatus(
                        row["frame_status"]
                    )
                except ValueError as error:
                    raise (
                        ControlledIlluminationRunBundleError(
                            "Invalid frame_status on CSV "
                            f"line {line_number}: "
                            f"{row['frame_status']}"
                        )
                    ) from error

                try:
                    record = RealtimeFrameRecord(
                        architecture=(
                            row["architecture"]
                        ),
                        algorithm=row["algorithm"],
                        resolution=row["resolution"],
                        trial=parse_required_integer(
                            row["trial"],
                            "trial",
                            line_number,
                        ),
                        frame_index=(
                            parse_required_integer(
                                row["frame_index"],
                                "frame_index",
                                line_number,
                            )
                        ),
                        scheduled_timestamp_ms=(
                            parse_required_number(
                                row[
                                    "scheduled_timestamp_ms"
                                ],
                                "scheduled_timestamp_ms",
                                line_number,
                            )
                        ),
                        enqueued_timestamp_ms=(
                            parse_optional_number(
                                row[
                                    "enqueued_timestamp_ms"
                                ],
                                "enqueued_timestamp_ms",
                                line_number,
                            )
                        ),
                        processing_start_timestamp_ms=(
                            parse_optional_number(
                                row[
                                    "processing_start_timestamp_ms"
                                ],
                                (
                                    "processing_start_"
                                    "timestamp_ms"
                                ),
                                line_number,
                            )
                        ),
                        processing_end_timestamp_ms=(
                            parse_optional_number(
                                row[
                                    "processing_end_timestamp_ms"
                                ],
                                (
                                    "processing_end_"
                                    "timestamp_ms"
                                ),
                                line_number,
                            )
                        ),
                        drop_timestamp_ms=(
                            parse_optional_number(
                                row["drop_timestamp_ms"],
                                "drop_timestamp_ms",
                                line_number,
                            )
                        ),
                        source_delay_ms=(
                            parse_optional_number(
                                row["source_delay_ms"],
                                "source_delay_ms",
                                line_number,
                            )
                        ),
                        queue_wait_time_ms=(
                            parse_optional_number(
                                row[
                                    "queue_wait_time_ms"
                                ],
                                "queue_wait_time_ms",
                                line_number,
                            )
                        ),
                        processing_time_ms=(
                            parse_optional_number(
                                row[
                                    "processing_time_ms"
                                ],
                                "processing_time_ms",
                                line_number,
                            )
                        ),
                        end_to_end_latency_ms=(
                            parse_optional_number(
                                row[
                                    "end_to_end_latency_ms"
                                ],
                                "end_to_end_latency_ms",
                                line_number,
                            )
                        ),
                        deadline_ms=(
                            parse_required_number(
                                row["deadline_ms"],
                                "deadline_ms",
                                line_number,
                            )
                        ),
                        deadline_missed=(
                            parse_optional_boolean(
                                row["deadline_missed"],
                                "deadline_missed",
                                line_number,
                            )
                        ),
                        frame_status=frame_status,
                    )
                except ValueError as error:
                    raise (
                        ControlledIlluminationRunBundleError(
                            "Invalid frame-results record "
                            f"on CSV line {line_number}: "
                            f"{error}"
                        )
                    ) from error

                if (
                    record.architecture
                    != planned_run.architecture
                ):
                    raise (
                        ControlledIlluminationRunBundleError(
                            "Frame architecture does not "
                            "match the planned run on "
                            f"CSV line {line_number}."
                        )
                    )

                if (
                    record.algorithm
                    != planned_run.algorithm
                ):
                    raise (
                        ControlledIlluminationRunBundleError(
                            "Frame algorithm does not "
                            "match the planned run on "
                            f"CSV line {line_number}."
                        )
                    )

                if (
                    record.resolution
                    != expected_resolution
                ):
                    raise (
                        ControlledIlluminationRunBundleError(
                            "Frame resolution does not "
                            "match the planned run on "
                            f"CSV line {line_number}."
                        )
                    )

                if (
                    record.trial
                    != planned_run.trial_number
                ):
                    raise (
                        ControlledIlluminationRunBundleError(
                            "Frame trial does not match "
                            "the planned run on CSV line "
                            f"{line_number}."
                        )
                    )

                if not math.isclose(
                    record.deadline_ms,
                    planned_run.frame_deadline_ms,
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                ):
                    raise (
                        ControlledIlluminationRunBundleError(
                            "Frame deadline does not "
                            "match the planned run on "
                            f"CSV line {line_number}."
                        )
                    )

                records.append(record)

    except UnicodeError as error:
        raise ControlledIlluminationRunBundleError(
            "Frame-results CSV is not valid UTF-8."
        ) from error

    if len(records) != summary.measured_frame_count:
        raise ControlledIlluminationRunBundleError(
            "Frame-results row count does not match "
            "measured_frame_count. "
            f"Expected: {summary.measured_frame_count}; "
            f"actual: {len(records)}"
        )

    actual_frame_indices = [
        record.frame_index
        for record in records
    ]
    expected_frame_indices = list(
        range(
            1,
            summary.measured_frame_count + 1,
        )
    )

    if actual_frame_indices != expected_frame_indices:
        raise ControlledIlluminationRunBundleError(
            "Frame indices must be sequential, "
            "unique and start at 1."
        )

    status_counts = Counter(
        record.frame_status
        for record in records
    )

    if (
        status_counts[FrameStatus.PROCESSED]
        != summary.processed_frame_count
    ):
        raise ControlledIlluminationRunBundleError(
            "Processed-frame count does not match "
            "the execution summary."
        )

    if (
        status_counts[FrameStatus.DROPPED]
        != summary.dropped_frame_count
    ):
        raise ControlledIlluminationRunBundleError(
            "Dropped-frame count does not match "
            "the execution summary."
        )

    if (
        status_counts[FrameStatus.SKIPPED]
        != summary.skipped_frame_count
    ):
        raise ControlledIlluminationRunBundleError(
            "Skipped-frame count does not match "
            "the execution summary."
        )

    deadline_miss_count = sum(
        record.frame_status == FrameStatus.PROCESSED
        and record.deadline_missed is True
        for record in records
    )
    deadline_met_count = sum(
        record.frame_status == FrameStatus.PROCESSED
        and record.deadline_missed is False
        for record in records
    )

    if (
        deadline_miss_count
        != summary.deadline_miss_count
    ):
        raise ControlledIlluminationRunBundleError(
            "Deadline-miss count does not match "
            "the execution summary."
        )

    if (
        deadline_met_count
        != summary.deadline_met_count
    ):
        raise ControlledIlluminationRunBundleError(
            "Deadline-met count does not match "
            "the execution summary."
        )

def validate_bundle_identifier(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise ControlledIlluminationRunBundleError(
            f"{field_name} must be a string."
        )

    try:
        validate_safe_identifier(
            value,
            field_name,
        )
    except ValueError as error:
        raise ControlledIlluminationRunBundleError(
            str(error)
        ) from error

    return value


def validate_bundle_timestamp(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise ControlledIlluminationRunBundleError(
            f"{field_name} must be a string."
        )

    try:
        validate_utc_timestamp(
            value,
            field_name,
        )
    except ValueError as error:
        raise ControlledIlluminationRunBundleError(
            str(error)
        ) from error

    return value


def validate_sha256(
    value: Any,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ControlledIlluminationRunBundleError(
            f"{field_name} must be a lowercase "
            "SHA-256 value."
        )

    return value

def resolve_run_directory(
    run_directory: str | Path,
) -> Path:
    if (
        not isinstance(run_directory, (str, Path))
        or (
            isinstance(run_directory, str)
            and not run_directory.strip()
        )
    ):
        raise ControlledIlluminationRunBundleError(
            "run_directory must be a non-empty path."
        )

    resolved_directory = Path(
        run_directory
    ).resolve()

    if not resolved_directory.is_dir():
        raise ControlledIlluminationRunBundleError(
            "Run directory was not found: "
            f"{resolved_directory}"
        )

    return resolved_directory


def require_bundle_artifact_path(
    run_directory: Path,
    file_name: str,
) -> Path:
    if file_name not in REQUIRED_BUNDLE_ARTIFACTS:
        raise ControlledIlluminationRunBundleError(
            f"Unsupported bundle artifact: {file_name}"
        )

    artifact_path = (
        run_directory / file_name
    ).resolve()

    try:
        artifact_path.relative_to(
            run_directory
        )
    except ValueError as error:
        raise ControlledIlluminationRunBundleError(
            "Bundle artifact must remain inside "
            "the run directory."
        ) from error

    if not artifact_path.is_file():
        raise ControlledIlluminationRunBundleError(
            "Required bundle artifact was not found: "
            f"{artifact_path}"
        )

    if artifact_path.stat().st_size <= 0:
        raise ControlledIlluminationRunBundleError(
            "Bundle artifact must not be empty: "
            f"{artifact_path}"
        )

    return artifact_path

@dataclass(frozen=True)
class RunBundleArtifact:
    file_name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.file_name, str)
            or self.file_name
            not in REQUIRED_BUNDLE_ARTIFACTS
        ):
            raise ControlledIlluminationRunBundleError(
                "Unsupported bundle artifact: "
                f"{self.file_name}"
            )

        validate_sha256(
            self.sha256,
            f"{self.file_name}.sha256",
        )

        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ControlledIlluminationRunBundleError(
                f"{self.file_name}.size_bytes must "
                "be a positive integer."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def collect_bundle_artifacts(
    run_directory: str | Path,
) -> tuple[RunBundleArtifact, ...]:
    resolved_directory = resolve_run_directory(
        run_directory
    )

    artifacts = []

    for file_name in (
        REQUIRED_BUNDLE_ARTIFACT_ORDER
    ):
        artifact_path = (
            require_bundle_artifact_path(
                resolved_directory,
                file_name,
            )
        )

        artifacts.append(
            RunBundleArtifact(
                file_name=file_name,
                sha256=calculate_file_sha256(
                    artifact_path
                ),
                size_bytes=(
                    artifact_path.stat().st_size
                ),
            )
        )

    return tuple(artifacts)

def require_summary_integer(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (positive and value <= 0)
        or (not positive and value < 0)
    ):
        requirement = (
            "positive"
            if positive
            else "non-negative"
        )

        raise ControlledIlluminationRunBundleError(
            f"{field_name} must be a "
            f"{requirement} integer."
        )

    return value


def require_optional_summary_number(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ControlledIlluminationRunBundleError(
            f"{field_name} must be null or a "
            "non-negative finite number."
        )

    return float(value)


@dataclass(frozen=True)
class ControlledIlluminationExecutionSummary:
    schema_version: int
    status: str
    experiment_id: str
    run_id: str
    execution_order: int
    phase: str
    platform: str
    architecture: str
    algorithm: str
    resolution: ResolutionMetadata
    trial_number: int
    started_at_utc: str
    finished_at_utc: str
    warmup_frame_count: int
    measured_frame_count: int
    processed_frame_count: int
    dropped_frame_count: int
    skipped_frame_count: int
    deadline_met_count: int
    deadline_miss_count: int
    mean_processing_time_ms: float | None
    mean_end_to_end_latency_ms: float | None
    frame_results_file: str
    frame_results_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ControlledIlluminationRunBundleError(
                "Execution summary schema_version "
                "must be 1."
            )

        if self.status != "completed":
            raise ControlledIlluminationRunBundleError(
                "Execution summary status must be "
                "completed."
            )

        for field_name in (
            "experiment_id",
            "run_id",
            "phase",
            "platform",
            "architecture",
            "algorithm",
        ):
            validate_bundle_identifier(
                getattr(self, field_name),
                field_name,
            )

        if not isinstance(
            self.resolution,
            ResolutionMetadata,
        ):
            raise ControlledIlluminationRunBundleError(
                "resolution must be "
                "ResolutionMetadata."
            )

        require_summary_integer(
            self.execution_order,
            "execution_order",
            positive=True,
        )
        require_summary_integer(
            self.trial_number,
            "trial_number",
            positive=True,
        )
        require_summary_integer(
            self.warmup_frame_count,
            "warmup_frame_count",
        )
        require_summary_integer(
            self.measured_frame_count,
            "measured_frame_count",
            positive=True,
        )

        for field_name in (
            "processed_frame_count",
            "dropped_frame_count",
            "skipped_frame_count",
            "deadline_met_count",
            "deadline_miss_count",
        ):
            require_summary_integer(
                getattr(self, field_name),
                field_name,
            )

        if (
            self.processed_frame_count
            + self.dropped_frame_count
            + self.skipped_frame_count
            != self.measured_frame_count
        ):
            raise ControlledIlluminationRunBundleError(
                "Processed, dropped and skipped frame "
                "counts must equal measured_frame_count."
            )

        if (
            self.deadline_met_count
            + self.deadline_miss_count
            != self.processed_frame_count
        ):
            raise ControlledIlluminationRunBundleError(
                "Deadline counts must equal "
                "processed_frame_count."
            )

        started_at = validate_bundle_timestamp(
            self.started_at_utc,
            "started_at_utc",
        )
        finished_at = validate_bundle_timestamp(
            self.finished_at_utc,
            "finished_at_utc",
        )

        started_datetime = datetime.fromisoformat(
            started_at.replace("Z", "+00:00")
        )
        finished_datetime = datetime.fromisoformat(
            finished_at.replace("Z", "+00:00")
        )

        if finished_datetime < started_datetime:
            raise ControlledIlluminationRunBundleError(
                "finished_at_utc cannot be earlier "
                "than started_at_utc."
            )

        require_optional_summary_number(
            self.mean_processing_time_ms,
            "mean_processing_time_ms",
        )
        require_optional_summary_number(
            self.mean_end_to_end_latency_ms,
            "mean_end_to_end_latency_ms",
        )

        if (
            self.processed_frame_count > 0
            and (
                self.mean_processing_time_ms is None
                or self.mean_end_to_end_latency_ms
                is None
            )
        ):
            raise ControlledIlluminationRunBundleError(
                "Processed frames require mean timing "
                "values."
            )

        if (
            self.processed_frame_count == 0
            and (
                self.mean_processing_time_ms
                is not None
                or self.mean_end_to_end_latency_ms
                is not None
            )
        ):
            raise ControlledIlluminationRunBundleError(
                "Mean timing values must be null when "
                "no frames were processed."
            )

        if (
            self.frame_results_file
            != FRAME_RESULTS_FILE_NAME
        ):
            raise ControlledIlluminationRunBundleError(
                "Execution summary references an "
                "unexpected frame-result file."
            )

        validate_sha256(
            self.frame_results_sha256,
            "frame_results_sha256",
        )


def execution_summary_from_dict(
    value: Any,
) -> ControlledIlluminationExecutionSummary:
    if not isinstance(value, dict):
        raise ControlledIlluminationRunBundleError(
            "Execution summary root must be an object."
        )

    actual_fields = set(value)

    if actual_fields != EXECUTION_SUMMARY_FIELDS:
        missing_fields = sorted(
            EXECUTION_SUMMARY_FIELDS - actual_fields
        )
        unexpected_fields = sorted(
            actual_fields - EXECUTION_SUMMARY_FIELDS
        )

        raise ControlledIlluminationRunBundleError(
            "Execution summary fields do not match "
            "the required schema. "
            f"Missing: {missing_fields}; "
            f"unexpected: {unexpected_fields}"
        )

    resolution_value = value["resolution"]

    if (
        not isinstance(resolution_value, dict)
        or set(resolution_value)
        != {"width", "height"}
    ):
        raise ControlledIlluminationRunBundleError(
            "Execution summary resolution must "
            "contain exactly width and height."
        )

    try:
        resolution = ResolutionMetadata(
            width=resolution_value["width"],
            height=resolution_value["height"],
        )
    except ValueError as error:
        raise ControlledIlluminationRunBundleError(
            f"Invalid execution resolution: {error}"
        ) from error

    summary_arguments = dict(value)
    summary_arguments["resolution"] = resolution

    try:
        return ControlledIlluminationExecutionSummary(
            **summary_arguments
        )
    except TypeError as error:
        raise ControlledIlluminationRunBundleError(
            "Invalid execution summary structure: "
            f"{error}"
        ) from error


def load_execution_summary(
    run_directory: str | Path,
) -> ControlledIlluminationExecutionSummary:
    resolved_directory = resolve_run_directory(
        run_directory
    )
    summary_path = require_bundle_artifact_path(
        resolved_directory,
        EXECUTION_SUMMARY_FILE_NAME,
    )

    try:
        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as summary_file:
            summary_value = json.load(
                summary_file
            )
    except json.JSONDecodeError as error:
        raise ControlledIlluminationRunBundleError(
            "Invalid execution summary JSON: "
            f"{error}"
        ) from error

    return execution_summary_from_dict(
        summary_value
    )

@dataclass(frozen=True)
class ControlledIlluminationRunBundleManifest:
    schema_version: int
    finalized_at_utc: str
    experiment_id: str
    run_id: str
    run_plan_sha256: str
    metadata_dry_run: bool
    artifacts: tuple[RunBundleArtifact, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version
            != RUN_BUNDLE_SCHEMA_VERSION
        ):
            raise ControlledIlluminationRunBundleError(
                "schema_version must be 1."
            )

        validate_bundle_timestamp(
            self.finalized_at_utc,
            "finalized_at_utc",
        )
        validate_bundle_identifier(
            self.experiment_id,
            "experiment_id",
        )
        validate_bundle_identifier(
            self.run_id,
            "run_id",
        )

        try:
            validate_run_plan_sha256(
                self.run_plan_sha256
            )
        except ValueError as error:
            raise ControlledIlluminationRunBundleError(
                str(error)
            ) from error

        if not isinstance(
            self.metadata_dry_run,
            bool,
        ):
            raise ControlledIlluminationRunBundleError(
                "metadata_dry_run must be boolean."
            )

        if (
            not isinstance(self.artifacts, tuple)
            or not self.artifacts
        ):
            raise ControlledIlluminationRunBundleError(
                "artifacts must be a non-empty tuple."
            )

        if not all(
            isinstance(artifact, RunBundleArtifact)
            for artifact in self.artifacts
        ):
            raise ControlledIlluminationRunBundleError(
                "Every artifact must be "
                "RunBundleArtifact."
            )

        artifact_names = [
            artifact.file_name
            for artifact in self.artifacts
        ]

        if len(artifact_names) != len(
            set(artifact_names)
        ):
            raise ControlledIlluminationRunBundleError(
                "Bundle artifact names must be unique."
            )

        if set(artifact_names) != set(
            REQUIRED_BUNDLE_ARTIFACTS
        ):
            missing_artifacts = sorted(
                REQUIRED_BUNDLE_ARTIFACTS
                - set(artifact_names)
            )
            unexpected_artifacts = sorted(
                set(artifact_names)
                - REQUIRED_BUNDLE_ARTIFACTS
            )

            raise ControlledIlluminationRunBundleError(
                "Bundle artifacts do not match the "
                "required set. "
                f"Missing: {missing_artifacts}; "
                f"unexpected: {unexpected_artifacts}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "finalized_at_utc": (
                self.finalized_at_utc
            ),
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "run_plan_sha256": (
                self.run_plan_sha256
            ),
            "metadata_dry_run": (
                self.metadata_dry_run
            ),
            "artifacts": [
                artifact.to_dict()
                for artifact in self.artifacts
            ],
        }

def write_run_bundle_manifest_atomic(
    manifest: ControlledIlluminationRunBundleManifest,
    run_directory: str | Path,
) -> Path:
    if not isinstance(
        manifest,
        ControlledIlluminationRunBundleManifest,
    ):
        raise ControlledIlluminationRunBundleError(
            "manifest must be "
            "ControlledIlluminationRunBundleManifest."
        )

    resolved_directory = resolve_run_directory(
        run_directory
    )
    output_path = (
        resolved_directory
        / RUN_BUNDLE_MANIFEST_FILE_NAME
    )

    if output_path.exists():
        raise ControlledIlluminationRunBundleError(
            "Run bundle has already been finalized: "
            f"{output_path}"
        )

    temporary_path = output_path.with_name(
        f".{output_path.name}."
        f"{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "x",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                manifest.to_dict(),
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
            missing_ok=True,
        )

    return output_path

def validate_run_bundle(
    run_directory: str | Path,
    planned_run: PlannedRun,
    config: dict[str, Any],
    run_plan_sha256: str,
    finalized_at_utc: str,
) -> ControlledIlluminationRunBundleManifest:
    if not isinstance(planned_run, PlannedRun):
        raise ControlledIlluminationRunBundleError(
            "planned_run must be PlannedRun."
        )

    if not isinstance(config, dict):
        raise ControlledIlluminationRunBundleError(
            "config must be a dictionary."
        )

    validated_finalized_at = (
        validate_bundle_timestamp(
            finalized_at_utc,
            "finalized_at_utc",
        )
    )

    try:
        validated_plan_sha256 = (
            validate_run_plan_sha256(
                run_plan_sha256
            )
        )
    except ValueError as error:
        raise ControlledIlluminationRunBundleError(
            str(error)
        ) from error

    resolved_directory = resolve_run_directory(
        run_directory
    )
    metadata_path = (
        resolved_directory
        / RUN_METADATA_FILE_NAME
    )

    try:
        metadata = load_run_metadata(
            metadata_path,
            config=config,
        )
    except (OSError, ValueError) as error:
        raise ControlledIlluminationRunBundleError(
            "Run metadata could not be loaded "
            f"or validated: {error}"
        ) from error

    summary = load_execution_summary(
        resolved_directory
    )
    artifacts = collect_bundle_artifacts(
        resolved_directory
    )

    validate_execution_summary_matches_run(
        summary,
        planned_run,
    )
    validate_metadata_matches_run(
        metadata,
        planned_run,
    )
    validate_summary_counts_against_config(
        summary,
        config,
    )
    validate_summary_frame_hash(
        summary,
        artifacts,
    )

    validate_frame_results_against_run(
        (
                resolved_directory
                / FRAME_RESULTS_FILE_NAME
        ),
        planned_run,
        summary,
    )

    finalized_datetime = datetime.fromisoformat(
        validated_finalized_at.replace(
            "Z",
            "+00:00",
        )
    )
    finished_datetime = datetime.fromisoformat(
        summary.finished_at_utc.replace(
            "Z",
            "+00:00",
        )
    )

    if finalized_datetime < finished_datetime:
        raise ControlledIlluminationRunBundleError(
            "finalized_at_utc cannot be earlier "
            "than the execution finish time."
        )

    manifest = (
        ControlledIlluminationRunBundleManifest(
            schema_version=(
                RUN_BUNDLE_SCHEMA_VERSION
            ),
            finalized_at_utc=(
                validated_finalized_at
            ),
            experiment_id=(
                planned_run.experiment_id
            ),
            run_id=planned_run.run_id,
            run_plan_sha256=(
                validated_plan_sha256
            ),
            metadata_dry_run=metadata.dry_run,
            artifacts=artifacts,
        )
    )

    return manifest

def finalize_run_bundle_atomic(
    run_directory: str | Path,
    planned_run: PlannedRun,
    config: dict[str, Any],
    run_plan_sha256: str,
    finalized_at_utc: str,
) -> tuple[
    ControlledIlluminationRunBundleManifest,
    Path,
]:
    manifest = validate_run_bundle(
        run_directory,
        planned_run,
        config,
        run_plan_sha256,
        finalized_at_utc,
    )

    manifest_path = (
        write_run_bundle_manifest_atomic(
            manifest,
            run_directory,
        )
    )

    return manifest, manifest_path