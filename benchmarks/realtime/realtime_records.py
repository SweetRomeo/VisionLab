import csv
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class FrameStatus(str, Enum):
    PROCESSED = "processed"
    DROPPED = "dropped"
    SKIPPED = "skipped"


CSV_FIELDNAMES = [
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


def require_non_empty_string(
    value: str,
    field_name: str,
) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string."
        )


def require_positive_integer(
    value: int,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be a positive integer."
        )


def require_non_negative_number(
    value: float,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"{field_name} must be a finite, "
            "non-negative number."
        )


@dataclass(frozen=True)
class RealtimeRunContext:
    architecture: str
    algorithm: str
    resolution: str
    trial: int
    origin_timestamp_ns: int
    deadline_ms: float

    def __post_init__(self) -> None:
        require_non_empty_string(
            self.architecture,
            "architecture",
        )
        require_non_empty_string(
            self.algorithm,
            "algorithm",
        )
        require_non_empty_string(
            self.resolution,
            "resolution",
        )
        require_positive_integer(
            self.trial,
            "trial",
        )

        if (
            isinstance(self.origin_timestamp_ns, bool)
            or not isinstance(
                self.origin_timestamp_ns,
                int,
            )
            or self.origin_timestamp_ns < 0
        ):
            raise ValueError(
                "origin_timestamp_ns must be a "
                "non-negative integer."
            )

        require_non_negative_number(
            self.deadline_ms,
            "deadline_ms",
        )

        if self.deadline_ms == 0.0:
            raise ValueError(
                "deadline_ms must be greater than zero."
            )


@dataclass(frozen=True)
class RealtimeFrameRecord:
    architecture: str
    algorithm: str
    resolution: str
    trial: int
    frame_index: int
    scheduled_timestamp_ms: float
    enqueued_timestamp_ms: float | None
    processing_start_timestamp_ms: float | None
    processing_end_timestamp_ms: float | None
    drop_timestamp_ms: float | None
    source_delay_ms: float | None
    queue_wait_time_ms: float | None
    processing_time_ms: float | None
    end_to_end_latency_ms: float | None
    deadline_ms: float
    deadline_missed: bool | None
    frame_status: FrameStatus

    def __post_init__(self) -> None:
        require_non_empty_string(
            self.architecture,
            "architecture",
        )
        require_non_empty_string(
            self.algorithm,
            "algorithm",
        )
        require_non_empty_string(
            self.resolution,
            "resolution",
        )
        require_positive_integer(
            self.trial,
            "trial",
        )
        require_positive_integer(
            self.frame_index,
            "frame_index",
        )

        require_non_negative_number(
            self.scheduled_timestamp_ms,
            "scheduled_timestamp_ms",
        )
        require_non_negative_number(
            self.deadline_ms,
            "deadline_ms",
        )

        if self.deadline_ms == 0.0:
            raise ValueError(
                "deadline_ms must be greater than zero."
            )

        optional_number_fields = (
            "enqueued_timestamp_ms",
            "processing_start_timestamp_ms",
            "processing_end_timestamp_ms",
            "drop_timestamp_ms",
            "source_delay_ms",
            "queue_wait_time_ms",
            "processing_time_ms",
            "end_to_end_latency_ms",
        )

        for field_name in optional_number_fields:
            value = getattr(self, field_name)

            if value is not None:
                require_non_negative_number(
                    value,
                    field_name,
                )

        if (
            self.deadline_missed is not None
            and not isinstance(
                self.deadline_missed,
                bool,
            )
        ):
            raise ValueError(
                "deadline_missed must be a boolean "
                "or None."
            )

        if not isinstance(
            self.frame_status,
            FrameStatus,
        ):
            raise ValueError(
                "frame_status must be a FrameStatus."
            )

        self._validate_status_fields()
        self._validate_timestamp_order()

    def _validate_status_fields(self) -> None:
        processing_fields = (
            self.processing_start_timestamp_ms,
            self.processing_end_timestamp_ms,
            self.queue_wait_time_ms,
            self.processing_time_ms,
            self.end_to_end_latency_ms,
        )

        if self.frame_status is FrameStatus.PROCESSED:
            required_values = (
                self.enqueued_timestamp_ms,
                self.processing_start_timestamp_ms,
                self.processing_end_timestamp_ms,
                self.source_delay_ms,
                self.queue_wait_time_ms,
                self.processing_time_ms,
                self.end_to_end_latency_ms,
                self.deadline_missed,
            )

            if any(
                value is None
                for value in required_values
            ):
                raise ValueError(
                    "Processed frame records require "
                    "all timing and deadline fields."
                )

            if self.drop_timestamp_ms is not None:
                raise ValueError(
                    "A processed frame cannot have "
                    "a drop timestamp."
                )

        elif self.frame_status is FrameStatus.DROPPED:
            required_values = (
                self.enqueued_timestamp_ms,
                self.drop_timestamp_ms,
                self.source_delay_ms,
            )

            if any(
                value is None
                for value in required_values
            ):
                raise ValueError(
                    "Dropped frame records require "
                    "enqueue, drop and source-delay "
                    "timestamps."
                )

            if any(
                value is not None
                for value in processing_fields
            ):
                raise ValueError(
                    "A dropped frame cannot contain "
                    "processing measurements."
                )

            if self.deadline_missed is not None:
                raise ValueError(
                    "A dropped frame cannot have a "
                    "deadline-missed result."
                )

        elif self.frame_status is FrameStatus.SKIPPED:
            skipped_optional_fields = (
                self.enqueued_timestamp_ms,
                self.processing_start_timestamp_ms,
                self.processing_end_timestamp_ms,
                self.drop_timestamp_ms,
                self.source_delay_ms,
                self.queue_wait_time_ms,
                self.processing_time_ms,
                self.end_to_end_latency_ms,
                self.deadline_missed,
            )

            if any(
                value is not None
                for value in skipped_optional_fields
            ):
                raise ValueError(
                    "A skipped frame cannot contain "
                    "queue or processing measurements."
                )

    def _validate_timestamp_order(self) -> None:
        if (
            self.enqueued_timestamp_ms is not None
            and self.enqueued_timestamp_ms
            < self.scheduled_timestamp_ms
        ):
            raise ValueError(
                "A frame cannot be enqueued before "
                "its scheduled time."
            )

        if (
            self.processing_start_timestamp_ms
            is not None
            and self.enqueued_timestamp_ms is not None
            and self.processing_start_timestamp_ms
            < self.enqueued_timestamp_ms
        ):
            raise ValueError(
                "Processing cannot start before "
                "the frame is enqueued."
            )

        if (
            self.processing_end_timestamp_ms
            is not None
            and self.processing_start_timestamp_ms
            is not None
            and self.processing_end_timestamp_ms
            < self.processing_start_timestamp_ms
        ):
            raise ValueError(
                "Processing cannot end before "
                "it starts."
            )

        if (
            self.drop_timestamp_ms is not None
            and self.enqueued_timestamp_ms is not None
            and self.drop_timestamp_ms
            < self.enqueued_timestamp_ms
        ):
            raise ValueError(
                "A frame cannot be dropped before "
                "it is enqueued."
            )

    def to_csv_row(
        self,
    ) -> dict[str, str | int]:
        return {
            "architecture": self.architecture,
            "algorithm": self.algorithm,
            "resolution": self.resolution,
            "trial": self.trial,
            "frame_index": self.frame_index,
            "scheduled_timestamp_ms": (
                format_optional_float(
                    self.scheduled_timestamp_ms
                )
            ),
            "enqueued_timestamp_ms": (
                format_optional_float(
                    self.enqueued_timestamp_ms
                )
            ),
            "processing_start_timestamp_ms": (
                format_optional_float(
                    self.processing_start_timestamp_ms
                )
            ),
            "processing_end_timestamp_ms": (
                format_optional_float(
                    self.processing_end_timestamp_ms
                )
            ),
            "drop_timestamp_ms": (
                format_optional_float(
                    self.drop_timestamp_ms
                )
            ),
            "source_delay_ms": (
                format_optional_float(
                    self.source_delay_ms
                )
            ),
            "queue_wait_time_ms": (
                format_optional_float(
                    self.queue_wait_time_ms
                )
            ),
            "processing_time_ms": (
                format_optional_float(
                    self.processing_time_ms
                )
            ),
            "end_to_end_latency_ms": (
                format_optional_float(
                    self.end_to_end_latency_ms
                )
            ),
            "deadline_ms": (
                format_optional_float(
                    self.deadline_ms
                )
            ),
            "deadline_missed": (
                ""
                if self.deadline_missed is None
                else str(
                    self.deadline_missed
                ).lower()
            ),
            "frame_status": self.frame_status.value,
        }


def relative_milliseconds(
    timestamp_ns: int,
    origin_timestamp_ns: int,
) -> float:
    if timestamp_ns < origin_timestamp_ns:
        raise ValueError(
            "Timestamp cannot be earlier than "
            "the run origin."
        )

    return (
        timestamp_ns - origin_timestamp_ns
    ) / 1_000_000.0


def create_processed_record(
    context: RealtimeRunContext,
    *,
    frame_index: int,
    scheduled_timestamp_ns: int,
    enqueued_timestamp_ns: int,
    processing_start_timestamp_ns: int,
    processing_end_timestamp_ns: int,
) -> RealtimeFrameRecord:
    scheduled_ms = relative_milliseconds(
        scheduled_timestamp_ns,
        context.origin_timestamp_ns,
    )
    enqueued_ms = relative_milliseconds(
        enqueued_timestamp_ns,
        context.origin_timestamp_ns,
    )
    processing_start_ms = relative_milliseconds(
        processing_start_timestamp_ns,
        context.origin_timestamp_ns,
    )
    processing_end_ms = relative_milliseconds(
        processing_end_timestamp_ns,
        context.origin_timestamp_ns,
    )

    end_to_end_latency_ms = (
        processing_end_ms - scheduled_ms
    )

    return RealtimeFrameRecord(
        architecture=context.architecture,
        algorithm=context.algorithm,
        resolution=context.resolution,
        trial=context.trial,
        frame_index=frame_index,
        scheduled_timestamp_ms=scheduled_ms,
        enqueued_timestamp_ms=enqueued_ms,
        processing_start_timestamp_ms=(
            processing_start_ms
        ),
        processing_end_timestamp_ms=(
            processing_end_ms
        ),
        drop_timestamp_ms=None,
        source_delay_ms=(
            enqueued_ms - scheduled_ms
        ),
        queue_wait_time_ms=(
            processing_start_ms - enqueued_ms
        ),
        processing_time_ms=(
            processing_end_ms
            - processing_start_ms
        ),
        end_to_end_latency_ms=(
            end_to_end_latency_ms
        ),
        deadline_ms=context.deadline_ms,
        deadline_missed=(
            end_to_end_latency_ms
            > context.deadline_ms
        ),
        frame_status=FrameStatus.PROCESSED,
    )


def create_dropped_record(
    context: RealtimeRunContext,
    *,
    frame_index: int,
    scheduled_timestamp_ns: int,
    enqueued_timestamp_ns: int,
    drop_timestamp_ns: int,
) -> RealtimeFrameRecord:
    scheduled_ms = relative_milliseconds(
        scheduled_timestamp_ns,
        context.origin_timestamp_ns,
    )
    enqueued_ms = relative_milliseconds(
        enqueued_timestamp_ns,
        context.origin_timestamp_ns,
    )
    drop_ms = relative_milliseconds(
        drop_timestamp_ns,
        context.origin_timestamp_ns,
    )

    return RealtimeFrameRecord(
        architecture=context.architecture,
        algorithm=context.algorithm,
        resolution=context.resolution,
        trial=context.trial,
        frame_index=frame_index,
        scheduled_timestamp_ms=scheduled_ms,
        enqueued_timestamp_ms=enqueued_ms,
        processing_start_timestamp_ms=None,
        processing_end_timestamp_ms=None,
        drop_timestamp_ms=drop_ms,
        source_delay_ms=(
            enqueued_ms - scheduled_ms
        ),
        queue_wait_time_ms=None,
        processing_time_ms=None,
        end_to_end_latency_ms=None,
        deadline_ms=context.deadline_ms,
        deadline_missed=None,
        frame_status=FrameStatus.DROPPED,
    )


def create_skipped_record(
    context: RealtimeRunContext,
    *,
    frame_index: int,
    scheduled_timestamp_ns: int,
) -> RealtimeFrameRecord:
    return RealtimeFrameRecord(
        architecture=context.architecture,
        algorithm=context.algorithm,
        resolution=context.resolution,
        trial=context.trial,
        frame_index=frame_index,
        scheduled_timestamp_ms=(
            relative_milliseconds(
                scheduled_timestamp_ns,
                context.origin_timestamp_ns,
            )
        ),
        enqueued_timestamp_ms=None,
        processing_start_timestamp_ms=None,
        processing_end_timestamp_ms=None,
        drop_timestamp_ms=None,
        source_delay_ms=None,
        queue_wait_time_ms=None,
        processing_time_ms=None,
        end_to_end_latency_ms=None,
        deadline_ms=context.deadline_ms,
        deadline_missed=None,
        frame_status=FrameStatus.SKIPPED,
    )


def format_optional_float(
    value: float | None,
) -> str:
    if value is None:
        return ""

    return f"{value:.6f}"


def write_frame_records(
    records: Iterable[RealtimeFrameRecord],
    output_path: Path,
) -> int:
    record_list = sorted(
        records,
        key=lambda record: (
            record.architecture,
            record.algorithm,
            record.resolution,
            record.trial,
            record.frame_index,
        ),
    )

    if not record_list:
        raise ValueError(
            "At least one frame record is required."
        )

    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "Frame results must be written "
            "to a CSV file."
        )

    record_keys = [
        (
            record.architecture,
            record.algorithm,
            record.resolution,
            record.trial,
            record.frame_index,
        )
        for record in record_list
    ]

    if len(record_keys) != len(set(record_keys)):
        raise ValueError(
            "Duplicate real-time frame records "
            "were detected."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.name}.tmp"
    )

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_FIELDNAMES,
        )
        writer.writeheader()

        for record in record_list:
            writer.writerow(
                record.to_csv_row()
            )

    temporary_path.replace(output_path)

    return len(record_list)