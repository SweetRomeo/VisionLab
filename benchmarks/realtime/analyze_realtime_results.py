import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from benchmarks.realtime.realtime_config import (
    RealtimeConfig,
    load_realtime_config,
)
from benchmarks.realtime.realtime_experiment_plan import (
    load_algorithms,
    load_benchmark_config,
    load_resolutions,
    validate_shared_execution_counts,
)
from benchmarks.realtime.realtime_records import (
    CSV_FIELDNAMES,
    FrameStatus,
    RealtimeFrameRecord,
)


ARCHITECTURES = (
    "pure_python",
    "hybrid",
    "pure_cpp",
)

TIMING_TOLERANCE_MS = 1e-3

SUMMARY_FIELDNAMES = [
    "architecture",
    "algorithm",
    "resolution",
    "trial_count",
    "frame_count",
    "processed_count",
    "dropped_count",
    "skipped_count",
    "processed_rate_percent",
    "drop_rate_percent",
    "deadline_miss_count",
    "deadline_miss_rate_percent",
    "on_time_rate_percent",
    "mean_source_delay_ms",
    "mean_queue_wait_time_ms",
    "mean_processing_time_ms",
    "p95_processing_time_ms",
    "mean_end_to_end_latency_ms",
    "p95_end_to_end_latency_ms",
    "effective_fps",
]


@dataclass(frozen=True)
class RealtimeSummary:
    architecture: str
    algorithm: str
    resolution: str
    trial_count: int
    frame_count: int
    processed_count: int
    dropped_count: int
    skipped_count: int
    processed_rate_percent: float
    drop_rate_percent: float
    deadline_miss_count: int
    deadline_miss_rate_percent: float
    on_time_rate_percent: float
    mean_source_delay_ms: float | None
    mean_queue_wait_time_ms: float | None
    mean_processing_time_ms: float | None
    p95_processing_time_ms: float | None
    mean_end_to_end_latency_ms: float | None
    p95_end_to_end_latency_ms: float | None
    effective_fps: float

    def to_csv_row(self) -> dict[str, str | int]:
        return {
            "architecture": self.architecture,
            "algorithm": self.algorithm,
            "resolution": self.resolution,
            "trial_count": self.trial_count,
            "frame_count": self.frame_count,
            "processed_count": self.processed_count,
            "dropped_count": self.dropped_count,
            "skipped_count": self.skipped_count,
            "processed_rate_percent": format_float(
                self.processed_rate_percent
            ),
            "drop_rate_percent": format_float(
                self.drop_rate_percent
            ),
            "deadline_miss_count": self.deadline_miss_count,
            "deadline_miss_rate_percent": format_float(
                self.deadline_miss_rate_percent
            ),
            "on_time_rate_percent": format_float(
                self.on_time_rate_percent
            ),
            "mean_source_delay_ms": format_optional_float(
                self.mean_source_delay_ms
            ),
            "mean_queue_wait_time_ms": format_optional_float(
                self.mean_queue_wait_time_ms
            ),
            "mean_processing_time_ms": format_optional_float(
                self.mean_processing_time_ms
            ),
            "p95_processing_time_ms": format_optional_float(
                self.p95_processing_time_ms
            ),
            "mean_end_to_end_latency_ms": format_optional_float(
                self.mean_end_to_end_latency_ms
            ),
            "p95_end_to_end_latency_ms": format_optional_float(
                self.p95_end_to_end_latency_ms
            ),
            "effective_fps": format_float(
                self.effective_fps
            ),
        }


def parse_integer(
    value: str | None,
    field_name: str,
) -> int:
    if value is None or not value.strip():
        raise ValueError(
            f"{field_name} must contain an integer."
        )

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise ValueError(
            f"{field_name} must contain an integer."
        ) from error

    return parsed_value


def parse_required_float(
    value: str | None,
    field_name: str,
) -> float:
    parsed_value = parse_optional_float(
        value,
        field_name,
    )

    if parsed_value is None:
        raise ValueError(
            f"{field_name} must contain a number."
        )

    return parsed_value


def parse_optional_float(
    value: str | None,
    field_name: str,
) -> float | None:
    if value is None or not value.strip():
        return None

    try:
        parsed_value = float(value)
    except ValueError as error:
        raise ValueError(
            f"{field_name} must contain a number."
        ) from error

    if not math.isfinite(parsed_value):
        raise ValueError(
            f"{field_name} must contain a finite number."
        )

    return parsed_value


def parse_optional_boolean(
    value: str | None,
    field_name: str,
) -> bool | None:
    if value is None or not value.strip():
        return None

    normalized_value = value.strip().lower()

    if normalized_value == "true":
        return True

    if normalized_value == "false":
        return False

    raise ValueError(
        f"{field_name} must contain true, false or an empty value."
    )


def parse_frame_status(value: str | None) -> FrameStatus:
    if value is None:
        raise ValueError(
            "frame_status must contain a supported value."
        )

    try:
        return FrameStatus(value.strip())
    except ValueError as error:
        raise ValueError(
            f"Unsupported frame_status: {value}"
        ) from error


def frame_record_from_row(
    row: dict[str, str],
) -> RealtimeFrameRecord:
    return RealtimeFrameRecord(
        architecture=row["architecture"],
        algorithm=row["algorithm"],
        resolution=row["resolution"],
        trial=parse_integer(
            row.get("trial"),
            "trial",
        ),
        frame_index=parse_integer(
            row.get("frame_index"),
            "frame_index",
        ),
        scheduled_timestamp_ms=parse_required_float(
            row.get("scheduled_timestamp_ms"),
            "scheduled_timestamp_ms",
        ),
        enqueued_timestamp_ms=parse_optional_float(
            row.get("enqueued_timestamp_ms"),
            "enqueued_timestamp_ms",
        ),
        processing_start_timestamp_ms=parse_optional_float(
            row.get("processing_start_timestamp_ms"),
            "processing_start_timestamp_ms",
        ),
        processing_end_timestamp_ms=parse_optional_float(
            row.get("processing_end_timestamp_ms"),
            "processing_end_timestamp_ms",
        ),
        drop_timestamp_ms=parse_optional_float(
            row.get("drop_timestamp_ms"),
            "drop_timestamp_ms",
        ),
        source_delay_ms=parse_optional_float(
            row.get("source_delay_ms"),
            "source_delay_ms",
        ),
        queue_wait_time_ms=parse_optional_float(
            row.get("queue_wait_time_ms"),
            "queue_wait_time_ms",
        ),
        processing_time_ms=parse_optional_float(
            row.get("processing_time_ms"),
            "processing_time_ms",
        ),
        end_to_end_latency_ms=parse_optional_float(
            row.get("end_to_end_latency_ms"),
            "end_to_end_latency_ms",
        ),
        deadline_ms=parse_required_float(
            row.get("deadline_ms"),
            "deadline_ms",
        ),
        deadline_missed=parse_optional_boolean(
            row.get("deadline_missed"),
            "deadline_missed",
        ),
        frame_status=parse_frame_status(
            row.get("frame_status")
        ),
    )


def load_frame_records(
    result_path: Path,
    expected_architecture: str,
) -> list[RealtimeFrameRecord]:
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Required real-time result file was not found: "
            f"{result_path}"
        )

    records = []

    with result_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as result_file:
        reader = csv.DictReader(result_file)

        if reader.fieldnames != CSV_FIELDNAMES:
            raise ValueError(
                "Unexpected real-time CSV header in "
                f"{result_path}."
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                record = frame_record_from_row(row)
            except (KeyError, ValueError) as error:
                raise ValueError(
                    "Invalid real-time frame record in "
                    f"{result_path} at row {row_number}: "
                    f"{error}"
                ) from error

            if record.architecture != expected_architecture:
                raise ValueError(
                    "Incorrect architecture label in "
                    f"{result_path} at row {row_number}: "
                    f"expected {expected_architecture}, "
                    f"received {record.architecture}."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No real-time frame records were found: {result_path}"
        )

    return records


def validate_deadlines(
    records_by_architecture: dict[
        str,
        list[RealtimeFrameRecord],
    ],
    realtime_config: RealtimeConfig,
) -> None:
    expected_deadline = realtime_config.deadline_ms

    for architecture, records in (
        records_by_architecture.items()
    ):
        for record in records:
            if not math.isclose(
                record.deadline_ms,
                expected_deadline,
                rel_tol=0.0,
                abs_tol=1e-3,
            ):
                raise ValueError(
                    "Real-time deadline does not match the "
                    "active configuration. "
                    f"Architecture: {architecture}; "
                    f"received: {record.deadline_ms}; "
                    f"expected: {expected_deadline}."
                )


def validate_measurement_consistency(
    records_by_architecture: dict[
        str,
        list[RealtimeFrameRecord],
    ],
) -> None:
    def require_close(
        actual: float,
        expected: float,
        field_name: str,
        record: RealtimeFrameRecord,
    ) -> None:
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=TIMING_TOLERANCE_MS,
        ):
            raise ValueError(
                "Inconsistent real-time timing field: "
                f"{field_name}. Architecture: "
                f"{record.architecture}; algorithm: "
                f"{record.algorithm}; resolution: "
                f"{record.resolution}; trial: "
                f"{record.trial}; frame: "
                f"{record.frame_index}."
            )

    for records in records_by_architecture.values():
        for record in records:
            if record.enqueued_timestamp_ms is not None:
                if record.source_delay_ms is None:
                    raise ValueError(
                        "An enqueued frame requires "
                        "source_delay_ms."
                    )

                require_close(
                    record.source_delay_ms,
                    record.enqueued_timestamp_ms
                    - record.scheduled_timestamp_ms,
                    "source_delay_ms",
                    record,
                )

            if record.frame_status is not FrameStatus.PROCESSED:
                continue

            processed_values = (
                record.enqueued_timestamp_ms,
                record.processing_start_timestamp_ms,
                record.processing_end_timestamp_ms,
                record.queue_wait_time_ms,
                record.processing_time_ms,
                record.end_to_end_latency_ms,
                record.deadline_missed,
            )

            if any(
                value is None
                for value in processed_values
            ):
                raise ValueError(
                    "Processed frame timing fields "
                    "cannot be empty."
                )

            enqueued_timestamp_ms = (
                record.enqueued_timestamp_ms
            )
            processing_start_timestamp_ms = (
                record.processing_start_timestamp_ms
            )
            processing_end_timestamp_ms = (
                record.processing_end_timestamp_ms
            )
            queue_wait_time_ms = (
                record.queue_wait_time_ms
            )
            processing_time_ms = (
                record.processing_time_ms
            )
            end_to_end_latency_ms = (
                record.end_to_end_latency_ms
            )

            assert enqueued_timestamp_ms is not None
            assert processing_start_timestamp_ms is not None
            assert processing_end_timestamp_ms is not None
            assert queue_wait_time_ms is not None
            assert processing_time_ms is not None
            assert end_to_end_latency_ms is not None

            require_close(
                queue_wait_time_ms,
                processing_start_timestamp_ms
                - enqueued_timestamp_ms,
                "queue_wait_time_ms",
                record,
            )
            require_close(
                processing_time_ms,
                processing_end_timestamp_ms
                - processing_start_timestamp_ms,
                "processing_time_ms",
                record,
            )
            require_close(
                end_to_end_latency_ms,
                processing_end_timestamp_ms
                - record.scheduled_timestamp_ms,
                "end_to_end_latency_ms",
                record,
            )

            deadline_difference = (
                end_to_end_latency_ms
                - record.deadline_ms
            )

            if (
                abs(deadline_difference)
                > TIMING_TOLERANCE_MS
                and record.deadline_missed
                != (deadline_difference > 0.0)
            ):
                raise ValueError(
                    "deadline_missed is inconsistent with "
                    "end_to_end_latency_ms. Architecture: "
                    f"{record.architecture}; algorithm: "
                    f"{record.algorithm}; resolution: "
                    f"{record.resolution}; trial: "
                    f"{record.trial}; frame: "
                    f"{record.frame_index}."
                )


def validate_experiment_coverage(
    records_by_architecture: dict[
        str,
        list[RealtimeFrameRecord],
    ],
    *,
    algorithm_names: list[str],
    resolution_names: list[str],
    trial_count: int,
    measured_frames: int,
) -> None:
    if set(records_by_architecture) != set(
        ARCHITECTURES
    ):
        raise ValueError(
            "Real-time results must contain exactly the "
            f"architectures: {list(ARCHITECTURES)}."
        )

    expected_keys = {
        (
            algorithm,
            resolution,
            trial,
            frame_index,
        )
        for algorithm in algorithm_names
        for resolution in resolution_names
        for trial in range(1, trial_count + 1)
        for frame_index in range(1, measured_frames + 1)
    }

    for architecture in ARCHITECTURES:
        records = records_by_architecture[architecture]
        observed_keys = [
            (
                record.algorithm,
                record.resolution,
                record.trial,
                record.frame_index,
            )
            for record in records
        ]
        observed_key_counts = Counter(observed_keys)
        observed_key_set = set(observed_key_counts)

        duplicate_keys = sorted(
            key
            for key, count
            in observed_key_counts.items()
            if count > 1
        )
        missing_keys = sorted(
            expected_keys - observed_key_set
        )
        unexpected_keys = sorted(
            observed_key_set - expected_keys
        )

        if (
            duplicate_keys
            or missing_keys
            or unexpected_keys
            or len(observed_keys) != len(expected_keys)
        ):
            raise ValueError(
                "Real-time experiment coverage validation "
                f"failed for {architecture}. "
                f"Missing: {missing_keys[:5]}; "
                f"duplicates: {duplicate_keys[:5]}; "
                f"unexpected: {unexpected_keys[:5]}."
            )


def calculate_percentile(
    values: list[float],
    fraction: float,
) -> float:
    if not values:
        raise ValueError(
            "At least one value is required for a percentile."
        )

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "Percentile fraction must be between zero and one."
        )

    ordered_values = sorted(values)

    if len(ordered_values) == 1:
        return ordered_values[0]

    position = (len(ordered_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered_values[lower_index]

    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]
    weight = position - lower_index

    return lower_value + (
        upper_value - lower_value
    ) * weight


def optional_mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return fmean(values)


def optional_percentile(
    values: list[float],
    fraction: float,
) -> float | None:
    if not values:
        return None

    return calculate_percentile(
        values,
        fraction,
    )


def build_realtime_summaries(
    records_by_architecture: dict[
        str,
        list[RealtimeFrameRecord],
    ],
    *,
    algorithm_names: list[str],
    resolution_names: list[str],
    target_fps: float,
) -> list[RealtimeSummary]:
    groups = defaultdict(list)

    for architecture, records in (
        records_by_architecture.items()
    ):
        for record in records:
            groups[
                (
                    architecture,
                    record.algorithm,
                    record.resolution,
                )
            ].append(record)

    summaries = []

    for architecture in ARCHITECTURES:
        for algorithm in algorithm_names:
            for resolution in resolution_names:
                key = (
                    architecture,
                    algorithm,
                    resolution,
                )
                records = groups.get(key, [])

                if not records:
                    raise ValueError(
                        "No real-time records were found for "
                        f"{key}."
                    )

                processed_records = [
                    record
                    for record in records
                    if record.frame_status
                    is FrameStatus.PROCESSED
                ]
                dropped_records = [
                    record
                    for record in records
                    if record.frame_status
                    is FrameStatus.DROPPED
                ]
                skipped_records = [
                    record
                    for record in records
                    if record.frame_status
                    is FrameStatus.SKIPPED
                ]

                frame_count = len(records)
                processed_count = len(
                    processed_records
                )
                dropped_count = len(
                    dropped_records
                )
                skipped_count = len(
                    skipped_records
                )
                deadline_miss_count = sum(
                    record.deadline_missed is True
                    for record in processed_records
                )
                on_time_count = (
                    processed_count
                    - deadline_miss_count
                )

                source_delay_values = [
                    record.source_delay_ms
                    for record in records
                    if record.source_delay_ms is not None
                ]
                queue_wait_values = [
                    record.queue_wait_time_ms
                    for record in processed_records
                    if record.queue_wait_time_ms
                    is not None
                ]
                processing_values = [
                    record.processing_time_ms
                    for record in processed_records
                    if record.processing_time_ms
                    is not None
                ]
                end_to_end_values = [
                    record.end_to_end_latency_ms
                    for record in processed_records
                    if record.end_to_end_latency_ms
                    is not None
                ]

                summaries.append(
                    RealtimeSummary(
                        architecture=architecture,
                        algorithm=algorithm,
                        resolution=resolution,
                        trial_count=len(
                            {
                                record.trial
                                for record in records
                            }
                        ),
                        frame_count=frame_count,
                        processed_count=processed_count,
                        dropped_count=dropped_count,
                        skipped_count=skipped_count,
                        processed_rate_percent=(
                            100.0
                            * processed_count
                            / frame_count
                        ),
                        drop_rate_percent=(
                            100.0
                            * dropped_count
                            / frame_count
                        ),
                        deadline_miss_count=(
                            deadline_miss_count
                        ),
                        deadline_miss_rate_percent=(
                            100.0
                            * deadline_miss_count
                            / processed_count
                            if processed_count
                            else 0.0
                        ),
                        on_time_rate_percent=(
                            100.0
                            * on_time_count
                            / frame_count
                        ),
                        mean_source_delay_ms=(
                            optional_mean(
                                source_delay_values
                            )
                        ),
                        mean_queue_wait_time_ms=(
                            optional_mean(
                                queue_wait_values
                            )
                        ),
                        mean_processing_time_ms=(
                            optional_mean(
                                processing_values
                            )
                        ),
                        p95_processing_time_ms=(
                            optional_percentile(
                                processing_values,
                                0.95,
                            )
                        ),
                        mean_end_to_end_latency_ms=(
                            optional_mean(
                                end_to_end_values
                            )
                        ),
                        p95_end_to_end_latency_ms=(
                            optional_percentile(
                                end_to_end_values,
                                0.95,
                            )
                        ),
                        effective_fps=(
                            target_fps
                            * processed_count
                            / frame_count
                        ),
                    )
                )

    return summaries


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_optional_float(
    value: float | None,
) -> str:
    if value is None:
        return ""

    return format_float(value)


def write_realtime_summaries(
    summaries: list[RealtimeSummary],
    output_path: Path,
) -> Path:
    if not summaries:
        raise ValueError(
            "At least one real-time summary is required."
        )

    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "Real-time summary output must be a CSV file."
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
            fieldnames=SUMMARY_FIELDNAMES,
        )
        writer.writeheader()

        for summary in summaries:
            writer.writerow(
                summary.to_csv_row()
            )

    temporary_path.replace(output_path)
    return output_path


def main() -> None:
    benchmark_config = load_benchmark_config()
    realtime_config = load_realtime_config()

    validate_shared_execution_counts(
        benchmark_config,
        realtime_config,
    )

    algorithm_names = [
        algorithm["name"]
        for algorithm in load_algorithms(
            benchmark_config
        )
    ]
    resolution_names = [
        f"{width}x{height}"
        for width, height in load_resolutions(
            benchmark_config
        )
    ]

    records_by_architecture = {
        architecture: load_frame_records(
            realtime_config.output_directory
            / architecture
            / realtime_config.frame_results_file,
            architecture,
        )
        for architecture in ARCHITECTURES
    }

    validate_deadlines(
        records_by_architecture,
        realtime_config,
    )
    validate_measurement_consistency(
        records_by_architecture
    )
    validate_experiment_coverage(
        records_by_architecture,
        algorithm_names=algorithm_names,
        resolution_names=resolution_names,
        trial_count=realtime_config.trial_count,
        measured_frames=(
            realtime_config.measured_frames
        ),
    )

    summaries = build_realtime_summaries(
        records_by_architecture,
        algorithm_names=algorithm_names,
        resolution_names=resolution_names,
        target_fps=realtime_config.target_fps,
    )
    output_path = write_realtime_summaries(
        summaries,
        realtime_config.summary_path,
    )

    total_record_count = sum(
        len(records)
        for records in records_by_architecture.values()
    )
    total_processed_count = sum(
        summary.processed_count
        for summary in summaries
    )
    total_dropped_count = sum(
        summary.dropped_count
        for summary in summaries
    )
    total_skipped_count = sum(
        summary.skipped_count
        for summary in summaries
    )

    print("Real-time result validation passed.")
    print(f"Frame records: {total_record_count}")
    print(f"Processed: {total_processed_count}")
    print(f"Dropped: {total_dropped_count}")
    print(f"Skipped: {total_skipped_count}")
    print(f"Summary rows: {len(summaries)}")
    print(f"Real-time summary created: {output_path}")


if __name__ == "__main__":
    main()