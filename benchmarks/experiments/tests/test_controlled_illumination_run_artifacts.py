from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.experiments.controlled_illumination_run_artifacts import (
    ControlledIlluminationArtifactError,
    calculate_file_sha256,
    validate_frame_records,
    write_completed_run_artifacts_atomic,
)
from benchmarks.experiments.controlled_illumination_runner_context import (
    load_runner_context_from_environment,
)
from benchmarks.realtime.realtime_records import (
    RealtimeRunContext,
    create_dropped_record,
    create_processed_record,
    create_skipped_record,
)


STARTED_AT = "2026-08-24T10:00:00Z"
FINISHED_AT = "2026-08-24T10:01:00Z"
ORIGIN_TIMESTAMP_NS = 1_000_000_000
DEADLINE_MS = 33.333333


class ControlledIlluminationRunArtifactTests(
    unittest.TestCase
):
    def create_context(
        self,
        results_root: Path,
    ):
        environment = {
            "VISIONLAB_EXPERIMENT_ID": (
                "controlled-illumination-test"
            ),
            "VISIONLAB_RUN_ID": "test-run-0001",
            "VISIONLAB_EXECUTION_ORDER": "1",
            "VISIONLAB_PHASE": "constant_lux",
            "VISIONLAB_PLATFORM": "desktop",
            "VISIONLAB_ARCHITECTURE": (
                "pure_python"
            ),
            "VISIONLAB_ALGORITHM": "original",
            "VISIONLAB_RESOLUTION_WIDTH": "640",
            "VISIONLAB_RESOLUTION_HEIGHT": "480",
            "VISIONLAB_TRIAL_NUMBER": "1",
            (
                "VISIONLAB_INCIDENCE_"
                "ANGLE_DEGREES"
            ): "0",
            "VISIONLAB_TARGET_ILLUMINANCE_LUX": (
                "50"
            ),
            "VISIONLAB_SOURCE_OUTPUT_SETTING": "",
            "VISIONLAB_TARGET_FPS": "30",
            "VISIONLAB_FRAME_DEADLINE_MS": str(
                DEADLINE_MS
            ),
            "VISIONLAB_RESULTS_ROOT": str(
                results_root
            ),
        }

        return load_runner_context_from_environment(
            environment,
            expected_architecture="pure_python",
        )

    def create_realtime_context(
        self,
        *,
        architecture: str = "pure_python",
        algorithm: str = "original",
        resolution: str = "640x480",
        trial: int = 1,
        deadline_ms: float = DEADLINE_MS,
    ) -> RealtimeRunContext:
        return RealtimeRunContext(
            architecture=architecture,
            algorithm=algorithm,
            resolution=resolution,
            trial=trial,
            origin_timestamp_ns=(
                ORIGIN_TIMESTAMP_NS
            ),
            deadline_ms=deadline_ms,
        )

    def create_records(
        self,
        realtime_context: (
            RealtimeRunContext | None
        ) = None,
    ):
        active_context = (
            realtime_context
            if realtime_context is not None
            else self.create_realtime_context()
        )

        return (
            create_processed_record(
                active_context,
                frame_index=1,
                scheduled_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                ),
                enqueued_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                ),
                processing_start_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 1_000_000
                ),
                processing_end_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 4_000_000
                ),
            ),
            create_processed_record(
                active_context,
                frame_index=2,
                scheduled_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 40_000_000
                ),
                enqueued_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 40_000_000
                ),
                processing_start_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 41_000_000
                ),
                processing_end_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 81_000_000
                ),
            ),
            create_dropped_record(
                active_context,
                frame_index=3,
                scheduled_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 80_000_000
                ),
                enqueued_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 81_000_000
                ),
                drop_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 82_000_000
                ),
            ),
            create_skipped_record(
                active_context,
                frame_index=4,
                scheduled_timestamp_ns=(
                    ORIGIN_TIMESTAMP_NS
                    + 120_000_000
                ),
            ),
        )

    def test_valid_records_are_accepted(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )
            records = self.create_records()

            validated_records = (
                validate_frame_records(
                    context,
                    records,
                )
            )

        self.assertEqual(
            validated_records,
            records,
        )

    def test_non_sequential_indices_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )
            records = self.create_records()

            with self.assertRaisesRegex(
                ControlledIlluminationArtifactError,
                "sequential",
            ):
                validate_frame_records(
                    context,
                    (
                        records[0],
                        records[2],
                    ),
                )

    def test_architecture_mismatch_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )
            records = self.create_records(
                self.create_realtime_context(
                    architecture="hybrid"
                )
            )

            with self.assertRaisesRegex(
                ControlledIlluminationArtifactError,
                "architecture",
            ):
                validate_frame_records(
                    context,
                    records,
                )

    def test_algorithm_mismatch_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )
            records = self.create_records(
                self.create_realtime_context(
                    algorithm="clahe"
                )
            )

            with self.assertRaisesRegex(
                ControlledIlluminationArtifactError,
                "algorithm",
            ):
                validate_frame_records(
                    context,
                    records,
                )

    def test_resolution_mismatch_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )
            records = self.create_records(
                self.create_realtime_context(
                    resolution="1280x720"
                )
            )

            with self.assertRaisesRegex(
                ControlledIlluminationArtifactError,
                "resolution",
            ):
                validate_frame_records(
                    context,
                    records,
                )

    def test_trial_mismatch_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )
            records = self.create_records(
                self.create_realtime_context(
                    trial=2
                )
            )

            with self.assertRaisesRegex(
                ControlledIlluminationArtifactError,
                "trial",
            ):
                validate_frame_records(
                    context,
                    records,
                )

    def test_deadline_mismatch_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )
            records = self.create_records(
                self.create_realtime_context(
                    deadline_ms=20.0
                )
            )

            with self.assertRaisesRegex(
                ControlledIlluminationArtifactError,
                "deadline",
            ):
                validate_frame_records(
                    context,
                    records,
                )

    def test_completed_artifacts_are_written(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )
            records = self.create_records()

            csv_path, summary_path = (
                write_completed_run_artifacts_atomic(
                    context,
                    records,
                    started_at_utc=STARTED_AT,
                    finished_at_utc=FINISHED_AT,
                    warmup_frame_count=30,
                )
            )

            with csv_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                csv_rows = list(
                    csv.DictReader(csv_file)
                )

            with summary_path.open(
                "r",
                encoding="utf-8",
            ) as summary_file:
                summary = json.load(
                    summary_file
                )

            self.assertEqual(len(csv_rows), 4)
            self.assertEqual(
                summary["status"],
                "completed",
            )
            self.assertEqual(
                summary["measured_frame_count"],
                4,
            )
            self.assertEqual(
                summary["processed_frame_count"],
                2,
            )
            self.assertEqual(
                summary["dropped_frame_count"],
                1,
            )
            self.assertEqual(
                summary["skipped_frame_count"],
                1,
            )
            self.assertEqual(
                summary["deadline_met_count"],
                1,
            )
            self.assertEqual(
                summary["deadline_miss_count"],
                1,
            )
            self.assertAlmostEqual(
                summary[
                    "mean_processing_time_ms"
                ],
                21.5,
            )
            self.assertAlmostEqual(
                summary[
                    "mean_end_to_end_latency_ms"
                ],
                22.5,
            )
            self.assertEqual(
                summary["frame_results_sha256"],
                calculate_file_sha256(
                    csv_path
                ),
            )
            self.assertEqual(
                list(
                    context.output_directory.glob(
                        ".*.tmp"
                    )
                ),
                [],
            )

    def test_existing_completed_artifacts_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )
            records = self.create_records()

            write_completed_run_artifacts_atomic(
                context,
                records,
                started_at_utc=STARTED_AT,
                finished_at_utc=FINISHED_AT,
                warmup_frame_count=30,
            )

            with self.assertRaisesRegex(
                ControlledIlluminationArtifactError,
                "already exist",
            ):
                write_completed_run_artifacts_atomic(
                    context,
                    records,
                    started_at_utc=STARTED_AT,
                    finished_at_utc=FINISHED_AT,
                    warmup_frame_count=30,
                )

    def test_invalid_timestamp_order_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )

            with self.assertRaisesRegex(
                ControlledIlluminationArtifactError,
                "cannot be earlier",
            ):
                write_completed_run_artifacts_atomic(
                    context,
                    self.create_records(),
                    started_at_utc=FINISHED_AT,
                    finished_at_utc=STARTED_AT,
                    warmup_frame_count=30,
                )

    def test_invalid_warmup_count_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )

            with self.assertRaisesRegex(
                ControlledIlluminationArtifactError,
                "warmup_frame_count",
            ):
                write_completed_run_artifacts_atomic(
                    context,
                    self.create_records(),
                    started_at_utc=STARTED_AT,
                    finished_at_utc=FINISHED_AT,
                    warmup_frame_count=-1,
                )


if __name__ == "__main__":
    unittest.main()