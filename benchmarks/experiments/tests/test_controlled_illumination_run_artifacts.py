from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.experiments.controlled_illumination_run_artifacts import (
    ControlledIlluminationArtifactError,
    ControlledIlluminationFrameRecord,
    calculate_file_sha256,
    validate_frame_records,
    write_completed_run_artifacts_atomic,
)
from benchmarks.experiments.controlled_illumination_runner_context import (
    load_runner_context_from_environment,
)


STARTED_AT = "2026-08-24T10:00:00Z"
FINISHED_AT = "2026-08-24T10:01:00Z"


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
            "VISIONLAB_FRAME_DEADLINE_MS": (
                "33.333333"
            ),
            "VISIONLAB_RESULTS_ROOT": str(
                results_root
            ),
        }

        return load_runner_context_from_environment(
            environment,
            expected_architecture="pure_python",
        )

    def create_processed_record(
        self,
        frame_index: int = 0,
        *,
        captured_at_ms: float = 0.0,
        processing_started_at_ms: float = 1.0,
        processing_finished_at_ms: float = 4.0,
        processing_time_ms: float = 3.0,
        end_to_end_latency_ms: float = 4.0,
        frame_deadline_ms: float = 33.333333,
        deadline_met: bool = True,
    ) -> ControlledIlluminationFrameRecord:
        return ControlledIlluminationFrameRecord(
            frame_index=frame_index,
            outcome="processed",
            captured_at_ms=captured_at_ms,
            processing_started_at_ms=(
                processing_started_at_ms
            ),
            processing_finished_at_ms=(
                processing_finished_at_ms
            ),
            processing_time_ms=processing_time_ms,
            end_to_end_latency_ms=(
                end_to_end_latency_ms
            ),
            frame_deadline_ms=frame_deadline_ms,
            deadline_met=deadline_met,
        )

    def create_dropped_record(
        self,
        frame_index: int,
    ) -> ControlledIlluminationFrameRecord:
        return ControlledIlluminationFrameRecord(
            frame_index=frame_index,
            outcome="dropped",
            captured_at_ms=66.0,
            processing_started_at_ms=None,
            processing_finished_at_ms=None,
            processing_time_ms=None,
            end_to_end_latency_ms=None,
            frame_deadline_ms=33.333333,
            deadline_met=False,
            reason="queue_capacity_exceeded",
        )

    def create_skipped_record(
        self,
        frame_index: int,
    ) -> ControlledIlluminationFrameRecord:
        return ControlledIlluminationFrameRecord(
            frame_index=frame_index,
            outcome="skipped",
            captured_at_ms=99.0,
            processing_started_at_ms=None,
            processing_finished_at_ms=None,
            processing_time_ms=None,
            end_to_end_latency_ms=None,
            frame_deadline_ms=33.333333,
            deadline_met=False,
            reason="frame_unavailable",
        )

    def test_valid_processed_record(
        self,
    ) -> None:
        record = self.create_processed_record()

        self.assertEqual(
            record.outcome,
            "processed",
        )
        self.assertTrue(record.deadline_met)

    def test_valid_dropped_record(
        self,
    ) -> None:
        record = self.create_dropped_record(0)

        self.assertEqual(
            record.reason,
            "queue_capacity_exceeded",
        )
        self.assertFalse(record.deadline_met)

    def test_processed_record_requires_timings(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationArtifactError,
            "require timing",
        ):
            ControlledIlluminationFrameRecord(
                frame_index=0,
                outcome="processed",
                captured_at_ms=0.0,
                processing_started_at_ms=None,
                processing_finished_at_ms=None,
                processing_time_ms=None,
                end_to_end_latency_ms=None,
                frame_deadline_ms=33.333333,
                deadline_met=False,
            )

    def test_processing_duration_mismatch_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationArtifactError,
            "does not match",
        ):
            self.create_processed_record(
                processing_time_ms=10.0,
            )

    def test_deadline_mismatch_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationArtifactError,
            "deadline_met",
        ):
            self.create_processed_record(
                processing_finished_at_ms=41.0,
                processing_time_ms=40.0,
                end_to_end_latency_ms=41.0,
                deadline_met=True,
            )

    def test_dropped_record_rejects_timings(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationArtifactError,
            "must not contain processing timings",
        ):
            ControlledIlluminationFrameRecord(
                frame_index=0,
                outcome="dropped",
                captured_at_ms=0.0,
                processing_started_at_ms=1.0,
                processing_finished_at_ms=None,
                processing_time_ms=None,
                end_to_end_latency_ms=None,
                frame_deadline_ms=33.333333,
                deadline_met=False,
                reason="queue_capacity_exceeded",
            )

    def test_dropped_record_requires_reason(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationArtifactError,
            "require a reason",
        ):
            ControlledIlluminationFrameRecord(
                frame_index=0,
                outcome="dropped",
                captured_at_ms=0.0,
                processing_started_at_ms=None,
                processing_finished_at_ms=None,
                processing_time_ms=None,
                end_to_end_latency_ms=None,
                frame_deadline_ms=33.333333,
                deadline_met=False,
            )

    def test_non_sequential_indices_are_rejected(
        self,
    ) -> None:
        records = (
            self.create_processed_record(
                frame_index=0
            ),
            self.create_dropped_record(
                frame_index=2
            ),
        )

        with self.assertRaisesRegex(
            ControlledIlluminationArtifactError,
            "sequential",
        ):
            validate_frame_records(records)

    def test_completed_artifacts_are_written(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            context = self.create_context(
                Path(temporary)
            )

            records = (
                self.create_processed_record(
                    frame_index=0,
                ),
                self.create_processed_record(
                    frame_index=1,
                    captured_at_ms=40.0,
                    processing_started_at_ms=41.0,
                    processing_finished_at_ms=81.0,
                    processing_time_ms=40.0,
                    end_to_end_latency_ms=41.0,
                    deadline_met=False,
                ),
                self.create_dropped_record(
                    frame_index=2
                ),
                self.create_skipped_record(
                    frame_index=3
                ),
            )

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
            records = (
                self.create_processed_record(),
            )

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
                    (
                        self.create_processed_record(),
                    ),
                    started_at_utc=FINISHED_AT,
                    finished_at_utc=STARTED_AT,
                    warmup_frame_count=30,
                )


if __name__ == "__main__":
    unittest.main()