from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlan,
    PlannedRun,
)
from benchmarks.experiments.controlled_illumination_run_state import (
    ControlledIlluminationProgress,
    ControlledIlluminationRunStateError,
    RunState,
    RunStatus,
    calculate_run_plan_sha256,
    initialize_run_progress,
    transition_progress_run,
    transition_run_state,
    load_or_initialize_run_progress,
    load_run_progress,
    progress_from_dict,
    save_run_progress_atomic,
)


STARTED_AT = "2026-08-23T10:00:00Z"
FINISHED_AT = "2026-08-23T10:05:00Z"


class ControlledIlluminationRunStateTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.planned_state = RunState(
            run_id="experiment-run-0001",
            execution_order=1,
        )

    def start_run(self) -> RunState:
        return transition_run_state(
            self.planned_state,
            RunStatus.RUNNING,
            STARTED_AT,
        )

    def test_planned_state_defaults(
        self,
    ) -> None:
        self.assertEqual(
            self.planned_state.status,
            RunStatus.PLANNED,
        )
        self.assertEqual(
            self.planned_state.attempt_count,
            0,
        )
        self.assertIsNone(
            self.planned_state.started_at_utc
        )

    def test_string_status_is_normalized(
        self,
    ) -> None:
        state = RunState(
            run_id="experiment-run-0001",
            execution_order=1,
            status="planned",  # type: ignore[arg-type]
        )

        self.assertEqual(
            state.status,
            RunStatus.PLANNED,
        )

    def test_planned_run_can_start(
        self,
    ) -> None:
        running = self.start_run()

        self.assertEqual(
            running.status,
            RunStatus.RUNNING,
        )
        self.assertEqual(running.attempt_count, 1)
        self.assertEqual(
            running.started_at_utc,
            STARTED_AT,
        )

    def test_running_run_can_complete(
        self,
    ) -> None:
        completed = transition_run_state(
            self.start_run(),
            RunStatus.COMPLETED,
            FINISHED_AT,
        )

        self.assertEqual(
            completed.status,
            RunStatus.COMPLETED,
        )
        self.assertEqual(
            completed.finished_at_utc,
            FINISHED_AT,
        )

    def test_running_run_can_fail(
        self,
    ) -> None:
        failed = transition_run_state(
            self.start_run(),
            RunStatus.FAILED,
            FINISHED_AT,
            reason="Camera disconnected.",
        )

        self.assertEqual(
            failed.status,
            RunStatus.FAILED,
        )
        self.assertEqual(
            failed.failure_reason,
            "Camera disconnected.",
        )

    def test_failed_transition_requires_reason(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunStateError
        ):
            transition_run_state(
                self.start_run(),
                RunStatus.FAILED,
                FINISHED_AT,
            )

    def test_planned_run_can_be_skipped(
        self,
    ) -> None:
        skipped = transition_run_state(
            self.planned_state,
            RunStatus.SKIPPED,
            FINISHED_AT,
            reason="Platform unavailable.",
        )

        self.assertEqual(
            skipped.status,
            RunStatus.SKIPPED,
        )
        self.assertEqual(
            skipped.skip_reason,
            "Platform unavailable.",
        )

    def test_skipped_transition_requires_reason(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunStateError
        ):
            transition_run_state(
                self.planned_state,
                RunStatus.SKIPPED,
                FINISHED_AT,
            )

    def test_invalid_transition_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "Invalid run-state transition",
        ):
            transition_run_state(
                self.planned_state,
                RunStatus.COMPLETED,
                FINISHED_AT,
            )

    def test_reason_is_rejected_when_not_allowed(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "reason is only allowed",
        ):
            transition_run_state(
                self.planned_state,
                RunStatus.RUNNING,
                STARTED_AT,
                reason="Unexpected reason.",
            )

    def test_failed_run_can_be_replanned(
        self,
    ) -> None:
        failed = transition_run_state(
            self.start_run(),
            RunStatus.FAILED,
            FINISHED_AT,
            reason="Temporary camera error.",
        )
        replanned = transition_run_state(
            failed,
            RunStatus.PLANNED,
            "2026-08-23T10:06:00Z",
        )
        restarted = transition_run_state(
            replanned,
            RunStatus.RUNNING,
            "2026-08-23T10:07:00Z",
        )

        self.assertEqual(
            replanned.status,
            RunStatus.PLANNED,
        )
        self.assertIsNone(
            replanned.failure_reason
        )
        self.assertEqual(
            restarted.attempt_count,
            2,
        )

    def test_invalid_timestamp_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunStateError
        ):
            transition_run_state(
                self.planned_state,
                RunStatus.RUNNING,
                "invalid-timestamp",
            )

    def test_inconsistent_running_state_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "Running run fields are inconsistent",
        ):
            RunState(
                run_id="experiment-run-0001",
                execution_order=1,
                status=RunStatus.RUNNING,
                attempt_count=1,
                started_at_utc=STARTED_AT,
                finished_at_utc=FINISHED_AT,
            )

    def build_run_plan(
        self,
    ) -> ControlledIlluminationRunPlan:
        planned_runs = tuple(
            PlannedRun(
                execution_order=execution_order,
                experiment_id="experiment-progress",
                run_id=run_id,
                phase="constant_lux",
                platform="desktop",
                architecture="pure_python",
                algorithm="original",
                resolution=ResolutionMetadata(
                    width=640,
                    height=480,
                ),
                trial_number=execution_order,
                incidence_angle_degrees=0.0,
                target_illuminance_lux=50.0,
                source_output_setting=None,
                target_fps=30.0,
                frame_deadline_ms=1000.0 / 30.0,
            )
            for execution_order, run_id in (
                (1, "experiment-progress-run-0001"),
                (2, "experiment-progress-run-0002"),
            )
        )

        return ControlledIlluminationRunPlan(
            schema_version=1,
            generated_at_utc=(
                "2026-08-23T09:00:00Z"
            ),
            experiment_id="experiment-progress",
            randomized=False,
            randomization_seed=20260821,
            runs=planned_runs,
        )

    def test_progress_is_initialized_from_plan(
            self,
    ) -> None:
        plan = self.build_run_plan()

        progress = initialize_run_progress(
            plan,
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )

        self.assertEqual(progress.run_count, 2)
        self.assertEqual(
            progress.experiment_id,
            plan.experiment_id,
        )
        self.assertEqual(
            len(progress.run_plan_sha256),
            64,
        )
        self.assertEqual(
            progress.next_planned_run.run_id,
            "experiment-progress-run-0001",
        )

    def test_run_plan_hash_is_deterministic(
            self,
    ) -> None:
        plan = self.build_run_plan()

        self.assertEqual(
            calculate_run_plan_sha256(plan),
            calculate_run_plan_sha256(plan),
        )

    def test_progress_transition_updates_run(
            self,
    ) -> None:
        progress = initialize_run_progress(
            self.build_run_plan(),
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )

        updated = transition_progress_run(
            progress,
            "experiment-progress-run-0001",
            RunStatus.RUNNING,
            STARTED_AT,
        )

        first_run = updated.get_run_state(
            "experiment-progress-run-0001"
        )

        self.assertEqual(
            first_run.status,
            RunStatus.RUNNING,
        )
        self.assertEqual(
            updated.updated_at_utc,
            STARTED_AT,
        )
        self.assertEqual(
            updated.next_planned_run.run_id,
            "experiment-progress-run-0002",
        )

    def test_only_one_run_can_be_running(
            self,
    ) -> None:
        progress = initialize_run_progress(
            self.build_run_plan(),
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )
        progress = transition_progress_run(
            progress,
            "experiment-progress-run-0001",
            RunStatus.RUNNING,
            STARTED_AT,
        )

        with self.assertRaisesRegex(
                ControlledIlluminationRunStateError,
                "already running",
        ):
            transition_progress_run(
                progress,
                "experiment-progress-run-0002",
                RunStatus.RUNNING,
                "2026-08-23T10:01:00Z",
            )

    def test_unknown_run_id_is_rejected(
            self,
    ) -> None:
        progress = initialize_run_progress(
            self.build_run_plan(),
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )

        with self.assertRaisesRegex(
                ControlledIlluminationRunStateError,
                "Unknown run ID",
        ):
            progress.get_run_state(
                "unknown-run"
            )

    def test_completed_run_advances_next_planned(
            self,
    ) -> None:
        progress = initialize_run_progress(
            self.build_run_plan(),
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )
        progress = transition_progress_run(
            progress,
            "experiment-progress-run-0001",
            RunStatus.RUNNING,
            STARTED_AT,
        )
        progress = transition_progress_run(
            progress,
            "experiment-progress-run-0001",
            RunStatus.COMPLETED,
            FINISHED_AT,
        )

        self.assertEqual(
            progress.next_planned_run.run_id,
            "experiment-progress-run-0002",
        )
        self.assertEqual(
            progress.status_counts["completed"],
            1,
        )
        self.assertEqual(
            progress.status_counts["planned"],
            1,
        )

    def test_progress_is_serialized(
            self,
    ) -> None:
        progress = initialize_run_progress(
            self.build_run_plan(),
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )

        serialized = progress.to_dict()

        self.assertEqual(
            serialized["run_count"],
            2,
        )
        self.assertEqual(
            serialized["status_counts"]["planned"],
            2,
        )
        self.assertEqual(
            len(serialized["runs"]),
            2,
        )

    def test_invalid_plan_type_is_rejected(
            self,
    ) -> None:
        with self.assertRaises(
                ControlledIlluminationRunStateError
        ):
            initialize_run_progress(
                "invalid-plan",  # type: ignore[arg-type]
                created_at_utc=(
                    "2026-08-23T09:30:00Z"
                ),
            )
    def test_progress_save_and_load_round_trip(
        self,
    ) -> None:
        plan = self.build_run_plan()
        progress = initialize_run_progress(
            plan,
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )

        with TemporaryDirectory() as temporary:
            progress_path = (
                Path(temporary)
                / "run_progress.json"
            )

            save_run_progress_atomic(
                progress,
                progress_path,
            )
            loaded = load_run_progress(
                progress_path,
                plan=plan,
            )

        self.assertEqual(
            loaded.to_dict(),
            progress.to_dict(),
        )

    def test_atomic_save_removes_temporary_file(
        self,
    ) -> None:
        progress = initialize_run_progress(
            self.build_run_plan(),
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )

        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            progress_path = (
                temporary_path
                / "run_progress.json"
            )

            save_run_progress_atomic(
                progress,
                progress_path,
            )

            temporary_files = [
                path
                for path in temporary_path.iterdir()
                if path.name.endswith(".tmp")
            ]

            self.assertEqual(
                temporary_files,
                [],
            )

    def test_load_or_initialize_creates_progress(
        self,
    ) -> None:
        plan = self.build_run_plan()

        with TemporaryDirectory() as temporary:
            progress_path = (
                Path(temporary)
                / "run_progress.json"
            )

            progress = (
                load_or_initialize_run_progress(
                    plan,
                    progress_path,
                    created_at_utc=(
                        "2026-08-23T09:30:00Z"
                    ),
                )
            )

            self.assertTrue(
                progress_path.is_file()
            )
            self.assertEqual(
                progress.run_count,
                2,
            )

    def test_load_or_initialize_resumes_progress(
        self,
    ) -> None:
        plan = self.build_run_plan()

        with TemporaryDirectory() as temporary:
            progress_path = (
                Path(temporary)
                / "run_progress.json"
            )

            progress = (
                load_or_initialize_run_progress(
                    plan,
                    progress_path,
                    created_at_utc=(
                        "2026-08-23T09:30:00Z"
                    ),
                )
            )
            progress = transition_progress_run(
                progress,
                "experiment-progress-run-0001",
                RunStatus.RUNNING,
                STARTED_AT,
            )
            save_run_progress_atomic(
                progress,
                progress_path,
            )

            resumed = (
                load_or_initialize_run_progress(
                    plan,
                    progress_path,
                    created_at_utc=(
                        "2026-08-23T11:00:00Z"
                    ),
                )
            )

        self.assertEqual(
            resumed.get_run_state(
                "experiment-progress-run-0001"
            ).status,
            RunStatus.RUNNING,
        )
        self.assertEqual(
            resumed.created_at_utc,
            "2026-08-23T09:30:00Z",
        )

    def test_different_run_plan_is_rejected(
        self,
    ) -> None:
        plan = self.build_run_plan()
        different_plan = replace(
            plan,
            randomization_seed=20260822,
        )
        progress = initialize_run_progress(
            plan,
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )

        with TemporaryDirectory() as temporary:
            progress_path = (
                Path(temporary)
                / "run_progress.json"
            )

            save_run_progress_atomic(
                progress,
                progress_path,
            )

            with self.assertRaisesRegex(
                ControlledIlluminationRunStateError,
                "different run plan",
            ):
                load_run_progress(
                    progress_path,
                    plan=different_plan,
                )

    def test_modified_run_count_is_rejected(
        self,
    ) -> None:
        progress = initialize_run_progress(
            self.build_run_plan(),
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )
        progress_data = progress.to_dict()
        progress_data["run_count"] = 999

        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "run_count",
        ):
            progress_from_dict(
                progress_data
            )

    def test_modified_status_counts_are_rejected(
        self,
    ) -> None:
        progress = initialize_run_progress(
            self.build_run_plan(),
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )
        progress_data = progress.to_dict()
        progress_data["status_counts"][
            "planned"
        ] = 999

        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "status_counts",
        ):
            progress_from_dict(
                progress_data
            )

    def test_invalid_progress_json_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            progress_path = (
                Path(temporary)
                / "run_progress.json"
            )
            progress_path.write_text(
                "{invalid-json",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ControlledIlluminationRunStateError,
                "could not be loaded",
            ):
                load_run_progress(
                    progress_path
                )
    def test_finished_timestamp_before_started_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "finished_at_utc must not be earlier",
        ):
            RunState(
                run_id="run-chronology",
                execution_order=1,
                status=RunStatus.COMPLETED,
                attempt_count=1,
                started_at_utc=(
                    "2026-08-23T10:00:00Z"
                ),
                finished_at_utc=(
                    "2026-08-23T09:59:59Z"
                ),
                failure_reason=None,
                skip_reason=None,
            )

    def test_progress_update_before_creation_is_rejected(
        self,
    ) -> None:
        planned_run = RunState(
            run_id="run-chronology",
            execution_order=1,
            status=RunStatus.PLANNED,
            attempt_count=0,
            started_at_utc=None,
            finished_at_utc=None,
            failure_reason=None,
            skip_reason=None,
        )

        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "updated_at_utc must not be earlier",
        ):
            ControlledIlluminationProgress(
                schema_version=1,
                experiment_id="experiment-chronology",
                run_plan_sha256="a" * 64,
                created_at_utc=(
                    "2026-08-23T10:00:00Z"
                ),
                updated_at_utc=(
                    "2026-08-23T09:59:59Z"
                ),
                runs=(planned_run,),
            )

    def test_progress_transition_rejects_earlier_timestamp(
            self,
    ) -> None:
        progress = initialize_run_progress(
            self.build_run_plan(),
            created_at_utc=(
                "2026-08-23T09:30:00Z"
            ),
        )

        progress = transition_progress_run(
            progress,
            "experiment-progress-run-0001",
            RunStatus.RUNNING,
            STARTED_AT,
        )

        progress = transition_progress_run(
            progress,
            "experiment-progress-run-0001",
            RunStatus.COMPLETED,
            FINISHED_AT,
        )

        with self.assertRaisesRegex(
                ControlledIlluminationRunStateError,
                "progress.updated_at_utc",
        ):
            transition_progress_run(
                progress,
                "experiment-progress-run-0002",
                RunStatus.RUNNING,
                "2026-08-23T10:04:00Z",
            )

if __name__ == "__main__":
    unittest.main()