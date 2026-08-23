from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlan,
    PlannedRun,
    write_run_plan_manifests_atomic,
)
from benchmarks.experiments.controlled_illumination_run_state import (
    ControlledIlluminationRunStateError,
    RunStatus,
    load_run_progress,
)
from benchmarks.experiments.manage_controlled_illumination_run import (
    create_argument_parser,
    determine_progress_path,
    run_cli,
)


CREATED_AT = "2026-08-23T09:30:00Z"
STARTED_AT = "2026-08-23T10:00:00Z"
COMPLETED_AT = "2026-08-23T10:05:00Z"
FAILED_AT = "2026-08-23T10:06:00Z"
SKIPPED_AT = "2026-08-23T10:07:00Z"
REPLANNED_AT = "2026-08-23T10:08:00Z"


class ManageControlledIlluminationRunTests(
    unittest.TestCase
):
    def create_run_plan(
        self,
        directory: Path,
    ) -> tuple[ControlledIlluminationRunPlan, Path]:
        planned_runs = tuple(
            PlannedRun(
                execution_order=execution_order,
                experiment_id="experiment-cli",
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
                (1, "experiment-cli-run-0001"),
                (2, "experiment-cli-run-0002"),
            )
        )

        plan = ControlledIlluminationRunPlan(
            schema_version=1,
            generated_at_utc=(
                "2026-08-23T09:00:00Z"
            ),
            experiment_id="experiment-cli",
            randomized=False,
            randomization_seed=20260821,
            runs=planned_runs,
        )

        json_path, _ = (
            write_run_plan_manifests_atomic(
                plan,
                directory,
            )
        )

        return plan, json_path

    def parse_arguments(
        self,
        plan_path: Path,
        progress_path: Path,
        command: str,
        *command_arguments: str,
    ):
        return create_argument_parser().parse_args(
            [
                "--plan",
                str(plan_path),
                "--progress",
                str(progress_path),
                command,
                *command_arguments,
            ]
        )

    def test_default_progress_path(
        self,
    ) -> None:
        plan_path = (
            Path("experiment")
            / "run_plan.json"
        )

        self.assertEqual(
            determine_progress_path(
                plan_path,
                None,
            ),
            (
                Path("experiment")
                / "run_progress.json"
            ),
        )

    def test_init_creates_progress_file(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path
                / "run_progress.json"
            )
            arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )

            with redirect_stdout(StringIO()):
                exit_code = run_cli(
                    arguments,
                    now_provider=lambda: CREATED_AT,
                )

            progress = load_run_progress(
                progress_path,
                plan=plan,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(progress.run_count, 2)
        self.assertEqual(
            progress.status_counts["planned"],
            2,
        )

    def test_init_resumes_existing_progress(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            _, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path
                / "run_progress.json"
            )
            arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )

            with redirect_stdout(StringIO()):
                run_cli(
                    arguments,
                    now_provider=lambda: CREATED_AT,
                )

            captured_output = StringIO()

            with redirect_stdout(captured_output):
                exit_code = run_cli(
                    arguments,
                    now_provider=lambda: (
                        "2026-08-23T11:00:00Z"
                    ),
                )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "Existing run progress loaded",
            captured_output.getvalue(),
        )

    def test_status_displays_progress(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            _, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path
                / "run_progress.json"
            )

            init_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )
            status_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "status",
            )

            with redirect_stdout(StringIO()):
                run_cli(
                    init_arguments,
                    now_provider=lambda: CREATED_AT,
                )

            captured_output = StringIO()

            with redirect_stdout(captured_output):
                exit_code = run_cli(
                    status_arguments
                )

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "Total runs: 2",
            captured_output.getvalue(),
        )
        self.assertIn(
            "planned: 2",
            captured_output.getvalue(),
        )

    def test_start_next_starts_first_run(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path
                / "run_progress.json"
            )

            init_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )
            start_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "start-next",
            )

            with redirect_stdout(StringIO()):
                run_cli(
                    init_arguments,
                    now_provider=lambda: CREATED_AT,
                )
                exit_code = run_cli(
                    start_arguments,
                    now_provider=lambda: STARTED_AT,
                )

            progress = load_run_progress(
                progress_path,
                plan=plan,
            )
            first_run = progress.get_run_state(
                "experiment-cli-run-0001"
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            first_run.status,
            RunStatus.RUNNING,
        )
        self.assertEqual(
            first_run.attempt_count,
            1,
        )

    def test_start_next_rejects_second_running_run(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            _, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path
                / "run_progress.json"
            )

            init_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )
            start_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "start-next",
            )

            with redirect_stdout(StringIO()):
                run_cli(
                    init_arguments,
                    now_provider=lambda: CREATED_AT,
                )
                run_cli(
                    start_arguments,
                    now_provider=lambda: STARTED_AT,
                )

            with self.assertRaisesRegex(
                ControlledIlluminationRunStateError,
                "already running",
            ):
                with redirect_stdout(StringIO()):
                    run_cli(
                        start_arguments,
                        now_provider=lambda: (
                            "2026-08-23T10:01:00Z"
                        ),
                    )
    def test_complete_marks_running_run_completed(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path / "run_progress.json"
            )

            init_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )
            start_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "start-next",
            )
            complete_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "complete",
                "--run-id",
                "experiment-cli-run-0001",
            )

            with redirect_stdout(StringIO()):
                run_cli(
                    init_arguments,
                    now_provider=lambda: CREATED_AT,
                )
                run_cli(
                    start_arguments,
                    now_provider=lambda: STARTED_AT,
                )
                exit_code = run_cli(
                    complete_arguments,
                    now_provider=lambda: COMPLETED_AT,
                )

            progress = load_run_progress(
                progress_path,
                plan=plan,
            )
            first_run = progress.get_run_state(
                "experiment-cli-run-0001"
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            first_run.status,
            RunStatus.COMPLETED,
        )
        self.assertEqual(
            first_run.finished_at_utc,
            COMPLETED_AT,
        )

    def test_fail_records_failure_reason(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path / "run_progress.json"
            )

            init_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )
            start_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "start-next",
            )
            fail_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "fail",
                "--run-id",
                "experiment-cli-run-0001",
                "--reason",
                "Camera disconnected.",
            )

            with redirect_stdout(StringIO()):
                run_cli(
                    init_arguments,
                    now_provider=lambda: CREATED_AT,
                )
                run_cli(
                    start_arguments,
                    now_provider=lambda: STARTED_AT,
                )
                exit_code = run_cli(
                    fail_arguments,
                    now_provider=lambda: FAILED_AT,
                )

            progress = load_run_progress(
                progress_path,
                plan=plan,
            )
            first_run = progress.get_run_state(
                "experiment-cli-run-0001"
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            first_run.status,
            RunStatus.FAILED,
        )
        self.assertEqual(
            first_run.failure_reason,
            "Camera disconnected.",
        )

    def test_skip_records_skip_reason(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path / "run_progress.json"
            )

            init_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )
            skip_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "skip",
                "--run-id",
                "experiment-cli-run-0002",
                "--reason",
                "Lighting unavailable.",
            )

            with redirect_stdout(StringIO()):
                run_cli(
                    init_arguments,
                    now_provider=lambda: CREATED_AT,
                )
                exit_code = run_cli(
                    skip_arguments,
                    now_provider=lambda: SKIPPED_AT,
                )

            progress = load_run_progress(
                progress_path,
                plan=plan,
            )
            second_run = progress.get_run_state(
                "experiment-cli-run-0002"
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            second_run.status,
            RunStatus.SKIPPED,
        )
        self.assertEqual(
            second_run.skip_reason,
            "Lighting unavailable.",
        )

    def test_replan_returns_failed_run_to_planned(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path / "run_progress.json"
            )

            init_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )
            start_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "start-next",
            )
            fail_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "fail",
                "--run-id",
                "experiment-cli-run-0001",
                "--reason",
                "Temporary failure.",
            )
            replan_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "replan",
                "--run-id",
                "experiment-cli-run-0001",
            )

            with redirect_stdout(StringIO()):
                run_cli(
                    init_arguments,
                    now_provider=lambda: CREATED_AT,
                )
                run_cli(
                    start_arguments,
                    now_provider=lambda: STARTED_AT,
                )
                run_cli(
                    fail_arguments,
                    now_provider=lambda: FAILED_AT,
                )
                exit_code = run_cli(
                    replan_arguments,
                    now_provider=lambda: REPLANNED_AT,
                )

            progress = load_run_progress(
                progress_path,
                plan=plan,
            )
            first_run = progress.get_run_state(
                "experiment-cli-run-0001"
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            first_run.status,
            RunStatus.PLANNED,
        )
        self.assertIsNone(
            first_run.failure_reason
        )
        self.assertEqual(
            first_run.attempt_count,
            1,
        )

    def test_complete_rejects_planned_run(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            _, plan_path = self.create_run_plan(
                temporary_path
            )
            progress_path = (
                temporary_path / "run_progress.json"
            )

            init_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "init",
            )
            complete_arguments = self.parse_arguments(
                plan_path,
                progress_path,
                "complete",
                "--run-id",
                "experiment-cli-run-0001",
            )

            with redirect_stdout(StringIO()):
                run_cli(
                    init_arguments,
                    now_provider=lambda: CREATED_AT,
                )

            with self.assertRaises(
                ControlledIlluminationRunStateError
            ):
                with redirect_stdout(StringIO()):
                    run_cli(
                        complete_arguments,
                        now_provider=lambda: COMPLETED_AT,
                    )


if __name__ == "__main__":
    unittest.main()