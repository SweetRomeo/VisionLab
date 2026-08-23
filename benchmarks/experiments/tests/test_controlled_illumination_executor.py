from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

from benchmarks.experiments.controlled_illumination_executor import (
    ArchitectureRunnerRegistry,
    ControlledIlluminationExecutionError,
    RunnerCommand,
    RunnerExecutionResult,
    SubprocessArchitectureRunner,
    execute_next_planned_run,
    select_next_planned_run,
    execute_next_planned_run_from_files,
)
from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlan,
    PlannedRun,
    write_run_plan_manifests_atomic,
)
from benchmarks.experiments.controlled_illumination_run_state import (
    RunStatus,
    initialize_run_progress,
    transition_progress_run,
    load_run_progress,
)


class ControlledIlluminationExecutorTests(
    unittest.TestCase
):
    @staticmethod
    def build_planned_run() -> PlannedRun:
        return PlannedRun(
            execution_order=1,
            experiment_id="experiment-executor",
            run_id="experiment-executor-run-0001",
            phase="constant_lux",
            platform="desktop",
            architecture="pure_python",
            algorithm="original",
            resolution=ResolutionMetadata(
                width=640,
                height=480,
            ),
            trial_number=1,
            incidence_angle_degrees=0.0,
            target_illuminance_lux=50.0,
            source_output_setting=None,
            target_fps=30.0,
            frame_deadline_ms=1000.0 / 30.0,
        )

    def build_run_plan(
        self,
    ) -> ControlledIlluminationRunPlan:
        first_run = self.build_planned_run()

        second_run = replace(
            first_run,
            execution_order=2,
            run_id="experiment-executor-run-0002",
            trial_number=2,
        )

        return ControlledIlluminationRunPlan(
            schema_version=1,
            generated_at_utc=(
                "2026-08-24T09:00:00Z"
            ),
            experiment_id="experiment-executor",
            randomized=False,
            randomization_seed=20260824,
            runs=(
                first_run,
                second_run,
            ),
        )

    @staticmethod
    def successful_runner(
        planned_run,
    ) -> RunnerExecutionResult:
        return RunnerExecutionResult(
            exit_code=0,
        )

    def build_runners(self) -> dict:
        return {
            "pure_python": self.successful_runner,
            "hybrid": self.successful_runner,
            "pure_cpp": self.successful_runner,
        }

    def test_successful_result(
        self,
    ) -> None:
        result = RunnerExecutionResult(
            exit_code=0,
            standard_output="completed",
        )

        self.assertTrue(result.succeeded)

    def test_failed_result(
        self,
    ) -> None:
        result = RunnerExecutionResult(
            exit_code=1,
            standard_error="runner failed",
        )

        self.assertFalse(result.succeeded)

    def test_invalid_exit_code_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationExecutionError
        ):
            RunnerExecutionResult(
                exit_code=True,
            )

    def test_registry_returns_runner(
        self,
    ) -> None:
        runners = self.build_runners()
        registry = ArchitectureRunnerRegistry(
            runners
        )

        self.assertIs(
            registry.get_runner("hybrid"),
            runners["hybrid"],
        )

    def test_missing_runner_is_rejected(
        self,
    ) -> None:
        runners = self.build_runners()
        del runners["pure_cpp"]

        with self.assertRaisesRegex(
            ControlledIlluminationExecutionError,
            "Missing architecture runners",
        ):
            ArchitectureRunnerRegistry(
                runners
            )

    def test_unexpected_runner_is_rejected(
        self,
    ) -> None:
        runners = self.build_runners()
        runners["unknown"] = (
            self.successful_runner
        )

        with self.assertRaisesRegex(
            ControlledIlluminationExecutionError,
            "Unexpected architecture runners",
        ):
            ArchitectureRunnerRegistry(
                runners
            )

    def test_unknown_architecture_is_rejected(
        self,
    ) -> None:
        registry = ArchitectureRunnerRegistry(
            self.build_runners()
        )

        with self.assertRaisesRegex(
            ControlledIlluminationExecutionError,
            "Unsupported execution architecture",
        ):
            registry.get_runner(
                "unknown"
            )

    def test_runner_command_rejects_empty_arguments(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationExecutionError,
            "arguments",
        ):
            RunnerCommand(
                arguments=(),
                working_directory=Path("."),
            )

    def test_subprocess_runner_captures_result(
        self,
    ) -> None:
        command = RunnerCommand(
            arguments=("python", "runner.py"),
            working_directory=Path("."),
            environment={
                "VISIONLAB_TEST": "enabled",
            },
            timeout_seconds=30.0,
        )

        completed_process = subprocess.CompletedProcess(
            args=command.arguments,
            returncode=0,
            stdout="completed\n",
            stderr="",
        )

        with patch(
            "benchmarks.experiments."
            "controlled_illumination_executor."
            "subprocess.run",
            return_value=completed_process,
        ) as run_mock:
            runner = SubprocessArchitectureRunner(
                lambda planned_run: command
            )

            result = runner(
                self.build_planned_run()
            )

        self.assertTrue(result.succeeded)
        self.assertEqual(
            result.standard_output,
            "completed\n",
        )

        run_mock.assert_called_once()

        call_arguments = run_mock.call_args

        self.assertEqual(
            call_arguments.args[0],
            command.arguments,
        )
        self.assertEqual(
            call_arguments.kwargs["cwd"],
            Path("."),
        )
        self.assertEqual(
            call_arguments.kwargs["env"][
                "VISIONLAB_TEST"
            ],
            "enabled",
        )
        self.assertEqual(
            call_arguments.kwargs["timeout"],
            30.0,
        )

    def test_subprocess_runner_rejects_timeout(
        self,
    ) -> None:
        command = RunnerCommand(
            arguments=("python", "runner.py"),
            working_directory=Path("."),
            timeout_seconds=1.0,
        )

        with patch(
            "benchmarks.experiments."
            "controlled_illumination_executor."
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(
                cmd=command.arguments,
                timeout=1.0,
            ),
        ):
            runner = SubprocessArchitectureRunner(
                lambda planned_run: command
            )

            with self.assertRaisesRegex(
                ControlledIlluminationExecutionError,
                "timed out",
            ):
                runner(
                    self.build_planned_run()
                )

    def test_subprocess_runner_rejects_start_failure(
        self,
    ) -> None:
        command = RunnerCommand(
            arguments=("missing-runner",),
            working_directory=Path("."),
        )

        with patch(
            "benchmarks.experiments."
            "controlled_illumination_executor."
            "subprocess.run",
            side_effect=FileNotFoundError(
                "runner not found"
            ),
        ):
            runner = SubprocessArchitectureRunner(
                lambda planned_run: command
            )

            with self.assertRaisesRegex(
                ControlledIlluminationExecutionError,
                "could not be started",
            ):
                runner(
                    self.build_planned_run()
                )

    def test_subprocess_runner_rejects_invalid_command(
        self,
    ) -> None:
        runner = SubprocessArchitectureRunner(
            lambda planned_run: "invalid"  # type: ignore[return-value]
        )

        with self.assertRaisesRegex(
            ControlledIlluminationExecutionError,
            "must return RunnerCommand",
        ):
            runner(
                self.build_planned_run()
            )

    def test_selects_first_planned_run(
        self,
    ) -> None:
        plan = self.build_run_plan()
        progress = initialize_run_progress(
            plan,
            created_at_utc="2026-08-24T09:01:00Z",
        )

        selected_run = select_next_planned_run(
            plan,
            progress,
        )

        self.assertIsNotNone(selected_run)
        self.assertEqual(
            selected_run.run_id,
            "experiment-executor-run-0001",
        )

    def test_selects_next_run_after_completion(
        self,
    ) -> None:
        plan = self.build_run_plan()
        progress = initialize_run_progress(
            plan,
            created_at_utc="2026-08-24T09:01:00Z",
        )

        progress = transition_progress_run(
            progress,
            plan.runs[0].run_id,
            RunStatus.RUNNING,
            "2026-08-24T09:02:00Z",
        )
        progress = transition_progress_run(
            progress,
            plan.runs[0].run_id,
            RunStatus.COMPLETED,
            "2026-08-24T09:03:00Z",
        )

        selected_run = select_next_planned_run(
            plan,
            progress,
        )

        self.assertIsNotNone(selected_run)
        self.assertEqual(
            selected_run.run_id,
            "experiment-executor-run-0002",
        )

    def test_returns_none_when_no_runs_remain(
        self,
    ) -> None:
        plan = self.build_run_plan()
        progress = initialize_run_progress(
            plan,
            created_at_utc="2026-08-24T09:01:00Z",
        )
        transition_timestamps = (
            (
                "2026-08-24T09:02:00Z",
                "2026-08-24T09:03:00Z",
            ),
            (
                "2026-08-24T09:04:00Z",
                "2026-08-24T09:05:00Z",
            ),
        )

        for planned_run, timestamps in zip(
            plan.runs,
            transition_timestamps,
            strict=True,
        ):
            started_at_utc, finished_at_utc = (
                timestamps
            )
            progress = transition_progress_run(
                progress,
                planned_run.run_id,
                RunStatus.RUNNING,
                started_at_utc,
            )
            progress = transition_progress_run(
                progress,
                planned_run.run_id,
                RunStatus.COMPLETED,
                finished_at_utc,
            )

        self.assertIsNone(
            select_next_planned_run(
                plan,
                progress,
            )
        )

    def test_successful_execution_completes_run(
        self,
    ) -> None:
        plan = self.build_run_plan()
        progress = initialize_run_progress(
            plan,
            created_at_utc="2026-08-24T09:01:00Z",
        )
        saved_progress = []
        timestamps = iter(
            (
                "2026-08-24T09:02:00Z",
                "2026-08-24T09:03:00Z",
            )
        )

        outcome = execute_next_planned_run(
            plan,
            progress,
            ArchitectureRunnerRegistry(
                self.build_runners()
            ),
            timestamp_provider=lambda: next(
                timestamps
            ),
            persist_progress=saved_progress.append,
        )

        self.assertIsNotNone(outcome)
        self.assertTrue(outcome.succeeded)
        self.assertEqual(
            len(saved_progress),
            2,
        )
        self.assertEqual(
            saved_progress[0].get_run_state(
                plan.runs[0].run_id
            ).status,
            RunStatus.RUNNING,
        )
        self.assertEqual(
            outcome.progress.get_run_state(
                plan.runs[0].run_id
            ).status,
            RunStatus.COMPLETED,
        )

    def test_failed_result_marks_run_failed(
        self,
    ) -> None:
        plan = self.build_run_plan()
        progress = initialize_run_progress(
            plan,
            created_at_utc="2026-08-24T09:01:00Z",
        )
        runners = self.build_runners()
        runners["pure_python"] = (
            lambda planned_run: RunnerExecutionResult(
                exit_code=5,
                standard_error="runner failed",
            )
        )
        saved_progress = []
        timestamps = iter(
            (
                "2026-08-24T09:02:00Z",
                "2026-08-24T09:03:00Z",
            )
        )

        outcome = execute_next_planned_run(
            plan,
            progress,
            ArchitectureRunnerRegistry(runners),
            timestamp_provider=lambda: next(
                timestamps
            ),
            persist_progress=saved_progress.append,
        )

        self.assertIsNotNone(outcome)
        final_state = outcome.progress.get_run_state(
            plan.runs[0].run_id
        )

        self.assertFalse(outcome.succeeded)
        self.assertEqual(
            final_state.status,
            RunStatus.FAILED,
        )
        self.assertIn(
            "runner failed",
            final_state.failure_reason,
        )
        self.assertEqual(
            len(saved_progress),
            2,
        )

    def test_runner_exception_marks_run_failed(
        self,
    ) -> None:
        plan = self.build_run_plan()
        progress = initialize_run_progress(
            plan,
            created_at_utc="2026-08-24T09:01:00Z",
        )

        def failing_runner(
            planned_run,
        ) -> RunnerExecutionResult:
            raise RuntimeError(
                "runner exploded"
            )

        runners = self.build_runners()
        runners["pure_python"] = failing_runner
        saved_progress = []
        timestamps = iter(
            (
                "2026-08-24T09:02:00Z",
                "2026-08-24T09:03:00Z",
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "runner exploded",
        ):
            execute_next_planned_run(
                plan,
                progress,
                ArchitectureRunnerRegistry(runners),
                timestamp_provider=lambda: next(
                    timestamps
                ),
                persist_progress=saved_progress.append,
            )

        failed_state = saved_progress[-1].get_run_state(
            plan.runs[0].run_id
        )

        self.assertEqual(
            failed_state.status,
            RunStatus.FAILED,
        )
        self.assertEqual(
            failed_state.failure_reason,
            "runner exploded",
        )

    def test_file_execution_creates_progress(
            self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan = self.build_run_plan()

            plan_path, _ = (
                write_run_plan_manifests_atomic(
                    plan,
                    temporary_path,
                )
            )
            progress_path = (
                    temporary_path
                    / "run_progress.json"
            )
            timestamps = iter(
                (
                    "2026-08-24T09:01:00Z",
                    "2026-08-24T09:02:00Z",
                    "2026-08-24T09:03:00Z",
                )
            )

            outcome = (
                execute_next_planned_run_from_files(
                    plan_path,
                    progress_path,
                    ArchitectureRunnerRegistry(
                        self.build_runners()
                    ),
                    timestamp_provider=lambda: next(
                        timestamps
                    ),
                )
            )

            persisted_progress = load_run_progress(
                progress_path,
                plan=plan,
            )

        self.assertIsNotNone(outcome)
        self.assertTrue(outcome.succeeded)
        self.assertEqual(
            persisted_progress.get_run_state(
                plan.runs[0].run_id
            ).status,
            RunStatus.COMPLETED,
        )
        self.assertEqual(
            persisted_progress.get_run_state(
                plan.runs[1].run_id
            ).status,
            RunStatus.PLANNED,
        )

    def test_file_execution_resumes_existing_progress(
            self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan = self.build_run_plan()

            plan_path, _ = (
                write_run_plan_manifests_atomic(
                    plan,
                    temporary_path,
                )
            )
            progress_path = (
                    temporary_path
                    / "run_progress.json"
            )
            runner_registry = (
                ArchitectureRunnerRegistry(
                    self.build_runners()
                )
            )

            first_timestamps = iter(
                (
                    "2026-08-24T09:01:00Z",
                    "2026-08-24T09:02:00Z",
                    "2026-08-24T09:03:00Z",
                )
            )

            first_outcome = (
                execute_next_planned_run_from_files(
                    plan_path,
                    progress_path,
                    runner_registry,
                    timestamp_provider=lambda: next(
                        first_timestamps
                    ),
                )
            )

            second_timestamps = iter(
                (
                    "2026-08-24T09:04:00Z",
                    "2026-08-24T09:05:00Z",
                    "2026-08-24T09:06:00Z",
                )
            )

            second_outcome = (
                execute_next_planned_run_from_files(
                    plan_path,
                    progress_path,
                    runner_registry,
                    timestamp_provider=lambda: next(
                        second_timestamps
                    ),
                )
            )

            no_run_outcome = (
                execute_next_planned_run_from_files(
                    plan_path,
                    progress_path,
                    runner_registry,
                    timestamp_provider=lambda: (
                        "2026-08-24T09:07:00Z"
                    ),
                )
            )

            persisted_progress = load_run_progress(
                progress_path,
                plan=plan,
            )

        self.assertIsNotNone(first_outcome)
        self.assertEqual(
            first_outcome.planned_run.run_id,
            plan.runs[0].run_id,
        )

        self.assertIsNotNone(second_outcome)
        self.assertEqual(
            second_outcome.planned_run.run_id,
            plan.runs[1].run_id,
        )

        self.assertIsNone(no_run_outcome)

        for planned_run in plan.runs:
            self.assertEqual(
                persisted_progress.get_run_state(
                    planned_run.run_id
                ).status,
                RunStatus.COMPLETED,
            )

if __name__ == "__main__":
    unittest.main()
