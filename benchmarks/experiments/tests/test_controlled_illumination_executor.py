from __future__ import annotations

from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from benchmarks.experiments.controlled_illumination_executor import (
    ArchitectureRunnerRegistry,
    ControlledIlluminationExecutionError,
    RunnerCommand,
    RunnerExecutionResult,
    SubprocessArchitectureRunner,
)
from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    PlannedRun,
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


if __name__ == "__main__":
    unittest.main()