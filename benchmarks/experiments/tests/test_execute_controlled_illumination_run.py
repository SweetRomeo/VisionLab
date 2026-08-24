from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from benchmarks.experiments.controlled_illumination_executor import (
    ControlledIlluminationExecutionError,
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
    load_run_progress,
)
from benchmarks.experiments.execute_controlled_illumination_run import (
    build_planned_run_environment,
    create_argument_parser,
    determine_progress_path,
    load_runner_registry,
    main,
    run_cli,
)


class ExecuteControlledIlluminationRunTests(
    unittest.TestCase
):
    @staticmethod
    def build_planned_run() -> PlannedRun:
        return PlannedRun(
            execution_order=1,
            experiment_id="experiment-cli-execution",
            run_id="experiment-cli-execution-run-0001",
            phase="constant_lux",
            platform="desktop",
            architecture="pure_python",
            algorithm="clahe",
            resolution=ResolutionMetadata(
                width=1280,
                height=720,
            ),
            trial_number=1,
            incidence_angle_degrees=30.0,
            target_illuminance_lux=50.0,
            source_output_setting=None,
            target_fps=30.0,
            frame_deadline_ms=1000.0 / 30.0,
        )

    def build_plan(
        self,
    ) -> ControlledIlluminationRunPlan:
        return ControlledIlluminationRunPlan(
            schema_version=1,
            generated_at_utc=(
                "2026-08-24T10:00:00Z"
            ),
            experiment_id=(
                "experiment-cli-execution"
            ),
            randomized=False,
            randomization_seed=20260824,
            runs=(
                self.build_planned_run(),
            ),
        )

    @staticmethod
    def write_runner_config(
        directory: Path,
        *,
        include_pure_cpp: bool = True,
    ) -> Path:
        runner_config = {
            "schema_version": 1,
            "runners": {
                "pure_python": {
                    "arguments": [
                        sys.executable,
                        "-c",
                        "raise SystemExit(0)",
                    ],
                    "working_directory": str(
                        directory
                    ),
                    "environment": {
                        "VISIONLAB_BASE": "configured",
                    },
                    "timeout_seconds": 30.0,
                },
                "hybrid": {
                    "arguments": [
                        sys.executable,
                        "-c",
                        "raise SystemExit(0)",
                    ],
                    "working_directory": str(
                        directory
                    ),
                },
            },
        }

        if include_pure_cpp:
            runner_config["runners"]["pure_cpp"] = {
                "arguments": [
                    sys.executable,
                    "-c",
                    "raise SystemExit(0)",
                ],
                "working_directory": str(
                    directory
                ),
            }

        config_path = (
            directory / "runner_config.json"
        )
        config_path.write_text(
            json.dumps(
                runner_config,
                indent=2,
            ),
            encoding="utf-8",
        )

        return config_path

    @staticmethod
    def parse_arguments(
        plan_path: Path,
        progress_path: Path,
        runner_config_path: Path,
    ):
        return create_argument_parser().parse_args(
            [
                "--plan",
                str(plan_path),
                "--progress",
                str(progress_path),
                "--runner-config",
                str(runner_config_path),
            ]
        )

    def test_default_progress_path(
        self,
    ) -> None:
        plan_path = Path("experiment/run_plan.json")

        self.assertEqual(
            determine_progress_path(
                plan_path,
                None,
            ),
            (
                plan_path.resolve().parent
                / "run_progress.json"
            ),
        )

    def test_planned_run_environment_is_complete(
        self,
    ) -> None:
        environment = build_planned_run_environment(
            self.build_planned_run()
        )

        self.assertEqual(
            environment["VISIONLAB_RUN_ID"],
            "experiment-cli-execution-run-0001",
        )
        self.assertEqual(
            environment["VISIONLAB_ALGORITHM"],
            "clahe",
        )
        self.assertEqual(
            environment[
                "VISIONLAB_RESOLUTION_WIDTH"
            ],
            "1280",
        )
        self.assertEqual(
            environment[
                "VISIONLAB_TARGET_ILLUMINANCE_LUX"
            ],
            "50.0",
        )
        self.assertEqual(
            environment[
                "VISIONLAB_SOURCE_OUTPUT_SETTING"
            ],
            "",
        )

    def test_runner_config_requires_all_architectures(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            config_path = self.write_runner_config(
                Path(temporary),
                include_pure_cpp=False,
            )

            with self.assertRaisesRegex(
                ControlledIlluminationExecutionError,
                "architectures do not match",
            ):
                load_runner_registry(
                    config_path
                )

    def test_runner_config_merges_run_environment(
        self,
    ) -> None:
        completed_process = subprocess.CompletedProcess(
            args=(sys.executable,),
            returncode=0,
            stdout="completed",
            stderr="",
        )

        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config_path = self.write_runner_config(
                temporary_path
            )
            registry = load_runner_registry(
                config_path
            )

            with patch(
                "benchmarks.experiments."
                "controlled_illumination_executor."
                "subprocess.run",
                return_value=completed_process,
            ) as run_mock:
                result = registry.get_runner(
                    "pure_python"
                )(
                    self.build_planned_run()
                )

        self.assertTrue(result.succeeded)
        execution_environment = (
            run_mock.call_args.kwargs["env"]
        )
        self.assertEqual(
            execution_environment["VISIONLAB_BASE"],
            "configured",
        )
        self.assertEqual(
            execution_environment["VISIONLAB_RUN_ID"],
            "experiment-cli-execution-run-0001",
        )

    def test_run_cli_completes_and_persists_run(
        self,
    ) -> None:
        completed_process = subprocess.CompletedProcess(
            args=(sys.executable,),
            returncode=0,
            stdout="completed",
            stderr="",
        )

        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan = self.build_plan()
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
            config_path = self.write_runner_config(
                temporary_path
            )
            arguments = self.parse_arguments(
                plan_path,
                progress_path,
                config_path,
            )
            timestamps = iter(
                (
                    "2026-08-24T10:01:00Z",
                    "2026-08-24T10:02:00Z",
                    "2026-08-24T10:03:00Z",
                )
            )

            with patch(
                "benchmarks.experiments."
                "controlled_illumination_executor."
                "subprocess.run",
                return_value=completed_process,
            ):
                with redirect_stdout(StringIO()):
                    exit_code = run_cli(
                        arguments,
                        timestamp_provider=lambda: next(
                            timestamps
                        ),
                    )

            progress = load_run_progress(
                progress_path,
                plan=plan,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            progress.get_run_state(
                plan.runs[0].run_id
            ).status,
            RunStatus.COMPLETED,
        )

    def test_run_cli_reports_runner_failure(
        self,
    ) -> None:
        failed_process = subprocess.CompletedProcess(
            args=(sys.executable,),
            returncode=5,
            stdout="",
            stderr="runner failed",
        )

        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan = self.build_plan()
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
            config_path = self.write_runner_config(
                temporary_path
            )
            arguments = self.parse_arguments(
                plan_path,
                progress_path,
                config_path,
            )
            timestamps = iter(
                (
                    "2026-08-24T10:01:00Z",
                    "2026-08-24T10:02:00Z",
                    "2026-08-24T10:03:00Z",
                )
            )

            with patch(
                "benchmarks.experiments."
                "controlled_illumination_executor."
                "subprocess.run",
                return_value=failed_process,
            ):
                with redirect_stderr(StringIO()):
                    exit_code = run_cli(
                        arguments,
                        timestamp_provider=lambda: next(
                            timestamps
                        ),
                    )

            progress = load_run_progress(
                progress_path,
                plan=plan,
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            progress.get_run_state(
                plan.runs[0].run_id
            ).status,
            RunStatus.FAILED,
        )

    def test_run_cli_reports_no_remaining_runs(
        self,
    ) -> None:
        completed_process = subprocess.CompletedProcess(
            args=(sys.executable,),
            returncode=0,
            stdout="",
            stderr="",
        )

        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            plan = self.build_plan()
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
            config_path = self.write_runner_config(
                temporary_path
            )
            arguments = self.parse_arguments(
                plan_path,
                progress_path,
                config_path,
            )
            timestamps = iter(
                (
                    "2026-08-24T10:01:00Z",
                    "2026-08-24T10:02:00Z",
                    "2026-08-24T10:03:00Z",
                )
            )

            with patch(
                "benchmarks.experiments."
                "controlled_illumination_executor."
                "subprocess.run",
                return_value=completed_process,
            ):
                with redirect_stdout(StringIO()):
                    first_exit_code = run_cli(
                        arguments,
                        timestamp_provider=lambda: next(
                            timestamps
                        ),
                    )

            captured_output = StringIO()

            with redirect_stdout(captured_output):
                second_exit_code = run_cli(
                    arguments,
                    timestamp_provider=lambda: (
                        "2026-08-24T10:04:00Z"
                    ),
                )

        self.assertEqual(first_exit_code, 0)
        self.assertEqual(second_exit_code, 0)
        self.assertIn(
            "No planned experiment runs remain",
            captured_output.getvalue(),
        )

    def test_main_returns_interrupt_exit_code(
        self,
    ) -> None:
        with patch(
            "benchmarks.experiments."
            "execute_controlled_illumination_run."
            "run_cli",
            side_effect=KeyboardInterrupt,
        ):
            with redirect_stderr(StringIO()):
                exit_code = main(
                    [
                        "--plan",
                        "run_plan.json",
                        "--runner-config",
                        "runner_config.json",
                    ]
                )

        self.assertEqual(exit_code, 130)

    def test_example_runner_configuration_is_valid(
            self,
    ) -> None:
        config_path = (
                Path(__file__).resolve().parents[1]
                / "config"
                / (
                    "controlled_illumination_"
                    "runner_config.example.json"
                )
        )

        registry = load_runner_registry(
            config_path
        )

        for architecture in (
                "pure_python",
                "hybrid",
                "pure_cpp",
        ):
            self.assertTrue(
                callable(
                    registry.get_runner(
                        architecture
                    )
                )
            )

if __name__ == "__main__":
    unittest.main()