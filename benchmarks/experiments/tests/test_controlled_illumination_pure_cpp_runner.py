from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_pure_cpp_runner import (
    CSV_FIELDS,
    ControlledIlluminationPureCppRunnerError,
    EXECUTABLE_ENVIRONMENT_VARIABLE,
    RAW_RESULTS_ENVIRONMENT_VARIABLE,
    execute_pure_cpp_run,
    find_pure_cpp_realtime_executable,
    load_cpp_frame_records,
    run_cli,
    validate_executable_path,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    PlannedRun,
)
from benchmarks.experiments.controlled_illumination_runner_context import (
    ControlledIlluminationRunnerContext,
)


RUNNER_MODULE = (
    "benchmarks.experiments."
    "controlled_illumination_pure_cpp_runner"
)

STARTED_AT = "2026-08-25T12:00:00Z"
FINISHED_AT = "2026-08-25T12:01:00Z"


class ControlledIlluminationPureCppRunnerTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        planned_run = PlannedRun(
            execution_order=1,
            experiment_id="experiment-test",
            run_id="experiment-test-run-0001",
            phase="constant_lux",
            platform="desktop",
            architecture="pure_cpp",
            algorithm="clahe",
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

        self.context = (
            ControlledIlluminationRunnerContext(
                planned_run=planned_run,
                results_root=Path("results"),
                output_directory=(
                    Path("results")
                    / "desktop"
                    / "experiment-test"
                    / "experiment-test-run-0001"
                ),
            )
        )

    def write_valid_csv(
        self,
        output_path: Path,
    ) -> None:
        header = ",".join(CSV_FIELDS)
        row = (
            "pure_cpp,clahe,640x480,1,1,"
            "0.000000,0.100000,0.200000,"
            "3.200000,,0.100000,0.100000,"
            "3.000000,3.200000,33.333333,"
            "false,processed"
        )

        output_path.write_text(
            f"{header}\n{row}\n",
            encoding="utf-8",
        )

    def test_configured_release_executable_is_used(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            release_directory = (
                Path(temporary)
                / "Release"
            )
            release_directory.mkdir()

            executable_path = (
                release_directory
                / "VisionLabCppRealtime.exe"
            )
            executable_path.write_text(
                "",
                encoding="utf-8",
            )

            selected_path = (
                find_pure_cpp_realtime_executable(
                    {
                        EXECUTABLE_ENVIRONMENT_VARIABLE:
                            str(executable_path),
                    }
                )
            )

        self.assertEqual(
            selected_path,
            executable_path.resolve(),
        )

    def test_debug_executable_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            debug_directory = (
                Path(temporary)
                / "Debug"
            )
            debug_directory.mkdir()

            executable_path = (
                debug_directory
                / "VisionLabCppRealtime.exe"
            )
            executable_path.write_text(
                "",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ControlledIlluminationPureCppRunnerError,
                "Release",
            ):
                validate_executable_path(
                    executable_path
                )

    def test_valid_cpp_csv_is_loaded(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            csv_path = (
                Path(temporary)
                / "records.csv"
            )
            self.write_valid_csv(csv_path)

            records = load_cpp_frame_records(
                csv_path
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].architecture,
            "pure_cpp",
        )
        self.assertEqual(
            records[0].algorithm,
            "clahe",
        )
        self.assertEqual(
            records[0].frame_index,
            1,
        )
        self.assertEqual(
            records[0].processing_time_ms,
            3.0,
        )
        self.assertFalse(
            records[0].deadline_missed
        )

    def test_invalid_csv_header_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            csv_path = (
                Path(temporary)
                / "records.csv"
            )
            csv_path.write_text(
                "architecture,algorithm\n"
                "pure_cpp,clahe\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ControlledIlluminationPureCppRunnerError,
                "header",
            ):
                load_cpp_frame_records(
                    csv_path
                )

    def test_missing_cpp_csv_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            missing_path = (
                Path(temporary)
                / "missing.csv"
            )

            with self.assertRaisesRegex(
                ControlledIlluminationPureCppRunnerError,
                "was not created",
            ):
                load_cpp_frame_records(
                    missing_path
                )

    def test_execute_runs_cpp_and_writes_artifacts(
        self,
    ) -> None:
        executable_path = Path(
            "VisionLabCppRealtime.exe"
        )
        project_root = Path("project-root")
        records = (object(),)
        expected_paths = (
            Path("realtime_frame_results.csv"),
            Path("execution_summary.json"),
        )
        realtime_config = SimpleNamespace(
            warmup_frames=30,
        )
        timestamps = iter(
            [
                STARTED_AT,
                FINISHED_AT,
            ]
        )

        completed_process = subprocess.CompletedProcess(
            args=[str(executable_path)],
            returncode=0,
            stdout="completed",
            stderr="",
        )

        with (
            patch(
                f"{RUNNER_MODULE}."
                "load_runner_context_from_environment",
                return_value=self.context,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "load_realtime_config",
                return_value=realtime_config,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "find_pure_cpp_realtime_executable",
                return_value=executable_path,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "subprocess.run",
                return_value=completed_process,
            ) as run_process,
            patch(
                f"{RUNNER_MODULE}."
                "load_cpp_frame_records",
                return_value=records,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "write_completed_run_artifacts_atomic",
                return_value=expected_paths,
            ) as write_artifacts,
        ):
            actual_paths = execute_pure_cpp_run(
                {},
                now_provider=lambda: next(
                    timestamps
                ),
                project_root=project_root,
            )

        self.assertEqual(
            actual_paths,
            expected_paths,
        )

        process_environment = (
            run_process.call_args.kwargs["env"]
        )

        self.assertIn(
            RAW_RESULTS_ENVIRONMENT_VARIABLE,
            process_environment,
        )
        self.assertTrue(
            process_environment[
                RAW_RESULTS_ENVIRONMENT_VARIABLE
            ].endswith(
                ".pure_cpp_frame_results.raw.csv"
            )
        )

        write_artifacts.assert_called_once_with(
            self.context,
            records,
            started_at_utc=STARTED_AT,
            finished_at_utc=FINISHED_AT,
            warmup_frame_count=30,
        )

    def test_failed_cpp_process_is_rejected(
        self,
    ) -> None:
        completed_process = subprocess.CompletedProcess(
            args=["VisionLabCppRealtime.exe"],
            returncode=1,
            stdout="",
            stderr="C++ failure",
        )

        with (
            patch(
                f"{RUNNER_MODULE}."
                "load_runner_context_from_environment",
                return_value=self.context,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "load_realtime_config",
                return_value=SimpleNamespace(
                    warmup_frames=30
                ),
            ),
            patch(
                f"{RUNNER_MODULE}."
                "find_pure_cpp_realtime_executable",
                return_value=Path(
                    "VisionLabCppRealtime.exe"
                ),
            ),
            patch(
                f"{RUNNER_MODULE}."
                "subprocess.run",
                return_value=completed_process,
            ),
        ):
            with self.assertRaisesRegex(
                ControlledIlluminationPureCppRunnerError,
                "exit code 1",
            ):
                execute_pure_cpp_run(
                    {},
                    now_provider=lambda: STARTED_AT,
                    project_root=Path(
                        "project-root"
                    ),
                )

    def test_cli_returns_failure_exit_code(
        self,
    ) -> None:
        captured_error = StringIO()

        with (
            patch(
                f"{RUNNER_MODULE}."
                "execute_pure_cpp_run",
                side_effect=RuntimeError(
                    "runner failure"
                ),
            ),
            redirect_stderr(captured_error),
        ):
            exit_code = run_cli()

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "runner failure",
            captured_error.getvalue(),
        )

    def test_cli_reports_success(
        self,
    ) -> None:
        captured_output = StringIO()

        with (
            patch(
                f"{RUNNER_MODULE}."
                "execute_pure_cpp_run",
                return_value=(
                    Path("frames.csv"),
                    Path("summary.json"),
                ),
            ),
            redirect_stdout(captured_output),
        ):
            exit_code = run_cli()

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "run completed",
            captured_output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()