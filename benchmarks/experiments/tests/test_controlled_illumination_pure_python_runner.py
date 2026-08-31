from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_pure_python_runner import (
    ControlledIlluminationPurePythonRunnerError,
    execute_pure_python_run,
    run_cli,
    select_algorithm_configuration,
    validate_context_against_configuration,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    PlannedRun,
)
from benchmarks.experiments.controlled_illumination_runner_context import (
    ControlledIlluminationRunnerContext,
)

from benchmarks.experiments import (
    controlled_illumination_pure_python_runner
    as pure_python_runner,
)

RUNNER_MODULE = (
    "benchmarks.experiments."
    "controlled_illumination_pure_python_runner"
)

STARTED_AT = "2026-08-25T10:00:00Z"
FINISHED_AT = "2026-08-25T10:01:00Z"


class ControlledIlluminationPurePythonRunnerTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.planned_run = PlannedRun(
            execution_order=1,
            experiment_id="experiment-test",
            run_id="experiment-test-run-0001",
            phase="constant_lux",
            platform="desktop",
            architecture="pure_python",
            algorithm="gamma_correction",
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
                planned_run=self.planned_run,
                results_root=Path("results"),
                output_directory=(
                    Path("results")
                    / "desktop"
                    / "experiment-test"
                    / "experiment-test-run-0001"
                ),
            )
        )

        self.realtime_config = SimpleNamespace(
            target_fps=30.0,
            deadline_ms=1000.0 / 30.0,
            trial_count=5,
            warmup_frames=30,
        )

    def test_algorithm_configuration_is_selected(
        self,
    ) -> None:
        algorithm_config = {
            "name": "gamma_correction",
            "parameters": {
                "gamma_value": 0.6,
            },
        }

        with patch(
            f"{RUNNER_MODULE}.load_algorithms",
            return_value=[
                {
                    "name": "original",
                    "parameters": {},
                },
                algorithm_config,
            ],
        ):
            selected = (
                select_algorithm_configuration(
                    {},
                    "gamma_correction",
                )
            )

        self.assertEqual(
            selected,
            algorithm_config,
        )

    def test_missing_algorithm_is_rejected(
        self,
    ) -> None:
        with patch(
            f"{RUNNER_MODULE}.load_algorithms",
            return_value=[
                {
                    "name": "original",
                    "parameters": {},
                },
            ],
        ):
            with self.assertRaises(
                ControlledIlluminationPurePythonRunnerError
            ):
                select_algorithm_configuration(
                    {},
                    "clahe",
                )

    def test_valid_context_matches_configuration(
        self,
    ) -> None:
        with patch(
            f"{RUNNER_MODULE}.load_resolutions",
            return_value=[
                (640, 480),
                (1280, 720),
            ],
        ):
            validate_context_against_configuration(
                self.context,
                {},
                self.realtime_config,
            )

    def test_unsupported_resolution_is_rejected(
        self,
    ) -> None:
        invalid_context = replace(
            self.context,
            planned_run=replace(
                self.planned_run,
                resolution=ResolutionMetadata(
                    width=1920,
                    height=1080,
                ),
            ),
        )

        with patch(
            f"{RUNNER_MODULE}.load_resolutions",
            return_value=[
                (640, 480),
            ],
        ):
            with self.assertRaises(
                ControlledIlluminationPurePythonRunnerError
            ):
                validate_context_against_configuration(
                    invalid_context,
                    {},
                    self.realtime_config,
                )

    def test_trial_number_is_validated(
        self,
    ) -> None:
        invalid_context = replace(
            self.context,
            planned_run=replace(
                self.planned_run,
                trial_number=6,
            ),
        )

        with patch(
            f"{RUNNER_MODULE}.load_resolutions",
            return_value=[
                (640, 480),
            ],
        ):
            with self.assertRaises(
                ControlledIlluminationPurePythonRunnerError
            ):
                validate_context_against_configuration(
                    invalid_context,
                    {},
                    self.realtime_config,
                )

    def test_target_fps_is_validated(
        self,
    ) -> None:
        invalid_context = replace(
            self.context,
            planned_run=replace(
                self.planned_run,
                target_fps=60.0,
                frame_deadline_ms=1000.0 / 60.0,
            ),
        )

        with patch(
            f"{RUNNER_MODULE}.load_resolutions",
            return_value=[
                (640, 480),
            ],
        ):
            with self.assertRaisesRegex(
                ControlledIlluminationPurePythonRunnerError,
                "target FPS",
            ):
                validate_context_against_configuration(
                    invalid_context,
                    {},
                    self.realtime_config,
                )

    def test_execute_writes_completed_artifacts(
        self,
    ) -> None:
        benchmark_config = {
            "test": "benchmark-config",
        }
        realtime_config = SimpleNamespace(
            target_fps=30.0,
            warmup_frames=30,
        )
        algorithm_config = {
            "name": "gamma_correction",
            "parameters": {
                "gamma_value": 0.6,
            },
        }
        processor = object()
        frame_source = object()
        environment = {
            "VISIONLAB_INPUT_SOURCE": "camera",
            "VISIONLAB_CAMERA_INDEX": "2",
        }
        records = (object(),)
        expected_paths = (
            Path("realtime_frame_results.csv"),
            Path("execution_summary.json"),
        )

        timestamps = iter(
            [
                STARTED_AT,
                FINISHED_AT,
            ]
        )

        with (
            patch(
                f"{RUNNER_MODULE}."
                "load_runner_context_from_environment",
                return_value=self.context,
            ) as load_context,
            patch(
                f"{RUNNER_MODULE}."
                "load_benchmark_config",
                return_value=benchmark_config,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "load_realtime_config",
                return_value=realtime_config,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "validate_shared_execution_counts",
            ),
            patch(
                f"{RUNNER_MODULE}."
                "validate_context_against_configuration",
            ),
            patch(
                f"{RUNNER_MODULE}."
                "select_algorithm_configuration",
                return_value=algorithm_config,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "create_frame_processor",
                return_value=processor,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "create_frame_source",
                return_value=frame_source,
            ) as create_source,
            patch(
                f"{RUNNER_MODULE}."
                "run_realtime_trial",
                return_value=records,
            ) as run_trial,
            patch(
                f"{RUNNER_MODULE}."
                "write_completed_run_artifacts_atomic",
                return_value=expected_paths,
            ) as write_artifacts,
        ):
            actual_paths = execute_pure_python_run(
                environment,
                now_provider=lambda: next(
                    timestamps
                ),
            )

        self.assertEqual(
            actual_paths,
            expected_paths,
        )
        load_context.assert_called_once_with(
            environment,
            expected_architecture="pure_python",
        )
        create_source.assert_called_once_with(
            benchmark_config,
            width=640,
            height=480,
            fps=30.0,
            environment=environment,
        )
        run_trial.assert_called_once_with(
            frame_source=frame_source,
            processor=processor,
            config=realtime_config,
            architecture="pure_python",
            algorithm="gamma_correction",
            width=640,
            height=480,
            trial=1,
        )
        write_artifacts.assert_called_once_with(
            self.context,
            records,
            started_at_utc=STARTED_AT,
            finished_at_utc=FINISHED_AT,
            warmup_frame_count=30,
        )

    def test_cli_returns_failure_exit_code(
        self,
    ) -> None:
        captured_error = StringIO()

        with (
            patch(
                f"{RUNNER_MODULE}."
                "execute_pure_python_run",
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
                "execute_pure_python_run",
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

    def test_video_input_is_default(
        self,
    ) -> None:
        benchmark_config = {
            "test": "benchmark-config",
        }
        video_path = Path(
            "benchmark_input.mp4"
        )
        frame_source = object()

        with (
            patch(
                f"{RUNNER_MODULE}."
                "resolve_video_path",
                return_value=video_path,
            ) as resolve_video,
            patch(
                f"{RUNNER_MODULE}."
                "iter_video_frames",
                return_value=frame_source,
            ) as iter_video,
        ):
            selected_source = (
                pure_python_runner
                .create_frame_source(
                    benchmark_config,
                    environment={},
                )
            )

        self.assertIs(
            selected_source,
            frame_source,
        )
        resolve_video.assert_called_once_with(
            benchmark_config
        )
        iter_video.assert_called_once_with(
            video_path
        )

    def test_unsupported_input_source_is_rejected(
        self,
    ) -> None:
        benchmark_config = {
            "test": "benchmark-config",
        }

        with (
            patch(
                f"{RUNNER_MODULE}."
                "resolve_video_path",
                return_value=Path(
                    "benchmark_input.mp4"
                ),
            ),
            patch(
                f"{RUNNER_MODULE}."
                "iter_video_frames",
                return_value=object(),
            ),
        ):
            with self.assertRaisesRegex(
                ControlledIlluminationPurePythonRunnerError,
                "Unsupported input source: usb",
            ):
                pure_python_runner.create_frame_source(
                    benchmark_config,
                    environment={
                        "VISIONLAB_INPUT_SOURCE": "usb",
                    },
                )

    def test_camera_input_uses_configured_index(
        self,
    ) -> None:
        benchmark_config = {
            "test": "benchmark-config",
        }
        frame_source = object()

        with (
            patch(
                f"{RUNNER_MODULE}."
                "iter_camera_frames",
                return_value=frame_source,
            ) as iter_camera,
            patch(
                f"{RUNNER_MODULE}."
                "resolve_video_path",
            ) as resolve_video,
            patch(
                f"{RUNNER_MODULE}."
                "iter_video_frames",
            ) as iter_video,
        ):
            selected_source = (
                pure_python_runner
                .create_frame_source(
                    benchmark_config,
                    width=640,
                    height=480,
                    fps=30.0,
                    environment={
                        "VISIONLAB_INPUT_SOURCE": (
                            "camera"
                        ),
                        "VISIONLAB_CAMERA_INDEX": "2",
                    },
                )
            )

        self.assertIs(
            selected_source,
            frame_source,
        )
        iter_camera.assert_called_once_with(
            2,
            width=640,
            height=480,
            fps=30.0,
        )
        resolve_video.assert_not_called()
        iter_video.assert_not_called()

    def test_invalid_camera_index_configuration_is_rejected(
        self,
    ) -> None:
        invalid_indices = (
            None,
            "",
            "-1",
            "1.5",
            "camera-zero",
        )

        for invalid_index in invalid_indices:
            with self.subTest(
                camera_index=invalid_index
            ):
                environment = {
                    "VISIONLAB_INPUT_SOURCE": (
                        "camera"
                    ),
                }

                if invalid_index is not None:
                    environment[
                        "VISIONLAB_CAMERA_INDEX"
                    ] = invalid_index

                with patch(
                    f"{RUNNER_MODULE}."
                    "iter_camera_frames",
                ) as iter_camera:
                    with self.assertRaisesRegex(
                        ControlledIlluminationPurePythonRunnerError,
                        (
                            "VISIONLAB_CAMERA_INDEX "
                            "must be a non-negative "
                            "integer"
                        ),
                    ):
                        (
                            pure_python_runner
                            .create_frame_source(
                                {},
                                environment=environment,
                            )
                        )

                iter_camera.assert_not_called()

    def test_failed_camera_run_does_not_write_artifacts(
        self,
    ) -> None:
        environment = {
            "VISIONLAB_INPUT_SOURCE": "camera",
            "VISIONLAB_CAMERA_INDEX": "0",
        }
        realtime_config = SimpleNamespace(
            target_fps=30.0,
            warmup_frames=30,
        )

        with (
            patch(
                f"{RUNNER_MODULE}."
                "load_runner_context_from_environment",
                return_value=self.context,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "load_benchmark_config",
                return_value={},
            ),
            patch(
                f"{RUNNER_MODULE}."
                "load_realtime_config",
                return_value=realtime_config,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "validate_shared_execution_counts",
            ),
            patch(
                f"{RUNNER_MODULE}."
                "validate_context_against_configuration",
            ),
            patch(
                f"{RUNNER_MODULE}."
                "select_algorithm_configuration",
                return_value={},
            ),
            patch(
                f"{RUNNER_MODULE}."
                "create_frame_processor",
                return_value=object(),
            ),
            patch(
                f"{RUNNER_MODULE}."
                "create_frame_source",
                return_value=object(),
            ),
            patch(
                f"{RUNNER_MODULE}."
                "run_realtime_trial",
                side_effect=RuntimeError(
                    "Camera 0 could not be opened."
                ),
            ),
            patch(
                f"{RUNNER_MODULE}."
                "write_completed_run_artifacts_atomic",
            ) as write_artifacts,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Camera 0 could not be opened",
            ):
                execute_pure_python_run(
                    environment,
                    now_provider=lambda: STARTED_AT,
                )

        write_artifacts.assert_not_called()

if __name__ == "__main__":
    unittest.main()
