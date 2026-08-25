from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.experiments.controlled_illumination_hybrid_runner import (
    ControlledIlluminationHybridRunnerError,
    execute_hybrid_run,
    run_cli,
    select_algorithm_configuration,
    validate_context_against_configuration,
)
from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    PlannedRun,
)
from benchmarks.experiments.controlled_illumination_runner_context import (
    ControlledIlluminationRunnerContext,
)


RUNNER_MODULE = (
    "benchmarks.experiments."
    "controlled_illumination_hybrid_runner"
)

STARTED_AT = "2026-08-25T11:00:00Z"
FINISHED_AT = "2026-08-25T11:01:00Z"


class ControlledIlluminationHybridRunnerTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.planned_run = PlannedRun(
            execution_order=1,
            experiment_id="experiment-test",
            run_id="experiment-test-run-0001",
            phase="constant_lux",
            platform="desktop",
            architecture="hybrid",
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
            "name": "clahe",
            "parameters": {
                "clip_limit": 4.0,
                "grid_size": 8,
            },
        }

        with patch(
            f"{RUNNER_MODULE}.load_algorithms",
            return_value=[algorithm_config],
        ):
            selected = (
                select_algorithm_configuration(
                    {},
                    "clahe",
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
                ControlledIlluminationHybridRunnerError
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
                ControlledIlluminationHybridRunnerError
            ):
                validate_context_against_configuration(
                    invalid_context,
                    {},
                    self.realtime_config,
                )

    def test_execute_loads_hybrid_module_and_writes_artifacts(
        self,
    ) -> None:
        benchmark_config = {
            "test": "benchmark-config",
        }
        realtime_config = SimpleNamespace(
            warmup_frames=30,
        )
        algorithm_config = {
            "name": "clahe",
            "parameters": {
                "clip_limit": 4.0,
                "grid_size": 8,
            },
        }
        hybrid_module = object()
        processor = object()
        frame_source = object()
        records = (object(),)
        video_path = Path("benchmark_input.mp4")
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
            ),
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
                "load_hybrid_module",
                return_value=hybrid_module,
            ) as load_module,
            patch(
                f"{RUNNER_MODULE}."
                "create_frame_processor",
                return_value=processor,
            ) as create_processor,
            patch(
                f"{RUNNER_MODULE}."
                "resolve_video_path",
                return_value=video_path,
            ),
            patch(
                f"{RUNNER_MODULE}."
                "iter_video_frames",
                return_value=frame_source,
            ),
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
            actual_paths = execute_hybrid_run(
                now_provider=lambda: next(
                    timestamps
                )
            )

        self.assertEqual(
            actual_paths,
            expected_paths,
        )
        load_module.assert_called_once_with()
        create_processor.assert_called_once_with(
            algorithm_config,
            hybrid_module,
        )
        run_trial.assert_called_once_with(
            frame_source=frame_source,
            processor=processor,
            config=realtime_config,
            architecture="hybrid",
            algorithm="clahe",
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
                "execute_hybrid_run",
                side_effect=RuntimeError(
                    "hybrid runner failure"
                ),
            ),
            redirect_stderr(captured_error),
        ):
            exit_code = run_cli()

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "hybrid runner failure",
            captured_error.getvalue(),
        )

    def test_cli_reports_success(
        self,
    ) -> None:
        captured_output = StringIO()

        with (
            patch(
                f"{RUNNER_MODULE}."
                "execute_hybrid_run",
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