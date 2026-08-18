import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from benchmarks.realtime import (
    pure_python_realtime as runner,
)
from benchmarks.realtime.realtime_config import (
    RealtimeConfig,
)


class PurePythonRealtimeRunnerTests(
    unittest.TestCase
):
    def create_realtime_config(
        self,
    ) -> RealtimeConfig:
        return RealtimeConfig(
            schema_version=1,
            target_fps=30.0,
            queue_capacity=1,
            warmup_frames=1,
            measured_frames=2,
            trial_count=2,
            drop_policy="latest_frame",
            deadline_multiplier=1.0,
            output_directory=Path(
                "benchmarks/results/realtime"
            ),
            frame_results_file=(
                "realtime_frame_results.csv"
            ),
            summary_file=(
                "realtime_summary.csv"
            ),
        )

    def create_benchmark_config(
        self,
    ) -> dict:
        return {
            "input": {
                "video_path": "video.mp4",
            },
            "benchmark": {
                "warmup_frames": 1,
                "measured_frames": 2,
                "trials": 2,
            },
            "resolutions": [
                {
                    "width": 16,
                    "height": 12,
                }
            ],
            "algorithms": [
                {
                    "name": "original",
                    "parameters": {},
                }
            ],
        }

    def test_mismatched_execution_counts_are_rejected(
        self,
    ) -> None:
        benchmark_config = (
            self.create_benchmark_config()
        )
        benchmark_config["benchmark"][
            "measured_frames"
        ] = 3

        with self.assertRaisesRegex(
            ValueError,
            "counts must match",
        ):
            runner.validate_shared_execution_counts(
                benchmark_config,
                self.create_realtime_config(),
            )

    def test_duplicate_resolutions_are_rejected(
        self,
    ) -> None:
        benchmark_config = {
            "resolutions": [
                {
                    "width": 640,
                    "height": 480,
                },
                {
                    "width": 640,
                    "height": 480,
                },
            ]
        }

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate resolutions",
        ):
            runner.load_resolutions(
                benchmark_config
            )

    def test_unsupported_algorithm_is_rejected(
        self,
    ) -> None:
        benchmark_config = {
            "algorithms": [
                {
                    "name": "unsupported",
                    "parameters": {},
                }
            ]
        }

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported algorithm",
        ):
            runner.load_algorithms(
                benchmark_config
            )

    def test_all_algorithm_processors_return_valid_images(
        self,
    ) -> None:
        source = (
            np.arange(
                12 * 16 * 3,
                dtype=np.int32,
            )
            .reshape((12, 16, 3))
            .astype(np.uint8)
        )
        source_before = source.copy()

        algorithm_configs = [
            {
                "name": "original",
                "parameters": {},
            },
            {
                "name": "gamma_correction",
                "parameters": {
                    "gamma_value": 0.6,
                },
            },
            {
                "name": (
                    "histogram_equalization"
                ),
                "parameters": {},
            },
            {
                "name": "clahe",
                "parameters": {
                    "clip_limit": 4.0,
                    "grid_size": 8,
                },
            },
        ]

        for algorithm_config in (
            algorithm_configs
        ):
            with self.subTest(
                algorithm=(
                    algorithm_config["name"]
                )
            ):
                processor = (
                    runner.create_frame_processor(
                        algorithm_config
                    )
                )
                result = processor(source)

                self.assertEqual(
                    result.shape,
                    source.shape,
                )
                self.assertEqual(
                    result.dtype,
                    source.dtype,
                )
                self.assertTrue(
                    np.array_equal(
                        source,
                        source_before,
                    )
                )

    def test_invalid_algorithm_parameter_is_rejected(
        self,
    ) -> None:
        algorithm_config = {
            "name": "gamma_correction",
            "parameters": {
                "gamma_value": True,
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            "gamma_value",
        ):
            runner.create_frame_processor(
                algorithm_config
            )

    def test_video_path_cannot_escape_repository(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            project_root = (
                temporary_root / "project"
            )
            project_root.mkdir()

            outside_video = (
                temporary_root / "outside.mp4"
            )
            outside_video.write_bytes(
                b"placeholder"
            )

            benchmark_config = {
                "input": {
                    "video_path": (
                        "../outside.mp4"
                    )
                }
            }

            with patch.object(
                runner,
                "PROJECT_ROOT",
                project_root,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "outside",
                ):
                    runner.resolve_video_path(
                        benchmark_config
                    )

    def test_all_experiments_delegate_to_shared_pipeline(
        self,
    ) -> None:
        realtime_config = (
            self.create_realtime_config()
        )
        benchmark_config = (
            self.create_benchmark_config()
        )

        with TemporaryDirectory() as directory:
            project_root = Path(directory)
            video_path = (
                project_root / "video.mp4"
            )
            video_path.write_bytes(
                b"placeholder"
            )

            with (
                patch.object(
                    runner,
                    "PROJECT_ROOT",
                    project_root,
                ),
                patch.object(
                    runner,
                    "iter_video_frames",
                    side_effect=lambda path: iter(()),
                ) as frame_source_mock,
                patch.object(
                    runner,
                    "run_realtime_trial",
                    return_value=[],
                ) as trial_mock,
            ):
                records = (
                    runner.run_all_experiments(
                        benchmark_config,
                        realtime_config,
                    )
                )

        self.assertEqual(records, [])
        self.assertEqual(
            trial_mock.call_count,
            2,
        )
        self.assertEqual(
            frame_source_mock.call_count,
            2,
        )

        first_trial_arguments = (
            trial_mock.call_args_list[0].kwargs
        )
        second_trial_arguments = (
            trial_mock.call_args_list[1].kwargs
        )

        self.assertEqual(
            first_trial_arguments[
                "architecture"
            ],
            "pure_python",
        )
        self.assertEqual(
            first_trial_arguments["algorithm"],
            "original",
        )
        self.assertEqual(
            first_trial_arguments["width"],
            16,
        )
        self.assertEqual(
            first_trial_arguments["height"],
            12,
        )
        self.assertEqual(
            first_trial_arguments["trial"],
            1,
        )
        self.assertEqual(
            second_trial_arguments["trial"],
            2,
        )
        self.assertTrue(
            callable(
                first_trial_arguments[
                    "processor"
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()