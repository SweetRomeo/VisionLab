import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from benchmarks.realtime import (
    hybrid_realtime as runner,
)
from benchmarks.realtime.realtime_config import (
    RealtimeConfig,
)


class FakeProcessingAlgorithm:
    ORIGINAL = "original-enum"
    GAMMA = "gamma-enum"
    HISTOGRAM = "histogram-enum"
    CLAHE = "clahe-enum"


class FakeHybridModule:
    ProcessingAlgorithm = (
        FakeProcessingAlgorithm
    )

    def __init__(self) -> None:
        self.calls = []

    def process_frame(
        self,
        frame,
        algorithm,
        **parameters,
    ):
        self.calls.append(
            {
                "algorithm": algorithm,
                "parameters": parameters,
            }
        )

        return frame.copy()


class HybridRealtimeRunnerTests(
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

    def test_release_directory_is_accepted(
        self,
    ) -> None:
        candidate = Path(
            "build/Release/visionlab_cpp.so"
        )

        self.assertTrue(
            runner.is_release_candidate(
                candidate,
                Path("build"),
            )
        )

    def test_single_config_release_cache_is_accepted(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            build_directory = Path(directory)
            cache_path = (
                build_directory
                / "CMakeCache.txt"
            )
            cache_path.write_text(
                "CMAKE_BUILD_TYPE:STRING=Release\n",
                encoding="utf-8",
            )

            candidate = (
                build_directory
                / "visionlab_cpp.so"
            )

            self.assertTrue(
                runner.is_release_candidate(
                    candidate,
                    build_directory,
                )
            )

    def test_configured_release_module_is_found(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            release_directory = (
                Path(directory) / "Release"
            )
            release_directory.mkdir()

            module_path = (
                release_directory
                / "visionlab_cpp.test.pyd"
            )
            module_path.write_bytes(b"module")

            with patch.dict(
                os.environ,
                {
                    "VISIONLAB_CPP_MODULE_DIR": (
                        str(release_directory)
                    )
                },
            ):
                result = (
                    runner
                    .find_hybrid_module_directory()
                )

            self.assertEqual(
                result,
                release_directory.resolve(),
            )

    def test_processor_calls_cpp_module_with_parameters(
        self,
    ) -> None:
        hybrid_module = FakeHybridModule()

        processor = (
            runner.create_frame_processor(
                {
                    "name": "clahe",
                    "parameters": {
                        "gamma_value": 0.7,
                        "clip_limit": 3.5,
                        "grid_size": 6,
                    },
                },
                hybrid_module,
            )
        )

        source = np.zeros(
            (12, 16, 3),
            dtype=np.uint8,
        )
        result = processor(source)

        self.assertTrue(
            np.array_equal(result, source)
        )
        self.assertEqual(
            len(hybrid_module.calls),
            1,
        )
        self.assertEqual(
            hybrid_module.calls[0][
                "algorithm"
            ],
            FakeProcessingAlgorithm.CLAHE,
        )
        self.assertEqual(
            hybrid_module.calls[0][
                "parameters"
            ],
            {
                "gamma_value": 0.7,
                "clahe_clip_limit": 3.5,
                "clahe_grid_size": 6,
            },
        )

    def test_invalid_parameter_is_rejected(
        self,
    ) -> None:
        hybrid_module = FakeHybridModule()

        with self.assertRaisesRegex(
            ValueError,
            "grid_size",
        ):
            runner.create_frame_processor(
                {
                    "name": "clahe",
                    "parameters": {
                        "grid_size": True,
                    },
                },
                hybrid_module,
            )

    def test_all_experiments_delegate_to_pipeline(
        self,
    ) -> None:
        realtime_config = (
            self.create_realtime_config()
        )
        benchmark_config = (
            self.create_benchmark_config()
        )
        hybrid_module = FakeHybridModule()

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
                        hybrid_module,
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

        first_arguments = (
            trial_mock.call_args_list[0].kwargs
        )
        second_arguments = (
            trial_mock.call_args_list[1].kwargs
        )

        self.assertEqual(
            first_arguments["architecture"],
            "hybrid",
        )
        self.assertEqual(
            first_arguments["algorithm"],
            "original",
        )
        self.assertEqual(
            first_arguments["trial"],
            1,
        )
        self.assertEqual(
            second_arguments["trial"],
            2,
        )
        self.assertTrue(
            callable(
                first_arguments["processor"]
            )
        )

    def test_load_module_accepts_expected_path(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            module_directory = Path(directory)
            module_path = (
                module_directory
                / "visionlab_cpp.test.pyd"
            )
            module_path.write_bytes(b"module")

            fake_module = SimpleNamespace(
                __file__=str(module_path),
                process_frame=lambda *args: None,
                ProcessingAlgorithm=(
                    FakeProcessingAlgorithm
                ),
            )

            with (
                patch.object(
                    runner,
                    "find_hybrid_module_directory",
                    return_value=module_directory,
                ),
                patch.object(
                    runner.importlib,
                    "import_module",
                    return_value=fake_module,
                ),
            ):
                loaded_module = (
                    runner.load_hybrid_module()
                )

            self.assertIs(
                loaded_module,
                fake_module,
            )

    def test_load_module_rejects_unexpected_path(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root_directory = Path(directory)
            expected_directory = (
                root_directory / "expected"
            )
            unexpected_directory = (
                root_directory / "unexpected"
            )
            expected_directory.mkdir()
            unexpected_directory.mkdir()

            unexpected_module_path = (
                unexpected_directory
                / "visionlab_cpp.test.pyd"
            )
            unexpected_module_path.write_bytes(
                b"module"
            )

            fake_module = SimpleNamespace(
                __file__=str(
                    unexpected_module_path
                ),
                process_frame=lambda *args: None,
                ProcessingAlgorithm=(
                    FakeProcessingAlgorithm
                ),
            )

            with (
                patch.object(
                    runner,
                    "find_hybrid_module_directory",
                    return_value=(
                        expected_directory
                    ),
                ),
                patch.object(
                    runner.importlib,
                    "import_module",
                    return_value=fake_module,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "unexpected directory",
                ):
                    runner.load_hybrid_module()


if __name__ == "__main__":
    unittest.main()