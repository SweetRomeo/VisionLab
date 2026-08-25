from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.experiments.controlled_illumination_runner_context import (
    ControlledIlluminationRunnerContextError,
    load_runner_context_from_environment,
)


class ControlledIlluminationRunnerContextTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.environment = {
            "VISIONLAB_EXPERIMENT_ID": (
                "controlled-illumination-test"
            ),
            "VISIONLAB_RUN_ID": "test-run-0001",
            "VISIONLAB_EXECUTION_ORDER": "1",
            "VISIONLAB_PHASE": "constant_lux",
            "VISIONLAB_PLATFORM": "desktop",
            "VISIONLAB_ARCHITECTURE": (
                "pure_python"
            ),
            "VISIONLAB_ALGORITHM": "original",
            "VISIONLAB_RESOLUTION_WIDTH": "640",
            "VISIONLAB_RESOLUTION_HEIGHT": "480",
            "VISIONLAB_TRIAL_NUMBER": "1",
            (
                "VISIONLAB_INCIDENCE_"
                "ANGLE_DEGREES"
            ): "0",
            "VISIONLAB_TARGET_ILLUMINANCE_LUX": (
                "50"
            ),
            "VISIONLAB_SOURCE_OUTPUT_SETTING": "",
            "VISIONLAB_TARGET_FPS": "30",
            "VISIONLAB_FRAME_DEADLINE_MS": (
                "33.333333"
            ),
        }

    def test_valid_constant_lux_context(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            self.environment[
                "VISIONLAB_RESULTS_ROOT"
            ] = temporary

            context = (
                load_runner_context_from_environment(
                    self.environment,
                    expected_architecture="pure_python",
                )
            )

            self.assertEqual(
                context.experiment_id,
                "controlled-illumination-test",
            )
            self.assertEqual(
                context.run_id,
                "test-run-0001",
            )
            self.assertEqual(
                context.architecture,
                "pure_python",
            )
            self.assertEqual(
                context.algorithm,
                "original",
            )
            self.assertEqual(
                context.resolution.width,
                640,
            )
            self.assertEqual(
                context.resolution.height,
                480,
            )
            self.assertEqual(
                (
                    context.planned_run
                    .target_illuminance_lux
                ),
                50.0,
            )
            self.assertIsNone(
                context.planned_run
                .source_output_setting
            )
            self.assertEqual(
                context.output_directory,
                (
                    Path(temporary).resolve()
                    / "desktop"
                    / "controlled-illumination-test"
                    / "test-run-0001"
                ),
            )

    def test_valid_constant_source_context(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_PHASE"
        ] = "constant_source"
        self.environment[
            "VISIONLAB_TARGET_ILLUMINANCE_LUX"
        ] = ""
        self.environment[
            "VISIONLAB_SOURCE_OUTPUT_SETTING"
        ] = "device_setting_low"

        context = load_runner_context_from_environment(
            self.environment
        )

        self.assertIsNone(
            context.planned_run
            .target_illuminance_lux
        )
        self.assertEqual(
            context.planned_run
            .source_output_setting,
            "device_setting_low",
        )

    def test_numeric_source_setting_is_parsed(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_PHASE"
        ] = "constant_source"
        self.environment[
            "VISIONLAB_TARGET_ILLUMINANCE_LUX"
        ] = ""
        self.environment[
            "VISIONLAB_SOURCE_OUTPUT_SETTING"
        ] = "50"

        context = load_runner_context_from_environment(
            self.environment
        )

        self.assertEqual(
            context.planned_run
            .source_output_setting,
            50.0,
        )

    def test_missing_environment_value_is_rejected(
        self,
    ) -> None:
        del self.environment[
            "VISIONLAB_RUN_ID"
        ]

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "VISIONLAB_RUN_ID",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_unsafe_identifier_is_rejected(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_RUN_ID"
        ] = "../outside"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "unsafe",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_invalid_integer_is_rejected(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_TRIAL_NUMBER"
        ] = "1.5"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "must be an integer",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_non_positive_integer_is_rejected(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_EXECUTION_ORDER"
        ] = "0"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "must be positive",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_non_finite_number_is_rejected(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_TARGET_FPS"
        ] = "nan"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "must be finite",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_negative_angle_is_rejected(
        self,
    ) -> None:
        self.environment[
            (
                "VISIONLAB_INCIDENCE_"
                "ANGLE_DEGREES"
            )
        ] = "-1"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "must be non-negative",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_unsupported_architecture_is_rejected(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_ARCHITECTURE"
        ] = "unknown"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "Unsupported architecture",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_expected_architecture_mismatch(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "architecture mismatch",
        ):
            load_runner_context_from_environment(
                self.environment,
                expected_architecture="hybrid",
            )

    def test_unsupported_algorithm_is_rejected(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_ALGORITHM"
        ] = "unknown"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "Unsupported algorithm",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_unsupported_phase_is_rejected(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_PHASE"
        ] = "unknown"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "Unsupported experiment phase",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_constant_lux_requires_target_lux(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_TARGET_ILLUMINANCE_LUX"
        ] = ""

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "target_illuminance_lux",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_constant_lux_rejects_source_setting(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_SOURCE_OUTPUT_SETTING"
        ] = "device_setting_low"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "source_output_setting",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_constant_source_requires_setting(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_PHASE"
        ] = "constant_source"
        self.environment[
            "VISIONLAB_TARGET_ILLUMINANCE_LUX"
        ] = ""
        self.environment[
            "VISIONLAB_SOURCE_OUTPUT_SETTING"
        ] = ""

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "source_output_setting",
        ):
            load_runner_context_from_environment(
                self.environment
            )

    def test_constant_source_rejects_target_lux(
        self,
    ) -> None:
        self.environment[
            "VISIONLAB_PHASE"
        ] = "constant_source"
        self.environment[
            "VISIONLAB_SOURCE_OUTPUT_SETTING"
        ] = "device_setting_low"

        with self.assertRaisesRegex(
            ControlledIlluminationRunnerContextError,
            "target_illuminance_lux",
        ):
            load_runner_context_from_environment(
                self.environment
            )


if __name__ == "__main__":
    unittest.main()