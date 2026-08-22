from dataclasses import replace
import json
from pathlib import Path
import unittest

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlan,
    ControlledIlluminationRunPlanError,
    PlannedRun,
    build_run_conditions,
)


class ControlledIlluminationRunPlannerTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.base_run = PlannedRun(
            execution_order=1,
            experiment_id="experiment-test",
            run_id="run-1",
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

    def build_plan(
        self,
        runs: tuple[PlannedRun, ...],
        *,
        generated_at_utc: str = (
            "2026-08-22T11:00:00Z"
        ),
    ) -> ControlledIlluminationRunPlan:
        return ControlledIlluminationRunPlan(
            schema_version=1,
            generated_at_utc=generated_at_utc,
            experiment_id="experiment-test",
            randomized=False,
            randomization_seed=20260821,
            runs=runs,
        )

    def test_valid_plan_is_serialized(
        self,
    ) -> None:
        plan = self.build_plan(
            (self.base_run,)
        )

        serialized_plan = plan.to_dict()

        self.assertEqual(plan.run_count, 1)
        self.assertEqual(
            serialized_plan["run_count"],
            1,
        )
        self.assertEqual(
            serialized_plan["runs"][0][
                "resolution"
            ],
            {
                "width": 640,
                "height": 480,
            },
        )

    def test_valid_constant_source_run(
        self,
    ) -> None:
        constant_source_run = replace(
            self.base_run,
            phase="constant_source",
            target_illuminance_lux=None,
            source_output_setting=50.0,
        )

        self.assertEqual(
            constant_source_run.phase,
            "constant_source",
        )

    def test_constant_lux_requires_target(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunPlanError
        ):
            replace(
                self.base_run,
                target_illuminance_lux=None,
            )

    def test_constant_source_requires_output_setting(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunPlanError
        ):
            replace(
                self.base_run,
                phase="constant_source",
                target_illuminance_lux=None,
                source_output_setting=None,
            )

    def test_duplicate_run_ids_are_rejected(
        self,
    ) -> None:
        duplicate_run = replace(
            self.base_run,
            execution_order=2,
        )

        with self.assertRaises(
            ControlledIlluminationRunPlanError
        ):
            self.build_plan(
                (
                    self.base_run,
                    duplicate_run,
                )
            )

    def test_non_sequential_orders_are_rejected(
        self,
    ) -> None:
        second_run = replace(
            self.base_run,
            execution_order=3,
            run_id="run-2",
        )

        with self.assertRaises(
            ControlledIlluminationRunPlanError
        ):
            self.build_plan(
                (
                    self.base_run,
                    second_run,
                )
            )

    def test_mismatched_experiment_id_is_rejected(
        self,
    ) -> None:
        second_run = replace(
            self.base_run,
            execution_order=2,
            experiment_id="another-experiment",
            run_id="run-2",
            trial_number=2,
        )

        with self.assertRaises(
            ControlledIlluminationRunPlanError
        ):
            self.build_plan(
                (
                    self.base_run,
                    second_run,
                )
            )

    def test_invalid_timestamp_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunPlanError
        ):
            self.build_plan(
                (self.base_run,),
                generated_at_utc="invalid",
            )

    def test_invalid_resolution_type_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunPlanError
        ):
            replace(
                self.base_run,
                resolution="640x480",  # type: ignore[arg-type]
            )

    def load_experiment_config(self) -> dict:
        config_path = (
                Path(__file__).resolve().parents[1]
                / "config"
                / "controlled_illumination_config.json"
        )

        with config_path.open(
                "r",
                encoding="utf-8",
        ) as config_file:
            return json.load(config_file)

    def test_constant_source_requires_output_settings(
            self,
    ) -> None:
        config = self.load_experiment_config()

        with self.assertRaisesRegex(
                ControlledIlluminationRunPlanError,
                "source_output_settings",
        ):
            build_run_conditions(config)

    def test_complete_experiment_matrix_is_expanded(
            self,
    ) -> None:
        config = self.load_experiment_config()

        source_output_settings = [25, 50]
        config["experiment_matrix"][
            "source_output_settings"
        ] = source_output_settings

        conditions = build_run_conditions(config)
        matrix = config["experiment_matrix"]

        common_condition_count = (
                len(matrix["platforms"])
                * len(matrix["architectures"])
                * len(matrix["algorithms"])
                * len(matrix["resolutions"])
                * len(matrix["incidence_angles_degrees"])
                * matrix["trial_count"]
        )

        expected_constant_lux_count = (
                common_condition_count
                * len(
            matrix[
                "target_illuminance_levels_lux"
            ]
        )
        )
        expected_constant_source_count = (
                common_condition_count
                * len(source_output_settings)
        )

        constant_lux_count = sum(
            condition.phase == "constant_lux"
            for condition in conditions
        )
        constant_source_count = sum(
            condition.phase == "constant_source"
            for condition in conditions
        )

        self.assertEqual(
            constant_lux_count,
            expected_constant_lux_count,
        )
        self.assertEqual(
            constant_source_count,
            expected_constant_source_count,
        )
        self.assertEqual(
            len(conditions),
            (
                    expected_constant_lux_count
                    + expected_constant_source_count
            ),
        )

    def test_duplicate_conditions_are_rejected(
            self,
    ) -> None:
        duplicate_condition = replace(
            self.base_run,
            execution_order=2,
            run_id="run-2",
        )

        with self.assertRaisesRegex(
                ControlledIlluminationRunPlanError,
                "Duplicate experimental conditions",
        ):
            self.build_plan(
                (
                    self.base_run,
                    duplicate_condition,
                )
            )


if __name__ == "__main__":
    unittest.main()