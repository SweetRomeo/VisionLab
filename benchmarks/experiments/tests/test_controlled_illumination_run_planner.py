from dataclasses import replace
import json
from pathlib import Path
import unittest
import csv
from tempfile import TemporaryDirectory

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlan,
    ControlledIlluminationRunPlanError,
    PlannedRun,
    build_run_conditions,
    build_run_plan,
    write_run_plan_manifests_atomic,
    load_run_plan_manifest,
    planned_run_from_dict,
    run_plan_from_dict,
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

    def build_small_matrix_config(self) -> dict:
        config = self.load_experiment_config()
        matrix = config["experiment_matrix"]

        matrix[
            "target_illuminance_levels_lux"
        ] = [5, 50]
        matrix[
            "source_output_settings"
        ] = [25, 50]
        matrix[
            "incidence_angles_degrees"
        ] = [0, 30]
        matrix["algorithms"] = [
            "original",
            "clahe",
        ]
        matrix["architectures"] = [
            "pure_python",
            "pure_cpp",
        ]
        matrix["platforms"] = ["desktop"]
        matrix["resolutions"] = [
            {
                "width": 640,
                "height": 480,
            }
        ]
        matrix["trial_count"] = 2

        config["execution"][
            "randomize_run_order"
        ] = True
        config["execution"][
            "randomization_seed"
        ] = 20260821

        return config

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

    def load_optical_screening_config(
        self,
    ) -> dict:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / (
                "controlled_illumination_"
                "optical_screening.json"
            )
        )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            return json.load(config_file)

    def test_optical_screening_profile_expands_to_300_conditions(
        self,
    ) -> None:
        config = (
            self.load_optical_screening_config()
        )

        conditions = build_run_conditions(
            config
        )

        self.assertEqual(
            len(conditions),
            300,
        )

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

    def test_same_seed_produces_same_run_order(
            self,
    ) -> None:
        config = self.build_small_matrix_config()

        first_plan = build_run_plan(
            config,
            experiment_id="experiment-seeded",
            generated_at_utc=(
                "2026-08-22T12:00:00Z"
            ),
        )
        second_plan = build_run_plan(
            config,
            experiment_id="experiment-seeded",
            generated_at_utc=(
                "2026-08-22T12:00:00Z"
            ),
        )

        self.assertEqual(
            [
                planned_run.condition_key
                for planned_run in first_plan.runs
            ],
            [
                planned_run.condition_key
                for planned_run in second_plan.runs
            ],
        )
        self.assertEqual(
            [
                planned_run.run_id
                for planned_run in first_plan.runs
            ],
            [
                planned_run.run_id
                for planned_run in second_plan.runs
            ],
        )

    def test_run_plan_assigns_sequential_unique_ids(
            self,
    ) -> None:
        config = self.build_small_matrix_config()

        plan = build_run_plan(
            config,
            experiment_id="experiment-sequence",
            generated_at_utc=(
                "2026-08-22T12:00:00Z"
            ),
        )

        self.assertEqual(plan.run_count, 64)
        self.assertEqual(
            [
                planned_run.execution_order
                for planned_run in plan.runs
            ],
            list(range(1, 65)),
        )

        run_ids = [
            planned_run.run_id
            for planned_run in plan.runs
        ]

        self.assertEqual(
            len(run_ids),
            len(set(run_ids)),
        )
        self.assertTrue(plan.randomized)

    def test_run_plan_manifests_are_written_atomically(
            self,
    ) -> None:
        config = self.build_small_matrix_config()

        plan = build_run_plan(
            config,
            experiment_id="experiment-manifest",
            generated_at_utc=(
                "2026-08-22T12:00:00Z"
            ),
        )

        with TemporaryDirectory() as temporary_directory:
            json_path, csv_path = (
                write_run_plan_manifests_atomic(
                    plan,
                    temporary_directory,
                )
            )

            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertEqual(
                json_path.name,
                "run_plan.json",
            )
            self.assertEqual(
                csv_path.name,
                "run_plan.csv",
            )

            with json_path.open(
                    "r",
                    encoding="utf-8",
            ) as json_file:
                json_data = json.load(json_file)

            self.assertEqual(
                json_data["run_count"],
                64,
            )
            self.assertEqual(
                len(json_data["runs"]),
                64,
            )

            with csv_path.open(
                    "r",
                    newline="",
                    encoding="utf-8",
            ) as csv_file:
                csv_rows = list(
                    csv.DictReader(csv_file)
                )

            self.assertEqual(len(csv_rows), 64)
            self.assertEqual(
                csv_rows[0]["run_id"],
                plan.runs[0].run_id,
            )

            temporary_files = [
                path
                for path in Path(
                    temporary_directory
                ).iterdir()
                if path.name.endswith(".tmp")
            ]

            self.assertEqual(temporary_files, [])

    def test_unsafe_experiment_id_is_rejected(
            self,
    ) -> None:
        config = self.build_small_matrix_config()

        with self.assertRaisesRegex(
                ControlledIlluminationRunPlanError,
                "unsafe characters",
        ):
            build_run_plan(
                config,
                experiment_id="../outside",
                generated_at_utc=(
                    "2026-08-22T12:00:00Z"
                ),
            )

    def test_planned_run_round_trip(
        self,
    ) -> None:
        plan = build_run_plan(
            self.build_small_matrix_config(),
            experiment_id="experiment-load",
            generated_at_utc=(
                "2026-08-23T09:00:00Z"
            ),
        )

        original_run = plan.runs[0]
        loaded_run = planned_run_from_dict(
            original_run.to_dict()
        )

        self.assertEqual(
            loaded_run.to_dict(),
            original_run.to_dict(),
        )

    def test_run_plan_round_trip(
        self,
    ) -> None:
        plan = build_run_plan(
            self.build_small_matrix_config(),
            experiment_id="experiment-load",
            generated_at_utc=(
                "2026-08-23T09:00:00Z"
            ),
        )

        loaded_plan = run_plan_from_dict(
            plan.to_dict()
        )

        self.assertEqual(
            loaded_plan.to_dict(),
            plan.to_dict(),
        )

    def test_written_manifest_can_be_loaded(
        self,
    ) -> None:
        plan = build_run_plan(
            self.build_small_matrix_config(),
            experiment_id="experiment-load",
            generated_at_utc=(
                "2026-08-23T09:00:00Z"
            ),
        )

        with TemporaryDirectory() as temporary:
            json_path, _ = (
                write_run_plan_manifests_atomic(
                    plan,
                    temporary,
                )
            )
            loaded_plan = load_run_plan_manifest(
                json_path
            )

        self.assertEqual(
            loaded_plan.to_dict(),
            plan.to_dict(),
        )

    def test_modified_manifest_run_count_is_rejected(
        self,
    ) -> None:
        plan = build_run_plan(
            self.build_small_matrix_config(),
            experiment_id="experiment-load",
            generated_at_utc=(
                "2026-08-23T09:00:00Z"
            ),
        )
        plan_data = plan.to_dict()
        plan_data["run_count"] = 999

        with self.assertRaisesRegex(
            ControlledIlluminationRunPlanError,
            "run_count",
        ):
            run_plan_from_dict(
                plan_data
            )

    def test_invalid_manifest_json_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            manifest_path = (
                Path(temporary) / "run_plan.json"
            )
            manifest_path.write_text(
                "{invalid-json",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ControlledIlluminationRunPlanError,
                "could not be loaded",
            ):
                load_run_plan_manifest(
                    manifest_path
                )

    def test_optical_screening_profile_uses_canonical_dimensions(
        self,
    ) -> None:
        config = (
            self.load_optical_screening_config()
        )

        conditions = build_run_conditions(
            config
        )

        self.assertEqual(
            {condition.phase for condition in conditions},
            {"constant_lux"},
        )
        self.assertEqual(
            {condition.platform for condition in conditions},
            {"desktop"},
        )
        self.assertEqual(
            {
                condition.architecture
                for condition in conditions
            },
            {"pure_python"},
        )
        self.assertEqual(
            {
                condition.algorithm
                for condition in conditions
            },
            {
                "original",
                "gamma_correction",
                "histogram_equalization",
                "clahe",
            },
        )
        self.assertEqual(
            {
                (
                    condition.resolution.width,
                    condition.resolution.height,
                )
                for condition in conditions
            },
            {(1280, 720)},
        )
        self.assertEqual(
            {
                condition.target_illuminance_lux
                for condition in conditions
            },
            {5.0, 50.0, 200.0, 500.0, 1000.0},
        )
        self.assertEqual(
            {
                condition.incidence_angle_degrees
                for condition in conditions
            },
            {0.0, 30.0, 60.0},
        )
        self.assertEqual(
            {
                condition.trial_number
                for condition in conditions
            },
            {1, 2, 3, 4, 5},
        )
        self.assertEqual(
            {
                condition.source_output_setting
                for condition in conditions
            },
            {None},
        )

    def test_optical_screening_profile_covers_each_condition_once(
        self,
    ) -> None:
        config = (
            self.load_optical_screening_config()
        )

        conditions = build_run_conditions(
            config
        )

        observed_conditions = {
            (
                condition.algorithm,
                condition.target_illuminance_lux,
                condition.incidence_angle_degrees,
                condition.trial_number,
            )
            for condition in conditions
        }

        expected_conditions = {
            (
                algorithm,
                illuminance_lux,
                incidence_angle,
                trial_number,
            )
            for algorithm in (
                "original",
                "gamma_correction",
                "histogram_equalization",
                "clahe",
            )
            for illuminance_lux in (
                5.0,
                50.0,
                200.0,
                500.0,
                1000.0,
            )
            for incidence_angle in (
                0.0,
                30.0,
                60.0,
            )
            for trial_number in range(1, 6)
        }

        self.assertEqual(
            observed_conditions,
            expected_conditions,
        )
        self.assertEqual(
            len(observed_conditions),
            len(conditions),
        )

    def test_optical_screening_profile_has_deterministic_order(
        self,
    ) -> None:
        config = (
            self.load_optical_screening_config()
        )

        first_plan = build_run_plan(
            config,
            experiment_id="optical-screening-test",
            generated_at_utc=(
                "2026-08-30T12:00:00Z"
            ),
        )
        second_plan = build_run_plan(
            config,
            experiment_id="optical-screening-test",
            generated_at_utc=(
                "2026-08-30T12:00:00Z"
            ),
        )

        self.assertTrue(
            first_plan.randomized
        )
        self.assertEqual(
            first_plan.randomization_seed,
            20260821,
        )
        self.assertEqual(
            [
                planned_run.condition_key
                for planned_run in first_plan.runs
            ],
            [
                planned_run.condition_key
                for planned_run in second_plan.runs
            ],
        )

if __name__ == "__main__":
    unittest.main()
