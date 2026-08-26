from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)
from benchmarks.experiments.controlled_illumination_run_bundle import (
    ControlledIlluminationRunBundleError,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlan,
    PlannedRun,
)
from benchmarks.experiments.finalize_controlled_illumination_run_bundle import (
    create_argument_parser,
    find_planned_run,
    run_cli,
)


MODULE_NAME = (
    "benchmarks.experiments."
    "finalize_controlled_illumination_run_bundle"
)
GENERATED_AT = "2026-08-26T09:00:00Z"
FINALIZED_AT = "2026-08-26T11:00:00Z"
PLAN_SHA256 = "a" * 64


class FinalizeControlledIlluminationRunBundleTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.planned_run = PlannedRun(
            execution_order=1,
            experiment_id="experiment-test",
            run_id="run-test-0001",
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
        self.plan = ControlledIlluminationRunPlan(
            schema_version=1,
            generated_at_utc=GENERATED_AT,
            experiment_id="experiment-test",
            randomized=False,
            randomization_seed=20260821,
            runs=(self.planned_run,),
        )

    def test_argument_parser_accepts_required_values(
        self,
    ) -> None:
        arguments = create_argument_parser().parse_args(
            [
                "--plan",
                "run_plan.json",
                "--run-directory",
                "run-directory",
                "--run-id",
                "run-test-0001",
            ]
        )

        self.assertEqual(
            arguments.plan,
            "run_plan.json",
        )
        self.assertEqual(
            arguments.run_directory,
            "run-directory",
        )
        self.assertEqual(
            arguments.run_id,
            "run-test-0001",
        )
        self.assertIsNone(arguments.config)
        self.assertFalse(
            arguments.validate_only
        )

    def test_argument_parser_accepts_validate_only(
        self,
    ) -> None:
        arguments = (
            create_argument_parser().parse_args(
                [
                    "--plan",
                    "run_plan.json",
                    "--run-directory",
                    "run-directory",
                    "--run-id",
                    "run-test-0001",
                    "--validate-only",
                ]
            )
        )

        self.assertTrue(
            arguments.validate_only
        )

    def test_planned_run_is_found(
        self,
    ) -> None:
        selected_run = find_planned_run(
            self.plan,
            "run-test-0001",
        )

        self.assertIs(
            selected_run,
            self.planned_run,
        )

    def test_unknown_run_id_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "not found",
        ):
            find_planned_run(
                self.plan,
                "missing-run",
            )

    def test_run_cli_finalizes_selected_run(
        self,
    ) -> None:
        config = {"schema_version": 1}
        manifest_path = Path(
            "run-directory/run_bundle_manifest.json"
        )
        manifest = SimpleNamespace(
            validate_only=False,
            experiment_id="experiment-test",
            run_id="run-test-0001",
        )
        arguments = SimpleNamespace(
            plan="run_plan.json",
            run_directory="run-directory",
            run_id="run-test-0001",
            config=None,
            validate_only=False,
        )

        with (
            patch(
                f"{MODULE_NAME}."
                "load_controlled_illumination_config",
                return_value=config,
            ) as load_config,
            patch(
                f"{MODULE_NAME}."
                "load_run_plan_manifest",
                return_value=self.plan,
            ) as load_plan,
            patch(
                f"{MODULE_NAME}."
                "calculate_run_plan_sha256",
                return_value=PLAN_SHA256,
            ) as calculate_hash,
            patch(
                f"{MODULE_NAME}."
                "finalize_run_bundle_atomic",
                return_value=(
                    manifest,
                    manifest_path,
                ),
            ) as finalize_bundle,
        ):
            captured_output = StringIO()

            with redirect_stdout(captured_output):
                exit_code = run_cli(
                    arguments,
                    now_provider=lambda: FINALIZED_AT,
                )

        self.assertEqual(exit_code, 0)
        load_config.assert_called_once_with(None)
        load_plan.assert_called_once_with(
            "run_plan.json"
        )
        calculate_hash.assert_called_once_with(
            self.plan
        )
        finalize_bundle.assert_called_once_with(
            "run-directory",
            self.planned_run,
            config,
            PLAN_SHA256,
            FINALIZED_AT,
        )
        self.assertIn(
            "Controlled-illumination run finalized.",
            captured_output.getvalue(),
        )
        self.assertIn(
            str(manifest_path),
            captured_output.getvalue(),
        )

        def test_argument_parser_accepts_validate_only(
                self,
        ) -> None:
            arguments = create_argument_parser().parse_args(
                [
                    "--plan",
                    "run_plan.json",
                    "--run-directory",
                    "run-directory",
                    "--run-id",
                    "run-test-0001",
                    "--validate-only",
                ]
            )

            self.assertTrue(
                arguments.validate_only
            )

    def test_run_cli_validates_without_finalizing(
            self,
    ) -> None:
        config = {"schema_version": 1}
        manifest = SimpleNamespace(
            experiment_id="experiment-test",
            run_id="run-test-0001",
        )
        arguments = SimpleNamespace(
            plan="run_plan.json",
            run_directory="run-directory",
            run_id="run-test-0001",
            config=None,
            validate_only=True,
        )

        with (
            patch(
                f"{MODULE_NAME}."
                "load_controlled_illumination_config",
                return_value=config,
            ),
            patch(
                f"{MODULE_NAME}."
                "load_run_plan_manifest",
                return_value=self.plan,
            ),
            patch(
                f"{MODULE_NAME}."
                "calculate_run_plan_sha256",
                return_value=PLAN_SHA256,
            ),
            patch(
                f"{MODULE_NAME}."
                "validate_run_bundle",
                return_value=manifest,
            ) as validate_bundle,
            patch(
                f"{MODULE_NAME}."
                "finalize_run_bundle_atomic",
            ) as finalize_bundle,
        ):
            captured_output = StringIO()

            with redirect_stdout(captured_output):
                exit_code = run_cli(
                    arguments,
                    now_provider=lambda: FINALIZED_AT,
                )

        self.assertEqual(exit_code, 0)
        validate_bundle.assert_called_once_with(
            "run-directory",
            self.planned_run,
            config,
            PLAN_SHA256,
            FINALIZED_AT,
        )
        finalize_bundle.assert_not_called()
        self.assertIn(
            "validation passed",
            captured_output.getvalue(),
        )
        self.assertNotIn(
            "Bundle manifest:",
            captured_output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()