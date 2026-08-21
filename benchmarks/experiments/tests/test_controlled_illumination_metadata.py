from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.experiments.controlled_illumination_metadata import (
    ControlledIlluminationMetadataError,
    ControlledIlluminationRunMetadata,
    IlluminanceMeasurements,
    ResolutionMetadata,
    create_unique_identifier,
    load_controlled_illumination_config,
    save_run_metadata_atomic,
    validate_run_metadata,
)


class ControlledIlluminationMetadataTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.config = (
            load_controlled_illumination_config()
        )

        self.illuminance = IlluminanceMeasurements(
            centre_lux=50.0,
            top_left_lux=49.0,
            top_right_lux=51.0,
            bottom_left_lux=50.0,
            bottom_right_lux=50.0,
            measured_at_utc=(
                "2026-08-21T09:00:00Z"
            ),
            lux_meter_model="dry-run-meter",
            lux_meter_range="0-2000 lux",
            lux_meter_resolution="1 lux",
        )

        self.metadata = (
            ControlledIlluminationRunMetadata(
                experiment_id="experiment-test",
                run_id="run-test",
                collected_at_utc=(
                    "2026-08-21T09:01:00Z"
                ),
                git_commit_sha="a" * 40,
                phase="constant_lux",
                device_id="desktop-test",
                operating_system="Windows 11",
                platform="desktop",
                architecture="pure_python",
                algorithm="gamma_correction",
                algorithm_parameters={
                    "gamma_value": 0.6,
                },
                resolution=ResolutionMetadata(
                    width=640,
                    height=480,
                ),
                trial_number=1,
                target_fps=30.0,
                frame_deadline_ms=(
                    1000.0 / 30.0
                ),
                target_illuminance_lux=50.0,
                measured_illuminance=(
                    self.illuminance
                ),
                incidence_angle_degrees=30.0,
                source_output_setting=50.0,
                camera_mode="controlled",
                camera_settings={
                    "exposure_time": "1/60 s",
                    "sensor_gain_or_iso": 100,
                    "white_balance": (
                        "manual_4500k"
                    ),
                    "focus": "manual",
                    "frame_rate": 30.0,
                    "resolution": "640x480",
                    "lens": "dry-run-lens",
                    "aperture": None,
                },
                camera_to_target_distance_metres=(
                    1.5
                ),
                light_to_target_distance_metres=(
                    2.0
                ),
                input_scene_id="dry-run-scene",
                power_mode="balanced",
                clock_configuration="default",
                starting_temperature_celsius=45.0,
                ending_temperature_celsius=48.0,
                maximum_temperature_celsius=50.0,
                thermal_throttling_detected=False,
                software_versions={
                    "python": "3.13.12",
                    "opencv": "4.11.0",
                },
                dry_run=True,
            )
        )

    def test_valid_metadata_passes_validation(
        self,
    ) -> None:
        validate_run_metadata(
            self.metadata,
            self.config,
        )

    def test_illuminance_summary_is_calculated(
        self,
    ) -> None:
        self.assertEqual(
            self.illuminance.mean_lux,
            50.0,
        )
        self.assertEqual(
            self.illuminance.minimum_lux,
            49.0,
        )
        self.assertEqual(
            self.illuminance.maximum_lux,
            51.0,
        )

    def test_metadata_is_saved_atomically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "run_metadata.json"
            )

            saved_path = save_run_metadata_atomic(
                self.metadata,
                config=self.config,
                output_path=output_path,
            )

            self.assertEqual(
                saved_path,
                output_path,
            )
            self.assertTrue(
                output_path.is_file()
            )

            with output_path.open(
                "r",
                encoding="utf-8",
            ) as metadata_file:
                saved_metadata = json.load(
                    metadata_file
                )

            self.assertEqual(
                saved_metadata["experiment_id"],
                "experiment-test",
            )
            self.assertEqual(
                saved_metadata[
                    "measured_illuminance"
                ]["mean_lux"],
                50.0,
            )

            temporary_files = list(
                Path(directory).glob("*.tmp")
            )

            self.assertEqual(
                temporary_files,
                [],
            )

    def test_invalid_trial_number_is_rejected(
        self,
    ) -> None:
        invalid_metadata = replace(
            self.metadata,
            trial_number=6,
        )

        with self.assertRaises(
            ControlledIlluminationMetadataError
        ):
            validate_run_metadata(
                invalid_metadata,
                self.config,
            )

    def test_invalid_incidence_angle_is_rejected(
        self,
    ) -> None:
        invalid_metadata = replace(
            self.metadata,
            incidence_angle_degrees=45.0,
        )

        with self.assertRaises(
            ControlledIlluminationMetadataError
        ):
            validate_run_metadata(
                invalid_metadata,
                self.config,
            )

    def test_constant_source_rejects_target_lux(
        self,
    ) -> None:
        invalid_metadata = replace(
            self.metadata,
            phase="constant_source",
            target_illuminance_lux=50.0,
        )

        with self.assertRaises(
            ControlledIlluminationMetadataError
        ):
            validate_run_metadata(
                invalid_metadata,
                self.config,
            )

    def test_missing_camera_setting_is_rejected(
        self,
    ) -> None:
        camera_settings = dict(
            self.metadata.camera_settings
        )
        camera_settings.pop("lens")

        invalid_metadata = replace(
            self.metadata,
            camera_settings=camera_settings,
        )

        with self.assertRaises(
            ControlledIlluminationMetadataError
        ):
            validate_run_metadata(
                invalid_metadata,
                self.config,
            )

    def test_negative_lux_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationMetadataError
        ):
            IlluminanceMeasurements(
                centre_lux=-1.0,
                top_left_lux=0.0,
                top_right_lux=0.0,
                bottom_left_lux=0.0,
                bottom_right_lux=0.0,
                measured_at_utc=(
                    "2026-08-21T09:00:00Z"
                ),
                lux_meter_model="test-meter",
                lux_meter_range="0-2000 lux",
                lux_meter_resolution="1 lux",
            )

    def test_generated_identifiers_are_unique(
        self,
    ) -> None:
        first_identifier = (
            create_unique_identifier("run")
        )
        second_identifier = (
            create_unique_identifier("run")
        )

        self.assertNotEqual(
            first_identifier,
            second_identifier,
        )
        self.assertTrue(
            first_identifier.startswith("run-")
        )


if __name__ == "__main__":
    unittest.main()