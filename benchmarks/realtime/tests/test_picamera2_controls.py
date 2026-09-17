import unittest
from unittest.mock import MagicMock

from benchmarks.realtime.picamera2_controls import (
    Picamera2ControlError,
    Picamera2ControlProfile,
    apply_picamera2_control_profile,
    create_picamera2_controls,
    picamera2_control_results_to_metadata,
    validate_picamera2_control_profile,
    verify_picamera2_control_profile,
)

from benchmarks.realtime.picamera2_controls import (
    Picamera2ControlProfile,
    create_picamera2_controls,
    validate_picamera2_control_profile,
)


class Picamera2ControlsTests(unittest.TestCase):
    def test_empty_profile_creates_empty_controls(
            self,
    ) -> None:
        self.assertEqual(
            create_picamera2_controls(
                Picamera2ControlProfile()
            ),
            {},
        )

    def test_manual_exposure_and_gain_are_mapped(
            self,
    ) -> None:
        controls = create_picamera2_controls(
            Picamera2ControlProfile(
                ae_enable=False,
                exposure_time_us=10000,
                analogue_gain=2.5,
            )
        )

        self.assertEqual(
            controls,
            {
                "AeEnable": False,
                "ExposureTime": 10000,
                "AnalogueGain": 2.5,
            },
        )

    def test_manual_white_balance_is_mapped(
            self,
    ) -> None:
        controls = create_picamera2_controls(
            Picamera2ControlProfile(
                awb_enable=False,
                colour_gains=(
                    1.5,
                    1.8,
                ),
            )
        )

        self.assertEqual(
            controls,
            {
                "AwbEnable": False,
                "ColourGains": (
                    1.5,
                    1.8,
                ),
            },
        )

    def test_manual_exposure_requires_explicit_ae(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "explicit ae_enable",
        ):
            validate_picamera2_control_profile(
                Picamera2ControlProfile(
                    exposure_time_us=10000,
                )
            )

    def test_manual_exposure_requires_ae_disabled(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "ae_enable=False",
        ):
            validate_picamera2_control_profile(
                Picamera2ControlProfile(
                    ae_enable=True,
                    exposure_time_us=10000,
                )
            )

    def test_manual_gain_requires_ae_disabled(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "ae_enable=False",
        ):
            validate_picamera2_control_profile(
                Picamera2ControlProfile(
                    ae_enable=True,
                    analogue_gain=2.0,
                )
            )

    def test_manual_colour_gains_require_explicit_awb(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "explicit awb_enable",
        ):
            validate_picamera2_control_profile(
                Picamera2ControlProfile(
                    colour_gains=(
                        1.5,
                        1.8,
                    ),
                )
            )

    def test_manual_colour_gains_require_awb_disabled(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "awb_enable=False",
        ):
            validate_picamera2_control_profile(
                Picamera2ControlProfile(
                    awb_enable=True,
                    colour_gains=(
                        1.5,
                        1.8,
                    ),
                )
            )

    def test_invalid_exposure_time_is_rejected(
            self,
    ) -> None:
        for value in (
            True,
            0,
            -1,
            1.5,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ValueError,
                        "exposure_time_us",
                ):
                    validate_picamera2_control_profile(
                        Picamera2ControlProfile(
                            ae_enable=False,
                            exposure_time_us=value,
                        )
                    )

    def test_invalid_analogue_gain_is_rejected(
            self,
    ) -> None:
        for value in (
            True,
            0.0,
            -1.0,
            float("nan"),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ValueError,
                        "analogue_gain",
                ):
                    validate_picamera2_control_profile(
                        Picamera2ControlProfile(
                            ae_enable=False,
                            analogue_gain=value,
                        )
                    )

    def test_invalid_colour_gains_are_rejected(
            self,
    ) -> None:
        invalid_values = (
            (1.0,),
            (1.0, 2.0, 3.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (float("nan"), 1.0),
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ValueError,
                        "colour_gains",
                ):
                    validate_picamera2_control_profile(
                        Picamera2ControlProfile(
                            awb_enable=False,
                            colour_gains=value,
                        )
                    )

    def test_profile_is_applied_to_supported_camera(
            self,
    ) -> None:
        camera = MagicMock()

        camera.camera_controls = {
            "AeEnable": object(),
            "ExposureTime": object(),
            "AnalogueGain": object(),
        }

        profile = Picamera2ControlProfile(
            ae_enable=False,
            exposure_time_us=10000,
            analogue_gain=2.0,
        )

        apply_picamera2_control_profile(
            camera,
            profile,
        )

        camera.set_controls.assert_called_once_with(
            {
                "AeEnable": False,
                "ExposureTime": 10000,
                "AnalogueGain": 2.0,
            }
        )

    def test_unsupported_required_control_is_rejected(
            self,
    ) -> None:
        camera = MagicMock()

        camera.camera_controls = {
            "AeEnable": object(),
        }

        with self.assertRaisesRegex(
                Picamera2ControlError,
                "unsupported",
        ):
            apply_picamera2_control_profile(
                camera,
                Picamera2ControlProfile(
                    ae_enable=False,
                    exposure_time_us=10000,
                ),
            )

        camera.set_controls.assert_not_called()

    def test_effective_controls_are_verified(
            self,
    ) -> None:
        results = verify_picamera2_control_profile(
            Picamera2ControlProfile(
                ae_enable=False,
                exposure_time_us=10000,
                analogue_gain=2.0,
            ),
            {
                "AeEnable": False,
                "ExposureTime": 10000,
                "AnalogueGain": 2.0,
            },
        )

        self.assertEqual(
            len(results),
            3,
        )

        self.assertTrue(
            all(
                result.verified
                for result in results
            )
        )

        self.assertTrue(
            all(
                result.matches_requested
                for result in results
            )
        )

    def test_missing_metadata_is_unverifiable(
            self,
    ) -> None:
        results = verify_picamera2_control_profile(
            Picamera2ControlProfile(
                ae_enable=False,
            ),
            {},
        )

        self.assertEqual(
            len(results),
            1,
        )
        self.assertFalse(
            results[0].verified
        )
        self.assertIsNone(
            results[0].matches_requested
        )
        self.assertIsNone(
            results[0].effective_value
        )

    def test_mismatched_exposure_is_rejected(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                Picamera2ControlError,
                "does not match",
        ):
            verify_picamera2_control_profile(
                Picamera2ControlProfile(
                    ae_enable=False,
                    exposure_time_us=10000,
                ),
                {
                    "AeEnable": False,
                    "ExposureTime": 9000,
                },
            )

    def test_exposure_tolerance_accepts_close_value(
            self,
    ) -> None:
        results = verify_picamera2_control_profile(
            Picamera2ControlProfile(
                ae_enable=False,
                exposure_time_us=10000,
            ),
            {
                "AeEnable": False,
                "ExposureTime": 9995,
            },
            exposure_time_abs_tolerance_us=5.0,
        )

        self.assertTrue(
            results[1].verified
        )
        self.assertTrue(
            results[1].matches_requested
        )

    def test_colour_gains_are_verified_and_serialized(
            self,
    ) -> None:
        results = verify_picamera2_control_profile(
            Picamera2ControlProfile(
                awb_enable=False,
                colour_gains=(
                    1.5,
                    1.8,
                ),
            ),
            {
                "AwbEnable": False,
                "ColourGains": (
                    1.5,
                    1.8,
                ),
            },
        )

        metadata = (
            picamera2_control_results_to_metadata(
                results
            )
        )

        self.assertEqual(
            metadata["colour_gains"][
                "control_name"
            ],
            "ColourGains",
        )

        self.assertEqual(
            metadata["colour_gains"][
                "effective"
            ],
            (
                1.5,
                1.8,
            ),
        )


if __name__ == "__main__":
    unittest.main()