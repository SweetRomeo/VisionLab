from __future__ import annotations

import math
import unittest

from benchmarks.realtime.jetson_controls import (
    JetsonControlProfile,
    create_applied_jetson_control_results,
    create_jetson_source_properties,
    jetson_control_results_to_metadata,
    serialize_jetson_source_properties,
    validate_jetson_control_profile,
)


class JetsonControlsTests(unittest.TestCase):
    def test_empty_profile_creates_no_properties(
        self,
    ) -> None:
        profile = JetsonControlProfile()

        self.assertEqual(
            create_jetson_source_properties(
                profile
            ),
            {},
        )

    def test_control_profile_creates_argus_properties(
        self,
    ) -> None:
        profile = JetsonControlProfile(
            ae_lock=True,
            exposure_time_ns=5_000_000,
            gain=2.5,
            awb_lock=True,
            wb_mode=0,
        )

        self.assertEqual(
            create_jetson_source_properties(
                profile
            ),
            {
                "aelock": True,
                "exposuretimerange": (
                    "5000000 5000000"
                ),
                "gainrange": "2.5 2.5",
                "awblock": True,
                "wbmode": 0,
            },
        )

    def test_source_properties_are_serialized(
        self,
    ) -> None:
        profile = JetsonControlProfile(
            ae_lock=True,
            exposure_time_ns=5_000_000,
            gain=2.5,
            awb_lock=False,
            wb_mode=1,
        )

        serialized = (
            serialize_jetson_source_properties(
                profile
            )
        )

        self.assertEqual(
            serialized,
            (
                'aelock=true '
                'exposuretimerange="5000000 5000000" '
                'gainrange="2.5 2.5" '
                'awblock=false '
                'wbmode=1'
            ),
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
                    "exposure_time_ns",
                ):
                    validate_jetson_control_profile(
                        JetsonControlProfile(
                            exposure_time_ns=value,
                        )
                    )

    def test_invalid_gain_is_rejected(
        self,
    ) -> None:
        for value in (
            True,
            0,
            -1.0,
            math.inf,
            math.nan,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "gain",
                ):
                    validate_jetson_control_profile(
                        JetsonControlProfile(
                            gain=value,
                        )
                    )

    def test_invalid_lock_value_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "ae_lock",
        ):
            validate_jetson_control_profile(
                JetsonControlProfile(
                    ae_lock=1,
                )
            )

    def test_invalid_wb_mode_is_rejected(
        self,
    ) -> None:
        for value in (
            True,
            -1,
            1.5,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "wb_mode",
                ):
                    validate_jetson_control_profile(
                        JetsonControlProfile(
                            wb_mode=value,
                        )
                    )

    def test_empty_profile_creates_no_control_results(
        self,
    ) -> None:
        results = (
            create_applied_jetson_control_results(
                JetsonControlProfile()
            )
        )

        self.assertEqual(
            results,
            (),
        )

    def test_applied_control_results_are_unverified(
        self,
    ) -> None:
        profile = JetsonControlProfile(
            ae_lock=True,
            exposure_time_ns=5_000_000,
            gain=2.5,
        )

        results = (
            create_applied_jetson_control_results(
                profile
            )
        )

        self.assertEqual(
            len(results),
            3,
        )

        exposure_result = results[1]

        self.assertEqual(
            exposure_result.name,
            "exposure_time_ns",
        )
        self.assertEqual(
            exposure_result.control_name,
            "exposuretimerange",
        )
        self.assertEqual(
            exposure_result.requested_value,
            5_000_000,
        )
        self.assertTrue(
            exposure_result.applied
        )
        self.assertIsNone(
            exposure_result.effective_value
        )
        self.assertFalse(
            exposure_result.verified
        )
        self.assertIsNone(
            exposure_result.matches_requested
        )

    def test_control_results_are_serialized_to_metadata(
        self,
    ) -> None:
        results = (
            create_applied_jetson_control_results(
                JetsonControlProfile(
                    ae_lock=True,
                    gain=2.5,
                )
            )
        )

        metadata = (
            jetson_control_results_to_metadata(
                results
            )
        )

        self.assertEqual(
            metadata,
            {
                "ae_lock": {
                    "backend": "jetson_gstreamer",
                    "control_name": "aelock",
                    "requested": True,
                    "effective": None,
                    "applied": True,
                    "verified": False,
                    "matches_requested": None,
                },
                "gain": {
                    "backend": "jetson_gstreamer",
                    "control_name": "gainrange",
                    "requested": 2.5,
                    "effective": None,
                    "applied": True,
                    "verified": False,
                    "matches_requested": None,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()