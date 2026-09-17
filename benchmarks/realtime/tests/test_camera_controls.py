import unittest
import cv2
from unittest.mock import MagicMock

from benchmarks.realtime.camera_controls import (
    CameraControlError,
    CameraControlProfile,
    CameraControlRequest,
    apply_camera_control,
    apply_camera_controls,
    create_camera_control_requests,
    create_opencv_camera_control_request,
)


class CameraControlTests(
    unittest.TestCase
):
    def test_required_control_is_applied_and_verified(
        self,
    ) -> None:
        capture = MagicMock()
        capture.set.return_value = True
        capture.get.return_value = 42.0

        result = apply_camera_control(
            capture,
            CameraControlRequest(
                name="exposure",
                property_id=15,
                requested_value=42.0,
            ),
        )

        self.assertTrue(
            result.applied
        )
        self.assertTrue(
            result.verified
        )
        self.assertTrue(
            result.matches_requested
        )
        self.assertEqual(
            result.effective_value,
            42.0,
        )

        capture.set.assert_called_once_with(
            15,
            42.0,
        )
        capture.get.assert_called_once_with(
            15
        )

    def test_required_set_failure_is_rejected(
        self,
    ) -> None:
        capture = MagicMock()
        capture.set.return_value = False

        with self.assertRaisesRegex(
            CameraControlError,
            "could not be applied",
        ):
            apply_camera_control(
                capture,
                CameraControlRequest(
                    name="gain",
                    property_id=14,
                    requested_value=5.0,
                ),
            )

    def test_required_mismatched_readback_is_rejected(
        self,
    ) -> None:
        capture = MagicMock()
        capture.set.return_value = True
        capture.get.return_value = 10.0

        with self.assertRaisesRegex(
            CameraControlError,
            "does not match",
        ):
            apply_camera_control(
                capture,
                CameraControlRequest(
                    name="focus",
                    property_id=28,
                    requested_value=20.0,
                ),
            )

    def test_tolerance_accepts_nearby_effective_value(
        self,
    ) -> None:
        capture = MagicMock()
        capture.set.return_value = True
        capture.get.return_value = 29.97

        result = apply_camera_control(
            capture,
            CameraControlRequest(
                name="example",
                property_id=1,
                requested_value=30.0,
                absolute_tolerance=0.1,
            ),
        )

        self.assertTrue(
            result.matches_requested
        )

    def test_optional_unsupported_control_is_reported(
        self,
    ) -> None:
        capture = MagicMock()
        capture.set.return_value = False

        result = apply_camera_control(
            capture,
            CameraControlRequest(
                name="white_balance",
                property_id=17,
                requested_value=4500.0,
                required=False,
            ),
        )

        self.assertFalse(
            result.applied
        )
        self.assertFalse(
            result.verified
        )
        self.assertIsNone(
            result.effective_value
        )
        self.assertIsNone(
            result.matches_requested
        )

    def test_unverified_control_does_not_read_back(
        self,
    ) -> None:
        capture = MagicMock()
        capture.set.return_value = True

        result = apply_camera_control(
            capture,
            CameraControlRequest(
                name="auto_exposure",
                property_id=21,
                requested_value=0.0,
                verify=False,
            ),
        )

        self.assertTrue(
            result.applied
        )
        self.assertFalse(
            result.verified
        )

        capture.get.assert_not_called()

    def test_duplicate_control_names_are_rejected(
        self,
    ) -> None:
        capture = MagicMock()

        requests = (
            CameraControlRequest(
                name="gain",
                property_id=1,
                requested_value=1.0,
            ),
            CameraControlRequest(
                name="gain",
                property_id=2,
                requested_value=2.0,
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate camera control name",
        ):
            apply_camera_controls(
                capture,
                requests,
            )

    def test_duplicate_property_ids_are_rejected(
        self,
    ) -> None:
        capture = MagicMock()

        requests = (
            CameraControlRequest(
                name="gain",
                property_id=1,
                requested_value=1.0,
            ),
            CameraControlRequest(
                name="exposure",
                property_id=1,
                requested_value=2.0,
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate camera control property_id",
        ):
            apply_camera_controls(
                capture,
                requests,
            )

    def test_opencv_exposure_request_uses_expected_property(
            self,
    ) -> None:
        request = (
            create_opencv_camera_control_request(
                "exposure",
                -6.0,
            )
        )

        self.assertEqual(
            request.name,
            "exposure",
        )
        self.assertEqual(
            request.property_id,
            cv2.CAP_PROP_EXPOSURE,
        )
        self.assertEqual(
            request.requested_value,
            -6.0,
        )

    def test_opencv_focus_request_preserves_options(
            self,
    ) -> None:
        request = (
            create_opencv_camera_control_request(
                "focus",
                25.0,
                required=False,
                verify=True,
                absolute_tolerance=1.0,
            )
        )

        self.assertEqual(
            request.property_id,
            cv2.CAP_PROP_FOCUS,
        )
        self.assertFalse(
            request.required
        )
        self.assertTrue(
            request.verify
        )
        self.assertEqual(
            request.absolute_tolerance,
            1.0,
        )

    def test_unknown_opencv_control_is_rejected(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "Unsupported OpenCV camera control",
        ):
            create_opencv_camera_control_request(
                "unknown-control",
                1.0,
            )

    def test_empty_camera_control_profile_creates_no_requests(
            self,
    ) -> None:
        requests = (
            create_camera_control_requests(
                CameraControlProfile()
            )
        )

        self.assertEqual(
            requests,
            (),
        )

    def test_manual_exposure_requires_auto_exposure(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "explicit auto_exposure",
        ):
            create_camera_control_requests(
                CameraControlProfile(
                    exposure=-6.0,
                )
            )

    def test_manual_gain_requires_auto_exposure(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "explicit auto_exposure",
        ):
            create_camera_control_requests(
                CameraControlProfile(
                    gain=1.0,
                )
            )

    def test_manual_white_balance_requires_auto_white_balance(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "explicit auto_white_balance",
        ):
            create_camera_control_requests(
                CameraControlProfile(
                    white_balance_temperature=4500.0,
                )
            )

    def test_manual_focus_requires_autofocus(
            self,
    ) -> None:
        with self.assertRaisesRegex(
                ValueError,
                "explicit autofocus",
        ):
            create_camera_control_requests(
                CameraControlProfile(
                    focus=20.0,
                )
            )

    def test_profile_creates_controls_in_deterministic_order(
            self,
    ) -> None:
        requests = (
            create_camera_control_requests(
                CameraControlProfile(
                    auto_exposure=0.25,
                    exposure=-6.0,
                    gain=1.0,
                    auto_white_balance=0.0,
                    white_balance_temperature=4500.0,
                    autofocus=0.0,
                    focus=20.0,
                )
            )
        )

        self.assertEqual(
            [
                request.name
                for request in requests
            ],
            [
                "auto_exposure",
                "exposure",
                "gain",
                "auto_white_balance",
                "white_balance_temperature",
                "autofocus",
                "focus",
            ],
        )

if __name__ == "__main__":
    unittest.main()