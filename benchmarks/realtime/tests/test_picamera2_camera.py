import unittest
from unittest.mock import MagicMock, call

import numpy as np

from benchmarks.realtime.picamera2_camera import (
    Picamera2CameraError,
    iter_picamera2_frames,
)

from benchmarks.realtime.picamera2_controls import (
    Picamera2ControlError,
    Picamera2ControlProfile,
)


class Picamera2CameraTests(unittest.TestCase):
    def create_camera(
            self,
    ) -> MagicMock:
        camera = MagicMock()

        camera.create_video_configuration.return_value = {
            "test": "configuration"
        }

        return camera

    def test_valid_frame_is_returned(
            self,
    ) -> None:
        camera = self.create_camera()

        frame = np.zeros(
            (12, 16, 3),
            dtype=np.uint8,
        )

        camera.capture_array.return_value = frame

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            picamera2_factory=lambda _: camera,
        )

        self.assertIs(
            next(source),
            frame,
        )

        source.close()

        camera.create_video_configuration.assert_called_once_with(
            main={
                "size": (
                    16,
                    12,
                ),
                "format": "RGB888",
            },
            controls={
                "FrameDurationLimits": (
                    33333,
                    33333,
                ),
            },
        )

        camera.configure.assert_called_once_with(
            {
                "test": "configuration"
            }
        )

        camera.start.assert_called_once_with()
        camera.capture_array.assert_called_once_with(
            "main"
        )
        camera.stop.assert_called_once_with()
        camera.close.assert_called_once_with()

    def test_camera_index_is_forwarded(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.capture_array.return_value = (
            np.zeros(
                (12, 16, 3),
                dtype=np.uint8,
            )
        )

        factory = MagicMock(
            return_value=camera
        )

        source = iter_picamera2_frames(
            2,
            width=16,
            height=12,
            fps=30.0,
            picamera2_factory=factory,
        )

        next(source)
        source.close()

        factory.assert_called_once_with(2)

    def test_invalid_camera_index_is_rejected(
            self,
    ) -> None:
        for invalid_index in (
            True,
            -1,
            1.5,
            "0",
        ):
            with self.subTest(
                    camera_index=invalid_index,
            ):
                with self.assertRaisesRegex(
                        ValueError,
                        "camera_index",
                ):
                    next(
                        iter_picamera2_frames(
                            invalid_index,
                            width=16,
                            height=12,
                            fps=30.0,
                        )
                    )

    def test_invalid_capture_mode_is_rejected(
            self,
    ) -> None:
        invalid_modes = (
            (0, 12, 30.0),
            (16, 0, 30.0),
            (16, 12, 0.0),
            (16, 12, float("nan")),
        )

        for (
            width,
            height,
            fps,
        ) in invalid_modes:
            with self.subTest(
                    width=width,
                    height=height,
                    fps=fps,
            ):
                with self.assertRaises(
                        ValueError,
                ):
                    next(
                        iter_picamera2_frames(
                            0,
                            width=width,
                            height=height,
                            fps=fps,
                        )
                    )

    def test_non_numpy_frame_is_rejected(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.capture_array.return_value = (
            "not-a-frame"
        )

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            picamera2_factory=lambda _: camera,
        )

        with self.assertRaisesRegex(
                Picamera2CameraError,
                "not a NumPy array",
        ):
            next(source)

        camera.stop.assert_called_once_with()
        camera.close.assert_called_once_with()

    def test_empty_frame_is_rejected(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.capture_array.return_value = (
            np.empty(
                (0, 0, 3),
                dtype=np.uint8,
            )
        )

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            picamera2_factory=lambda _: camera,
        )

        with self.assertRaisesRegex(
                Picamera2CameraError,
                "invalid frame",
        ):
            next(source)

        camera.stop.assert_called_once_with()
        camera.close.assert_called_once_with()

    def test_wrong_frame_dimensions_are_rejected(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.capture_array.return_value = (
            np.zeros(
                (10, 10, 3),
                dtype=np.uint8,
            )
        )

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            picamera2_factory=lambda _: camera,
        )

        with self.assertRaisesRegex(
                Picamera2CameraError,
                "dimensions",
        ):
            next(source)

        camera.stop.assert_called_once_with()
        camera.close.assert_called_once_with()

    def test_configuration_failure_closes_camera(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.configure.side_effect = (
            RuntimeError(
                "configuration failure"
            )
        )

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            picamera2_factory=lambda _: camera,
        )

        with self.assertRaisesRegex(
                RuntimeError,
                "configuration failure",
        ):
            next(source)

        camera.stop.assert_not_called()
        camera.close.assert_called_once_with()

    def test_capture_failure_releases_camera(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.capture_array.side_effect = (
            RuntimeError(
                "capture failure"
            )
        )

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            picamera2_factory=lambda _: camera,
        )

        with self.assertRaisesRegex(
                RuntimeError,
                "capture failure",
        ):
            next(source)

        camera.stop.assert_called_once_with()
        camera.close.assert_called_once_with()

    def test_camera_controls_are_applied_and_reported(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.camera_controls = {
            "AeEnable": object(),
            "ExposureTime": object(),
            "AnalogueGain": object(),
        }

        camera.capture_metadata.return_value = {
            "AeEnable": False,
            "ExposureTime": 10000,
            "AnalogueGain": 2.0,
        }

        camera.capture_array.return_value = (
            np.zeros(
                (12, 16, 3),
                dtype=np.uint8,
            )
        )

        reported_results = []

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            control_profile=(
                Picamera2ControlProfile(
                    ae_enable=False,
                    exposure_time_us=10000,
                    analogue_gain=2.0,
                )
            ),
            control_reporter=(
                reported_results.append
            ),
            picamera2_factory=lambda _: camera,
        )

        next(source)
        source.close()

        camera.set_controls.assert_called_once_with(
            {
                "AeEnable": False,
                "ExposureTime": 10000,
                "AnalogueGain": 2.0,
            }
        )

        camera.capture_metadata.assert_called_once_with()

        self.assertEqual(
            len(reported_results),
            1,
        )

        self.assertEqual(
            len(reported_results[0]),
            3,
        )

    def test_controls_are_applied_before_camera_start(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.camera_controls = {
            "AeEnable": object(),
        }

        camera.capture_metadata.return_value = {
            "AeEnable": False,
        }

        camera.capture_array.return_value = (
            np.zeros(
                (12, 16, 3),
                dtype=np.uint8,
            )
        )

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            control_profile=(
                Picamera2ControlProfile(
                    ae_enable=False,
                )
            ),
            picamera2_factory=lambda _: camera,
        )

        next(source)
        source.close()

        self.assertLess(
            camera.mock_calls.index(
                call.set_controls(
                    {
                        "AeEnable": False,
                    }
                )
            ),
            camera.mock_calls.index(
                call.start()
            ),
        )

    def test_control_verification_failure_releases_camera(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.camera_controls = {
            "AeEnable": object(),
            "ExposureTime": object(),
        }

        camera.capture_metadata.return_value = {
            "AeEnable": False,
            "ExposureTime": 9000,
        }

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            control_profile=(
                Picamera2ControlProfile(
                    ae_enable=False,
                    exposure_time_us=10000,
                )
            ),
            picamera2_factory=lambda _: camera,
        )

        with self.assertRaisesRegex(
                Picamera2ControlError,
                "does not match",
        ):
            next(source)

        camera.stop.assert_called_once_with()
        camera.close.assert_called_once_with()
        camera.capture_array.assert_not_called()

    def test_missing_effective_metadata_is_reported_unverifiable(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.camera_controls = {
            "AeEnable": object(),
        }

        camera.capture_metadata.return_value = {}

        camera.capture_array.return_value = (
            np.zeros(
                (12, 16, 3),
                dtype=np.uint8,
            )
        )

        reported_results = []

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            control_profile=(
                Picamera2ControlProfile(
                    ae_enable=False,
                )
            ),
            control_reporter=(
                reported_results.append
            ),
            picamera2_factory=lambda _: camera,
        )

        next(source)
        source.close()

        result = reported_results[0][0]

        self.assertTrue(result.applied)
        self.assertFalse(result.verified)
        self.assertIsNone(
            result.effective_value
        )
        self.assertIsNone(
            result.matches_requested
        )

    def test_empty_control_profile_skips_metadata_capture(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.capture_array.return_value = (
            np.zeros(
                (12, 16, 3),
                dtype=np.uint8,
            )
        )

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            picamera2_factory=lambda _: camera,
        )

        next(source)
        source.close()

        camera.set_controls.assert_not_called()
        camera.capture_metadata.assert_not_called()

    def test_effective_capture_mode_is_reported(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.camera_configuration.return_value = {
            "main": {
                "size": (
                    16,
                    12,
                ),
            },
        }

        camera.capture_metadata.return_value = {
            "FrameDuration": 40000,
        }

        camera.capture_array.return_value = (
            np.zeros(
                (12, 16, 3),
                dtype=np.uint8,
            )
        )

        reporter = MagicMock()

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            capture_mode_reporter=reporter,
            picamera2_factory=lambda _: camera,
        )

        next(source)
        source.close()

        reporter.assert_called_once_with(
            16,
            12,
            25.0,
        )

        camera.capture_metadata.assert_called_once_with()
        camera.camera_configuration.assert_called_once_with()

    def test_invalid_effective_capture_mode_releases_camera(
            self,
    ) -> None:
        camera = self.create_camera()

        camera.camera_configuration.return_value = {
            "main": {
                "size": (
                    16,
                    12,
                ),
            },
        }

        camera.capture_metadata.return_value = {}

        source = iter_picamera2_frames(
            0,
            width=16,
            height=12,
            fps=30.0,
            capture_mode_reporter=MagicMock(),
            picamera2_factory=lambda _: camera,
        )

        with self.assertRaisesRegex(
                Picamera2CameraError,
                "effective capture mode",
        ):
            next(source)

        camera.stop.assert_called_once_with()
        camera.close.assert_called_once_with()
        camera.capture_array.assert_not_called()


if __name__ == "__main__":
    unittest.main()