import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import ANY, MagicMock, patch
from contextlib import (
    redirect_stderr,
    redirect_stdout,
)
from io import StringIO

from benchmarks.experiments import (
    controlled_illumination_camera_preflight
    as camera_preflight,
)

from benchmarks.realtime.camera_controls import (
    CameraControlRequest,
    CameraControlResult,
)

from benchmarks.realtime.picamera2_controls import (
    Picamera2ControlProfile,
    Picamera2ControlResult,
)

PREFLIGHT_MODULE = (
    "benchmarks.experiments."
    "controlled_illumination_camera_preflight"
)


class ControlledIlluminationCameraPreflightTests(
    unittest.TestCase
):
    def test_preflight_samples_frames_and_reports_mode(
        self,
    ) -> None:
        frame_source = MagicMock()
        frame_source.__next__.side_effect = (
            object(),
            object(),
            object(),
        )

        def create_frame_source(
                camera_index: int,
                **arguments,
        ):
            arguments[
                "capture_mode_reporter"
            ](
                16,
                12,
                29.97,
            )

            arguments[
                "camera_controls_reporter"
            ](
                (
                    CameraControlResult(
                        name="exposure",
                        property_id=15,
                        requested_value=-6.0,
                        applied=True,
                        effective_value=-6.0,
                        verified=True,
                        matches_requested=True,
                    ),
                )
            )

            return frame_source

        with patch(
            f"{PREFLIGHT_MODULE}."
            "iter_camera_frames",
            side_effect=create_frame_source,
        ) as iter_camera:
            result = (
                camera_preflight
                .run_camera_preflight(
                    camera_index=0,
                    width=16,
                    height=12,
                    fps=30.0,
                    sample_frames=3,
                )
            )

        self.assertEqual(result.camera_index, 0)
        self.assertEqual(result.effective_width, 16)
        self.assertEqual(result.effective_height, 12)
        self.assertEqual(result.effective_fps, 29.97)
        self.assertEqual(
            result.sampled_frame_count,
            3,
        )

        iter_camera.assert_called_once_with(
            0,
            width=16,
            height=12,
            fps=30.0,
            camera_controls=ANY,
            capture_mode_reporter=ANY,
            camera_controls_reporter=ANY,
        )
        self.assertEqual(
            frame_source.__next__.call_count,
            3,
        )
        frame_source.close.assert_called_once_with()

    def test_invalid_sample_frame_counts_are_rejected_before_camera_open(
            self,
    ) -> None:
        invalid_counts = (
            True,
            0,
            -1,
            1.5,
            "3",
        )

        with patch(
                f"{PREFLIGHT_MODULE}."
                "iter_camera_frames",
        ) as iter_camera:
            for invalid_count in invalid_counts:
                with self.subTest(
                        sample_frames=invalid_count
                ):
                    with self.assertRaisesRegex(
                            ValueError,
                            (
                                    "sample_frames must be "
                                    "a positive integer"
                            ),
                    ):
                        (
                            camera_preflight
                            .run_camera_preflight(
                                camera_index=0,
                                width=16,
                                height=12,
                                fps=30.0,
                                sample_frames=(
                                    invalid_count
                                ),
                            )
                        )

        iter_camera.assert_not_called()

    def test_cli_reports_successful_preflight(
            self,
    ) -> None:
        result = (
            camera_preflight.CameraPreflightResult(
                camera_backend="opencv",
                camera_index=0,
                effective_width=16,
                effective_height=12,
                effective_fps=29.97,
                sampled_frame_count=3,
                camera_controls=(
                    CameraControlResult(
                        name="exposure",
                        property_id=15,
                        requested_value=-6.0,
                        applied=True,
                        effective_value=-6.0,
                        verified=True,
                        matches_requested=True,
                    ),
                ),
            )
        )
        captured_output = StringIO()

        with (
            patch(
                f"{PREFLIGHT_MODULE}."
                "run_camera_preflight",
                return_value=result,
            ) as run_preflight,
            redirect_stdout(captured_output),
        ):
            exit_code = camera_preflight.run_cli(
                [
                    "--camera-index",
                    "0",
                    "--width",
                    "16",
                    "--height",
                    "12",
                    "--fps",
                    "30",
                    "--sample-frames",
                    "3",
                ]
            )

        self.assertEqual(exit_code, 0)
        run_preflight.assert_called_once_with(
            camera_backend="opencv",
            camera_index=0,
            width=16,
            height=12,
            fps=30.0,
            sample_frames=3,
            camera_controls=(),
            picamera2_control_profile=None,
        )

        output = captured_output.getvalue()

        self.assertIn(
            "Camera backend: opencv",
            output,
        )
        self.assertIn(
            "Camera preflight passed",
            output,
        )
        self.assertIn(
            "Effective resolution: 16x12",
            output,
        )
        self.assertIn(
            "Effective FPS: 29.97",
            output,
        )
        self.assertIn(
            "Sampled frames: 3",
            output,
        )
        self.assertIn(
            "No experiment artifacts were written",
            output,
        )

    def test_cli_reports_preflight_failure(
            self,
    ) -> None:
        captured_error = StringIO()

        with (
            patch(
                f"{PREFLIGHT_MODULE}."
                "run_camera_preflight",
                side_effect=RuntimeError(
                    "Camera 0 could not be opened."
                ),
            ),
            redirect_stderr(captured_error),
        ):
            exit_code = camera_preflight.run_cli(
                [
                    "--camera-index",
                    "0",
                    "--width",
                    "16",
                    "--height",
                    "12",
                    "--fps",
                    "30",
                    "--sample-frames",
                    "3",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "Camera preflight failed",
            captured_error.getvalue(),
        )
        self.assertIn(
            "Camera 0 could not be opened",
            captured_error.getvalue(),
        )

    def test_main_exits_with_cli_status(
            self,
    ) -> None:
        with patch(
                f"{PREFLIGHT_MODULE}.run_cli",
                return_value=1,
        ) as run_cli:
            with self.assertRaises(
                    SystemExit
            ) as raised:
                camera_preflight.main()

        self.assertEqual(
            raised.exception.code,
            1,
        )
        run_cli.assert_called_once_with()

    def test_preflight_closes_frame_source_after_read_failure(
            self,
    ) -> None:
        frame_source = MagicMock()
        frame_source.__next__.side_effect = RuntimeError(
            "camera read failure"
        )

        with patch(
                f"{PREFLIGHT_MODULE}.iter_camera_frames",
                return_value=frame_source,
        ):
            with self.assertRaisesRegex(
                    RuntimeError,
                    "camera read failure",
            ):
                camera_preflight.run_camera_preflight(
                    camera_index=0,
                    width=16,
                    height=12,
                    fps=30.0,
                    sample_frames=1,
                )

        frame_source.close.assert_called_once_with()

    def test_successful_preflight_writes_no_experiment_artifacts(
            self,
    ) -> None:
        frame_source = MagicMock()
        frame_source.__next__.return_value = object()

        def create_frame_source(
                camera_index: int,
                **arguments,
        ):
            arguments["capture_mode_reporter"](
                16,
                12,
                30.0,
            )
            arguments["camera_controls_reporter"](
                ()
            )
            return frame_source

        with TemporaryDirectory() as temporary:
            original_directory = os.getcwd()

            try:
                os.chdir(temporary)

                with patch(
                        f"{PREFLIGHT_MODULE}."
                        "iter_camera_frames",
                        side_effect=create_frame_source,
                ):
                    camera_preflight.run_camera_preflight(
                        camera_index=0,
                        width=16,
                        height=12,
                        fps=30.0,
                        sample_frames=1,
                    )

                self.assertEqual(
                    list(Path(temporary).iterdir()),
                    [],
                )
            finally:
                os.chdir(original_directory)

    def test_failed_preflight_writes_no_experiment_artifacts(
            self,
    ) -> None:
        frame_source = MagicMock()
        frame_source.__next__.side_effect = RuntimeError(
            "camera read failure"
        )

        with TemporaryDirectory() as temporary:
            original_directory = os.getcwd()

            try:
                os.chdir(temporary)

                with patch(
                        f"{PREFLIGHT_MODULE}."
                        "iter_camera_frames",
                        return_value=frame_source,
                ):
                    with self.assertRaisesRegex(
                            RuntimeError,
                            "camera read failure",
                    ):
                        camera_preflight.run_camera_preflight(
                            camera_index=0,
                            width=16,
                            height=12,
                            fps=30.0,
                            sample_frames=1,
                        )

                self.assertEqual(
                    list(Path(temporary).iterdir()),
                    [],
                )
            finally:
                os.chdir(original_directory)

    def test_cli_rejects_manual_exposure_without_auto_exposure(
            self,
    ) -> None:
        captured_error = StringIO()

        with redirect_stderr(
                captured_error
        ):
            exit_code = (
                camera_preflight.run_cli(
                    [
                        "--camera-index",
                        "0",
                        "--width",
                        "16",
                        "--height",
                        "12",
                        "--fps",
                        "30",
                        "--sample-frames",
                        "1",
                        "--exposure",
                        "-6",
                    ]
                )
            )

        self.assertEqual(
            exit_code,
            1,
        )

        self.assertIn(
            "explicit auto_exposure",
            captured_error.getvalue(),
        )

    def test_preflight_uses_opencv_backend_by_default(
            self,
    ) -> None:
        frame_source = MagicMock()
        frame_source.__next__.return_value = object()

        def create_frame_source(
                camera_index: int,
                **arguments,
        ):
            arguments["capture_mode_reporter"](
                16,
                12,
                30.0,
            )
            arguments["camera_controls_reporter"](
                ()
            )
            return frame_source

        with (
            patch(
                f"{PREFLIGHT_MODULE}."
                "iter_camera_frames",
                side_effect=create_frame_source,
            ) as iter_camera,
            patch(
                f"{PREFLIGHT_MODULE}."
                "iter_picamera2_frames",
            ) as iter_picamera2,
        ):
            result = (
                camera_preflight
                .run_camera_preflight(
                    camera_index=0,
                    width=16,
                    height=12,
                    fps=30.0,
                    sample_frames=1,
                )
            )

        self.assertEqual(
            result.camera_backend,
            "opencv",
        )

        iter_camera.assert_called_once()
        iter_picamera2.assert_not_called()

    def test_preflight_uses_picamera2_backend(
            self,
    ) -> None:
        frame_source = MagicMock()
        frame_source.__next__.return_value = object()

        profile = Picamera2ControlProfile(
            ae_enable=False,
            exposure_time_us=10000,
        )

        control_result = Picamera2ControlResult(
            name="exposure_time_us",
            control_name="ExposureTime",
            requested_value=10000,
            applied=True,
            effective_value=10000,
            verified=True,
            matches_requested=True,
        )

        def create_frame_source(
                camera_index: int,
                **arguments,
        ):
            arguments["capture_mode_reporter"](
                16,
                12,
                29.97,
            )
            arguments["control_reporter"](
                (
                    control_result,
                )
            )
            return frame_source

        with (
            patch(
                f"{PREFLIGHT_MODULE}."
                "iter_picamera2_frames",
                side_effect=create_frame_source,
            ) as iter_picamera2,
            patch(
                f"{PREFLIGHT_MODULE}."
                "iter_camera_frames",
            ) as iter_camera,
        ):
            result = (
                camera_preflight
                .run_camera_preflight(
                    camera_backend="picamera2",
                    camera_index=0,
                    width=16,
                    height=12,
                    fps=30.0,
                    sample_frames=1,
                    picamera2_control_profile=profile,
                )
            )

        self.assertEqual(
            result.camera_backend,
            "picamera2",
        )
        self.assertEqual(
            result.effective_width,
            16,
        )
        self.assertEqual(
            result.effective_height,
            12,
        )
        self.assertAlmostEqual(
            result.effective_fps,
            29.97,
        )
        self.assertEqual(
            result.camera_controls,
            (
                control_result,
            ),
        )

        iter_picamera2.assert_called_once_with(
            0,
            width=16,
            height=12,
            fps=30.0,
            control_profile=profile,
            control_reporter=ANY,
            capture_mode_reporter=ANY,
        )

        iter_camera.assert_not_called()

    def test_unsupported_camera_backend_is_rejected(
            self,
    ) -> None:
        with (
            patch(
                f"{PREFLIGHT_MODULE}."
                "iter_camera_frames",
            ) as iter_camera,
            patch(
                f"{PREFLIGHT_MODULE}."
                "iter_picamera2_frames",
            ) as iter_picamera2,
        ):
            with self.assertRaisesRegex(
                    ValueError,
                    "Unsupported camera backend",
            ):
                camera_preflight.run_camera_preflight(
                    camera_backend="unknown",
                    camera_index=0,
                    width=16,
                    height=12,
                    fps=30.0,
                    sample_frames=1,
                )

        iter_camera.assert_not_called()
        iter_picamera2.assert_not_called()

    def test_cli_routes_picamera2_backend(
            self,
    ) -> None:
        result = (
            camera_preflight.CameraPreflightResult(
                camera_backend="picamera2",
                camera_index=0,
                effective_width=16,
                effective_height=12,
                effective_fps=30.0,
                sampled_frame_count=1,
                camera_controls=(),
            )
        )

        with patch(
                f"{PREFLIGHT_MODULE}."
                "run_camera_preflight",
                return_value=result,
        ) as run_preflight:
            exit_code = camera_preflight.run_cli(
                [
                    "--camera-backend",
                    "picamera2",
                    "--camera-index",
                    "0",
                    "--width",
                    "16",
                    "--height",
                    "12",
                    "--fps",
                    "30",
                    "--sample-frames",
                    "1",
                ]
            )

        self.assertEqual(
            exit_code,
            0,
        )

        run_preflight.assert_called_once_with(
            camera_backend="picamera2",
            camera_index=0,
            width=16,
            height=12,
            fps=30.0,
            sample_frames=1,
            camera_controls=(),
            picamera2_control_profile=(
                Picamera2ControlProfile()
            ),
        )

    def test_cli_routes_picamera2_controls(
            self,
    ) -> None:
        result = (
            camera_preflight.CameraPreflightResult(
                camera_backend="picamera2",
                camera_index=0,
                effective_width=16,
                effective_height=12,
                effective_fps=30.0,
                sampled_frame_count=1,
                camera_controls=(),
            )
        )

        with patch(
                f"{PREFLIGHT_MODULE}."
                "run_camera_preflight",
                return_value=result,
        ) as run_preflight:
            exit_code = camera_preflight.run_cli(
                [
                    "--camera-backend",
                    "picamera2",
                    "--camera-index",
                    "0",
                    "--width",
                    "16",
                    "--height",
                    "12",
                    "--fps",
                    "30",
                    "--sample-frames",
                    "1",
                    "--ae-enable",
                    "false",
                    "--exposure-time-us",
                    "10000",
                    "--analogue-gain",
                    "2.0",
                    "--awb-enable",
                    "false",
                    "--colour-gains",
                    "1.5",
                    "1.25",
                ]
            )

        self.assertEqual(exit_code, 0)

        run_preflight.assert_called_once_with(
            camera_backend="picamera2",
            camera_index=0,
            width=16,
            height=12,
            fps=30.0,
            sample_frames=1,
            camera_controls=(),
            picamera2_control_profile=(
                Picamera2ControlProfile(
                    ae_enable=False,
                    exposure_time_us=10000,
                    analogue_gain=2.0,
                    awb_enable=False,
                    colour_gains=(
                        1.5,
                        1.25,
                    ),
                )
            ),
        )

    def test_cli_rejects_picamera2_manual_exposure_without_ae_disabled(
            self,
    ) -> None:
        captured_error = StringIO()

        with (
            patch(
                f"{PREFLIGHT_MODULE}."
                "run_camera_preflight",
            ) as run_preflight,
            redirect_stderr(captured_error),
        ):
            exit_code = camera_preflight.run_cli(
                [
                    "--camera-backend",
                    "picamera2",
                    "--camera-index",
                    "0",
                    "--width",
                    "16",
                    "--height",
                    "12",
                    "--fps",
                    "30",
                    "--sample-frames",
                    "1",
                    "--exposure-time-us",
                    "10000",
                ]
            )

        self.assertEqual(exit_code, 1)
        run_preflight.assert_not_called()

        self.assertIn(
            "ae_enable",
            captured_error.getvalue(),
        )

if __name__ == "__main__":
    unittest.main()
