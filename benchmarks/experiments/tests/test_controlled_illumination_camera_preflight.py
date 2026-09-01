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
            capture_mode_reporter=ANY,
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
                camera_index=0,
                effective_width=16,
                effective_height=12,
                effective_fps=29.97,
                sampled_frame_count=3,
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
            camera_index=0,
            width=16,
            height=12,
            fps=30.0,
            sample_frames=3,
        )

        output = captured_output.getvalue()

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

if __name__ == "__main__":
    unittest.main()
