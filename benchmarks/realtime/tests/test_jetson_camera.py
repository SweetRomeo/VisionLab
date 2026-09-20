from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import cv2

from benchmarks.realtime.jetson_camera import (
    JetsonCameraError,
    build_jetson_gstreamer_pipeline,
    iter_jetson_gstreamer_frames,
)

from benchmarks.realtime.jetson_controls import (
    JetsonControlProfile,
    JetsonControlResult,
)


class JetsonCameraTests(unittest.TestCase):
    def test_build_pipeline_contains_requested_mode(
        self,
    ) -> None:
        pipeline = build_jetson_gstreamer_pipeline(
            1,
            width=1280,
            height=720,
            fps=30.0,
        )

        self.assertIn(
            "nvarguscamerasrc sensor-id=1",
            pipeline,
        )
        self.assertIn(
            "width=(int)1280",
            pipeline,
        )
        self.assertIn(
            "height=(int)720",
            pipeline,
        )
        self.assertIn(
            "framerate=(fraction)30/1",
            pipeline,
        )
        self.assertIn(
            "format=(string)BGR",
            pipeline,
        )
        self.assertIn(
            "appsink",
            pipeline,
        )

    def test_invalid_camera_index_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "camera_index",
        ):
            build_jetson_gstreamer_pipeline(
                -1,
                width=1280,
                height=720,
                fps=30.0,
            )

    def test_frames_are_yielded_and_capture_released(
        self,
    ) -> None:
        frame = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        )

        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True,
            frame,
        )

        capture_factory = Mock(
            return_value=capture
        )

        frame_source = iter_jetson_gstreamer_frames(
            0,
            width=1280,
            height=720,
            fps=30.0,
            capture_factory=capture_factory,
        )

        received_frame = next(frame_source)
        frame_source.close()

        self.assertIs(
            received_frame,
            frame,
        )
        capture.release.assert_called_once()

    def test_camera_open_failure_releases_capture(
        self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = False

        frame_source = iter_jetson_gstreamer_frames(
            0,
            width=1280,
            height=720,
            fps=30.0,
            capture_factory=lambda *_: capture,
        )

        with self.assertRaisesRegex(
            JetsonCameraError,
            "could not be opened",
        ):
            next(frame_source)

        capture.release.assert_called_once()

    def test_empty_frame_is_rejected(
        self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True,
            None,
        )

        frame_source = iter_jetson_gstreamer_frames(
            0,
            width=1280,
            height=720,
            fps=30.0,
            capture_factory=lambda *_: capture,
        )

        with self.assertRaisesRegex(
            JetsonCameraError,
            "empty frame",
        ):
            next(frame_source)

        capture.release.assert_called_once()

    def test_wrong_dimensions_are_rejected(
        self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True,
            np.zeros(
                (480, 640, 3),
                dtype=np.uint8,
            ),
        )

        frame_source = iter_jetson_gstreamer_frames(
            0,
            width=1280,
            height=720,
            fps=30.0,
            capture_factory=lambda *_: capture,
        )

        with self.assertRaisesRegex(
            JetsonCameraError,
            "dimensions",
        ):
            next(frame_source)

        capture.release.assert_called_once()

    def test_effective_capture_mode_is_reported(
        self,
    ) -> None:
        frame = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        )

        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True,
            frame,
        )
        capture.get.return_value = 29.97

        reporter = Mock()

        frame_source = iter_jetson_gstreamer_frames(
            0,
            width=1280,
            height=720,
            fps=30.0,
            capture_mode_reporter=reporter,
            capture_factory=lambda *_: capture,
        )

        next(frame_source)
        frame_source.close()

        reporter.assert_called_once_with(
            1280,
            720,
            29.97,
        )

        capture.get.assert_called_once_with(
            cv2.CAP_PROP_FPS
        )

    def test_invalid_effective_fps_is_rejected(
        self,
    ) -> None:
        frame = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        )

        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True,
            frame,
        )
        capture.get.return_value = 0.0

        reporter = Mock()

        frame_source = iter_jetson_gstreamer_frames(
            0,
            width=1280,
            height=720,
            fps=30.0,
            capture_mode_reporter=reporter,
            capture_factory=lambda *_: capture,
        )

        with self.assertRaisesRegex(
            JetsonCameraError,
            "effective FPS",
        ):
            next(frame_source)

        reporter.assert_not_called()
        capture.release.assert_called_once()

    def test_pipeline_contains_jetson_control_properties(
        self,
    ) -> None:
        profile = JetsonControlProfile(
            ae_lock=True,
            exposure_time_ns=5_000_000,
            gain=2.5,
            awb_lock=False,
            wb_mode=1,
        )

        pipeline = build_jetson_gstreamer_pipeline(
            0,
            width=1280,
            height=720,
            fps=30.0,
            control_profile=profile,
        )

        self.assertIn(
            "aelock=true",
            pipeline,
        )
        self.assertIn(
            'exposuretimerange="5000000 5000000"',
            pipeline,
        )
        self.assertIn(
            'gainrange="2.5 2.5"',
            pipeline,
        )
        self.assertIn(
            "awblock=false",
            pipeline,
        )
        self.assertIn(
            "wbmode=1",
            pipeline,
        )

    def test_frame_source_forwards_control_profile(
        self,
    ) -> None:
        frame = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        )

        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True,
            frame,
        )

        capture_factory = Mock(
            return_value=capture
        )

        profile = JetsonControlProfile(
            ae_lock=True,
            exposure_time_ns=5_000_000,
        )

        frame_source = iter_jetson_gstreamer_frames(
            0,
            width=1280,
            height=720,
            fps=30.0,
            control_profile=profile,
            capture_factory=capture_factory,
        )

        next(frame_source)
        frame_source.close()

        pipeline = (
            capture_factory.call_args.args[0]
        )

        self.assertIn(
            "aelock=true",
            pipeline,
        )
        self.assertIn(
            'exposuretimerange="5000000 5000000"',
            pipeline,
        )

    def test_applied_controls_are_reported_after_open(
        self,
    ) -> None:
        frame = np.zeros(
            (720, 1280, 3),
            dtype=np.uint8,
        )

        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True,
            frame,
        )

        reporter = Mock()

        profile = JetsonControlProfile(
            ae_lock=True,
            gain=2.5,
        )

        frame_source = iter_jetson_gstreamer_frames(
            0,
            width=1280,
            height=720,
            fps=30.0,
            control_profile=profile,
            control_reporter=reporter,
            capture_factory=lambda *_: capture,
        )

        next(frame_source)
        frame_source.close()

        reporter.assert_called_once_with(
            (
                JetsonControlResult(
                    name="ae_lock",
                    control_name="aelock",
                    requested_value=True,
                    applied=True,
                    effective_value=None,
                    verified=False,
                    matches_requested=None,
                ),
                JetsonControlResult(
                    name="gain",
                    control_name="gainrange",
                    requested_value=2.5,
                    applied=True,
                    effective_value=None,
                    verified=False,
                    matches_requested=None,
                ),
            )
        )

    def test_invalid_control_reporter_is_rejected(
        self,
    ) -> None:
        frame_source = (
            iter_jetson_gstreamer_frames(
                0,
                width=1280,
                height=720,
                fps=30.0,
                control_reporter="invalid",
            )
        )

        with self.assertRaisesRegex(
            TypeError,
            "control_reporter must be callable",
        ):
            next(frame_source)