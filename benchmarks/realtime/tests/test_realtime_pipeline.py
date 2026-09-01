import time
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import Mock, call, patch

import numpy as np

from benchmarks.realtime import realtime_pipeline

from benchmarks.realtime.realtime_config import (
    RealtimeConfig,
)
from benchmarks.realtime.realtime_pipeline import (
    iter_video_frames,
    run_realtime_trial,
)
from benchmarks.realtime.realtime_records import (
    FrameStatus,
)


class RealtimePipelineTests(unittest.TestCase):
    def create_config(
        self,
        *,
        target_fps: float = 20.0,
        warmup_frames: int = 0,
        measured_frames: int = 3,
        deadline_multiplier: float = 1.0,
    ) -> RealtimeConfig:
        return RealtimeConfig(
            schema_version=1,
            target_fps=target_fps,
            queue_capacity=1,
            warmup_frames=warmup_frames,
            measured_frames=measured_frames,
            trial_count=1,
            drop_policy="latest_frame",
            deadline_multiplier=(
                deadline_multiplier
            ),
            output_directory=Path(
                "benchmarks/results/realtime"
            ),
            frame_results_file=(
                "realtime_frame_results.csv"
            ),
            summary_file=(
                "realtime_summary.csv"
            ),
        )

    def create_frames(
        self,
        count: int,
    ) -> list[np.ndarray]:
        return [
            np.full(
                (12, 16, 3),
                fill_value=frame_index,
                dtype=np.uint8,
            )
            for frame_index in range(count)
        ]

    @staticmethod
    def copy_processor(
        frame: np.ndarray,
    ) -> np.ndarray:
        return frame.copy()

    def test_fast_processor_records_all_frames(
        self,
    ) -> None:
        config = self.create_config(
            measured_frames=3,
        )

        records = run_realtime_trial(
            frame_source=self.create_frames(3),
            processor=self.copy_processor,
            config=config,
            architecture="pure_python",
            algorithm="original",
            width=16,
            height=12,
            trial=1,
        )

        self.assertEqual(len(records), 3)
        self.assertEqual(
            [
                record.frame_index
                for record in records
            ],
            [1, 2, 3],
        )
        self.assertTrue(
            all(
                record.frame_status
                is FrameStatus.PROCESSED
                for record in records
            )
        )

    def test_slow_processor_misses_deadline(
        self,
    ) -> None:
        config = self.create_config(
            target_fps=1000.0,
            measured_frames=1,
        )

        def slow_processor(
            frame: np.ndarray,
        ) -> np.ndarray:
            time.sleep(0.005)
            return frame.copy()

        records = run_realtime_trial(
            frame_source=self.create_frames(1),
            processor=slow_processor,
            config=config,
            architecture="pure_python",
            algorithm="original",
            width=16,
            height=12,
            trial=1,
        )

        self.assertEqual(len(records), 1)
        self.assertTrue(
            records[0].deadline_missed
        )
        self.assertEqual(
            records[0].frame_status,
            FrameStatus.PROCESSED,
        )

    def test_latest_frame_policy_records_drops(
        self,
    ) -> None:
        config = self.create_config(
            target_fps=50.0,
            measured_frames=4,
        )
        release_processor = Event()
        processor_call_count = 0

        def controlled_frame_source():
            for frame_index, frame in enumerate(
                self.create_frames(4)
            ):
                if frame_index == 3:
                    release_processor.set()

                yield frame

        def controlled_processor(
            frame: np.ndarray,
        ) -> np.ndarray:
            nonlocal processor_call_count
            processor_call_count += 1

            if processor_call_count == 1:
                released = (
                    release_processor.wait(
                        timeout=2.0
                    )
                )
                self.assertTrue(released)

            return frame.copy()

        records = run_realtime_trial(
            frame_source=controlled_frame_source(),
            processor=controlled_processor,
            config=config,
            architecture="pure_python",
            algorithm="original",
            width=16,
            height=12,
            trial=1,
        )

        statuses = [
            record.frame_status
            for record in records
        ]

        self.assertEqual(len(records), 4)
        self.assertIn(
            FrameStatus.DROPPED,
            statuses,
        )
        self.assertIn(
            FrameStatus.PROCESSED,
            statuses,
        )
        self.assertEqual(
            sorted(
                record.frame_index
                for record in records
            ),
            [1, 2, 3, 4],
        )

    def test_warmup_frames_are_not_recorded(
        self,
    ) -> None:
        config = self.create_config(
            warmup_frames=2,
            measured_frames=3,
        )

        records = run_realtime_trial(
            frame_source=self.create_frames(5),
            processor=self.copy_processor,
            config=config,
            architecture="hybrid",
            algorithm="clahe",
            width=16,
            height=12,
            trial=1,
        )

        self.assertEqual(len(records), 3)
        self.assertEqual(
            [
                record.frame_index
                for record in records
            ],
            [1, 2, 3],
        )

    def test_short_frame_source_is_rejected(
        self,
    ) -> None:
        config = self.create_config(
            measured_frames=3,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "ended before",
        ):
            run_realtime_trial(
                frame_source=self.create_frames(2),
                processor=self.copy_processor,
                config=config,
                architecture="pure_python",
                algorithm="original",
                width=16,
                height=12,
                trial=1,
            )

    def test_invalid_processor_output_is_rejected(
        self,
    ) -> None:
        config = self.create_config(
            measured_frames=1,
        )

        def invalid_processor(
            frame: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(
                (4, 4),
                dtype=np.uint8,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "shape",
        ):
            run_realtime_trial(
                frame_source=self.create_frames(1),
                processor=invalid_processor,
                config=config,
                architecture="pure_python",
                algorithm="original",
                width=16,
                height=12,
                trial=1,
            )

    def test_missing_video_is_rejected(
        self,
    ) -> None:
        frame_iterator = iter_video_frames(
            Path("missing-video.mp4")
        )

        with self.assertRaises(
            FileNotFoundError
        ):
            next(frame_iterator)

    def test_camera_open_failure_is_rejected(
        self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = False

        with patch.object(
            realtime_pipeline.cv2,
            "VideoCapture",
            return_value=capture,
        ) as video_capture:
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0
                )
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Camera 0 could not be opened",
            ):
                next(frame_iterator)

        video_capture.assert_called_once_with(0)
        capture.release.assert_called_once_with()

    def test_camera_frame_is_yielded_and_released(
        self,
    ) -> None:
        expected_frame = self.create_frames(
            1
        )[0]

        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True,
            expected_frame,
        )

        with patch.object(
            realtime_pipeline.cv2,
            "VideoCapture",
            return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0
                )
            )

            actual_frame = next(
                frame_iterator
            )
            frame_iterator.close()

        self.assertIs(
            actual_frame,
            expected_frame,
        )
        capture.read.assert_called_once_with()
        capture.release.assert_called_once_with()

    def test_empty_camera_frame_is_rejected(
        self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            True,
            np.empty(
                (0, 0, 3),
                dtype=np.uint8,
            ),
        )

        with patch.object(
            realtime_pipeline.cv2,
            "VideoCapture",
            return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0
                )
            )

            try:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "empty frame",
                ):
                    next(frame_iterator)
            finally:
                frame_iterator.close()

        capture.release.assert_called_once_with()

    def test_camera_frame_read_failure_is_rejected(
        self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (
            False,
            None,
        )

        with patch.object(
            realtime_pipeline.cv2,
            "VideoCapture",
            return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    2
                )
            )

            with self.assertRaisesRegex(
                RuntimeError,
                (
                    "could not be read "
                    "from camera 2"
                ),
            ):
                next(frame_iterator)

        capture.release.assert_called_once_with()

    def test_negative_camera_index_is_rejected_before_open(
        self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = False

        with patch.object(
            realtime_pipeline.cv2,
            "VideoCapture",
            return_value=capture,
        ) as video_capture:
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    -1
                )
            )

            with self.assertRaisesRegex(
                ValueError,
                (
                    "camera_index must be a "
                    "non-negative integer"
                ),
            ):
                next(frame_iterator)

        video_capture.assert_not_called()

    def test_non_integer_camera_indices_are_rejected_before_open(
        self,
    ) -> None:
        invalid_indices = (
            True,
            1.5,
            "0",
            None,
        )

        with patch.object(
            realtime_pipeline.cv2,
            "VideoCapture",
        ) as video_capture:
            for invalid_index in invalid_indices:
                with self.subTest(
                    camera_index=invalid_index
                ):
                    frame_iterator = (
                        realtime_pipeline
                        .iter_camera_frames(
                            invalid_index
                        )
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        (
                            "camera_index must be a "
                            "non-negative integer"
                        ),
                    ):
                        next(frame_iterator)

        video_capture.assert_not_called()

    def test_camera_width_is_configured_before_frame_read(
            self,
    ) -> None:
        expected_frame = self.create_frames(1)[0]

        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = True
        capture.get.return_value = 16.0
        capture.read.return_value = (
            True,
            expected_frame,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    width=16,
                )
            )

            try:
                next(frame_iterator)
            finally:
                frame_iterator.close()

        capture.assert_has_calls(
            [
                call.isOpened(),
                call.set(
                    realtime_pipeline.cv2
                    .CAP_PROP_FRAME_WIDTH,
                    16.0,
                ),
                call.get(
                    realtime_pipeline.cv2
                    .CAP_PROP_FRAME_WIDTH,
                ),
                call.read(),
            ]
        )
        capture.release.assert_called_once_with()

    def test_rejected_camera_width_configuration_is_reported(
            self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = False
        capture.read.return_value = (
            False,
            None,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    width=16,
                )
            )

            with self.assertRaisesRegex(
                    RuntimeError,
                    "Camera width could not be configured",
            ):
                next(frame_iterator)

        capture.read.assert_not_called()
        capture.release.assert_called_once_with()

    def test_invalid_camera_widths_are_rejected_before_open(
            self,
    ) -> None:
        invalid_widths = (
            True,
            0,
            -1,
            16.5,
            "16",
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
        ) as video_capture:
            for invalid_width in invalid_widths:
                with self.subTest(
                        width=invalid_width
                ):
                    frame_iterator = (
                        realtime_pipeline
                        .iter_camera_frames(
                            0,
                            width=invalid_width,
                        )
                    )

                    with self.assertRaisesRegex(
                            ValueError,
                            (
                                    "width must be a "
                                    "positive integer"
                            ),
                    ):
                        next(frame_iterator)

        video_capture.assert_not_called()

    def test_camera_height_is_configured_before_frame_read(
            self,
    ) -> None:
        expected_frame = self.create_frames(1)[0]

        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = True
        capture.get.return_value = 12.0
        capture.read.return_value = (
            True,
            expected_frame,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    height=12,
                )
            )

            try:
                next(frame_iterator)
            finally:
                frame_iterator.close()

        capture.assert_has_calls(
            [
                call.isOpened(),
                call.set(
                    realtime_pipeline.cv2
                    .CAP_PROP_FRAME_HEIGHT,
                    12.0,
                ),
                call.get(
                    realtime_pipeline.cv2
                    .CAP_PROP_FRAME_HEIGHT,
                ),
                call.read(),
            ]
        )
        capture.release.assert_called_once_with()

    def test_rejected_camera_height_configuration_is_reported(
            self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = False
        capture.read.return_value = (
            False,
            None,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    height=12,
                )
            )

            with self.assertRaisesRegex(
                    RuntimeError,
                    "Camera height could not be configured",
            ):
                next(frame_iterator)

        capture.read.assert_not_called()
        capture.release.assert_called_once_with()

    def test_invalid_camera_heights_are_rejected_before_open(
            self,
    ) -> None:
        invalid_heights = (
            True,
            0,
            -1,
            12.5,
            "12",
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
        ) as video_capture:
            for invalid_height in invalid_heights:
                with self.subTest(
                        height=invalid_height
                ):
                    frame_iterator = (
                        realtime_pipeline
                        .iter_camera_frames(
                            0,
                            height=invalid_height,
                        )
                    )

                    with self.assertRaisesRegex(
                            ValueError,
                            (
                                    "height must be a "
                                    "positive integer"
                            ),
                    ):
                        next(frame_iterator)

        video_capture.assert_not_called()

    def test_camera_fps_is_configured_before_frame_read(
            self,
    ) -> None:
        expected_frame = self.create_frames(1)[0]

        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = True
        capture.get.return_value = 30.0
        capture.read.return_value = (
            True,
            expected_frame,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    fps=30.0,
                )
            )

            try:
                next(frame_iterator)
            finally:
                frame_iterator.close()

        capture.assert_has_calls(
            [
                call.isOpened(),
                call.set(
                    realtime_pipeline.cv2
                    .CAP_PROP_FPS,
                    30.0,
                ),
                call.get(
                    realtime_pipeline.cv2.CAP_PROP_FPS,
                ),
                call.read(),
            ]
        )
        capture.release.assert_called_once_with()

    def test_rejected_camera_fps_configuration_is_reported(
            self,
    ) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = False
        capture.read.return_value = (
            False,
            None,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    fps=30.0,
                )
            )

            with self.assertRaisesRegex(
                    RuntimeError,
                    "Camera FPS could not be configured",
            ):
                next(frame_iterator)

        capture.read.assert_not_called()
        capture.release.assert_called_once_with()

    def test_invalid_camera_fps_values_are_rejected_before_open(
            self,
    ) -> None:
        invalid_fps_values = (
            True,
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            "30",
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
        ) as video_capture:
            for invalid_fps in invalid_fps_values:
                with self.subTest(
                        fps=invalid_fps
                ):
                    frame_iterator = (
                        realtime_pipeline
                        .iter_camera_frames(
                            0,
                            fps=invalid_fps,
                        )
                    )

                    with self.assertRaisesRegex(
                            ValueError,
                            (
                                    "fps must be a positive "
                                    "finite number"
                            ),
                    ):
                        next(frame_iterator)

        video_capture.assert_not_called()

    def test_mismatched_camera_width_readback_is_rejected(
            self,
    ) -> None:
        expected_frame = self.create_frames(1)[0]

        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = True
        capture.get.return_value = 15.0
        capture.read.return_value = (
            True,
            expected_frame,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    width=16,
                )
            )

            with self.assertRaisesRegex(
                    RuntimeError,
                    (
                            "Camera width does not match "
                            "the requested value"
                    ),
            ):
                next(frame_iterator)

        capture.get.assert_called_once_with(
            realtime_pipeline.cv2
            .CAP_PROP_FRAME_WIDTH
        )
        capture.read.assert_not_called()
        capture.release.assert_called_once_with()

    def test_mismatched_camera_height_readback_is_rejected(
            self,
    ) -> None:
        expected_frame = self.create_frames(1)[0]

        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = True
        capture.get.return_value = 11.0
        capture.read.return_value = (
            True,
            expected_frame,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    height=12,
                )
            )

            with self.assertRaisesRegex(
                    RuntimeError,
                    (
                            "Camera height does not match "
                            "the requested value"
                    ),
            ):
                next(frame_iterator)

        capture.get.assert_called_once_with(
            realtime_pipeline.cv2
            .CAP_PROP_FRAME_HEIGHT
        )
        capture.read.assert_not_called()
        capture.release.assert_called_once_with()

    def test_mismatched_camera_fps_readback_is_rejected(
            self,
    ) -> None:
        expected_frame = self.create_frames(1)[0]

        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = True
        capture.get.return_value = 25.0
        capture.read.return_value = (
            True,
            expected_frame,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    fps=30.0,
                )
            )

            with self.assertRaisesRegex(
                    RuntimeError,
                    (
                            "Camera FPS does not match "
                            "the requested value"
                    ),
            ):
                next(frame_iterator)

        capture.get.assert_called_once_with(
            realtime_pipeline.cv2.CAP_PROP_FPS
        )
        capture.read.assert_not_called()
        capture.release.assert_called_once_with()

    def test_camera_fps_readback_accepts_standard_rounding(
            self,
    ) -> None:
        expected_frame = self.create_frames(1)[0]

        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.return_value = True
        capture.get.return_value = 29.97
        capture.read.return_value = (
            True,
            expected_frame,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    fps=30.0,
                )
            )

            try:
                actual_frame = next(frame_iterator)
            finally:
                frame_iterator.close()

        self.assertIs(
            actual_frame,
            expected_frame,
        )
        capture.get.assert_called_once_with(
            realtime_pipeline.cv2.CAP_PROP_FPS
        )
        capture.read.assert_called_once_with()
        capture.release.assert_called_once_with()

    def test_invalid_camera_width_readbacks_are_rejected(
            self,
    ) -> None:
        invalid_readbacks = (
            None,
            "16",
            0.0,
            float("nan"),
            float("inf"),
        )

        for invalid_readback in invalid_readbacks:
            with self.subTest(
                    readback=invalid_readback
            ):
                capture = Mock()
                capture.isOpened.return_value = True
                capture.set.return_value = True
                capture.get.return_value = (
                    invalid_readback
                )

                with patch.object(
                        realtime_pipeline.cv2,
                        "VideoCapture",
                        return_value=capture,
                ):
                    frame_iterator = (
                        realtime_pipeline
                        .iter_camera_frames(
                            0,
                            width=16,
                        )
                    )

                    with self.assertRaisesRegex(
                            RuntimeError,
                            (
                                    "Camera width returned "
                                    "an invalid value"
                            ),
                    ):
                        next(frame_iterator)

                capture.read.assert_not_called()
                capture.release.assert_called_once_with()

    def test_invalid_camera_height_readbacks_are_rejected(
            self,
    ) -> None:
        invalid_readbacks = (
            None,
            "12",
            0.0,
            float("nan"),
            float("inf"),
        )

        for invalid_readback in invalid_readbacks:
            with self.subTest(
                    readback=invalid_readback
            ):
                capture = Mock()
                capture.isOpened.return_value = True
                capture.set.return_value = True
                capture.get.return_value = (
                    invalid_readback
                )

                with patch.object(
                        realtime_pipeline.cv2,
                        "VideoCapture",
                        return_value=capture,
                ):
                    frame_iterator = (
                        realtime_pipeline
                        .iter_camera_frames(
                            0,
                            height=12,
                        )
                    )

                    with self.assertRaisesRegex(
                            RuntimeError,
                            (
                                    "Camera height returned "
                                    "an invalid value"
                            ),
                    ):
                        next(frame_iterator)

                capture.read.assert_not_called()
                capture.release.assert_called_once_with()

    def test_invalid_camera_fps_readbacks_are_rejected(
            self,
    ) -> None:
        invalid_readbacks = (
            None,
            "30",
            0.0,
            float("nan"),
            float("inf"),
        )

        for invalid_readback in invalid_readbacks:
            with self.subTest(
                    readback=invalid_readback
            ):
                capture = Mock()
                capture.isOpened.return_value = True
                capture.set.return_value = True
                capture.get.return_value = (
                    invalid_readback
                )

                with patch.object(
                        realtime_pipeline.cv2,
                        "VideoCapture",
                        return_value=capture,
                ):
                    frame_iterator = (
                        realtime_pipeline
                        .iter_camera_frames(
                            0,
                            fps=30.0,
                        )
                    )

                    with self.assertRaisesRegex(
                            RuntimeError,
                            (
                                    "Camera FPS returned "
                                    "an invalid value"
                            ),
                    ):
                        next(frame_iterator)

                capture.read.assert_not_called()
                capture.release.assert_called_once_with()

    def test_camera_frame_dimension_mismatch_is_rejected(
            self,
    ) -> None:
        mismatched_frame = np.zeros(
            (10, 16, 3),
            dtype=np.uint8,
        )

        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.side_effect = (
            True,
            True,
        )
        capture.get.side_effect = (
            16.0,
            12.0,
        )
        capture.read.return_value = (
            True,
            mismatched_frame,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    width=16,
                    height=12,
                )
            )

            try:
                with self.assertRaisesRegex(
                        RuntimeError,
                        (
                                "Camera frame dimensions "
                                "do not match"
                        ),
                ):
                    next(frame_iterator)
            finally:
                frame_iterator.close()

        capture.read.assert_called_once_with()
        capture.release.assert_called_once_with()

    def test_camera_reports_effective_capture_mode(
            self,
    ) -> None:
        expected_frame = self.create_frames(1)[0]
        capture_mode_reporter = Mock()

        capture = Mock()
        capture.isOpened.return_value = True
        capture.set.side_effect = (
            True,
            True,
            True,
        )
        capture.get.side_effect = (
            16.0,
            12.0,
            29.97,
        )
        capture.read.return_value = (
            True,
            expected_frame,
        )

        with patch.object(
                realtime_pipeline.cv2,
                "VideoCapture",
                return_value=capture,
        ):
            frame_iterator = (
                realtime_pipeline.iter_camera_frames(
                    0,
                    width=16,
                    height=12,
                    fps=30.0,
                    capture_mode_reporter=(
                        capture_mode_reporter
                    ),
                )
            )

            try:
                next(frame_iterator)
            finally:
                frame_iterator.close()

        capture_mode_reporter.assert_called_once_with(
            16,
            12,
            29.97,
        )
        capture.release.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
