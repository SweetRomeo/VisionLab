from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from fractions import Fraction
from typing import Any

import cv2
import numpy as np

from benchmarks.realtime.jetson_controls import (
    JetsonControlProfile,
    JetsonControlResult,
    create_applied_jetson_control_results,
    serialize_jetson_source_properties,
)


class JetsonCameraError(RuntimeError):
    """Raised when the Jetson camera backend cannot operate."""


def _validate_camera_index(
    camera_index: int,
) -> None:
    if (
        isinstance(camera_index, bool)
        or not isinstance(camera_index, int)
        or camera_index < 0
    ):
        raise ValueError(
            "camera_index must be a "
            "non-negative integer."
        )


def _validate_capture_mode(
    width: int,
    height: int,
    fps: float,
) -> None:
    for field_name, value in (
        ("width", width),
        ("height", height),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be a "
                "positive integer."
            )

    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or fps <= 0
    ):
        raise ValueError(
            "fps must be a positive finite number."
        )


def build_jetson_gstreamer_pipeline(
    camera_index: int,
    *,
    width: int,
    height: int,
    fps: float,
    control_profile: (
        JetsonControlProfile | None
    ) = None,
) -> str:
    _validate_camera_index(camera_index)
    _validate_capture_mode(
        width,
        height,
        fps,
    )

    fps_fraction = Fraction(
        float(fps)
    ).limit_denominator(1000)

    active_control_profile = (
        JetsonControlProfile()
        if control_profile is None
        else control_profile
    )

    source_properties = (
        serialize_jetson_source_properties(
            active_control_profile
        )
    )

    source = (
        f"nvarguscamerasrc sensor-id={camera_index}"
    )

    if source_properties:
        source = (
            f"{source} {source_properties}"
        )

    return (
        f"{source} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){width}, "
        f"height=(int){height}, "
        "format=(string)NV12, "
        f"framerate=(fraction)"
        f"{fps_fraction.numerator}/"
        f"{fps_fraction.denominator} ! "
        "nvvidconv ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true sync=false"
    )


def iter_jetson_gstreamer_frames(
    camera_index: int,
    *,
    width: int,
    height: int,
    fps: float,
    control_profile: (
        JetsonControlProfile | None
    ) = None,
    control_reporter: (
        Callable[
            [
                tuple[
                    JetsonControlResult,
                    ...,
                ]
            ],
            None,
        ]
        | None
    ) = None,
    capture_mode_reporter: (
        Callable[[int, int, float], None]
        | None
    ) = None,
    capture_factory: (
        Callable[[str, int], Any] | None
    ) = None,
) -> Iterator[np.ndarray]:
    pipeline = build_jetson_gstreamer_pipeline(
        camera_index,
        width=width,
        height=height,
        fps=fps,
        control_profile=control_profile,
    )

    if (
        capture_mode_reporter is not None
        and not callable(capture_mode_reporter)
    ):
        raise TypeError(
            "capture_mode_reporter must be callable."
        )

    if (
        control_reporter is not None
        and not callable(control_reporter)
    ):
        raise TypeError(
            "control_reporter must be callable."
        )

    factory = (
        cv2.VideoCapture
        if capture_factory is None
        else capture_factory
    )

    capture = factory(
        pipeline,
        cv2.CAP_GSTREAMER,
    )

    capture_mode_reported = False

    try:
        if not capture.isOpened():
            raise JetsonCameraError(
                "Jetson GStreamer camera "
                f"{camera_index} could not be opened."
            )

        if control_reporter is not None:
            active_control_profile = (
                JetsonControlProfile()
                if control_profile is None
                else control_profile
            )

            control_reporter(
                create_applied_jetson_control_results(
                    active_control_profile
                )
            )

        while True:
            frame_received, frame = capture.read()

            if not frame_received:
                raise JetsonCameraError(
                    "Jetson GStreamer camera "
                    "frame acquisition failed."
                )

            if (
                not isinstance(frame, np.ndarray)
                or frame.size == 0
            ):
                raise JetsonCameraError(
                    "Jetson GStreamer camera "
                    "returned an empty frame."
                )

            if (
                frame.ndim != 3
                or frame.shape[2] != 3
            ):
                raise JetsonCameraError(
                    "Jetson GStreamer camera "
                    "must return HxWx3 frames."
                )

            if frame.dtype != np.uint8:
                raise JetsonCameraError(
                    "Jetson GStreamer camera "
                    "must return uint8 frames."
                )

            if (
                frame.shape[1] != width
                or frame.shape[0] != height
            ):
                raise JetsonCameraError(
                    "Jetson GStreamer camera frame "
                    "dimensions do not match the "
                    "requested capture mode."
                )

            if (
                capture_mode_reporter is not None
                and not capture_mode_reported
            ):
                effective_fps = capture.get(
                    cv2.CAP_PROP_FPS
                )

                if (
                    isinstance(effective_fps, bool)
                    or not isinstance(
                        effective_fps,
                        (int, float),
                    )
                    or not math.isfinite(
                        float(effective_fps)
                    )
                    or effective_fps <= 0
                ):
                    raise JetsonCameraError(
                        "Jetson GStreamer effective FPS "
                        "could not be determined."
                    )

                capture_mode_reporter(
                    int(frame.shape[1]),
                    int(frame.shape[0]),
                    float(effective_fps),
                )

                capture_mode_reported = True

            yield frame

    finally:
        capture.release()