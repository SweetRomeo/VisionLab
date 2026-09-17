from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from benchmarks.realtime.picamera2_controls import (
    Picamera2ControlProfile,
    Picamera2ControlResult,
    apply_picamera2_control_profile,
    create_picamera2_controls,
    verify_picamera2_control_profile,
)


class Picamera2CameraError(RuntimeError):
    """Raised when the Picamera2 camera backend cannot operate."""


def load_picamera2_factory() -> Callable[[int], Any]:
    try:
        from picamera2 import Picamera2
    except ImportError as error:
        raise Picamera2CameraError(
            "Picamera2 is required for the "
            "Raspberry Pi camera backend."
        ) from error

    return Picamera2


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


def iter_picamera2_frames(
    camera_index: int,
    *,
    width: int,
    height: int,
    fps: float,
    control_profile: Picamera2ControlProfile | None = None,
    control_reporter: (
        Callable[
            [tuple[Picamera2ControlResult, ...]],
            None,
        ]
        | None
    ) = None,
    capture_mode_reporter: (
        Callable[[int, int, float], None]
        | None
    ) = None,
    picamera2_factory: (
        Callable[[int], Any] | None
    ) = None,
) -> Iterator[np.ndarray]:
    _validate_camera_index(camera_index)
    _validate_capture_mode(
        width,
        height,
        fps,
    )

    active_control_profile = (
        Picamera2ControlProfile()
        if control_profile is None
        else control_profile
    )

    if (
        control_reporter is not None
        and not callable(control_reporter)
    ):
        raise TypeError(
            "control_reporter must be callable."
        )

    factory = (
        load_picamera2_factory()
        if picamera2_factory is None
        else picamera2_factory
    )

    frame_duration_us = round(
        1_000_000.0 / float(fps)
    )

    camera = factory(camera_index)
    started = False

    try:
        configuration = (
            camera.create_video_configuration(
                main={
                    "size": (
                        width,
                        height,
                    ),
                    "format": "RGB888",
                },
                controls={
                    "FrameDurationLimits": (
                        frame_duration_us,
                        frame_duration_us,
                    ),
                },
            )
        )

        camera.configure(configuration)

        apply_picamera2_control_profile(
            camera,
            active_control_profile,
        )

        camera.start()
        started = True

        requested_controls = (
            create_picamera2_controls(
                active_control_profile
            )
        )

        effective_metadata = None

        if (
                requested_controls
                or capture_mode_reporter is not None
        ):
            effective_metadata = (
                camera.capture_metadata()
            )

        if capture_mode_reporter is not None:
            effective_configuration = (
                camera.camera_configuration()
            )

            try:
                effective_size = (
                    effective_configuration[
                        "main"
                    ]["size"]
                )

                effective_width = int(
                    effective_size[0]
                )
                effective_height = int(
                    effective_size[1]
                )

                frame_duration_us = (
                    effective_metadata[
                        "FrameDuration"
                    ]
                )

                if (
                        isinstance(frame_duration_us, bool)
                        or not isinstance(
                    frame_duration_us,
                    (int, float),
                )
                        or not math.isfinite(
                    float(frame_duration_us)
                )
                        or frame_duration_us <= 0
                ):
                    raise ValueError

            except (
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
            ) as error:
                raise Picamera2CameraError(
                    "Picamera2 effective capture mode "
                    "could not be determined."
                ) from error

            effective_fps = (
                    1_000_000.0
                    / float(frame_duration_us)
            )

            capture_mode_reporter(
                effective_width,
                effective_height,
                effective_fps,
            )

        if requested_controls:
            control_results = (
                verify_picamera2_control_profile(
                    active_control_profile,
                    effective_metadata,
                )
            )

            if control_reporter is not None:
                control_reporter(
                    control_results
                )

        while True:
            frame = camera.capture_array(
                "main"
            )

            if not isinstance(
                    frame,
                    np.ndarray,
            ):
                raise Picamera2CameraError(
                    "Picamera2 returned a frame "
                    "that is not a NumPy array."
                )

            if (
                frame.size == 0
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.dtype != np.uint8
            ):
                raise Picamera2CameraError(
                    "Picamera2 returned an invalid "
                    "frame. Expected a non-empty "
                    "H x W x 3 uint8 image."
                )

            if (
                frame.shape[1] != width
                or frame.shape[0] != height
            ):
                raise Picamera2CameraError(
                    "Picamera2 frame dimensions do "
                    "not match the requested "
                    "capture mode."
                )

            yield frame

    finally:
        try:
            if started:
                camera.stop()
        finally:
            camera.close()