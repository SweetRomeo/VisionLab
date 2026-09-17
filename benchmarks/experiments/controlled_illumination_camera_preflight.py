from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from benchmarks.realtime.realtime_pipeline import (
    iter_camera_frames,
)
from benchmarks.realtime.camera_controls import (
    CameraControlProfile,
    CameraControlRequest,
    CameraControlResult,
    create_camera_control_requests,
)
from benchmarks.realtime.picamera2_camera import (
    iter_picamera2_frames,
)
from benchmarks.realtime.picamera2_controls import (
    Picamera2ControlProfile,
    Picamera2ControlResult,
    validate_picamera2_control_profile,
)

OPENCV_CAMERA_BACKEND = "opencv"
PICAMERA2_CAMERA_BACKEND = "picamera2"

@dataclass(frozen=True)
class CameraPreflightResult:
    camera_backend: str
    camera_index: int
    effective_width: int
    effective_height: int
    effective_fps: float
    sampled_frame_count: int
    camera_controls: tuple[
        CameraControlResult | Picamera2ControlResult,
        ...,
    ]


def run_camera_preflight(
    *,
    camera_index: int,
    width: int,
    height: int,
    fps: float,
    sample_frames: int,
    camera_backend: str = OPENCV_CAMERA_BACKEND,
    camera_controls: tuple[
        CameraControlRequest,
        ...,
    ] = (),
    picamera2_control_profile: (
        Picamera2ControlProfile | None
    ) = None,
) -> CameraPreflightResult:
    if (
        isinstance(sample_frames, bool)
        or not isinstance(sample_frames, int)
        or sample_frames <= 0
    ):
        raise ValueError(
            "sample_frames must be "
            "a positive integer."
        )

    effective_mode: (
        tuple[int, int, float] | None
    ) = None

    effective_camera_controls: tuple[
        CameraControlResult | Picamera2ControlResult,
        ...,
    ] = ()

    def report_capture_mode(
        effective_width: int,
        effective_height: int,
        effective_fps: float,
    ) -> None:
        nonlocal effective_mode

        effective_mode = (
            effective_width,
            effective_height,
            effective_fps,
        )

    def report_camera_controls(
            results: tuple[
                CameraControlResult | Picamera2ControlResult,
                ...,
            ],
    ) -> None:
        nonlocal effective_camera_controls

        effective_camera_controls = results

    if camera_backend == OPENCV_CAMERA_BACKEND:
        frame_source = iter_camera_frames(
            camera_index,
            width=width,
            height=height,
            fps=fps,
            camera_controls=camera_controls,
            capture_mode_reporter=(
                report_capture_mode
            ),
            camera_controls_reporter=(
                report_camera_controls
            ),
        )

    elif camera_backend == PICAMERA2_CAMERA_BACKEND:
        frame_source = iter_picamera2_frames(
            camera_index,
            width=width,
            height=height,
            fps=fps,
            control_profile=(
                picamera2_control_profile
            ),
            control_reporter=(
                report_camera_controls
            ),
            capture_mode_reporter=(
                report_capture_mode
            ),
        )

    else:
        raise ValueError(
            "Unsupported camera backend: "
            f"{camera_backend}"
        )

    sampled_frame_count = 0

    try:
        for _ in range(sample_frames):
            next(frame_source)
            sampled_frame_count += 1
    finally:
        frame_source.close()

    if effective_mode is None:
        raise RuntimeError(
            "Camera capture mode was not reported."
        )

    (
        effective_width,
        effective_height,
        effective_fps,
    ) = effective_mode

    return CameraPreflightResult(
        camera_backend=camera_backend,
        camera_index=camera_index,
        effective_width=effective_width,
        effective_height=effective_height,
        effective_fps=effective_fps,
        sampled_frame_count=(
            sampled_frame_count
        ),
        camera_controls=(
            effective_camera_controls
        ),
    )

def create_argument_parser() -> (
    argparse.ArgumentParser
):
    parser = argparse.ArgumentParser(
        description=(
            "Validate a controlled-illumination "
            "live-camera capture mode."
        )
    )
    parser.add_argument(
        "--ae-enable",
        type=lambda value: (
            value.lower() == "true"
            if value.lower() in ("true", "false")
            else parser.error(
                "--ae-enable must be true or false."
            )
        ),
    )

    parser.add_argument(
        "--exposure-time-us",
        type=int,
    )

    parser.add_argument(
        "--analogue-gain",
        type=float,
    )

    parser.add_argument(
        "--awb-enable",
        type=lambda value: (
            value.lower() == "true"
            if value.lower() in ("true", "false")
            else parser.error(
                "--awb-enable must be true or false."
            )
        ),
    )

    parser.add_argument(
        "--colour-gains",
        type=float,
        nargs=2,
    )
    parser.add_argument(
        "--camera-backend",
        choices=(
            OPENCV_CAMERA_BACKEND,
            PICAMERA2_CAMERA_BACKEND,
        ),
        default=OPENCV_CAMERA_BACKEND,
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--fps",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--auto-exposure",
        type=float,
    )

    parser.add_argument(
        "--exposure",
        type=float,
    )

    parser.add_argument(
        "--gain",
        type=float,
    )

    parser.add_argument(
        "--auto-white-balance",
        type=float,
    )

    parser.add_argument(
        "--white-balance-temperature",
        type=float,
    )

    parser.add_argument(
        "--autofocus",
        type=float,
    )

    parser.add_argument(
        "--focus",
        type=float,
    )

    return parser


def run_cli(
    arguments: list[str] | None = None,
) -> int:
    parser = create_argument_parser()

    parsed_arguments = parser.parse_args(
        arguments
    )

    try:
        camera_control_profile = (
            CameraControlProfile(
                auto_exposure=(
                    parsed_arguments.auto_exposure
                ),
                exposure=(
                    parsed_arguments.exposure
                ),
                gain=(
                    parsed_arguments.gain
                ),
                auto_white_balance=(
                    parsed_arguments
                    .auto_white_balance
                ),
                white_balance_temperature=(
                    parsed_arguments
                    .white_balance_temperature
                ),
                autofocus=(
                    parsed_arguments.autofocus
                ),
                focus=(
                    parsed_arguments.focus
                ),
            )
        )

        camera_controls = (
            create_camera_control_requests(
                camera_control_profile
            )
        )

        camera_controls: tuple[
            CameraControlRequest,
            ...,
        ] = ()

        picamera2_control_profile = None

        if (
                parsed_arguments.camera_backend
                == OPENCV_CAMERA_BACKEND
        ):
            camera_control_profile = (
                CameraControlProfile(
                    auto_exposure=(
                        parsed_arguments.auto_exposure
                    ),
                    exposure=(
                        parsed_arguments.exposure
                    ),
                    gain=(
                        parsed_arguments.gain
                    ),
                    auto_white_balance=(
                        parsed_arguments
                        .auto_white_balance
                    ),
                    white_balance_temperature=(
                        parsed_arguments
                        .white_balance_temperature
                    ),
                    autofocus=(
                        parsed_arguments.autofocus
                    ),
                    focus=(
                        parsed_arguments.focus
                    ),
                )
            )

            camera_controls = (
                create_camera_control_requests(
                    camera_control_profile
                )
            )

        else:
            colour_gains = None

            if parsed_arguments.colour_gains is not None:
                colour_gains = (
                    parsed_arguments.colour_gains[0],
                    parsed_arguments.colour_gains[1],
                )

            picamera2_control_profile = (
                Picamera2ControlProfile(
                    ae_enable=(
                        parsed_arguments.ae_enable
                    ),
                    exposure_time_us=(
                        parsed_arguments.exposure_time_us
                    ),
                    analogue_gain=(
                        parsed_arguments.analogue_gain
                    ),
                    awb_enable=(
                        parsed_arguments.awb_enable
                    ),
                    colour_gains=colour_gains,
                )
            )

            validate_picamera2_control_profile(
                picamera2_control_profile
            )

        result = run_camera_preflight(
            camera_backend=(
                parsed_arguments.camera_backend
            ),
            camera_index=(
                parsed_arguments.camera_index
            ),
            width=parsed_arguments.width,
            height=parsed_arguments.height,
            fps=parsed_arguments.fps,
            sample_frames=(
                parsed_arguments.sample_frames
            ),
            camera_controls=(
                camera_controls
            ),
            picamera2_control_profile=(
                picamera2_control_profile
            ),
        )

    except Exception as error:
        print(
            f"Camera preflight failed: {error}",
            file=sys.stderr,
        )
        return 1

    print("Camera preflight passed.")
    print(
        "Camera backend: "
        f"{result.camera_backend}"
    )
    print(
        f"Camera index: {result.camera_index}"
    )
    print(
        "Effective resolution: "
        f"{result.effective_width}"
        f"x{result.effective_height}"
    )
    print(
        "Effective FPS: "
        f"{result.effective_fps:.2f}"
    )
    print(
        "Sampled frames: "
        f"{result.sampled_frame_count}"
    )
    if result.camera_controls:
        print("Camera controls:")

        for control in result.camera_controls:
            effective = (
                "unavailable"
                if control.effective_value is None
                else str(
                    control.effective_value
                )
            )

            print(
                f"  {control.name}: "
                f"requested="
                f"{control.requested_value}, "
                f"effective={effective}, "
                f"applied={control.applied}, "
                f"verified={control.verified}"
            )
    else:
        print(
            "Camera controls: none requested."
        )
    print(
        "No experiment artifacts were written."
    )

    return 0

def main() -> None:
    raise SystemExit(run_cli())

if __name__ == "__main__":
    main()
