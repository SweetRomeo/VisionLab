from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from benchmarks.realtime.realtime_pipeline import (
    iter_camera_frames,
)


@dataclass(frozen=True)
class CameraPreflightResult:
    camera_index: int
    effective_width: int
    effective_height: int
    effective_fps: float
    sampled_frame_count: int


def run_camera_preflight(
    *,
    camera_index: int,
    width: int,
    height: int,
    fps: float,
    sample_frames: int,
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

    frame_source = iter_camera_frames(
        camera_index,
        width=width,
        height=height,
        fps=fps,
        capture_mode_reporter=(
            report_capture_mode
        ),
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
        camera_index=camera_index,
        effective_width=effective_width,
        effective_height=effective_height,
        effective_fps=effective_fps,
        sampled_frame_count=(
            sampled_frame_count
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

    return parser


def run_cli(
    arguments: list[str] | None = None,
) -> int:
    parser = create_argument_parser()
    parsed_arguments = parser.parse_args(
        arguments
    )

    try:
        result = run_camera_preflight(
            camera_index=(
                parsed_arguments.camera_index
            ),
            width=parsed_arguments.width,
            height=parsed_arguments.height,
            fps=parsed_arguments.fps,
            sample_frames=(
                parsed_arguments.sample_frames
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
    print(
        "No experiment artifacts were written."
    )

    return 0

def main() -> None:
    raise SystemExit(run_cli())

if __name__ == "__main__":
    main()
