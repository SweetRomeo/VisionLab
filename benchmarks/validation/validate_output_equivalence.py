import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

VALIDATION_CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "output_validation_config.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration must contain a JSON object: {path}"
        )

    return data


def validate_frame_indices(
    validation_config: dict[str, Any],
) -> list[int]:
    frame_indices = validation_config.get(
        "frame_indices"
    )

    if (
        not isinstance(frame_indices, list)
        or not frame_indices
    ):
        raise ValueError(
            "frame_indices must be a non-empty list."
        )

    if any(
        not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
        or frame_index <= 0
        for frame_index in frame_indices
    ):
        raise ValueError(
            "Frame indices must be positive integers."
        )

    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError(
            "frame_indices must not contain duplicates."
        )

    return sorted(frame_indices)


def read_selected_frames(
    video_path: Path,
    frame_indices: list[int],
) -> dict[int, np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Input video was not found: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Input video could not be opened: {video_path}"
        )

    requested_indices = set(frame_indices)
    selected_frames: dict[int, np.ndarray] = {}

    current_frame_index = 0
    final_frame_index = max(frame_indices)

    try:
        while current_frame_index < final_frame_index:
            frame_received, frame = capture.read()

            if not frame_received:
                break

            # Validation configuration uses 1-based indices.
            current_frame_index += 1

            if current_frame_index in requested_indices:
                selected_frames[current_frame_index] = (
                    frame.copy()
                )
    finally:
        capture.release()

    missing_indices = (
        requested_indices - selected_frames.keys()
    )

    if missing_indices:
        raise RuntimeError(
            "The following frames could not be read: "
            f"{sorted(missing_indices)}"
        )

    return selected_frames


def save_validation_inputs(
    frames: dict[int, np.ndarray],
    resolutions: list[dict[str, Any]],
    output_directory: Path,
) -> int:
    input_directory = output_directory / "inputs"
    saved_image_count = 0

    for resolution in resolutions:
        width = resolution.get("width")
        height = resolution.get("height")

        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ValueError(
                f"Invalid resolution: {resolution}"
            )

        resolution_name = f"{width}x{height}"
        resolution_directory = (
            input_directory / resolution_name
        )
        resolution_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for frame_index, frame in frames.items():
            resized_frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            output_path = (
                resolution_directory
                / f"frame_{frame_index:06d}.png"
            )

            image_written = cv2.imwrite(
                str(output_path),
                resized_frame,
            )

            if not image_written:
                raise RuntimeError(
                    f"Image could not be written: "
                    f"{output_path}"
                )

            saved_image_count += 1

    return saved_image_count


def main() -> None:
    benchmark_config = load_json(
        BENCHMARK_CONFIG_PATH
    )
    validation_config = load_json(
        VALIDATION_CONFIG_PATH
    )

    frame_indices = validate_frame_indices(
        validation_config
    )

    video_relative_path = benchmark_config["input"][
        "video_path"
    ]
    video_path = PROJECT_ROOT / video_relative_path

    resolutions = benchmark_config["resolutions"]

    output_relative_path = validation_config[
        "output_directory"
    ]
    output_directory = (
        PROJECT_ROOT / output_relative_path
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_frames = read_selected_frames(
        video_path,
        frame_indices,
    )

    saved_image_count = save_validation_inputs(
        selected_frames,
        resolutions,
        output_directory,
    )

    print(
        f"Selected deterministic frames: "
        f"{len(selected_frames)}"
    )
    print(
        f"Prepared validation input images: "
        f"{saved_image_count}"
    )
    print(
        f"Output directory: {output_directory}"
    )


if __name__ == "__main__":
    main()