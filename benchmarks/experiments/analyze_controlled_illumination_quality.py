from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from benchmarks.experiments.controlled_illumination_quality_capture import (
    QUALITY_SAMPLES_MANIFEST_FILE_NAME,
    calculate_bytes_sha256,
)


DARK_CLIPPING_THRESHOLD = 0
BRIGHT_CLIPPING_THRESHOLD = 255
PIXEL_VALUE_RANGE = 255.0


class ControlledIlluminationQualityAnalysisError(
    RuntimeError
):
    """Raised when optical-quality analysis fails."""


@dataclass(frozen=True)
class ImageQualityMetrics:
    mean_intensity: float
    intensity_std: float
    rms_contrast: float
    dark_clipping_pct: float
    bright_clipping_pct: float


@dataclass(frozen=True)
class ImagePairQualityMetrics:
    mae: float
    max_absolute_error: float
    mse: float
    psnr: float


@dataclass(frozen=True)
class ValidatedQualitySample:
    measured_frame_index: int
    input_path: Path
    processed_path: Path
    input_image: np.ndarray
    processed_image: np.ndarray


@dataclass(frozen=True)
class ValidatedQualityRun:
    experiment_id: str
    run_id: str
    algorithm: str
    width: int
    height: int
    samples: tuple[
        ValidatedQualitySample,
        ...,
    ]


def convert_to_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise ControlledIlluminationQualityAnalysisError(
            "Image must be a NumPy array."
        )

    if image.size == 0:
        raise ControlledIlluminationQualityAnalysisError(
            "Image must not be empty."
        )

    if image.dtype != np.uint8:
        raise ControlledIlluminationQualityAnalysisError(
            "Only uint8 images are currently supported."
        )

    if image.ndim == 2:
        return image

    if (
        image.ndim == 3
        and image.shape[2] == 3
    ):
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    raise ControlledIlluminationQualityAnalysisError(
        "Image must be grayscale or three-channel BGR."
    )


def calculate_image_quality_metrics(
    image: np.ndarray,
) -> ImageQualityMetrics:
    grayscale = convert_to_grayscale(
        image
    )

    values = grayscale.astype(
        np.float64
    )

    mean_intensity = float(
        np.mean(values)
    )

    intensity_std = float(
        np.std(values)
    )

    rms_contrast = (
        intensity_std
        / PIXEL_VALUE_RANGE
    )

    pixel_count = grayscale.size

    dark_clipping_pct = (
        float(
            np.count_nonzero(
                grayscale
                <= DARK_CLIPPING_THRESHOLD
            )
        )
        / pixel_count
        * 100.0
    )

    bright_clipping_pct = (
        float(
            np.count_nonzero(
                grayscale
                >= BRIGHT_CLIPPING_THRESHOLD
            )
        )
        / pixel_count
        * 100.0
    )

    return ImageQualityMetrics(
        mean_intensity=mean_intensity,
        intensity_std=intensity_std,
        rms_contrast=rms_contrast,
        dark_clipping_pct=dark_clipping_pct,
        bright_clipping_pct=bright_clipping_pct,
    )


def calculate_image_pair_quality_metrics(
    input_image: np.ndarray,
    processed_image: np.ndarray,
) -> ImagePairQualityMetrics:
    if (
        not isinstance(input_image, np.ndarray)
        or not isinstance(
            processed_image,
            np.ndarray,
        )
    ):
        raise ControlledIlluminationQualityAnalysisError(
            "Input and processed images must "
            "be NumPy arrays."
        )

    if input_image.shape != processed_image.shape:
        raise ControlledIlluminationQualityAnalysisError(
            "Input and processed images must "
            "have identical shapes."
        )

    if (
        input_image.dtype != np.uint8
        or processed_image.dtype != np.uint8
    ):
        raise ControlledIlluminationQualityAnalysisError(
            "Input and processed images must "
            "use uint8 dtype."
        )

    input_values = input_image.astype(
        np.float64
    )
    processed_values = (
        processed_image.astype(
            np.float64
        )
    )

    difference = (
        processed_values
        - input_values
    )

    absolute_difference = np.abs(
        difference
    )

    mae = float(
        np.mean(
            absolute_difference
        )
    )

    max_absolute_error = float(
        np.max(
            absolute_difference
        )
    )

    mse = float(
        np.mean(
            difference ** 2
        )
    )

    if mse == 0.0:
        psnr = math.inf
    else:
        psnr = (
            20.0
            * math.log10(
                PIXEL_VALUE_RANGE
                / math.sqrt(mse)
            )
        )

    return ImagePairQualityMetrics(
        mae=mae,
        max_absolute_error=max_absolute_error,
        mse=mse,
        psnr=psnr,
    )

def require_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlledIlluminationQualityAnalysisError(
            f"{field_name} must be an object."
        )

    return value


def require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ControlledIlluminationQualityAnalysisError(
            f"{field_name} must be a non-empty string."
        )

    return value.strip()


def require_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ControlledIlluminationQualityAnalysisError(
            f"{field_name} must be a positive integer."
        )

    return value


def require_non_negative_integer(
    value: Any,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ControlledIlluminationQualityAnalysisError(
            f"{field_name} must be a non-negative integer."
        )

    return value


def validate_sha256(
    value: Any,
    field_name: str,
) -> str:
    sha256 = require_non_empty_string(
        value,
        field_name,
    ).lower()

    if (
        len(sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in sha256
        )
    ):
        raise ControlledIlluminationQualityAnalysisError(
            f"{field_name} must contain a valid SHA-256 hash."
        )

    return sha256


def resolve_quality_artifact_path(
    run_directory: Path,
    relative_path: str,
    field_name: str,
) -> Path:
    candidate_path = Path(
        relative_path
    )

    if candidate_path.is_absolute():
        raise ControlledIlluminationQualityAnalysisError(
            f"{field_name} must be relative to the run directory."
        )

    resolved_run_directory = (
        run_directory.resolve()
    )

    resolved_path = (
        resolved_run_directory
        / candidate_path
    ).resolve()

    try:
        resolved_path.relative_to(
            resolved_run_directory
        )
    except ValueError as error:
        raise ControlledIlluminationQualityAnalysisError(
            f"{field_name} escapes the run directory."
        ) from error

    if not resolved_path.is_file():
        raise ControlledIlluminationQualityAnalysisError(
            f"Quality sample image was not found: "
            f"{resolved_path}"
        )

    return resolved_path


def load_and_validate_sample_image(
    run_directory: Path,
    metadata: dict[str, Any],
    field_name: str,
    *,
    expected_width: int,
    expected_height: int,
) -> tuple[Path, np.ndarray]:
    relative_path = require_non_empty_string(
        metadata.get("path"),
        f"{field_name}.path",
    )

    expected_image_width = (
        require_positive_integer(
            metadata.get("width"),
            f"{field_name}.width",
        )
    )

    expected_image_height = (
        require_positive_integer(
            metadata.get("height"),
            f"{field_name}.height",
        )
    )

    expected_dtype = require_non_empty_string(
        metadata.get("dtype"),
        f"{field_name}.dtype",
    )

    expected_sha256 = validate_sha256(
        metadata.get("sha256"),
        f"{field_name}.sha256",
    )

    if (
        expected_image_width != expected_width
        or expected_image_height != expected_height
    ):
        raise ControlledIlluminationQualityAnalysisError(
            f"{field_name} dimensions do not match "
            "the run resolution."
        )

    image_path = resolve_quality_artifact_path(
        run_directory,
        relative_path,
        f"{field_name}.path",
    )

    try:
        image_bytes = image_path.read_bytes()
    except OSError as error:
        raise ControlledIlluminationQualityAnalysisError(
            f"Quality sample image could not be read: "
            f"{image_path}"
        ) from error

    actual_sha256 = calculate_bytes_sha256(
        image_bytes
    )

    if actual_sha256 != expected_sha256:
        raise ControlledIlluminationQualityAnalysisError(
            f"SHA-256 mismatch for quality sample: "
            f"{image_path}"
        )

    image_buffer = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_buffer,
        cv2.IMREAD_UNCHANGED,
    )

    if image is None or image.size == 0:
        raise ControlledIlluminationQualityAnalysisError(
            f"Quality sample image could not be decoded: "
            f"{image_path}"
        )

    actual_height = int(
        image.shape[0]
    )
    actual_width = int(
        image.shape[1]
    )

    if (
        actual_width != expected_image_width
        or actual_height != expected_image_height
    ):
        raise ControlledIlluminationQualityAnalysisError(
            f"Image dimensions do not match manifest "
            f"metadata for {image_path}."
        )

    if str(image.dtype) != expected_dtype:
        raise ControlledIlluminationQualityAnalysisError(
            f"Image dtype does not match manifest "
            f"metadata for {image_path}."
        )

    return image_path, image


def load_and_validate_quality_capture_run(
    run_directory: Path,
) -> ValidatedQualityRun:
    run_directory = Path(
        run_directory
    ).resolve()

    manifest_path = (
        run_directory
        / QUALITY_SAMPLES_MANIFEST_FILE_NAME
    )

    if not manifest_path.is_file():
        raise ControlledIlluminationQualityAnalysisError(
            "Quality-capture manifest was not found: "
            f"{manifest_path}"
        )

    try:
        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as manifest_file:
            manifest_value = json.load(
                manifest_file
            )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise ControlledIlluminationQualityAnalysisError(
            "Quality-capture manifest could not be loaded: "
            f"{manifest_path}"
        ) from error

    manifest = require_mapping(
        manifest_value,
        "quality-capture manifest",
    )

    if manifest.get("schema_version") != 1:
        raise ControlledIlluminationQualityAnalysisError(
            "quality-capture manifest schema_version "
            "must be 1."
        )

    experiment_id = require_non_empty_string(
        manifest.get("experiment_id"),
        "experiment_id",
    )

    run_id = require_non_empty_string(
        manifest.get("run_id"),
        "run_id",
    )

    algorithm = require_non_empty_string(
        manifest.get("algorithm"),
        "algorithm",
    )

    resolution = require_mapping(
        manifest.get("resolution"),
        "resolution",
    )

    width = require_positive_integer(
        resolution.get("width"),
        "resolution.width",
    )

    height = require_positive_integer(
        resolution.get("height"),
        "resolution.height",
    )

    capture_configuration = require_mapping(
        manifest.get("capture_configuration"),
        "capture_configuration",
    )

    if (
        capture_configuration.get("enabled")
        is not True
    ):
        raise ControlledIlluminationQualityAnalysisError(
            "capture_configuration.enabled "
            "must be true."
        )

    image_format = require_non_empty_string(
        capture_configuration.get(
            "image_format"
        ),
        "capture_configuration.image_format",
    ).lower()

    if image_format != "png":
        raise ControlledIlluminationQualityAnalysisError(
            "Only PNG quality samples are supported."
        )

    configured_indices_value = (
        capture_configuration.get(
            "measured_frame_indices"
        )
    )

    if not isinstance(
        configured_indices_value,
        list,
    ):
        raise ControlledIlluminationQualityAnalysisError(
            "capture_configuration."
            "measured_frame_indices must be a list."
        )

    configured_indices = tuple(
        require_non_negative_integer(
            frame_index,
            (
                "capture_configuration."
                "measured_frame_indices"
            ),
        )
        for frame_index
        in configured_indices_value
    )

    if not configured_indices:
        raise ControlledIlluminationQualityAnalysisError(
            "capture_configuration."
            "measured_frame_indices must not be empty."
        )

    if len(configured_indices) != len(
        set(configured_indices)
    ):
        raise ControlledIlluminationQualityAnalysisError(
            "capture_configuration."
            "measured_frame_indices contains duplicates."
        )

    samples_value = manifest.get(
        "samples"
    )

    if (
        not isinstance(samples_value, list)
        or not samples_value
    ):
        raise ControlledIlluminationQualityAnalysisError(
            "samples must be a non-empty list."
        )

    samples: list[
        ValidatedQualitySample
    ] = []

    discovered_indices: set[int] = set()

    for sample_number, sample_value in enumerate(
        samples_value
    ):
        sample = require_mapping(
            sample_value,
            f"samples[{sample_number}]",
        )

        measured_frame_index = (
            require_non_negative_integer(
                sample.get(
                    "measured_frame_index"
                ),
                (
                    f"samples[{sample_number}]."
                    "measured_frame_index"
                ),
            )
        )

        if measured_frame_index in discovered_indices:
            raise ControlledIlluminationQualityAnalysisError(
                "Duplicate quality sample for measured "
                f"frame index {measured_frame_index}."
            )

        discovered_indices.add(
            measured_frame_index
        )

        input_metadata = require_mapping(
            sample.get("input"),
            f"samples[{sample_number}].input",
        )

        processed_metadata = require_mapping(
            sample.get("processed"),
            f"samples[{sample_number}].processed",
        )

        input_path, input_image = (
            load_and_validate_sample_image(
                run_directory,
                input_metadata,
                f"samples[{sample_number}].input",
                expected_width=width,
                expected_height=height,
            )
        )

        processed_path, processed_image = (
            load_and_validate_sample_image(
                run_directory,
                processed_metadata,
                f"samples[{sample_number}].processed",
                expected_width=width,
                expected_height=height,
            )
        )

        if (
            input_image.shape
            != processed_image.shape
        ):
            raise ControlledIlluminationQualityAnalysisError(
                "Input and processed quality samples "
                "must have identical shapes."
            )

        samples.append(
            ValidatedQualitySample(
                measured_frame_index=(
                    measured_frame_index
                ),
                input_path=input_path,
                processed_path=processed_path,
                input_image=input_image,
                processed_image=processed_image,
            )
        )

    actual_indices = tuple(
        sorted(
            discovered_indices
        )
    )

    expected_indices = tuple(
        sorted(
            configured_indices
        )
    )

    if actual_indices != expected_indices:
        raise ControlledIlluminationQualityAnalysisError(
            "Quality samples do not match configured "
            "measured frame indices."
        )

    return ValidatedQualityRun(
        experiment_id=experiment_id,
        run_id=run_id,
        algorithm=algorithm,
        width=width,
        height=height,
        samples=tuple(
            sorted(
                samples,
                key=lambda sample: (
                    sample.measured_frame_index
                ),
            )
        ),
    )