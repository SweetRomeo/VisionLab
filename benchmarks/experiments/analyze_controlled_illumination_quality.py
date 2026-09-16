from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from benchmarks.experiments.controlled_illumination_quality_capture import (
    QUALITY_SAMPLES_MANIFEST_FILE_NAME,
    calculate_bytes_sha256,
)


DARK_CLIPPING_THRESHOLD = 0
BRIGHT_CLIPPING_THRESHOLD = 255
PIXEL_VALUE_RANGE = 255.0

OPTICAL_QUALITY_SAMPLES_FILE_NAME = (
    "optical_quality_samples.csv"
)

OPTICAL_QUALITY_TRIAL_SUMMARY_FILE_NAME = (
    "optical_quality_trial_summary.csv"
)

QUALITY_SAMPLE_CSV_FIELDS = (
    "experiment_id",
    "run_id",
    "algorithm",
    "resolution_width",
    "resolution_height",
    "measured_frame_index",
    "input_path",
    "processed_path",
    "input_sha256",
    "processed_sha256",
    "input_mean_intensity",
    "processed_mean_intensity",
    "input_intensity_std",
    "processed_intensity_std",
    "input_rms_contrast",
    "processed_rms_contrast",
    "input_dark_clipping_pct",
    "processed_dark_clipping_pct",
    "input_bright_clipping_pct",
    "processed_bright_clipping_pct",
    "mae",
    "max_absolute_error",
    "mse",
    "psnr",
)

QUALITY_TRIAL_SUMMARY_CSV_FIELDS = (
    "experiment_id",
    "run_id",
    "algorithm",
    "resolution_width",
    "resolution_height",
    "sample_count",
    "mean_input_intensity",
    "mean_processed_intensity",
    "mean_input_rms_contrast",
    "mean_processed_rms_contrast",
    "mean_input_dark_clipping_pct",
    "mean_processed_dark_clipping_pct",
    "mean_input_bright_clipping_pct",
    "mean_processed_bright_clipping_pct",
    "mean_mae",
    "mean_mse",
    "mean_psnr",
)




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
    input_sha256: str
    processed_sha256: str
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

@dataclass(frozen=True)
class QualitySampleAnalysis:
    experiment_id: str
    run_id: str
    algorithm: str
    resolution_width: int
    resolution_height: int
    measured_frame_index: int
    input_path: str
    processed_path: str
    input_sha256: str
    processed_sha256: str

    input_mean_intensity: float
    processed_mean_intensity: float

    input_intensity_std: float
    processed_intensity_std: float

    input_rms_contrast: float
    processed_rms_contrast: float

    input_dark_clipping_pct: float
    processed_dark_clipping_pct: float

    input_bright_clipping_pct: float
    processed_bright_clipping_pct: float

    mae: float
    max_absolute_error: float
    mse: float
    psnr: float

@dataclass(frozen=True)
class QualityTrialSummary:
    experiment_id: str
    run_id: str
    algorithm: str
    resolution_width: int
    resolution_height: int
    sample_count: int

    mean_input_intensity: float
    mean_processed_intensity: float

    mean_input_rms_contrast: float
    mean_processed_rms_contrast: float

    mean_input_dark_clipping_pct: float
    mean_processed_dark_clipping_pct: float

    mean_input_bright_clipping_pct: float
    mean_processed_bright_clipping_pct: float

    mean_mae: float
    mean_mse: float
    mean_psnr: float

def analyze_quality_run(
    run: ValidatedQualityRun,
) -> tuple[QualitySampleAnalysis, ...]:
    analyses: list[
        QualitySampleAnalysis
    ] = []

    for sample in run.samples:
        input_metrics = (
            calculate_image_quality_metrics(
                sample.input_image
            )
        )

        processed_metrics = (
            calculate_image_quality_metrics(
                sample.processed_image
            )
        )

        pair_metrics = (
            calculate_image_pair_quality_metrics(
                sample.input_image,
                sample.processed_image,
            )
        )

        analyses.append(
            QualitySampleAnalysis(
                experiment_id=run.experiment_id,
                run_id=run.run_id,
                algorithm=run.algorithm,
                resolution_width=run.width,
                resolution_height=run.height,
                measured_frame_index=(
                    sample.measured_frame_index
                ),
                input_path=str(
                    sample.input_path
                ),
                processed_path=str(
                    sample.processed_path
                ),
                input_sha256=(
                    sample.input_sha256
                ),
                processed_sha256=(
                    sample.processed_sha256
                ),
                input_mean_intensity=(
                    input_metrics.mean_intensity
                ),
                processed_mean_intensity=(
                    processed_metrics.mean_intensity
                ),
                input_intensity_std=(
                    input_metrics.intensity_std
                ),
                processed_intensity_std=(
                    processed_metrics.intensity_std
                ),
                input_rms_contrast=(
                    input_metrics.rms_contrast
                ),
                processed_rms_contrast=(
                    processed_metrics.rms_contrast
                ),
                input_dark_clipping_pct=(
                    input_metrics.dark_clipping_pct
                ),
                processed_dark_clipping_pct=(
                    processed_metrics.dark_clipping_pct
                ),
                input_bright_clipping_pct=(
                    input_metrics.bright_clipping_pct
                ),
                processed_bright_clipping_pct=(
                    processed_metrics.bright_clipping_pct
                ),
                mae=pair_metrics.mae,
                max_absolute_error=(
                    pair_metrics.max_absolute_error
                ),
                mse=pair_metrics.mse,
                psnr=pair_metrics.psnr,
            )
        )

    return tuple(analyses)

def summarize_quality_trial(
    analyses: tuple[
        QualitySampleAnalysis,
        ...,
    ],
) -> QualityTrialSummary:
    if not analyses:
        raise ControlledIlluminationQualityAnalysisError(
            "Quality trial analysis must not be empty."
        )

    first = analyses[0]

    for analysis in analyses[1:]:
        if (
            analysis.experiment_id
            != first.experiment_id
            or analysis.run_id
            != first.run_id
            or analysis.algorithm
            != first.algorithm
            or analysis.resolution_width
            != first.resolution_width
            or analysis.resolution_height
            != first.resolution_height
        ):
            raise ControlledIlluminationQualityAnalysisError(
                "All quality samples in a trial summary "
                "must belong to the same run."
            )

    return QualityTrialSummary(
        experiment_id=first.experiment_id,
        run_id=first.run_id,
        algorithm=first.algorithm,
        resolution_width=(
            first.resolution_width
        ),
        resolution_height=(
            first.resolution_height
        ),
        sample_count=len(analyses),
        mean_input_intensity=float(
            np.mean(
                [
                    analysis.input_mean_intensity
                    for analysis in analyses
                ]
            )
        ),
        mean_processed_intensity=float(
            np.mean(
                [
                    analysis.processed_mean_intensity
                    for analysis in analyses
                ]
            )
        ),
        mean_input_rms_contrast=float(
            np.mean(
                [
                    analysis.input_rms_contrast
                    for analysis in analyses
                ]
            )
        ),
        mean_processed_rms_contrast=float(
            np.mean(
                [
                    analysis.processed_rms_contrast
                    for analysis in analyses
                ]
            )
        ),
        mean_input_dark_clipping_pct=float(
            np.mean(
                [
                    analysis.input_dark_clipping_pct
                    for analysis in analyses
                ]
            )
        ),
        mean_processed_dark_clipping_pct=float(
            np.mean(
                [
                    analysis.processed_dark_clipping_pct
                    for analysis in analyses
                ]
            )
        ),
        mean_input_bright_clipping_pct=float(
            np.mean(
                [
                    analysis.input_bright_clipping_pct
                    for analysis in analyses
                ]
            )
        ),
        mean_processed_bright_clipping_pct=float(
            np.mean(
                [
                    analysis.processed_bright_clipping_pct
                    for analysis in analyses
                ]
            )
        ),
        mean_mae=float(
            np.mean(
                [
                    analysis.mae
                    for analysis in analyses
                ]
            )
        ),
        mean_mse=float(
            np.mean(
                [
                    analysis.mse
                    for analysis in analyses
                ]
            )
        ),
        mean_psnr=float(
            np.mean(
                [
                    analysis.psnr
                    for analysis in analyses
                ]
            )
        ),
    )

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
) -> tuple[Path, str, np.ndarray]:
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

    return (image_path, actual_sha256, image)


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

        input_path, input_sha256, input_image = (
            load_and_validate_sample_image(
                run_directory,
                input_metadata,
                f"samples[{sample_number}].input",
                expected_width=width,
                expected_height=height,
            )
        )

        processed_path, processed_sha256, processed_image = (
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
                measured_frame_index=measured_frame_index,
                input_path=input_path,
                processed_path=processed_path,
                input_sha256=input_sha256,
                processed_sha256=processed_sha256,
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

def discover_quality_capture_runs(
    results_directory: Path,
) -> tuple[Path, ...]:
    results_directory = Path(
        results_directory
    ).resolve()

    if not results_directory.exists():
        raise ControlledIlluminationQualityAnalysisError(
            "Controlled-illumination results "
            "directory was not found: "
            f"{results_directory}"
        )

    if not results_directory.is_dir():
        raise ControlledIlluminationQualityAnalysisError(
            "Controlled-illumination results path "
            "must be a directory: "
            f"{results_directory}"
        )

    manifest_paths = tuple(
        sorted(
            results_directory.rglob(
                QUALITY_SAMPLES_MANIFEST_FILE_NAME
            ),
            key=lambda path: (
                path.as_posix()
            ),
        )
    )

    run_directories = tuple(
        manifest_path.parent.resolve()
        for manifest_path in manifest_paths
    )

    return run_directories

def write_quality_sample_analysis_csv(
    output_path: Path,
    analyses: tuple[
        QualitySampleAnalysis,
        ...,
    ],
) -> Path:
    if not analyses:
        raise ControlledIlluminationQualityAnalysisError(
            "Quality-sample analysis must not be empty."
        )

    identities: set[
        tuple[str, str, int]
    ] = set()

    for analysis in analyses:
        identity = (
            analysis.experiment_id,
            analysis.run_id,
            analysis.measured_frame_index,
        )

        if identity in identities:
            raise ControlledIlluminationQualityAnalysisError(
                "Duplicate quality-sample analysis "
                f"identity: {identity}"
            )

        identities.add(identity)

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.parent
        / (
            f".{output_path.name}."
            f"{uuid4().hex}.tmp"
        )
    )

    try:
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=QUALITY_SAMPLE_CSV_FIELDS,
            )

            writer.writeheader()

            for analysis in analyses:
                writer.writerow(
                    {
                        "experiment_id": analysis.experiment_id,
                        "run_id": analysis.run_id,
                        "algorithm": analysis.algorithm,
                        "resolution_width": (
                            analysis.resolution_width
                        ),
                        "resolution_height": (
                            analysis.resolution_height
                        ),
                        "measured_frame_index": (
                            analysis.measured_frame_index
                        ),
                        "input_path": analysis.input_path,
                        "processed_path": (
                            analysis.processed_path
                        ),
                        "input_sha256": (
                            analysis.input_sha256
                        ),
                        "processed_sha256": (
                            analysis.processed_sha256
                        ),
                        "input_mean_intensity": (
                            analysis.input_mean_intensity
                        ),
                        "processed_mean_intensity": (
                            analysis.processed_mean_intensity
                        ),
                        "input_intensity_std": (
                            analysis.input_intensity_std
                        ),
                        "processed_intensity_std": (
                            analysis.processed_intensity_std
                        ),
                        "input_rms_contrast": (
                            analysis.input_rms_contrast
                        ),
                        "processed_rms_contrast": (
                            analysis.processed_rms_contrast
                        ),
                        "input_dark_clipping_pct": (
                            analysis.input_dark_clipping_pct
                        ),
                        "processed_dark_clipping_pct": (
                            analysis.processed_dark_clipping_pct
                        ),
                        "input_bright_clipping_pct": (
                            analysis.input_bright_clipping_pct
                        ),
                        "processed_bright_clipping_pct": (
                            analysis.processed_bright_clipping_pct
                        ),
                        "mae": analysis.mae,
                        "max_absolute_error": (
                            analysis.max_absolute_error
                        ),
                        "mse": analysis.mse,
                        "psnr": analysis.psnr,
                    }
                )

            output_file.flush()
            os.fsync(
                output_file.fileno()
            )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )

    return output_path

def analyze_quality_capture_results(
    results_directory: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    run_directories = (
        discover_quality_capture_runs(
            results_directory
        )
    )

    if not run_directories:
        raise ControlledIlluminationQualityAnalysisError(
            "No completed quality-capture runs "
            "were found."
        )

    all_analyses: list[
        QualitySampleAnalysis
    ] = []

    trial_summaries: list[
        QualityTrialSummary
    ] = []

    discovered_run_identities: set[
        tuple[str, str]
    ] = set()

    for run_directory in run_directories:
        run = (
            load_and_validate_quality_capture_run(
                run_directory
            )
        )

        run_identity = (
            run.experiment_id,
            run.run_id,
        )

        if (
            run_identity
            in discovered_run_identities
        ):
            raise ControlledIlluminationQualityAnalysisError(
                "Duplicate quality-capture run "
                f"identity: {run_identity}"
            )

        discovered_run_identities.add(
            run_identity
        )

        analyses = analyze_quality_run(
            run
        )

        if not analyses:
            raise ControlledIlluminationQualityAnalysisError(
                "Validated quality-capture run "
                "contains no analyzable samples: "
                f"{run_directory}"
            )

        summary = summarize_quality_trial(
            analyses
        )

        all_analyses.extend(
            analyses
        )

        trial_summaries.append(
            summary
        )

    output_directory = Path(
        output_directory
    )

    sample_output_path = (
        output_directory
        / OPTICAL_QUALITY_SAMPLES_FILE_NAME
    )

    summary_output_path = (
        output_directory
        / OPTICAL_QUALITY_TRIAL_SUMMARY_FILE_NAME
    )

    sample_written = False

    try:
        write_quality_sample_analysis_csv(
            sample_output_path,
            tuple(all_analyses),
        )

        sample_written = True

        write_quality_trial_summary_csv(
            summary_output_path,
            tuple(trial_summaries),
        )

    except Exception:
        if sample_written:
            sample_output_path.unlink(
                missing_ok=True
            )

        summary_output_path.unlink(
            missing_ok=True
        )

        raise

    return (
        sample_output_path,
        summary_output_path,
    )

def write_quality_trial_summary_csv(
    output_path: Path,
    summaries: tuple[
        QualityTrialSummary,
        ...,
    ],
) -> Path:
    if not summaries:
        raise ControlledIlluminationQualityAnalysisError(
            "Quality trial summaries must not be empty."
        )

    identities: set[
        tuple[str, str]
    ] = set()

    for summary in summaries:
        identity = (
            summary.experiment_id,
            summary.run_id,
        )

        if identity in identities:
            raise ControlledIlluminationQualityAnalysisError(
                "Duplicate quality trial summary "
                f"identity: {identity}"
            )

        identities.add(identity)

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.parent
        / (
            f".{output_path.name}."
            f"{uuid4().hex}.tmp"
        )
    )

    try:
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=(
                    QUALITY_TRIAL_SUMMARY_CSV_FIELDS
                ),
            )

            writer.writeheader()

            for summary in summaries:
                writer.writerow(
                    {
                        "experiment_id": (
                            summary.experiment_id
                        ),
                        "run_id": (
                            summary.run_id
                        ),
                        "algorithm": (
                            summary.algorithm
                        ),
                        "resolution_width": (
                            summary.resolution_width
                        ),
                        "resolution_height": (
                            summary.resolution_height
                        ),
                        "sample_count": (
                            summary.sample_count
                        ),
                        "mean_input_intensity": (
                            summary.mean_input_intensity
                        ),
                        "mean_processed_intensity": (
                            summary.mean_processed_intensity
                        ),
                        "mean_input_rms_contrast": (
                            summary.mean_input_rms_contrast
                        ),
                        "mean_processed_rms_contrast": (
                            summary.mean_processed_rms_contrast
                        ),
                        "mean_input_dark_clipping_pct": (
                            summary.mean_input_dark_clipping_pct
                        ),
                        "mean_processed_dark_clipping_pct": (
                            summary.mean_processed_dark_clipping_pct
                        ),
                        "mean_input_bright_clipping_pct": (
                            summary.mean_input_bright_clipping_pct
                        ),
                        "mean_processed_bright_clipping_pct": (
                            summary.mean_processed_bright_clipping_pct
                        ),
                        "mean_mae": (
                            summary.mean_mae
                        ),
                        "mean_mse": (
                            summary.mean_mse
                        ),
                        "mean_psnr": (
                            summary.mean_psnr
                        ),
                    }
                )

            output_file.flush()
            os.fsync(
                output_file.fileno()
            )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )

    return output_path