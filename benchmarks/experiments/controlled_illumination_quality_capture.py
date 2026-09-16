from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import cv2
import numpy as np

SUPPORTED_IMAGE_FORMATS = {
    "png",
}

QUALITY_SAMPLES_DIRECTORY_NAME = "quality_samples"
QUALITY_SAMPLES_MANIFEST_FILE_NAME = (
    "quality_samples_manifest.json"
)


class QualityCaptureConfigError(ValueError):
    """Raised when quality-capture configuration is invalid."""

class QualityCaptureArtifactError(RuntimeError):
    """Raised when quality-capture artifacts cannot be published."""

@dataclass(frozen=True)
class QualityCaptureConfig:
    enabled: bool
    measured_frame_indices: tuple[int, ...]
    image_format: str

@dataclass(frozen=True)
class CapturedQualitySample:
    measured_frame_index: int
    input_frame: np.ndarray
    processed_frame: np.ndarray


def load_quality_capture_config(
    config: Mapping[str, Any],
    *,
    measured_frames: int,
) -> QualityCaptureConfig:
    if (
        isinstance(measured_frames, bool)
        or not isinstance(measured_frames, int)
        or measured_frames <= 0
    ):
        raise QualityCaptureConfigError(
            "measured_frames must be a positive integer."
        )

    raw_quality_capture = config.get(
        "quality_capture"
    )

    # Backward compatibility:
    # existing configurations without quality_capture
    # behave exactly as before.
    if raw_quality_capture is None:
        return QualityCaptureConfig(
            enabled=False,
            measured_frame_indices=(),
            image_format="png",
        )

    if not isinstance(
        raw_quality_capture,
        Mapping,
    ):
        raise QualityCaptureConfigError(
            "quality_capture must be an object."
        )

    enabled = raw_quality_capture.get(
        "enabled",
        False,
    )

    if not isinstance(enabled, bool):
        raise QualityCaptureConfigError(
            "quality_capture.enabled must be boolean."
        )

    image_format = raw_quality_capture.get(
        "image_format",
        "png",
    )

    if (
        not isinstance(image_format, str)
        or not image_format.strip()
    ):
        raise QualityCaptureConfigError(
            "quality_capture.image_format must "
            "be a non-empty string."
        )

    image_format = image_format.strip().lower()

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise QualityCaptureConfigError(
            "Unsupported quality-capture image format: "
            f"{image_format}"
        )

    raw_indices = raw_quality_capture.get(
        "measured_frame_indices",
        [],
    )

    if not isinstance(raw_indices, list):
        raise QualityCaptureConfigError(
            "quality_capture.measured_frame_indices "
            "must be a list."
        )

    validated_indices: list[int] = []

    for index, frame_index in enumerate(
        raw_indices
    ):
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
        ):
            raise QualityCaptureConfigError(
                "quality_capture."
                "measured_frame_indices"
                f"[{index}] must be an integer."
            )

        if (
            frame_index < 0
            or frame_index >= measured_frames
        ):
            raise QualityCaptureConfigError(
                "quality_capture."
                "measured_frame_indices"
                f"[{index}] must be between 0 and "
                f"{measured_frames - 1}."
            )

        validated_indices.append(
            frame_index
        )

    if len(validated_indices) != len(
        set(validated_indices)
    ):
        raise QualityCaptureConfigError(
            "quality_capture.measured_frame_indices "
            "contains duplicate values."
        )

    if enabled and not validated_indices:
        raise QualityCaptureConfigError(
            "quality_capture.measured_frame_indices "
            "must not be empty when quality capture "
            "is enabled."
        )

    return QualityCaptureConfig(
        enabled=enabled,
        measured_frame_indices=tuple(
            validated_indices
        ),
        image_format=image_format,
    )

class QualityCaptureBuffer:
    def __init__(
        self,
        config: QualityCaptureConfig,
    ) -> None:
        self._config = config
        self._selected_indices = set(
            config.measured_frame_indices
        )
        self._samples: dict[
            int,
            CapturedQualitySample,
        ] = {}

    @property
    def samples(
        self,
    ) -> tuple[CapturedQualitySample, ...]:
        return tuple(
            self._samples[index]
            for index in sorted(
                self._samples
            )
        )

    @property
    def missing_indices(
        self,
    ) -> tuple[int, ...]:
        if not self._config.enabled:
            return ()

        return tuple(
            sorted(
                self._selected_indices
                - set(self._samples)
            )
        )

    def capture(
        self,
        measured_frame_index: int,
        input_frame: np.ndarray,
        processed_frame: np.ndarray,
    ) -> None:
        if not self._config.enabled:
            return

        if (
            measured_frame_index
            not in self._selected_indices
        ):
            return

        if measured_frame_index in self._samples:
            raise RuntimeError(
                "Quality sample already captured for "
                "measured frame index "
                f"{measured_frame_index}."
            )

        self._samples[
            measured_frame_index
        ] = CapturedQualitySample(
            measured_frame_index=(
                measured_frame_index
            ),
            input_frame=input_frame.copy(),
            processed_frame=(
                processed_frame.copy()
            ),
        )

    def write_bytes_durable(
            output_path: Path,
            value: bytes,
    ) -> None:
        with output_path.open("wb") as output_file:
            output_file.write(value)
            output_file.flush()
            os.fsync(output_file.fileno())

    def write_json_durable(
            output_path: Path,
            value: dict[str, Any],
    ) -> None:
        with output_path.open(
                "w",
                encoding="utf-8",
        ) as output_file:
            json.dump(
                value,
                output_file,
                indent=2,
                ensure_ascii=False,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())

    def write_quality_capture_artifacts_atomic(
            *,
            output_directory: Path,
            experiment_id: str,
            run_id: str,
            algorithm: str,
            width: int,
            height: int,
            config: QualityCaptureConfig,
            samples: Sequence[CapturedQualitySample],
    ) -> tuple[Path, Path]:
        if not config.enabled:
            raise QualityCaptureArtifactError(
                "Quality capture must be enabled before "
                "artifacts can be written."
            )

        normalized_samples = tuple(samples)

        expected_indices = tuple(
            sorted(config.measured_frame_indices)
        )
        actual_indices = tuple(
            sample.measured_frame_index
            for sample in normalized_samples
        )

        if actual_indices != expected_indices:
            raise QualityCaptureArtifactError(
                "Captured quality samples do not match "
                "the configured measured frame indices."
            )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        samples_directory = (
                output_directory
                / QUALITY_SAMPLES_DIRECTORY_NAME
        )
        manifest_path = (
                output_directory
                / QUALITY_SAMPLES_MANIFEST_FILE_NAME
        )

        if (
                samples_directory.exists()
                or manifest_path.exists()
        ):
            raise QualityCaptureArtifactError(
                "Quality-capture artifacts already exist."
            )

        staging_directory = (
                output_directory
                / (
                    ".quality_capture."
                    f"{uuid4().hex}.tmp"
                )
        )
        staging_samples_directory = (
                staging_directory
                / QUALITY_SAMPLES_DIRECTORY_NAME
        )
        staging_manifest_path = (
                staging_directory
                / QUALITY_SAMPLES_MANIFEST_FILE_NAME
        )

        published_samples = False
        published_manifest = False

        try:
            staging_samples_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            manifest_samples: list[
                dict[str, Any]
            ] = []

            for sample in normalized_samples:
                frame_index = (
                    sample.measured_frame_index
                )

                input_file_name = (
                    f"frame_{frame_index:06d}_input.png"
                )
                processed_file_name = (
                    f"frame_{frame_index:06d}_processed.png"
                )

                input_bytes = encode_png_lossless(
                    sample.input_frame
                )
                processed_bytes = encode_png_lossless(
                    sample.processed_frame
                )

                input_path = (
                        staging_samples_directory
                        / input_file_name
                )
                processed_path = (
                        staging_samples_directory
                        / processed_file_name
                )

                write_bytes_durable(
                    input_path,
                    input_bytes,
                )
                write_bytes_durable(
                    processed_path,
                    processed_bytes,
                )

                manifest_samples.append(
                    {
                        "measured_frame_index": (
                            frame_index
                        ),
                        "input": {
                            "path": (
                                f"{QUALITY_SAMPLES_DIRECTORY_NAME}/"
                                f"{input_file_name}"
                            ),
                            "width": int(
                                sample.input_frame.shape[1]
                            ),
                            "height": int(
                                sample.input_frame.shape[0]
                            ),
                            "dtype": str(
                                sample.input_frame.dtype
                            ),
                            "sha256": (
                                calculate_bytes_sha256(
                                    input_bytes
                                )
                            ),
                        },
                        "processed": {
                            "path": (
                                f"{QUALITY_SAMPLES_DIRECTORY_NAME}/"
                                f"{processed_file_name}"
                            ),
                            "width": int(
                                sample.processed_frame.shape[1]
                            ),
                            "height": int(
                                sample.processed_frame.shape[0]
                            ),
                            "dtype": str(
                                sample.processed_frame.dtype
                            ),
                            "sha256": (
                                calculate_bytes_sha256(
                                    processed_bytes
                                )
                            ),
                        },
                    }
                )

            manifest = {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "algorithm": algorithm,
                "resolution": {
                    "width": width,
                    "height": height,
                },
                "capture_configuration": {
                    "enabled": config.enabled,
                    "measured_frame_indices": list(
                        config.measured_frame_indices
                    ),
                    "image_format": (
                        config.image_format
                    ),
                },
                "samples": manifest_samples,
            }

            write_json_durable(
                staging_manifest_path,
                manifest,
            )

            os.replace(
                staging_samples_directory,
                samples_directory,
            )
            published_samples = True

            # Manifest is published last and acts as the
            # quality-capture completion marker.
            os.replace(
                staging_manifest_path,
                manifest_path,
            )
            published_manifest = True

        except Exception:
            if published_manifest:
                manifest_path.unlink(
                    missing_ok=True
                )

            if published_samples:
                shutil.rmtree(
                    samples_directory,
                    ignore_errors=True,
                )

            raise

        finally:
            shutil.rmtree(
                staging_directory,
                ignore_errors=True,
            )

        return samples_directory, manifest_path

def encode_png_lossless(
        frame: np.ndarray,
) -> bytes:
    if not isinstance(frame, np.ndarray):
        raise QualityCaptureArtifactError(
            "Quality-capture frame must be a NumPy array."
        )

    if frame.size == 0:
        raise QualityCaptureArtifactError(
            "Quality-capture frame must not be empty."
        )

    encoded, buffer = cv2.imencode(
        ".png",
        frame,
    )

    if not encoded:
        raise QualityCaptureArtifactError(
            "Quality-capture PNG encoding failed."
        )

    return buffer.tobytes()

def calculate_bytes_sha256(
        value: bytes,
) -> str:
    return hashlib.sha256(value).hexdigest()

def write_bytes_durable(
    output_path: Path,
    value: bytes,
) -> None:
    with output_path.open("wb") as output_file:
        output_file.write(value)
        output_file.flush()
        os.fsync(output_file.fileno())


def write_json_durable(
    output_path: Path,
    value: dict[str, Any],
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            value,
            output_file,
            indent=2,
            ensure_ascii=False,
        )
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def write_quality_capture_artifacts_atomic(
    *,
    output_directory: Path,
    experiment_id: str,
    run_id: str,
    algorithm: str,
    width: int,
    height: int,
    config: QualityCaptureConfig,
    samples: Sequence[CapturedQualitySample],
) -> tuple[Path, Path]:
    if not config.enabled:
        raise QualityCaptureArtifactError(
            "Quality capture must be enabled before "
            "artifacts can be written."
        )

    normalized_samples = tuple(samples)

    expected_indices = tuple(
        sorted(config.measured_frame_indices)
    )
    actual_indices = tuple(
        sample.measured_frame_index
        for sample in normalized_samples
    )

    if actual_indices != expected_indices:
        raise QualityCaptureArtifactError(
            "Captured quality samples do not match "
            "the configured measured frame indices."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples_directory = (
        output_directory
        / QUALITY_SAMPLES_DIRECTORY_NAME
    )
    manifest_path = (
        output_directory
        / QUALITY_SAMPLES_MANIFEST_FILE_NAME
    )

    if (
        samples_directory.exists()
        or manifest_path.exists()
    ):
        raise QualityCaptureArtifactError(
            "Quality-capture artifacts already exist."
        )

    staging_directory = (
        output_directory
        / (
            ".quality_capture."
            f"{uuid4().hex}.tmp"
        )
    )
    staging_samples_directory = (
        staging_directory
        / QUALITY_SAMPLES_DIRECTORY_NAME
    )
    staging_manifest_path = (
        staging_directory
        / QUALITY_SAMPLES_MANIFEST_FILE_NAME
    )

    published_samples = False
    published_manifest = False

    try:
        staging_samples_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        manifest_samples: list[
            dict[str, Any]
        ] = []

        for sample in normalized_samples:
            frame_index = (
                sample.measured_frame_index
            )

            input_file_name = (
                f"frame_{frame_index:06d}_input.png"
            )
            processed_file_name = (
                f"frame_{frame_index:06d}_processed.png"
            )

            input_bytes = encode_png_lossless(
                sample.input_frame
            )
            processed_bytes = encode_png_lossless(
                sample.processed_frame
            )

            input_path = (
                staging_samples_directory
                / input_file_name
            )
            processed_path = (
                staging_samples_directory
                / processed_file_name
            )

            write_bytes_durable(
                input_path,
                input_bytes,
            )
            write_bytes_durable(
                processed_path,
                processed_bytes,
            )

            manifest_samples.append(
                {
                    "measured_frame_index": (
                        frame_index
                    ),
                    "input": {
                        "path": (
                            f"{QUALITY_SAMPLES_DIRECTORY_NAME}/"
                            f"{input_file_name}"
                        ),
                        "width": int(
                            sample.input_frame.shape[1]
                        ),
                        "height": int(
                            sample.input_frame.shape[0]
                        ),
                        "dtype": str(
                            sample.input_frame.dtype
                        ),
                        "sha256": (
                            calculate_bytes_sha256(
                                input_bytes
                            )
                        ),
                    },
                    "processed": {
                        "path": (
                            f"{QUALITY_SAMPLES_DIRECTORY_NAME}/"
                            f"{processed_file_name}"
                        ),
                        "width": int(
                            sample.processed_frame.shape[1]
                        ),
                        "height": int(
                            sample.processed_frame.shape[0]
                        ),
                        "dtype": str(
                            sample.processed_frame.dtype
                        ),
                        "sha256": (
                            calculate_bytes_sha256(
                                processed_bytes
                            )
                        ),
                    },
                }
            )

        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "algorithm": algorithm,
            "resolution": {
                "width": width,
                "height": height,
            },
            "capture_configuration": {
                "enabled": config.enabled,
                "measured_frame_indices": list(
                    config.measured_frame_indices
                ),
                "image_format": config.image_format,
            },
            "samples": manifest_samples,
        }

        write_json_durable(
            staging_manifest_path,
            manifest,
        )

        os.replace(
            staging_samples_directory,
            samples_directory,
        )
        published_samples = True

        os.replace(
            staging_manifest_path,
            manifest_path,
        )
        published_manifest = True

    except Exception:
        if published_manifest:
            manifest_path.unlink(
                missing_ok=True
            )

        if published_samples:
            shutil.rmtree(
                samples_directory,
                ignore_errors=True,
            )

        raise

    finally:
        shutil.rmtree(
            staging_directory,
            ignore_errors=True,
        )

    return samples_directory, manifest_path