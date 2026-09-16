from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SUPPORTED_IMAGE_FORMATS = {
    "png",
}


class QualityCaptureConfigError(ValueError):
    """Raised when quality-capture configuration is invalid."""


@dataclass(frozen=True)
class QualityCaptureConfig:
    enabled: bool
    measured_frame_indices: tuple[int, ...]
    image_format: str


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