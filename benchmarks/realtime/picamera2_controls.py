from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any



@dataclass(frozen=True)
class Picamera2ControlProfile:
    ae_enable: bool | None = None
    exposure_time_us: int | None = None
    analogue_gain: float | None = None
    awb_enable: bool | None = None
    colour_gains: tuple[float, float] | None = None


def _validate_optional_boolean(
        name: str,
        value: bool | None,
) -> None:
    if (
        value is not None
        and not isinstance(value, bool)
    ):
        raise ValueError(
            f"{name} must be boolean or None."
        )


def _validate_exposure_time(
        value: int | None,
) -> None:
    if value is None:
        return

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            "exposure_time_us must be a "
            "positive integer."
        )


def _validate_analogue_gain(
        value: float | None,
) -> None:
    if value is None:
        return

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(
            "analogue_gain must be a "
            "positive finite number."
        )


def _validate_colour_gains(
        value: tuple[float, float] | None,
) -> None:
    if value is None:
        return

    if (
        not isinstance(value, tuple)
        or len(value) != 2
    ):
        raise ValueError(
            "colour_gains must contain exactly "
            "two values."
        )

    for gain in value:
        if (
            isinstance(gain, bool)
            or not isinstance(
                gain,
                (int, float),
            )
            or not math.isfinite(float(gain))
            or gain <= 0
        ):
            raise ValueError(
                "colour_gains values must be "
                "positive finite numbers."
            )


def validate_picamera2_control_profile(
        profile: Picamera2ControlProfile,
) -> None:
    if not isinstance(
            profile,
            Picamera2ControlProfile,
    ):
        raise TypeError(
            "profile must be a "
            "Picamera2ControlProfile."
        )

    _validate_optional_boolean(
        "ae_enable",
        profile.ae_enable,
    )
    _validate_optional_boolean(
        "awb_enable",
        profile.awb_enable,
    )

    _validate_exposure_time(
        profile.exposure_time_us
    )
    _validate_analogue_gain(
        profile.analogue_gain
    )
    _validate_colour_gains(
        profile.colour_gains
    )

    if (
        profile.exposure_time_us is not None
        or profile.analogue_gain is not None
    ):
        if profile.ae_enable is None:
            raise ValueError(
                "Manual exposure or analogue gain "
                "requires explicit ae_enable."
            )

        if profile.ae_enable:
            raise ValueError(
                "Manual exposure or analogue gain "
                "requires ae_enable=False."
            )

    if profile.colour_gains is not None:
        if profile.awb_enable is None:
            raise ValueError(
                "Manual colour gains require "
                "explicit awb_enable."
            )

        if profile.awb_enable:
            raise ValueError(
                "Manual colour gains require "
                "awb_enable=False."
            )


def create_picamera2_controls(
        profile: Picamera2ControlProfile,
) -> dict[str, Any]:
    validate_picamera2_control_profile(
        profile
    )

    controls: dict[str, Any] = {}

    if profile.ae_enable is not None:
        controls["AeEnable"] = (
            profile.ae_enable
        )

    if profile.exposure_time_us is not None:
        controls["ExposureTime"] = (
            profile.exposure_time_us
        )

    if profile.analogue_gain is not None:
        controls["AnalogueGain"] = float(
            profile.analogue_gain
        )

    if profile.awb_enable is not None:
        controls["AwbEnable"] = (
            profile.awb_enable
        )

    if profile.colour_gains is not None:
        controls["ColourGains"] = (
            float(profile.colour_gains[0]),
            float(profile.colour_gains[1]),
        )

    return controls

Picamera2ControlValue = (
    bool
    | int
    | float
    | tuple[float, float]
)


class Picamera2ControlError(RuntimeError):
    """Raised when a required Picamera2 control fails."""


@dataclass(frozen=True)
class Picamera2ControlResult:
    name: str
    control_name: str
    requested_value: Picamera2ControlValue
    applied: bool
    effective_value: (
        Picamera2ControlValue | None
    )
    verified: bool
    matches_requested: bool | None


PICAMERA2_LOGICAL_NAMES = {
    "AeEnable": "ae_enable",
    "ExposureTime": "exposure_time_us",
    "AnalogueGain": "analogue_gain",
    "AwbEnable": "awb_enable",
    "ColourGains": "colour_gains",
}


def apply_picamera2_control_profile(
        camera: Any,
        profile: Picamera2ControlProfile,
) -> None:
    controls = create_picamera2_controls(
        profile
    )

    if not controls:
        return

    available_controls = getattr(
        camera,
        "camera_controls",
        None,
    )

    if not isinstance(
            available_controls,
            Mapping,
    ):
        raise Picamera2ControlError(
            "Picamera2 camera control "
            "capabilities are unavailable."
        )

    unsupported = [
        control_name
        for control_name in controls
        if control_name not in available_controls
    ]

    if unsupported:
        raise Picamera2ControlError(
            "Required Picamera2 controls are "
            "unsupported: "
            f"{unsupported}"
        )

    camera.set_controls(
        controls
    )


def _validate_tolerance(
        name: str,
        value: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative "
            "finite number."
        )

    return float(value)


def _picamera2_values_match(
        control_name: str,
        requested: Picamera2ControlValue,
        effective: Any,
        *,
        exposure_time_abs_tolerance_us: float,
        analogue_gain_abs_tolerance: float,
        colour_gain_abs_tolerance: float,
) -> bool:
    if isinstance(requested, bool):
        return (
            isinstance(effective, bool)
            and effective is requested
        )

    if control_name == "ColourGains":
        if (
            not isinstance(
                effective,
                (tuple, list),
            )
            or len(effective) != 2
        ):
            return False

        return all(
            math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=colour_gain_abs_tolerance,
            )
            for expected, actual in zip(
                requested,
                effective,
            )
        )

    if (
        isinstance(effective, bool)
        or not isinstance(
            effective,
            (int, float),
        )
        or not math.isfinite(float(effective))
    ):
        return False

    if control_name == "ExposureTime":
        tolerance = (
            exposure_time_abs_tolerance_us
        )
    elif control_name == "AnalogueGain":
        tolerance = (
            analogue_gain_abs_tolerance
        )
    else:
        tolerance = 0.0

    return math.isclose(
        float(effective),
        float(requested),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def verify_picamera2_control_profile(
        profile: Picamera2ControlProfile,
        metadata: Mapping[str, Any],
        *,
        exposure_time_abs_tolerance_us: float = 0.0,
        analogue_gain_abs_tolerance: float = 0.0,
        colour_gain_abs_tolerance: float = 0.0,
) -> tuple[Picamera2ControlResult, ...]:
    controls = create_picamera2_controls(
        profile
    )

    if not isinstance(metadata, Mapping):
        raise TypeError(
            "metadata must be a mapping."
        )

    exposure_tolerance = _validate_tolerance(
        "exposure_time_abs_tolerance_us",
        exposure_time_abs_tolerance_us,
    )
    gain_tolerance = _validate_tolerance(
        "analogue_gain_abs_tolerance",
        analogue_gain_abs_tolerance,
    )
    colour_tolerance = _validate_tolerance(
        "colour_gain_abs_tolerance",
        colour_gain_abs_tolerance,
    )

    results: list[
        Picamera2ControlResult
    ] = []

    for control_name, requested in (
        controls.items()
    ):
        logical_name = (
            PICAMERA2_LOGICAL_NAMES[
                control_name
            ]
        )

        if control_name not in metadata:
            results.append(
                Picamera2ControlResult(
                    name=logical_name,
                    control_name=control_name,
                    requested_value=requested,
                    applied=True,
                    effective_value=None,
                    verified=False,
                    matches_requested=None,
                )
            )
            continue

        effective = metadata[
            control_name
        ]

        matches = _picamera2_values_match(
            control_name,
            requested,
            effective,
            exposure_time_abs_tolerance_us=(
                exposure_tolerance
            ),
            analogue_gain_abs_tolerance=(
                gain_tolerance
            ),
            colour_gain_abs_tolerance=(
                colour_tolerance
            ),
        )

        if not matches:
            raise Picamera2ControlError(
                "Required Picamera2 control "
                "effective value does not match "
                "the requested value: "
                f"{control_name}; "
                f"requested={requested}, "
                f"effective={effective}"
            )

        normalized_effective = effective

        if control_name == "ColourGains":
            normalized_effective = (
                float(effective[0]),
                float(effective[1]),
            )

        results.append(
            Picamera2ControlResult(
                name=logical_name,
                control_name=control_name,
                requested_value=requested,
                applied=True,
                effective_value=(
                    normalized_effective
                ),
                verified=True,
                matches_requested=True,
            )
        )

    return tuple(results)


def picamera2_control_results_to_metadata(
        results: tuple[
            Picamera2ControlResult,
            ...,
        ],
) -> dict[str, dict[str, Any]]:
    metadata: dict[
        str,
        dict[str, Any],
    ] = {}

    for result in results:
        if not isinstance(
                result,
                Picamera2ControlResult,
        ):
            raise TypeError(
                "Every result must be a "
                "Picamera2ControlResult."
            )

        metadata[result.name] = {
            "backend": "picamera2",
            "control_name": (
                result.control_name
            ),
            "requested": (
                result.requested_value
            ),
            "effective": (
                result.effective_value
            ),
            "applied": result.applied,
            "verified": result.verified,
            "matches_requested": (
                result.matches_requested
            ),
        }

    return metadata