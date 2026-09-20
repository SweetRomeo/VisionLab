from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class JetsonControlProfile:
    ae_lock: bool | None = None
    exposure_time_ns: int | None = None
    gain: float | None = None
    awb_lock: bool | None = None
    wb_mode: int | None = None


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
            "exposure_time_ns must be a "
            "positive integer."
        )


def _validate_gain(
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
            "gain must be a positive finite number."
        )


def _validate_wb_mode(
    value: int | None,
) -> None:
    if value is None:
        return

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            "wb_mode must be a "
            "non-negative integer."
        )


def validate_jetson_control_profile(
    profile: JetsonControlProfile,
) -> None:
    if not isinstance(
        profile,
        JetsonControlProfile,
    ):
        raise TypeError(
            "profile must be a "
            "JetsonControlProfile."
        )

    _validate_optional_boolean(
        "ae_lock",
        profile.ae_lock,
    )
    _validate_optional_boolean(
        "awb_lock",
        profile.awb_lock,
    )

    _validate_exposure_time(
        profile.exposure_time_ns
    )
    _validate_gain(
        profile.gain
    )
    _validate_wb_mode(
        profile.wb_mode
    )


def create_jetson_source_properties(
    profile: JetsonControlProfile,
) -> dict[str, Any]:
    validate_jetson_control_profile(
        profile
    )

    properties: dict[str, Any] = {}

    if profile.ae_lock is not None:
        properties["aelock"] = (
            profile.ae_lock
        )

    if profile.exposure_time_ns is not None:
        properties["exposuretimerange"] = (
            f"{profile.exposure_time_ns} "
            f"{profile.exposure_time_ns}"
        )

    if profile.gain is not None:
        gain = float(profile.gain)

        properties["gainrange"] = (
            f"{gain:g} {gain:g}"
        )

    if profile.awb_lock is not None:
        properties["awblock"] = (
            profile.awb_lock
        )

    if profile.wb_mode is not None:
        properties["wbmode"] = (
            profile.wb_mode
        )

    return properties


def serialize_jetson_source_properties(
    profile: JetsonControlProfile,
) -> str:
    properties = create_jetson_source_properties(
        profile
    )

    serialized: list[str] = []

    for name, value in properties.items():
        if isinstance(value, bool):
            serialized_value = (
                "true"
                if value
                else "false"
            )

        elif isinstance(value, str):
            serialized_value = (
                f'"{value}"'
            )

        else:
            serialized_value = str(value)

        serialized.append(
            f"{name}={serialized_value}"
        )

    return " ".join(serialized)

JetsonControlValue = (
    bool
    | int
    | float
)


@dataclass(frozen=True)
class JetsonControlResult:
    name: str
    control_name: str
    requested_value: JetsonControlValue
    applied: bool
    effective_value: (
        JetsonControlValue | None
    )
    verified: bool
    matches_requested: bool | None


JETSON_CONTROL_NAMES = {
    "ae_lock": "aelock",
    "exposure_time_ns": "exposuretimerange",
    "gain": "gainrange",
    "awb_lock": "awblock",
    "wb_mode": "wbmode",
}


def create_applied_jetson_control_results(
    profile: JetsonControlProfile,
) -> tuple[JetsonControlResult, ...]:
    validate_jetson_control_profile(
        profile
    )

    requested_controls = (
        (
            "ae_lock",
            profile.ae_lock,
        ),
        (
            "exposure_time_ns",
            profile.exposure_time_ns,
        ),
        (
            "gain",
            profile.gain,
        ),
        (
            "awb_lock",
            profile.awb_lock,
        ),
        (
            "wb_mode",
            profile.wb_mode,
        ),
    )

    results: list[JetsonControlResult] = []

    for name, requested_value in requested_controls:
        if requested_value is None:
            continue

        if name == "gain":
            requested_value = float(
                requested_value
            )

        results.append(
            JetsonControlResult(
                name=name,
                control_name=(
                    JETSON_CONTROL_NAMES[name]
                ),
                requested_value=requested_value,
                applied=True,
                effective_value=None,
                verified=False,
                matches_requested=None,
            )
        )

    return tuple(results)


def jetson_control_results_to_metadata(
    results: tuple[
        JetsonControlResult,
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
            JetsonControlResult,
        ):
            raise TypeError(
                "Every result must be a "
                "JetsonControlResult."
            )

        metadata[result.name] = {
            "backend": "jetson_gstreamer",
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