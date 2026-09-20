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