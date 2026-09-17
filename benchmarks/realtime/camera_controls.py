from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
from typing import Protocol


class CameraCaptureProtocol(Protocol):
    def set(
        self,
        property_id: int,
        value: float,
    ) -> bool:
        ...

    def get(
        self,
        property_id: int,
    ) -> float:
        ...


class CameraControlError(RuntimeError):
    """Raised when a required camera control cannot be established."""


@dataclass(frozen=True)
class CameraControlRequest:
    name: str
    property_id: int
    requested_value: float
    required: bool = True
    verify: bool = True
    absolute_tolerance: float = 0.0


@dataclass(frozen=True)
class CameraControlResult:
    name: str
    property_id: int
    requested_value: float
    applied: bool
    effective_value: float | None
    verified: bool
    matches_requested: bool | None

OPENCV_CAMERA_CONTROL_PROPERTIES: dict[
    str,
    int,
] = {
    "auto_exposure": (
        cv2.CAP_PROP_AUTO_EXPOSURE
    ),
    "exposure": (
        cv2.CAP_PROP_EXPOSURE
    ),
    "gain": (
        cv2.CAP_PROP_GAIN
    ),
    "auto_white_balance": (
        cv2.CAP_PROP_AUTO_WB
    ),
    "white_balance_temperature": (
        cv2.CAP_PROP_WB_TEMPERATURE
    ),
    "autofocus": (
        cv2.CAP_PROP_AUTOFOCUS
    ),
    "focus": (
        cv2.CAP_PROP_FOCUS
    ),
}

def create_opencv_camera_control_request(
    name: str,
    requested_value: float,
    *,
    required: bool = True,
    verify: bool = True,
    absolute_tolerance: float = 0.0,
) -> CameraControlRequest:
    if (
        not isinstance(name, str)
        or not name.strip()
    ):
        raise ValueError(
            "Camera control name must be a "
            "non-empty string."
        )

    normalized_name = (
        name.strip().lower()
    )

    property_id = (
        OPENCV_CAMERA_CONTROL_PROPERTIES
        .get(normalized_name)
    )

    if property_id is None:
        raise ValueError(
            "Unsupported OpenCV camera control: "
            f"{normalized_name}"
        )

    request = CameraControlRequest(
        name=normalized_name,
        property_id=property_id,
        requested_value=requested_value,
        required=required,
        verify=verify,
        absolute_tolerance=(
            absolute_tolerance
        ),
    )

    validate_camera_control_request(
        request
    )

    return request

def validate_camera_control_request(
    request: CameraControlRequest,
) -> None:
    if (
        not isinstance(request.name, str)
        or not request.name.strip()
    ):
        raise ValueError(
            "Camera control name must be a "
            "non-empty string."
        )

    if (
        isinstance(request.property_id, bool)
        or not isinstance(
            request.property_id,
            int,
        )
        or request.property_id < 0
    ):
        raise ValueError(
            "Camera control property_id must be "
            "a non-negative integer."
        )

    if (
        isinstance(
            request.requested_value,
            bool,
        )
        or not isinstance(
            request.requested_value,
            (int, float),
        )
        or not math.isfinite(
            float(
                request.requested_value
            )
        )
    ):
        raise ValueError(
            "Camera control requested_value must "
            "be a finite number."
        )

    if not isinstance(
        request.required,
        bool,
    ):
        raise ValueError(
            "Camera control required must be boolean."
        )

    if not isinstance(
        request.verify,
        bool,
    ):
        raise ValueError(
            "Camera control verify must be boolean."
        )

    if (
        isinstance(
            request.absolute_tolerance,
            bool,
        )
        or not isinstance(
            request.absolute_tolerance,
            (int, float),
        )
        or not math.isfinite(
            float(
                request.absolute_tolerance
            )
        )
        or request.absolute_tolerance < 0
    ):
        raise ValueError(
            "Camera control absolute_tolerance "
            "must be a non-negative finite number."
        )


def apply_camera_control(
    capture: CameraCaptureProtocol,
    request: CameraControlRequest,
) -> CameraControlResult:
    validate_camera_control_request(
        request
    )

    applied = capture.set(
        request.property_id,
        float(
            request.requested_value
        ),
    )

    if not applied:
        if request.required:
            raise CameraControlError(
                "Required camera control could not "
                f"be applied: {request.name}"
            )

        return CameraControlResult(
            name=request.name,
            property_id=request.property_id,
            requested_value=float(
                request.requested_value
            ),
            applied=False,
            effective_value=None,
            verified=False,
            matches_requested=None,
        )

    if not request.verify:
        return CameraControlResult(
            name=request.name,
            property_id=request.property_id,
            requested_value=float(
                request.requested_value
            ),
            applied=True,
            effective_value=None,
            verified=False,
            matches_requested=None,
        )

    effective_value = capture.get(
        request.property_id
    )

    if (
        isinstance(effective_value, bool)
        or not isinstance(
            effective_value,
            (int, float),
        )
        or not math.isfinite(
            float(effective_value)
        )
    ):
        if request.required:
            raise CameraControlError(
                "Required camera control returned "
                "an invalid effective value: "
                f"{request.name}"
            )

        return CameraControlResult(
            name=request.name,
            property_id=request.property_id,
            requested_value=float(
                request.requested_value
            ),
            applied=True,
            effective_value=None,
            verified=False,
            matches_requested=None,
        )

    normalized_effective_value = float(
        effective_value
    )

    matches_requested = math.isclose(
        normalized_effective_value,
        float(
            request.requested_value
        ),
        rel_tol=0.0,
        abs_tol=float(
            request.absolute_tolerance
        ),
    )

    if (
        request.required
        and not matches_requested
    ):
        raise CameraControlError(
            "Required camera control effective value "
            "does not match the requested value: "
            f"{request.name}; requested="
            f"{request.requested_value}, "
            f"effective={normalized_effective_value}"
        )

    return CameraControlResult(
        name=request.name,
        property_id=request.property_id,
        requested_value=float(
            request.requested_value
        ),
        applied=True,
        effective_value=(
            normalized_effective_value
        ),
        verified=True,
        matches_requested=(
            matches_requested
        ),
    )

def apply_camera_controls(
    capture: CameraCaptureProtocol,
    requests: tuple[
        CameraControlRequest,
        ...,
    ],
) -> tuple[
    CameraControlResult,
    ...,
]:
    names: set[str] = set()
    property_ids: set[int] = set()

    # Validate the complete request set before
    # modifying the camera.
    for request in requests:
        validate_camera_control_request(
            request
        )

        normalized_name = (
            request.name.strip()
        )

        if normalized_name in names:
            raise ValueError(
                "Duplicate camera control name: "
                f"{normalized_name}"
            )

        if (
            request.property_id
            in property_ids
        ):
            raise ValueError(
                "Duplicate camera control property_id: "
                f"{request.property_id}"
            )

        names.add(
            normalized_name
        )
        property_ids.add(
            request.property_id
        )

    # Only apply controls after the entire request
    # collection has been validated.
    results: list[
        CameraControlResult
    ] = []

    for request in requests:
        results.append(
            apply_camera_control(
                capture,
                request,
            )
        )

    return tuple(results)
