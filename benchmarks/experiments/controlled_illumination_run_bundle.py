from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.experiments.controlled_illumination_metadata import (
    validate_safe_identifier,
    validate_utc_timestamp,
)
from benchmarks.experiments.controlled_illumination_run_state import (
    validate_run_plan_sha256,
)


RUN_BUNDLE_SCHEMA_VERSION = 1

FRAME_RESULTS_FILE_NAME = (
    "realtime_frame_results.csv"
)
EXECUTION_SUMMARY_FILE_NAME = (
    "execution_summary.json"
)
RUN_METADATA_FILE_NAME = "run_metadata.json"
RUN_BUNDLE_MANIFEST_FILE_NAME = (
    "run_bundle_manifest.json"
)

REQUIRED_BUNDLE_ARTIFACTS = frozenset(
    {
        FRAME_RESULTS_FILE_NAME,
        EXECUTION_SUMMARY_FILE_NAME,
        RUN_METADATA_FILE_NAME,
    }
)


class ControlledIlluminationRunBundleError(
    ValueError
):
    """Raised when a controlled run bundle is invalid."""


def validate_bundle_identifier(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise ControlledIlluminationRunBundleError(
            f"{field_name} must be a string."
        )

    try:
        validate_safe_identifier(
            value,
            field_name,
        )
    except ValueError as error:
        raise ControlledIlluminationRunBundleError(
            str(error)
        ) from error

    return value


def validate_bundle_timestamp(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise ControlledIlluminationRunBundleError(
            f"{field_name} must be a string."
        )

    try:
        validate_utc_timestamp(
            value,
            field_name,
        )
    except ValueError as error:
        raise ControlledIlluminationRunBundleError(
            str(error)
        ) from error

    return value


def validate_sha256(
    value: Any,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ControlledIlluminationRunBundleError(
            f"{field_name} must be a lowercase "
            "SHA-256 value."
        )

    return value


@dataclass(frozen=True)
class RunBundleArtifact:
    file_name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.file_name, str)
            or self.file_name
            not in REQUIRED_BUNDLE_ARTIFACTS
        ):
            raise ControlledIlluminationRunBundleError(
                "Unsupported bundle artifact: "
                f"{self.file_name}"
            )

        validate_sha256(
            self.sha256,
            f"{self.file_name}.sha256",
        )

        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ControlledIlluminationRunBundleError(
                f"{self.file_name}.size_bytes must "
                "be a positive integer."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledIlluminationRunBundleManifest:
    schema_version: int
    finalized_at_utc: str
    experiment_id: str
    run_id: str
    run_plan_sha256: str
    metadata_dry_run: bool
    artifacts: tuple[RunBundleArtifact, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version
            != RUN_BUNDLE_SCHEMA_VERSION
        ):
            raise ControlledIlluminationRunBundleError(
                "schema_version must be 1."
            )

        validate_bundle_timestamp(
            self.finalized_at_utc,
            "finalized_at_utc",
        )
        validate_bundle_identifier(
            self.experiment_id,
            "experiment_id",
        )
        validate_bundle_identifier(
            self.run_id,
            "run_id",
        )

        try:
            validate_run_plan_sha256(
                self.run_plan_sha256
            )
        except ValueError as error:
            raise ControlledIlluminationRunBundleError(
                str(error)
            ) from error

        if not isinstance(
            self.metadata_dry_run,
            bool,
        ):
            raise ControlledIlluminationRunBundleError(
                "metadata_dry_run must be boolean."
            )

        if (
            not isinstance(self.artifacts, tuple)
            or not self.artifacts
        ):
            raise ControlledIlluminationRunBundleError(
                "artifacts must be a non-empty tuple."
            )

        if not all(
            isinstance(artifact, RunBundleArtifact)
            for artifact in self.artifacts
        ):
            raise ControlledIlluminationRunBundleError(
                "Every artifact must be "
                "RunBundleArtifact."
            )

        artifact_names = [
            artifact.file_name
            for artifact in self.artifacts
        ]

        if len(artifact_names) != len(
            set(artifact_names)
        ):
            raise ControlledIlluminationRunBundleError(
                "Bundle artifact names must be unique."
            )

        if set(artifact_names) != set(
            REQUIRED_BUNDLE_ARTIFACTS
        ):
            missing_artifacts = sorted(
                REQUIRED_BUNDLE_ARTIFACTS
                - set(artifact_names)
            )
            unexpected_artifacts = sorted(
                set(artifact_names)
                - REQUIRED_BUNDLE_ARTIFACTS
            )

            raise ControlledIlluminationRunBundleError(
                "Bundle artifacts do not match the "
                "required set. "
                f"Missing: {missing_artifacts}; "
                f"unexpected: {unexpected_artifacts}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "finalized_at_utc": (
                self.finalized_at_utc
            ),
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "run_plan_sha256": (
                self.run_plan_sha256
            ),
            "metadata_dry_run": (
                self.metadata_dry_run
            ),
            "artifacts": [
                artifact.to_dict()
                for artifact in self.artifacts
            ],
        }