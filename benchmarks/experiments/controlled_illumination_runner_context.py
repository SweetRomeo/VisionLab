from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Any

from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlanError,
    PlannedRun,
)
from benchmarks.experiments.controlled_illumination_metadata import (
    ResolutionMetadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
    / "controlled_illumination"
)

RESULTS_ROOT_VARIABLE = "VISIONLAB_RESULTS_ROOT"

SUPPORTED_RUNNER_ARCHITECTURES = frozenset(
    {
        "pure_python",
        "hybrid",
        "pure_cpp",
    }
)

SUPPORTED_RUNNER_ALGORITHMS = frozenset(
    {
        "original",
        "gamma_correction",
        "histogram_equalization",
        "clahe",
    }
)

SUPPORTED_EXPERIMENT_PHASES = frozenset(
    {
        "constant_lux",
        "constant_source",
    }
)

SAFE_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
)


class ControlledIlluminationRunnerContextError(
    ValueError
):
    """Raised when runner environment values are invalid."""


@dataclass(frozen=True)
class ControlledIlluminationRunnerContext:
    planned_run: PlannedRun
    results_root: Path
    output_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.planned_run, PlannedRun):
            raise ControlledIlluminationRunnerContextError(
                "planned_run must be PlannedRun."
            )

        if not isinstance(self.results_root, Path):
            raise ControlledIlluminationRunnerContextError(
                "results_root must be a Path."
            )

        if not isinstance(self.output_directory, Path):
            raise ControlledIlluminationRunnerContextError(
                "output_directory must be a Path."
            )

        resolved_results_root = (
            self.results_root.resolve()
        )
        resolved_output_directory = (
            self.output_directory.resolve()
        )

        try:
            resolved_output_directory.relative_to(
                resolved_results_root
            )
        except ValueError as error:
            raise ControlledIlluminationRunnerContextError(
                "output_directory must be inside "
                "results_root."
            ) from error

        object.__setattr__(
            self,
            "results_root",
            resolved_results_root,
        )
        object.__setattr__(
            self,
            "output_directory",
            resolved_output_directory,
        )

    @property
    def experiment_id(self) -> str:
        return self.planned_run.experiment_id

    @property
    def run_id(self) -> str:
        return self.planned_run.run_id

    @property
    def architecture(self) -> str:
        return self.planned_run.architecture

    @property
    def algorithm(self) -> str:
        return self.planned_run.algorithm

    @property
    def resolution(self) -> ResolutionMetadata:
        return self.planned_run.resolution


def require_environment_value(
    environment: Mapping[str, str],
    variable_name: str,
) -> str:
    try:
        value = environment[variable_name]
    except KeyError as error:
        raise ControlledIlluminationRunnerContextError(
            "Required environment variable is missing: "
            f"{variable_name}"
        ) from error

    if not isinstance(value, str) or not value.strip():
        raise ControlledIlluminationRunnerContextError(
            "Environment variable must contain a "
            f"non-empty string: {variable_name}"
        )

    return value.strip()


def read_optional_environment_value(
    environment: Mapping[str, str],
    variable_name: str,
) -> str | None:
    value = environment.get(variable_name)

    if value is None:
        return None

    if not isinstance(value, str):
        raise ControlledIlluminationRunnerContextError(
            "Environment variable must be a string: "
            f"{variable_name}"
        )

    normalized_value = value.strip()

    return normalized_value or None


def require_safe_identifier(
    value: str,
    field_name: str,
) -> str:
    if SAFE_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ControlledIlluminationRunnerContextError(
            f"{field_name} contains unsafe characters."
        )

    return value


def parse_positive_integer(
    environment: Mapping[str, str],
    variable_name: str,
) -> int:
    raw_value = require_environment_value(
        environment,
        variable_name,
    )

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be an integer."
        ) from error

    if value <= 0:
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be positive."
        )

    return value


def parse_finite_number(
    environment: Mapping[str, str],
    variable_name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    raw_value = require_environment_value(
        environment,
        variable_name,
    )

    try:
        value = float(raw_value)
    except ValueError as error:
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be numeric."
        ) from error

    if not math.isfinite(value):
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be finite."
        )

    if positive and value <= 0.0:
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be positive."
        )

    if non_negative and value < 0.0:
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be non-negative."
        )

    return value


def parse_optional_positive_number(
    environment: Mapping[str, str],
    variable_name: str,
) -> float | None:
    raw_value = read_optional_environment_value(
        environment,
        variable_name,
    )

    if raw_value is None:
        return None

    try:
        value = float(raw_value)
    except ValueError as error:
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be numeric."
        ) from error

    if not math.isfinite(value) or value <= 0.0:
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be positive and finite."
        )

    return value


def parse_source_output_setting(
    environment: Mapping[str, str],
) -> str | float | None:
    variable_name = (
        "VISIONLAB_SOURCE_OUTPUT_SETTING"
    )

    raw_value = read_optional_environment_value(
        environment,
        variable_name,
    )

    if raw_value is None:
        return None

    try:
        numeric_value = float(raw_value)
    except ValueError:
        return raw_value

    if not math.isfinite(numeric_value):
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be finite."
        )

    if numeric_value < 0.0:
        raise ControlledIlluminationRunnerContextError(
            f"{variable_name} must be non-negative."
        )

    return numeric_value


def resolve_results_root(
    environment: Mapping[str, str],
) -> Path:
    configured_path = read_optional_environment_value(
        environment,
        RESULTS_ROOT_VARIABLE,
    )

    if configured_path is None:
        return DEFAULT_RESULTS_ROOT.resolve()

    results_root = Path(configured_path)

    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root

    return results_root.resolve()


def load_runner_context_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    expected_architecture: str | None = None,
) -> ControlledIlluminationRunnerContext:
    active_environment: Mapping[str, str] = (
        os.environ
        if environment is None
        else environment
    )

    if not isinstance(active_environment, Mapping):
        raise ControlledIlluminationRunnerContextError(
            "environment must be a mapping."
        )

    experiment_id = require_safe_identifier(
        require_environment_value(
            active_environment,
            "VISIONLAB_EXPERIMENT_ID",
        ),
        "experiment_id",
    )
    run_id = require_safe_identifier(
        require_environment_value(
            active_environment,
            "VISIONLAB_RUN_ID",
        ),
        "run_id",
    )
    platform = require_safe_identifier(
        require_environment_value(
            active_environment,
            "VISIONLAB_PLATFORM",
        ),
        "platform",
    )

    phase = require_environment_value(
        active_environment,
        "VISIONLAB_PHASE",
    )
    architecture = require_environment_value(
        active_environment,
        "VISIONLAB_ARCHITECTURE",
    )
    algorithm = require_environment_value(
        active_environment,
        "VISIONLAB_ALGORITHM",
    )

    if phase not in SUPPORTED_EXPERIMENT_PHASES:
        raise ControlledIlluminationRunnerContextError(
            f"Unsupported experiment phase: {phase}"
        )

    if architecture not in (
        SUPPORTED_RUNNER_ARCHITECTURES
    ):
        raise ControlledIlluminationRunnerContextError(
            f"Unsupported architecture: {architecture}"
        )

    if (
        expected_architecture is not None
        and architecture != expected_architecture
    ):
        raise ControlledIlluminationRunnerContextError(
            "Runner architecture mismatch. "
            f"Expected {expected_architecture}, "
            f"received {architecture}."
        )

    if algorithm not in SUPPORTED_RUNNER_ALGORITHMS:
        raise ControlledIlluminationRunnerContextError(
            f"Unsupported algorithm: {algorithm}"
        )

    target_illuminance_lux = (
        parse_optional_positive_number(
            active_environment,
            "VISIONLAB_TARGET_ILLUMINANCE_LUX",
        )
    )
    source_output_setting = (
        parse_source_output_setting(
            active_environment
        )
    )

    try:
        planned_run = PlannedRun(
            execution_order=parse_positive_integer(
                active_environment,
                "VISIONLAB_EXECUTION_ORDER",
            ),
            experiment_id=experiment_id,
            run_id=run_id,
            phase=phase,
            platform=platform,
            architecture=architecture,
            algorithm=algorithm,
            resolution=ResolutionMetadata(
                width=parse_positive_integer(
                    active_environment,
                    "VISIONLAB_RESOLUTION_WIDTH",
                ),
                height=parse_positive_integer(
                    active_environment,
                    "VISIONLAB_RESOLUTION_HEIGHT",
                ),
            ),
            trial_number=parse_positive_integer(
                active_environment,
                "VISIONLAB_TRIAL_NUMBER",
            ),
            incidence_angle_degrees=(
                parse_finite_number(
                    active_environment,
                    (
                        "VISIONLAB_INCIDENCE_"
                        "ANGLE_DEGREES"
                    ),
                    non_negative=True,
                )
            ),
            target_illuminance_lux=(
                target_illuminance_lux
            ),
            source_output_setting=(
                source_output_setting
            ),
            target_fps=parse_finite_number(
                active_environment,
                "VISIONLAB_TARGET_FPS",
                positive=True,
            ),
            frame_deadline_ms=parse_finite_number(
                active_environment,
                "VISIONLAB_FRAME_DEADLINE_MS",
                positive=True,
            ),
        )
    except ControlledIlluminationRunPlanError as error:
        raise ControlledIlluminationRunnerContextError(
            f"Invalid planned-run environment: {error}"
        ) from error

    results_root = resolve_results_root(
        active_environment
    )
    output_directory = (
        results_root
        / platform
        / experiment_id
        / run_id
    )

    return ControlledIlluminationRunnerContext(
        planned_run=planned_run,
        results_root=results_root,
        output_directory=output_directory,
    )