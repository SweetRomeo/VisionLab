from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from benchmarks.experiments.controlled_illumination_executor import (
    SUPPORTED_EXECUTION_ARCHITECTURES,
    ArchitectureRunnerRegistry,
    ControlledIlluminationExecutionError,
    RunnerCommand,
    SubprocessArchitectureRunner,
    execute_next_planned_run_from_files,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    PlannedRun,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_CONFIG_SCHEMA_VERSION = 1


def current_utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def determine_progress_path(
    plan_path: str | Path,
    progress_path: str | Path | None,
) -> Path:
    if progress_path is not None:
        return Path(progress_path).resolve()

    return (
        Path(plan_path).resolve().parent
        / "run_progress.json"
    )


def require_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlledIlluminationExecutionError(
            f"{field_name} must be an object."
        )

    return value


def require_command_arguments(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(argument, str)
            or not argument.strip()
            for argument in value
        )
    ):
        raise ControlledIlluminationExecutionError(
            f"{field_name} must be a non-empty "
            "array of non-empty strings."
        )

    return tuple(value)


def require_environment(
    value: Any,
    field_name: str,
) -> dict[str, str]:
    if value is None:
        return {}

    environment = require_mapping(
        value,
        field_name,
    )

    for name, environment_value in environment.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(environment_value, str)
        ):
            raise ControlledIlluminationExecutionError(
                f"{field_name} must use non-empty "
                "string names and string values."
            )

    return dict(environment)


def resolve_working_directory(
    value: Any,
    field_name: str,
) -> Path:
    if value is None:
        return PROJECT_ROOT

    if not isinstance(value, str) or not value.strip():
        raise ControlledIlluminationExecutionError(
            f"{field_name} must be a non-empty string."
        )

    working_directory = Path(value)

    if not working_directory.is_absolute():
        working_directory = (
            PROJECT_ROOT / working_directory
        )

    return working_directory.resolve()


def require_optional_timeout(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ControlledIlluminationExecutionError(
            f"{field_name} must be a positive number."
        )

    return float(value)


def serialize_optional_value(
    value: str | int | float | None,
) -> str:
    if value is None:
        return ""

    return str(value)


def build_planned_run_environment(
    planned_run: PlannedRun,
) -> dict[str, str]:
    if not isinstance(planned_run, PlannedRun):
        raise ControlledIlluminationExecutionError(
            "planned_run must be PlannedRun."
        )

    return {
        "VISIONLAB_EXPERIMENT_ID": (
            planned_run.experiment_id
        ),
        "VISIONLAB_RUN_ID": planned_run.run_id,
        "VISIONLAB_EXECUTION_ORDER": str(
            planned_run.execution_order
        ),
        "VISIONLAB_PHASE": planned_run.phase,
        "VISIONLAB_PLATFORM": planned_run.platform,
        "VISIONLAB_ARCHITECTURE": (
            planned_run.architecture
        ),
        "VISIONLAB_ALGORITHM": planned_run.algorithm,
        "VISIONLAB_RESOLUTION_WIDTH": str(
            planned_run.resolution.width
        ),
        "VISIONLAB_RESOLUTION_HEIGHT": str(
            planned_run.resolution.height
        ),
        "VISIONLAB_TRIAL_NUMBER": str(
            planned_run.trial_number
        ),
        "VISIONLAB_INCIDENCE_ANGLE_DEGREES": str(
            planned_run.incidence_angle_degrees
        ),
        "VISIONLAB_TARGET_ILLUMINANCE_LUX": (
            serialize_optional_value(
                planned_run.target_illuminance_lux
            )
        ),
        "VISIONLAB_SOURCE_OUTPUT_SETTING": (
            serialize_optional_value(
                planned_run.source_output_setting
            )
        ),
        "VISIONLAB_TARGET_FPS": str(
            planned_run.target_fps
        ),
        "VISIONLAB_FRAME_DEADLINE_MS": str(
            planned_run.frame_deadline_ms
        ),
    }


def load_runner_registry(
    config_path: str | Path,
) -> ArchitectureRunnerRegistry:
    resolved_config_path = Path(
        config_path
    ).resolve()

    try:
        with resolved_config_path.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            config_value = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ControlledIlluminationExecutionError(
            "Runner configuration could not be "
            f"loaded: {error}"
        ) from error

    config = require_mapping(
        config_value,
        "runner configuration",
    )

    if config.get("schema_version") != (
        RUNNER_CONFIG_SCHEMA_VERSION
    ):
        raise ControlledIlluminationExecutionError(
            "runner configuration schema_version "
            "must be 1."
        )

    runner_values = require_mapping(
        config.get("runners"),
        "runners",
    )

    configured_architectures = set(
        runner_values
    )

    if configured_architectures != (
        SUPPORTED_EXECUTION_ARCHITECTURES
    ):
        missing = sorted(
            SUPPORTED_EXECUTION_ARCHITECTURES
            - configured_architectures
        )
        unexpected = sorted(
            configured_architectures
            - SUPPORTED_EXECUTION_ARCHITECTURES
        )

        raise ControlledIlluminationExecutionError(
            "Runner configuration architectures "
            f"do not match. Missing: {missing}; "
            f"unexpected: {unexpected}."
        )

    architecture_runners = {}

    for architecture in sorted(
        SUPPORTED_EXECUTION_ARCHITECTURES
    ):
        runner_value = require_mapping(
            runner_values[architecture],
            f"runners.{architecture}",
        )

        allowed_fields = {
            "arguments",
            "working_directory",
            "environment",
            "timeout_seconds",
        }
        unexpected_fields = (
            set(runner_value) - allowed_fields
        )

        if unexpected_fields:
            raise ControlledIlluminationExecutionError(
                f"Unexpected fields in runners."
                f"{architecture}: "
                f"{sorted(unexpected_fields)}"
            )

        arguments = require_command_arguments(
            runner_value.get("arguments"),
            f"runners.{architecture}.arguments",
        )
        working_directory = (
            resolve_working_directory(
                runner_value.get(
                    "working_directory"
                ),
                (
                    f"runners.{architecture}."
                    "working_directory"
                ),
            )
        )
        base_environment = require_environment(
            runner_value.get("environment"),
            f"runners.{architecture}.environment",
        )
        timeout_seconds = require_optional_timeout(
            runner_value.get("timeout_seconds"),
            (
                f"runners.{architecture}."
                "timeout_seconds"
            ),
        )

        def command_builder(
            planned_run: PlannedRun,
            *,
            command_arguments: tuple[str, ...] = (
                arguments
            ),
            command_working_directory: Path = (
                working_directory
            ),
            command_environment: Mapping[str, str] = (
                base_environment
            ),
            command_timeout_seconds: float | None = (
                timeout_seconds
            ),
        ) -> RunnerCommand:
            execution_environment = dict(
                command_environment
            )
            execution_environment.update(
                build_planned_run_environment(
                    planned_run
                )
            )

            return RunnerCommand(
                arguments=command_arguments,
                working_directory=(
                    command_working_directory
                ),
                environment=execution_environment,
                timeout_seconds=(
                    command_timeout_seconds
                ),
            )

        architecture_runners[architecture] = (
            SubprocessArchitectureRunner(
                command_builder
            )
        )

    return ArchitectureRunnerRegistry(
        architecture_runners
    )


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the next planned controlled-"
            "illumination experiment run."
        )
    )
    parser.add_argument(
        "--plan",
        required=True,
        help="Path to run_plan.json.",
    )
    parser.add_argument(
        "--progress",
        help=(
            "Path to run_progress.json. Defaults "
            "to the run-plan directory."
        ),
    )
    parser.add_argument(
        "--runner-config",
        required=True,
        help=(
            "Path to the architecture runner "
            "configuration JSON file."
        ),
    )
    return parser


def run_cli(
    arguments: argparse.Namespace,
    *,
    timestamp_provider: Callable[[], str] = (
        current_utc_timestamp
    ),
) -> int:
    plan_path = Path(arguments.plan).resolve()
    progress_path = determine_progress_path(
        plan_path,
        arguments.progress,
    )
    runner_registry = load_runner_registry(
        arguments.runner_config
    )

    outcome = execute_next_planned_run_from_files(
        plan_path,
        progress_path,
        runner_registry,
        timestamp_provider=timestamp_provider,
    )

    if outcome is None:
        print("No planned experiment runs remain.")
        return 0

    if outcome.succeeded:
        print(
            "Controlled-illumination run completed: "
            f"{outcome.planned_run.run_id}"
        )
        return 0

    print(
        "Controlled-illumination run failed: "
        f"{outcome.planned_run.run_id}",
        file=sys.stderr,
    )

    if outcome.execution_result.standard_error:
        print(
            outcome.execution_result.standard_error,
            file=sys.stderr,
        )

    return 1


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = create_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        return run_cli(arguments)
    except KeyboardInterrupt:
        print(
            "Controlled-illumination execution "
            "interrupted.",
            file=sys.stderr,
        )
        return 130
    except (
        ControlledIlluminationExecutionError,
        OSError,
        ValueError,
    ) as error:
        print(
            f"Controlled-illumination execution "
            f"failed: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())