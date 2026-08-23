from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import TypeAlias

from benchmarks.experiments.controlled_illumination_run_planner import (
    PlannedRun,
)


SUPPORTED_EXECUTION_ARCHITECTURES = frozenset(
    {
        "pure_python",
        "hybrid",
        "pure_cpp",
    }
)


class ControlledIlluminationExecutionError(
    RuntimeError
):
    """Raised when an experiment run cannot be executed."""


@dataclass(frozen=True)
class RunnerExecutionResult:
    exit_code: int
    standard_output: str = ""
    standard_error: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.exit_code, bool)
            or not isinstance(self.exit_code, int)
        ):
            raise ControlledIlluminationExecutionError(
                "exit_code must be an integer."
            )

        if not isinstance(self.standard_output, str):
            raise ControlledIlluminationExecutionError(
                "standard_output must be a string."
            )

        if not isinstance(self.standard_error, str):
            raise ControlledIlluminationExecutionError(
                "standard_error must be a string."
            )

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class RunnerCommand:
    arguments: tuple[str, ...]
    working_directory: Path
    environment: Mapping[str, str] = field(
        default_factory=dict
    )
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.arguments, tuple)
            or not self.arguments
        ):
            raise ControlledIlluminationExecutionError(
                "arguments must be a non-empty tuple."
            )

        if any(
            not isinstance(argument, str)
            or not argument.strip()
            for argument in self.arguments
        ):
            raise ControlledIlluminationExecutionError(
                "Every command argument must be "
                "a non-empty string."
            )

        if not isinstance(
            self.working_directory,
            Path,
        ):
            raise ControlledIlluminationExecutionError(
                "working_directory must be a Path."
            )

        if not isinstance(self.environment, Mapping):
            raise ControlledIlluminationExecutionError(
                "environment must be a mapping."
            )

        for name, value in self.environment.items():
            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(value, str)
            ):
                raise ControlledIlluminationExecutionError(
                    "Environment variables must use "
                    "non-empty string names and "
                    "string values."
                )

        object.__setattr__(
            self,
            "environment",
            MappingProxyType(
                dict(self.environment)
            ),
        )

        if self.timeout_seconds is not None:
            if (
                isinstance(self.timeout_seconds, bool)
                or not isinstance(
                    self.timeout_seconds,
                    (int, float),
                )
                or not math.isfinite(
                    self.timeout_seconds
                )
                or self.timeout_seconds <= 0.0
            ):
                raise ControlledIlluminationExecutionError(
                    "timeout_seconds must be positive "
                    "and finite."
                )

            object.__setattr__(
                self,
                "timeout_seconds",
                float(self.timeout_seconds),
            )


RunnerCommandBuilder: TypeAlias = Callable[
    [PlannedRun],
    RunnerCommand,
]

ArchitectureRunner: TypeAlias = Callable[
    [PlannedRun],
    RunnerExecutionResult,
]

class ArchitectureRunnerRegistry:
    def __init__(
        self,
        runners: Mapping[
            str,
            ArchitectureRunner,
        ],
    ) -> None:
        if not isinstance(runners, Mapping):
            raise ControlledIlluminationExecutionError(
                "runners must be a mapping."
            )

        architecture_names = set(runners)

        missing_architectures = (
            SUPPORTED_EXECUTION_ARCHITECTURES
            - architecture_names
        )
        unexpected_architectures = (
            architecture_names
            - SUPPORTED_EXECUTION_ARCHITECTURES
        )

        if missing_architectures:
            raise ControlledIlluminationExecutionError(
                "Missing architecture runners: "
                f"{sorted(missing_architectures)}"
            )

        if unexpected_architectures:
            raise ControlledIlluminationExecutionError(
                "Unexpected architecture runners: "
                f"{sorted(unexpected_architectures)}"
            )

        for architecture, runner in runners.items():
            if not callable(runner):
                raise ControlledIlluminationExecutionError(
                    "Architecture runner must be callable: "
                    f"{architecture}"
                )

        self._runners: Mapping[
            str,
            ArchitectureRunner,
        ] = MappingProxyType(
            dict(runners)
        )

    def get_runner(
        self,
        architecture: str,
    ) -> ArchitectureRunner:
        if (
            not isinstance(architecture, str)
            or not architecture.strip()
        ):
            raise ControlledIlluminationExecutionError(
                "architecture must be a non-empty string."
            )

        try:
            return self._runners[architecture]
        except KeyError as error:
            raise ControlledIlluminationExecutionError(
                "Unsupported execution architecture: "
                f"{architecture}"
            ) from error

class SubprocessArchitectureRunner:
    def __init__(
        self,
        command_builder: RunnerCommandBuilder,
    ) -> None:
        if not callable(command_builder):
            raise ControlledIlluminationExecutionError(
                "command_builder must be callable."
            )

        self._command_builder = command_builder

    def __call__(
        self,
        planned_run: PlannedRun,
    ) -> RunnerExecutionResult:
        if not isinstance(planned_run, PlannedRun):
            raise ControlledIlluminationExecutionError(
                "planned_run must be PlannedRun."
            )

        runner_command = self._command_builder(
            planned_run
        )

        if not isinstance(
            runner_command,
            RunnerCommand,
        ):
            raise ControlledIlluminationExecutionError(
                "command_builder must return "
                "RunnerCommand."
            )

        execution_environment = os.environ.copy()
        execution_environment.update(
            runner_command.environment
        )

        try:
            completed_process = subprocess.run(
                runner_command.arguments,
                cwd=runner_command.working_directory,
                env=execution_environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=(
                    runner_command.timeout_seconds
                ),
            )
        except subprocess.TimeoutExpired as error:
            raise ControlledIlluminationExecutionError(
                "Architecture runner timed out after "
                f"{runner_command.timeout_seconds} "
                "seconds."
            ) from error
        except OSError as error:
            raise ControlledIlluminationExecutionError(
                "Architecture runner could not be "
                f"started: {error}"
            ) from error

        return RunnerExecutionResult(
            exit_code=completed_process.returncode,
            standard_output=(
                completed_process.stdout or ""
            ),
            standard_error=(
                completed_process.stderr or ""
            ),
        )
