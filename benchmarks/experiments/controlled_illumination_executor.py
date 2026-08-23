from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
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