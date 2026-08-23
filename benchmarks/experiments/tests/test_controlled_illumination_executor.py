from __future__ import annotations

import unittest

from benchmarks.experiments.controlled_illumination_executor import (
    ArchitectureRunnerRegistry,
    ControlledIlluminationExecutionError,
    RunnerExecutionResult,
)


class ControlledIlluminationExecutorTests(
    unittest.TestCase
):
    @staticmethod
    def successful_runner(
        planned_run,
    ) -> RunnerExecutionResult:
        return RunnerExecutionResult(
            exit_code=0,
        )

    def build_runners(self) -> dict:
        return {
            "pure_python": self.successful_runner,
            "hybrid": self.successful_runner,
            "pure_cpp": self.successful_runner,
        }

    def test_successful_result(
        self,
    ) -> None:
        result = RunnerExecutionResult(
            exit_code=0,
            standard_output="completed",
        )

        self.assertTrue(result.succeeded)

    def test_failed_result(
        self,
    ) -> None:
        result = RunnerExecutionResult(
            exit_code=1,
            standard_error="runner failed",
        )

        self.assertFalse(result.succeeded)

    def test_invalid_exit_code_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationExecutionError
        ):
            RunnerExecutionResult(
                exit_code=True,
            )

    def test_registry_returns_runner(
        self,
    ) -> None:
        runners = self.build_runners()
        registry = ArchitectureRunnerRegistry(
            runners
        )

        self.assertIs(
            registry.get_runner("hybrid"),
            runners["hybrid"],
        )

    def test_missing_runner_is_rejected(
        self,
    ) -> None:
        runners = self.build_runners()
        del runners["pure_cpp"]

        with self.assertRaisesRegex(
            ControlledIlluminationExecutionError,
            "Missing architecture runners",
        ):
            ArchitectureRunnerRegistry(
                runners
            )

    def test_unexpected_runner_is_rejected(
        self,
    ) -> None:
        runners = self.build_runners()
        runners["unknown"] = (
            self.successful_runner
        )

        with self.assertRaisesRegex(
            ControlledIlluminationExecutionError,
            "Unexpected architecture runners",
        ):
            ArchitectureRunnerRegistry(
                runners
            )

    def test_unknown_architecture_is_rejected(
        self,
    ) -> None:
        registry = ArchitectureRunnerRegistry(
            self.build_runners()
        )

        with self.assertRaisesRegex(
            ControlledIlluminationExecutionError,
            "Unsupported execution architecture",
        ):
            registry.get_runner(
                "unknown"
            )


if __name__ == "__main__":
    unittest.main()