from __future__ import annotations

import unittest

from benchmarks.experiments.controlled_illumination_run_state import (
    ControlledIlluminationRunStateError,
    RunState,
    RunStatus,
    transition_run_state,
)


STARTED_AT = "2026-08-23T10:00:00Z"
FINISHED_AT = "2026-08-23T10:05:00Z"


class ControlledIlluminationRunStateTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.planned_state = RunState(
            run_id="experiment-run-0001",
            execution_order=1,
        )

    def start_run(self) -> RunState:
        return transition_run_state(
            self.planned_state,
            RunStatus.RUNNING,
            STARTED_AT,
        )

    def test_planned_state_defaults(
        self,
    ) -> None:
        self.assertEqual(
            self.planned_state.status,
            RunStatus.PLANNED,
        )
        self.assertEqual(
            self.planned_state.attempt_count,
            0,
        )
        self.assertIsNone(
            self.planned_state.started_at_utc
        )

    def test_string_status_is_normalized(
        self,
    ) -> None:
        state = RunState(
            run_id="experiment-run-0001",
            execution_order=1,
            status="planned",  # type: ignore[arg-type]
        )

        self.assertEqual(
            state.status,
            RunStatus.PLANNED,
        )

    def test_planned_run_can_start(
        self,
    ) -> None:
        running = self.start_run()

        self.assertEqual(
            running.status,
            RunStatus.RUNNING,
        )
        self.assertEqual(running.attempt_count, 1)
        self.assertEqual(
            running.started_at_utc,
            STARTED_AT,
        )

    def test_running_run_can_complete(
        self,
    ) -> None:
        completed = transition_run_state(
            self.start_run(),
            RunStatus.COMPLETED,
            FINISHED_AT,
        )

        self.assertEqual(
            completed.status,
            RunStatus.COMPLETED,
        )
        self.assertEqual(
            completed.finished_at_utc,
            FINISHED_AT,
        )

    def test_running_run_can_fail(
        self,
    ) -> None:
        failed = transition_run_state(
            self.start_run(),
            RunStatus.FAILED,
            FINISHED_AT,
            reason="Camera disconnected.",
        )

        self.assertEqual(
            failed.status,
            RunStatus.FAILED,
        )
        self.assertEqual(
            failed.failure_reason,
            "Camera disconnected.",
        )

    def test_failed_transition_requires_reason(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunStateError
        ):
            transition_run_state(
                self.start_run(),
                RunStatus.FAILED,
                FINISHED_AT,
            )

    def test_planned_run_can_be_skipped(
        self,
    ) -> None:
        skipped = transition_run_state(
            self.planned_state,
            RunStatus.SKIPPED,
            FINISHED_AT,
            reason="Platform unavailable.",
        )

        self.assertEqual(
            skipped.status,
            RunStatus.SKIPPED,
        )
        self.assertEqual(
            skipped.skip_reason,
            "Platform unavailable.",
        )

    def test_skipped_transition_requires_reason(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunStateError
        ):
            transition_run_state(
                self.planned_state,
                RunStatus.SKIPPED,
                FINISHED_AT,
            )

    def test_invalid_transition_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "Invalid run-state transition",
        ):
            transition_run_state(
                self.planned_state,
                RunStatus.COMPLETED,
                FINISHED_AT,
            )

    def test_reason_is_rejected_when_not_allowed(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "reason is only allowed",
        ):
            transition_run_state(
                self.planned_state,
                RunStatus.RUNNING,
                STARTED_AT,
                reason="Unexpected reason.",
            )

    def test_failed_run_can_be_replanned(
        self,
    ) -> None:
        failed = transition_run_state(
            self.start_run(),
            RunStatus.FAILED,
            FINISHED_AT,
            reason="Temporary camera error.",
        )
        replanned = transition_run_state(
            failed,
            RunStatus.PLANNED,
            "2026-08-23T10:06:00Z",
        )
        restarted = transition_run_state(
            replanned,
            RunStatus.RUNNING,
            "2026-08-23T10:07:00Z",
        )

        self.assertEqual(
            replanned.status,
            RunStatus.PLANNED,
        )
        self.assertIsNone(
            replanned.failure_reason
        )
        self.assertEqual(
            restarted.attempt_count,
            2,
        )

    def test_invalid_timestamp_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunStateError
        ):
            transition_run_state(
                self.planned_state,
                RunStatus.RUNNING,
                "invalid-timestamp",
            )

    def test_inconsistent_running_state_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationRunStateError,
            "Running run fields are inconsistent",
        ):
            RunState(
                run_id="experiment-run-0001",
                execution_order=1,
                status=RunStatus.RUNNING,
                attempt_count=1,
                started_at_utc=STARTED_AT,
                finished_at_utc=FINISHED_AT,
            )


if __name__ == "__main__":
    unittest.main()