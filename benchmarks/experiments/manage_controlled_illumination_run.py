from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlanError,
    load_run_plan_manifest,
)
from benchmarks.experiments.controlled_illumination_run_state import (
    RUN_PROGRESS_FILE_NAME,
    ControlledIlluminationProgress,
    ControlledIlluminationRunStateError,
    RunStatus,
    load_or_initialize_run_progress,
    load_run_progress,
    save_run_progress_atomic,
    transition_progress_run,
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def create_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manage controlled-illumination "
            "experiment run progress."
        )
    )

    parser.add_argument(
        "--plan",
        required=True,
        help="Path to the run_plan.json manifest.",
    )
    parser.add_argument(
        "--progress",
        help=(
            "Path to run_progress.json. "
            "Defaults to the run-plan directory."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "init",
        help=(
            "Create progress or validate and load "
            "existing progress."
        ),
    )
    subparsers.add_parser(
        "status",
        help="Display experiment progress.",
    )
    subparsers.add_parser(
        "start-next",
        help="Start the next planned run.",
    )

    return parser


def determine_progress_path(
    plan_path: Path,
    progress_argument: str | None,
) -> Path:
    if progress_argument is not None:
        return Path(progress_argument)

    return (
        plan_path.parent
        / RUN_PROGRESS_FILE_NAME
    )


def print_progress_summary(
    progress: ControlledIlluminationProgress,
    progress_path: Path,
) -> None:
    print(f"Experiment: {progress.experiment_id}")
    print(f"Progress file: {progress_path}")
    print(f"Total runs: {progress.run_count}")

    for status in RunStatus:
        print(
            f"{status.value}: "
            f"{progress.status_counts[status.value]}"
        )

    next_run = progress.next_planned_run

    if next_run is None:
        print("Next planned run: none")
    else:
        print(
            "Next planned run: "
            f"{next_run.run_id} "
            f"(order {next_run.execution_order})"
        )


def initialize_progress_command(
    plan_path: Path,
    progress_path: Path,
    *,
    now_provider: Callable[[], str],
) -> ControlledIlluminationProgress:
    plan = load_run_plan_manifest(
        plan_path
    )
    progress_existed = progress_path.is_file()

    progress = load_or_initialize_run_progress(
        plan,
        progress_path,
        created_at_utc=now_provider(),
    )

    if progress_existed:
        print(
            "Existing run progress loaded and "
            "validated."
        )
    else:
        print("Run progress initialized.")

    print_progress_summary(
        progress,
        progress_path,
    )

    return progress


def status_command(
    plan_path: Path,
    progress_path: Path,
) -> ControlledIlluminationProgress:
    plan = load_run_plan_manifest(
        plan_path
    )
    progress = load_run_progress(
        progress_path,
        plan=plan,
    )

    print_progress_summary(
        progress,
        progress_path,
    )

    return progress


def start_next_command(
    plan_path: Path,
    progress_path: Path,
    *,
    now_provider: Callable[[], str],
) -> ControlledIlluminationProgress:
    plan = load_run_plan_manifest(
        plan_path
    )
    progress = load_run_progress(
        progress_path,
        plan=plan,
    )

    running_run = next(
        (
            run_state
            for run_state in progress.runs
            if run_state.status
            == RunStatus.RUNNING
        ),
        None,
    )

    if running_run is not None:
        raise ControlledIlluminationRunStateError(
            "A run is already running: "
            f"{running_run.run_id}"
        )

    next_run = progress.next_planned_run

    if next_run is None:
        raise ControlledIlluminationRunStateError(
            "No planned runs remain."
        )

    updated_progress = transition_progress_run(
        progress,
        next_run.run_id,
        RunStatus.RUNNING,
        now_provider(),
    )

    save_run_progress_atomic(
        updated_progress,
        progress_path,
    )

    print(f"Started run: {next_run.run_id}")
    print(
        "Execution order: "
        f"{next_run.execution_order}"
    )

    return updated_progress


def run_cli(
    arguments: argparse.Namespace,
    *,
    now_provider: Callable[[], str] = utc_now,
) -> int:
    plan_path = Path(arguments.plan)
    progress_path = determine_progress_path(
        plan_path,
        arguments.progress,
    )

    if arguments.command == "init":
        initialize_progress_command(
            plan_path,
            progress_path,
            now_provider=now_provider,
        )
        return 0

    if arguments.command == "status":
        status_command(
            plan_path,
            progress_path,
        )
        return 0

    if arguments.command == "start-next":
        start_next_command(
            plan_path,
            progress_path,
            now_provider=now_provider,
        )
        return 0

    raise ControlledIlluminationRunStateError(
        f"Unsupported command: {arguments.command}"
    )


def main() -> None:
    parser = create_argument_parser()
    arguments = parser.parse_args()

    try:
        exit_code = run_cli(arguments)
    except (
        ControlledIlluminationRunPlanError,
        ControlledIlluminationRunStateError,
        OSError,
    ) as error:
        parser.error(str(error))

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()