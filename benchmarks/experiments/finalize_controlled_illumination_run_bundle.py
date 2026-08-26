from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import sys

from benchmarks.experiments.controlled_illumination_metadata import (
    load_controlled_illumination_config,
)
from benchmarks.experiments.controlled_illumination_run_bundle import (
    ControlledIlluminationRunBundleError,
    finalize_run_bundle_atomic,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlan,
    PlannedRun,
    load_run_plan_manifest,
)
from benchmarks.experiments.controlled_illumination_run_state import (
    calculate_run_plan_sha256,
)


def current_utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def find_planned_run(
    plan: ControlledIlluminationRunPlan,
    run_id: str,
) -> PlannedRun:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ControlledIlluminationRunBundleError(
            "run_id must be a non-empty string."
        )

    for planned_run in plan.runs:
        if planned_run.run_id == run_id:
            return planned_run

    raise ControlledIlluminationRunBundleError(
        f"Run ID was not found in the plan: {run_id}"
    )


def create_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and finalize a completed "
            "controlled-illumination run bundle."
        )
    )

    parser.add_argument(
        "--plan",
        required=True,
        help="Path to run_plan.json.",
    )
    parser.add_argument(
        "--run-directory",
        required=True,
        help=(
            "Directory containing the completed "
            "run artifacts."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier to finalize.",
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional controlled-illumination "
            "configuration path."
        ),
    )

    return parser


def run_cli(
    arguments: argparse.Namespace,
    *,
    now_provider: Callable[[], str] = (
        current_utc_timestamp
    ),
) -> int:
    config = load_controlled_illumination_config(
        arguments.config
    )
    plan = load_run_plan_manifest(
        arguments.plan
    )
    planned_run = find_planned_run(
        plan,
        arguments.run_id,
    )
    run_plan_sha256 = calculate_run_plan_sha256(
        plan
    )

    manifest, manifest_path = (
        finalize_run_bundle_atomic(
            arguments.run_directory,
            planned_run,
            config,
            run_plan_sha256,
            now_provider(),
        )
    )

    print("Controlled-illumination run finalized.")
    print(f"Experiment ID: {manifest.experiment_id}")
    print(f"Run ID: {manifest.run_id}")
    print(f"Bundle manifest: {manifest_path}")

    return 0


def main() -> None:
    parser = create_argument_parser()
    arguments = parser.parse_args()

    try:
        exit_code = run_cli(arguments)
    except (
        ControlledIlluminationRunBundleError,
        OSError,
        ValueError,
    ) as error:
        print(
            "Controlled-illumination run "
            f"finalization failed: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()