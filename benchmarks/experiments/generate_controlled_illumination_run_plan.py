from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmarks.experiments.controlled_illumination_run_planner import (
    ControlledIlluminationRunPlanError,
    build_run_plan,
    write_run_plan_manifests_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "experiments"
    / "config"
    / "controlled_illumination_config.json"
)


def resolve_project_path(
    path_value: str | Path,
) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_plan_config(
    config_path: str | Path,
) -> dict[str, Any]:
    resolved_path = resolve_project_path(
        config_path
    )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Configuration file was not found: "
            f"{resolved_path}"
        )

    with resolved_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ControlledIlluminationRunPlanError(
            "Configuration root must be an object."
        )

    return config


def create_plan_identity() -> tuple[str, str]:
    current_time = datetime.now(timezone.utc)

    generated_at_utc = (
        current_time.isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
    )
    identifier_timestamp = current_time.strftime(
        "%Y%m%dT%H%M%SZ"
    )
    experiment_id = (
        f"controlled-illumination-"
        f"{identifier_timestamp}-"
        f"{uuid4().hex[:8]}"
    )

    return experiment_id, generated_at_utc


def determine_output_directory(
    config: dict[str, Any],
    experiment_id: str,
    output_override: str | Path | None,
) -> Path:
    if output_override is not None:
        return resolve_project_path(
            output_override
        )

    output_config = config.get("output")

    if not isinstance(output_config, dict):
        raise ControlledIlluminationRunPlanError(
            "output must be an object."
        )

    configured_directory = output_config.get(
        "directory"
    )

    if (
        not isinstance(configured_directory, str)
        or not configured_directory.strip()
    ):
        raise ControlledIlluminationRunPlanError(
            "output.directory must be a "
            "non-empty string."
        )

    return (
        resolve_project_path(configured_directory)
        / experiment_id
    )


def create_argument_parser() -> (
    argparse.ArgumentParser
):
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic controlled-"
            "illumination experiment run plan."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "Controlled-illumination configuration "
            "JSON path."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        help=(
            "Explicit experiment identifier. "
            "A unique identifier is generated when "
            "omitted."
        ),
    )
    parser.add_argument(
        "--output-directory",
        help=(
            "Manifest output directory override."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and summarize the plan without "
            "writing manifest files."
        ),
    )

    return parser


def run_cli(
    arguments: argparse.Namespace,
) -> int:
    config = load_plan_config(
        arguments.config
    )

    generated_experiment_id, generated_at_utc = (
        create_plan_identity()
    )
    experiment_id = (
        arguments.experiment_id
        if arguments.experiment_id is not None
        else generated_experiment_id
    )

    plan = build_run_plan(
        config,
        experiment_id=experiment_id,
        generated_at_utc=generated_at_utc,
    )

    print("Controlled-illumination run plan valid.")
    print(f"Experiment ID: {plan.experiment_id}")
    print(f"Run count: {plan.run_count}")
    print(f"Randomized: {plan.randomized}")
    print(
        "Randomization seed: "
        f"{plan.randomization_seed}"
    )

    if arguments.dry_run:
        print("Dry run: no manifest files written.")
        return 0

    output_directory = determine_output_directory(
        config,
        plan.experiment_id,
        arguments.output_directory,
    )

    json_path, csv_path = (
        write_run_plan_manifests_atomic(
            plan,
            output_directory,
        )
    )

    print(f"JSON manifest: {json_path}")
    print(f"CSV manifest: {csv_path}")

    return 0


def main() -> None:
    parser = create_argument_parser()
    arguments = parser.parse_args()

    try:
        exit_code = run_cli(arguments)
    except (
        ControlledIlluminationRunPlanError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
    ) as error:
        parser.error(str(error))

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
