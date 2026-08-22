from __future__ import annotations

import argparse
from pathlib import Path
import platform as platform_module
import subprocess
from typing import Any

from benchmarks.experiments.controlled_illumination_metadata import (
    PROJECT_ROOT,
    ControlledIlluminationRunMetadata,
    IlluminanceMeasurements,
    ResolutionMetadata,
    create_unique_identifier,
    create_utc_timestamp,
    load_controlled_illumination_config,
    save_run_metadata_atomic,
    validate_run_metadata,
)


def read_git_commit_sha() -> str:
    try:
        completed_process = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ) as error:
        raise RuntimeError(
            "Unable to determine the Git commit SHA."
        ) from error

    commit_sha = completed_process.stdout.strip()

    if not commit_sha:
        raise RuntimeError(
            "Git returned an empty commit SHA."
        )

    return commit_sha


def build_dry_run_metadata(
    config: dict[str, Any],
    git_commit_sha: str,
) -> ControlledIlluminationRunMetadata:
    matrix = config["experiment_matrix"]
    execution = config["execution"]

    resolution_config = matrix["resolutions"][0]
    target_illuminance_lux = float(
        matrix["target_illuminance_levels_lux"][0]
    )
    incidence_angle_degrees = float(
        matrix["incidence_angles_degrees"][0]
    )

    collected_at_utc = create_utc_timestamp()

    illuminance = IlluminanceMeasurements(
        centre_lux=target_illuminance_lux,
        top_left_lux=target_illuminance_lux,
        top_right_lux=target_illuminance_lux,
        bottom_left_lux=target_illuminance_lux,
        bottom_right_lux=target_illuminance_lux,
        measured_at_utc=collected_at_utc,
        lux_meter_model="dry-run-meter",
        lux_meter_range="dry-run",
        lux_meter_resolution="dry-run",
    )

    target_fps = float(
        execution["target_fps"]
    )
    frame_deadline_ms = (
        1000.0
        / target_fps
        * execution["deadline_multiplier"]
    )

    metadata = ControlledIlluminationRunMetadata(
        experiment_id=create_unique_identifier(
            "controlled-illumination"
        ),
        run_id=create_unique_identifier("run"),
        collected_at_utc=collected_at_utc,
        git_commit_sha=git_commit_sha,
        phase="constant_lux",
        device_id=(
            platform_module.node()
            or "dry-run-device"
        ),
        operating_system=platform_module.platform(),
        platform="desktop",
        architecture="pure_python",
        algorithm="original",
        algorithm_parameters={},
        resolution=ResolutionMetadata(
            width=resolution_config["width"],
            height=resolution_config["height"],
        ),
        trial_number=1,
        target_fps=target_fps,
        frame_deadline_ms=frame_deadline_ms,
        target_illuminance_lux=(
            target_illuminance_lux
        ),
        measured_illuminance=illuminance,
        incidence_angle_degrees=(
            incidence_angle_degrees
        ),
        source_output_setting=(
            "dry-run-adjusted-to-target-lux"
        ),
        camera_mode="controlled",
        camera_settings={
            "exposure_time": "dry-run",
            "sensor_gain_or_iso": "dry-run",
            "white_balance": "dry-run",
            "focus": "dry-run",
            "frame_rate": target_fps,
            "resolution": (
                f"{resolution_config['width']}x"
                f"{resolution_config['height']}"
            ),
            "lens": "dry-run",
            "aperture": None,
        },
        camera_to_target_distance_metres=1.0,
        light_to_target_distance_metres=1.0,
        input_scene_id="dry-run-scene",
        power_mode="dry-run",
        clock_configuration="dry-run",
        starting_temperature_celsius=25.0,
        ending_temperature_celsius=25.0,
        maximum_temperature_celsius=25.0,
        thermal_throttling_detected=False,
        software_versions={
            "python": (
                platform_module.python_version()
            ),
            "python_implementation": (
                platform_module.python_implementation()
            ),
            "metadata_schema": str(
                config["schema_version"]
            ),
        },
        dry_run=True,
    )

    validate_run_metadata(
        metadata,
        config,
    )

    return metadata


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate validated controlled-illumination "
            "dry-run metadata."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. The configured "
            "results directory is used by default."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_controlled_illumination_config()

    metadata = build_dry_run_metadata(
        config,
        read_git_commit_sha(),
    )

    output_path = save_run_metadata_atomic(
        metadata,
        config=config,
        output_path=arguments.output,
    )

    print(
        "Controlled-illumination dry run passed."
    )
    print(
        f"Experiment ID: {metadata.experiment_id}"
    )
    print(
        f"Run ID: {metadata.run_id}"
    )
    print(
        f"Metadata created: {output_path}"
    )


if __name__ == "__main__":
    main()