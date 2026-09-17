from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import math
import os
import sys
from typing import Any
from pathlib import Path

from benchmarks.experiments.controlled_illumination_run_artifacts import (
    write_completed_run_artifacts_atomic,
)
from benchmarks.experiments.controlled_illumination_runner_context import (
    ControlledIlluminationRunnerContext,
    load_runner_context_from_environment,
)
from benchmarks.realtime.pure_python_realtime import (
    create_frame_processor,
)
from benchmarks.realtime.realtime_config import (
    RealtimeConfig,
    load_realtime_config,
)
from benchmarks.realtime.realtime_experiment_plan import (
    load_algorithms,
    load_benchmark_config,
    load_resolutions,
    resolve_video_path,
    validate_shared_execution_counts,
)
from benchmarks.realtime.realtime_pipeline import (
    iter_camera_frames,
    iter_video_frames,
    run_realtime_trial,
)
from benchmarks.experiments.controlled_illumination_metadata import (
    load_controlled_illumination_config,
)

from benchmarks.experiments.controlled_illumination_quality_capture import (
    QualityCaptureBuffer,
    cleanup_quality_capture_artifacts,
    load_quality_capture_config,
    write_quality_capture_artifacts_atomic,
)
from benchmarks.realtime.camera_controls import (
    CameraControlProfile,
    CameraControlResult,
    camera_control_results_to_metadata,
    create_camera_control_requests,
)

PURE_PYTHON_ARCHITECTURE = "pure_python"

INPUT_SOURCE_VARIABLE = (
    "VISIONLAB_INPUT_SOURCE"
)

VIDEO_INPUT_SOURCE = "video"
CAMERA_INPUT_SOURCE = "camera"
CAMERA_INDEX_VARIABLE = (
    "VISIONLAB_CAMERA_INDEX"
)
EXPERIMENT_CONFIG_VARIABLE = (
    "VISIONLAB_EXPERIMENT_CONFIG"
)

class ControlledIlluminationPurePythonRunnerError(
    RuntimeError
):
    """Raised when the Pure Python runner cannot execute."""


def current_utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

def load_experiment_config(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    active_environment = (
        os.environ
        if environment is None
        else environment
    )

    raw_config_path = active_environment.get(
        EXPERIMENT_CONFIG_VARIABLE
    )

    if raw_config_path is None:
        return load_controlled_illumination_config()

    if (
        not isinstance(raw_config_path, str)
        or not raw_config_path.strip()
    ):
        raise ControlledIlluminationPurePythonRunnerError(
            f"{EXPERIMENT_CONFIG_VARIABLE} must contain "
            "a non-empty path."
        )

    return load_controlled_illumination_config(
        Path(raw_config_path.strip())
    )

def load_camera_control_profile(
    experiment_config: dict[str, Any],
) -> CameraControlProfile:
    raw_profile = experiment_config.get(
        "camera_control_profile"
    )

    if raw_profile is None:
        return CameraControlProfile()

    if not isinstance(
        raw_profile,
        dict,
    ):
        raise ControlledIlluminationPurePythonRunnerError(
            "camera_control_profile must be an object."
        )

    supported_fields = {
        "auto_exposure",
        "exposure",
        "gain",
        "auto_white_balance",
        "white_balance_temperature",
        "autofocus",
        "focus",
    }

    unknown_fields = (
        set(raw_profile)
        - supported_fields
    )

    if unknown_fields:
        raise ControlledIlluminationPurePythonRunnerError(
            "Unsupported camera control profile fields: "
            f"{sorted(unknown_fields)}"
        )

    try:
        profile = CameraControlProfile(
            auto_exposure=(
                raw_profile.get(
                    "auto_exposure"
                )
            ),
            exposure=(
                raw_profile.get(
                    "exposure"
                )
            ),
            gain=(
                raw_profile.get(
                    "gain"
                )
            ),
            auto_white_balance=(
                raw_profile.get(
                    "auto_white_balance"
                )
            ),
            white_balance_temperature=(
                raw_profile.get(
                    "white_balance_temperature"
                )
            ),
            autofocus=(
                raw_profile.get(
                    "autofocus"
                )
            ),
            focus=(
                raw_profile.get(
                    "focus"
                )
            ),
        )

        # Also validates relationships such as:
        # manual exposure -> explicit auto_exposure.
        create_camera_control_requests(
            profile
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ControlledIlluminationPurePythonRunnerError(
            "Invalid camera_control_profile: "
            f"{error}"
        ) from error

    return profile

def create_frame_source(
    benchmark_config: dict[str, Any],
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    camera_controls: tuple[
        CameraControlRequest,
        ...,
    ] = (),
    camera_controls_reporter: (
        Callable[
            [
                tuple[
                    CameraControlResult,
                    ...,
                ]
            ],
            None,
        ]
        | None
    ) = None,
    environment: Mapping[str, str] | None = None,
) -> Any:
    active_environment = (
        os.environ
        if environment is None
        else environment
    )

    raw_input_source = active_environment.get(
        INPUT_SOURCE_VARIABLE,
        VIDEO_INPUT_SOURCE,
    )

    if (
        not isinstance(raw_input_source, str)
        or not raw_input_source.strip()
    ):
        raise ControlledIlluminationPurePythonRunnerError(
            f"{INPUT_SOURCE_VARIABLE} must contain "
            "a non-empty string."
        )

    input_source = (
        raw_input_source.strip().lower()
    )

    if input_source == VIDEO_INPUT_SOURCE:
        video_path = resolve_video_path(
            benchmark_config
        )

        return iter_video_frames(
            video_path
        )

    if input_source == CAMERA_INPUT_SOURCE:
        raw_camera_index = (
            active_environment.get(
                CAMERA_INDEX_VARIABLE
            )
        )

        if (
            not isinstance(raw_camera_index, str)
            or not raw_camera_index.strip()
        ):
            raise ControlledIlluminationPurePythonRunnerError(
                f"{CAMERA_INDEX_VARIABLE} must be "
                "a non-negative integer."
            )

        try:
            camera_index = int(
                raw_camera_index
            )
        except ValueError as error:
            raise ControlledIlluminationPurePythonRunnerError(
                f"{CAMERA_INDEX_VARIABLE} must be "
                "a non-negative integer."
            ) from error

        if camera_index < 0:
            raise ControlledIlluminationPurePythonRunnerError(
                f"{CAMERA_INDEX_VARIABLE} must be "
                "a non-negative integer."
            )

        return iter_camera_frames(
            camera_index,
            width=width,
            height=height,
            fps=fps,
            camera_controls=camera_controls,
            camera_controls_reporter=(
                camera_controls_reporter
            ),
        )

    raise ControlledIlluminationPurePythonRunnerError(
        "Unsupported input source: "
        f"{input_source}"
    )


def select_algorithm_configuration(
    benchmark_config: dict[str, Any],
    algorithm_name: str,
) -> dict[str, Any]:
    matching_algorithms = [
        algorithm_config
        for algorithm_config in load_algorithms(
            benchmark_config
        )
        if algorithm_config["name"] == algorithm_name
    ]

    if len(matching_algorithms) != 1:
        raise ControlledIlluminationPurePythonRunnerError(
            "Exactly one algorithm configuration must "
            f"match: {algorithm_name}"
        )

    return matching_algorithms[0]


def validate_context_against_configuration(
    context: ControlledIlluminationRunnerContext,
    benchmark_config: dict[str, Any],
    realtime_config: RealtimeConfig,
) -> None:
    planned_run = context.planned_run

    supported_resolutions = set(
        load_resolutions(benchmark_config)
    )
    selected_resolution = (
        planned_run.resolution.width,
        planned_run.resolution.height,
    )

    if selected_resolution not in supported_resolutions:
        raise ControlledIlluminationPurePythonRunnerError(
            "The planned resolution is not present in "
            "the benchmark configuration: "
            f"{selected_resolution[0]}x"
            f"{selected_resolution[1]}"
        )

    if planned_run.trial_number > realtime_config.trial_count:
        raise ControlledIlluminationPurePythonRunnerError(
            "The planned trial number exceeds the "
            "configured trial count."
        )

    if not math.isclose(
        planned_run.target_fps,
        realtime_config.target_fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ControlledIlluminationPurePythonRunnerError(
            "The planned target FPS does not match "
            "the real-time configuration."
        )

    if not math.isclose(
        planned_run.frame_deadline_ms,
        realtime_config.deadline_ms,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ControlledIlluminationPurePythonRunnerError(
            "The planned frame deadline does not match "
            "the real-time configuration."
        )


def execute_pure_python_run(
    environment: Mapping[str, str] | None = None,
    *,
    now_provider: Callable[[], str] = (
        current_utc_timestamp
    ),
) -> tuple:
    context = load_runner_context_from_environment(
        environment,
        expected_architecture=(
            PURE_PYTHON_ARCHITECTURE
        ),
    )
    planned_run = context.planned_run

    benchmark_config = load_benchmark_config()
    realtime_config = load_realtime_config()

    experiment_config = load_experiment_config(
        environment
    )

    camera_control_profile = (
        load_camera_control_profile(
            experiment_config
        )
    )

    camera_controls = (
        create_camera_control_requests(
            camera_control_profile
        )
    )

    effective_camera_controls: tuple[
        CameraControlResult,
        ...,
    ] = ()

    def report_camera_controls(
            results: tuple[
                CameraControlResult,
                ...,
            ],
    ) -> None:
        nonlocal effective_camera_controls
        effective_camera_controls = results

    quality_capture_config = (
        load_quality_capture_config(
            experiment_config,
            measured_frames=(
                realtime_config.measured_frames
            ),
        )
    )

    quality_capture_buffer = QualityCaptureBuffer(
        quality_capture_config
    )

    validate_shared_execution_counts(
        benchmark_config,
        realtime_config,
    )
    validate_context_against_configuration(
        context,
        benchmark_config,
        realtime_config,
    )

    algorithm_config = (
        select_algorithm_configuration(
            benchmark_config,
            planned_run.algorithm,
        )
    )
    frame_processor = create_frame_processor(
        algorithm_config
    )
    frame_source = create_frame_source(
        benchmark_config,
        width=planned_run.resolution.width,
        height=planned_run.resolution.height,
        fps=realtime_config.target_fps,
        camera_controls=camera_controls,
        camera_controls_reporter=(
            report_camera_controls
        ),
        environment=environment,
    )

    started_at_utc = now_provider()

    frame_records = run_realtime_trial(
        frame_source=frame_source,
        processor=frame_processor,
        config=realtime_config,
        architecture=planned_run.architecture,
        algorithm=planned_run.algorithm,
        width=planned_run.resolution.width,
        height=planned_run.resolution.height,
        trial=planned_run.trial_number,
        frame_capture_callback=(
            quality_capture_buffer.capture
            if quality_capture_config.enabled
            else None
        ),
    )

    camera_controls_metadata = (
        camera_control_results_to_metadata(
            effective_camera_controls
        )
    )

    if (
            quality_capture_config.enabled
            and quality_capture_buffer.missing_indices
    ):
        raise ControlledIlluminationPurePythonRunnerError(
            "Configured quality samples were not "
            "captured. Missing measured frame indices: "
            f"{list(quality_capture_buffer.missing_indices)}"
        )

    finished_at_utc = now_provider()

    quality_artifacts_written = False

    if quality_capture_config.enabled:
        write_quality_capture_artifacts_atomic(
            output_directory=(
                context.output_directory
            ),
            experiment_id=context.experiment_id,
            run_id=context.run_id,
            algorithm=planned_run.algorithm,
            width=planned_run.resolution.width,
            height=planned_run.resolution.height,
            config=quality_capture_config,
            samples=quality_capture_buffer.samples,
        )

        quality_artifacts_written = True

    try:
        artifact_paths = (
            write_completed_run_artifacts_atomic(
                context,
                frame_records,
                started_at_utc=started_at_utc,
                finished_at_utc=finished_at_utc,
                warmup_frame_count=(
                    realtime_config.warmup_frames
                ),
                camera_controls=(
                    camera_controls_metadata
                ),
            )
        )
    except Exception:
        if quality_artifacts_written:
            cleanup_quality_capture_artifacts(
                context.output_directory
            )

        raise

    return artifact_paths


def run_cli() -> int:
    try:
        frame_results_path, summary_path = (
            execute_pure_python_run()
        )
    except Exception as error:
        print(
            "Pure Python controlled-illumination "
            f"run failed: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        "Pure Python controlled-illumination "
        "run completed."
    )
    print(
        f"Frame results: {frame_results_path}"
    )
    print(
        f"Execution summary: {summary_path}"
    )

    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()