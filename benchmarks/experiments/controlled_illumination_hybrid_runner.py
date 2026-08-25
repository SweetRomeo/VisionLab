from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any

from benchmarks.experiments.controlled_illumination_run_artifacts import (
    write_completed_run_artifacts_atomic,
)
from benchmarks.experiments.controlled_illumination_runner_context import (
    ControlledIlluminationRunnerContext,
    load_runner_context_from_environment,
)
from benchmarks.realtime.hybrid_realtime import (
    create_frame_processor,
    load_hybrid_module,
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
    iter_video_frames,
    run_realtime_trial,
)


HYBRID_ARCHITECTURE = "hybrid"


class ControlledIlluminationHybridRunnerError(
    RuntimeError
):
    """Raised when the Hybrid runner cannot execute."""


def current_utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
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
        raise ControlledIlluminationHybridRunnerError(
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
        raise ControlledIlluminationHybridRunnerError(
            "The planned resolution is not present in "
            "the benchmark configuration: "
            f"{selected_resolution[0]}x"
            f"{selected_resolution[1]}"
        )

    if planned_run.trial_number > realtime_config.trial_count:
        raise ControlledIlluminationHybridRunnerError(
            "The planned trial number exceeds the "
            "configured trial count."
        )

    if not math.isclose(
        planned_run.target_fps,
        realtime_config.target_fps,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ControlledIlluminationHybridRunnerError(
            "The planned target FPS does not match "
            "the real-time configuration."
        )

    if not math.isclose(
        planned_run.frame_deadline_ms,
        realtime_config.deadline_ms,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ControlledIlluminationHybridRunnerError(
            "The planned frame deadline does not match "
            "the real-time configuration."
        )


def execute_hybrid_run(
    environment: Mapping[str, str] | None = None,
    *,
    now_provider: Callable[[], str] = (
        current_utc_timestamp
    ),
) -> tuple[Path, Path]:
    context = load_runner_context_from_environment(
        environment,
        expected_architecture=HYBRID_ARCHITECTURE,
    )
    planned_run = context.planned_run

    benchmark_config = load_benchmark_config()
    realtime_config = load_realtime_config()

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

    hybrid_module = load_hybrid_module()
    frame_processor = create_frame_processor(
        algorithm_config,
        hybrid_module,
    )
    video_path = resolve_video_path(
        benchmark_config
    )

    started_at_utc = now_provider()

    frame_records = run_realtime_trial(
        frame_source=iter_video_frames(
            video_path
        ),
        processor=frame_processor,
        config=realtime_config,
        architecture=planned_run.architecture,
        algorithm=planned_run.algorithm,
        width=planned_run.resolution.width,
        height=planned_run.resolution.height,
        trial=planned_run.trial_number,
    )

    finished_at_utc = now_provider()

    return write_completed_run_artifacts_atomic(
        context,
        frame_records,
        started_at_utc=started_at_utc,
        finished_at_utc=finished_at_utc,
        warmup_frame_count=(
            realtime_config.warmup_frames
        ),
    )


def run_cli() -> int:
    try:
        frame_results_path, summary_path = (
            execute_hybrid_run()
        )
    except Exception as error:
        print(
            "Hybrid controlled-illumination "
            f"run failed: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        "Hybrid controlled-illumination "
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