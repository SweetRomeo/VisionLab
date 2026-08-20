import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PURE_PYTHON_DIRECTORY = (
    PROJECT_ROOT / "pure-python"
)

for import_directory in (
    PROJECT_ROOT,
    PURE_PYTHON_DIRECTORY,
):
    import_path = str(import_directory)

    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from benchmarks.realtime.realtime_config import (  # noqa: E402
    RealtimeConfig,
    load_realtime_config,
)
from benchmarks.realtime.realtime_experiment_plan import (  # noqa: E402
    load_algorithms,
    load_benchmark_config,
    load_resolutions,
    require_positive_integer,
    require_positive_number,
    resolve_video_path,
    validate_shared_execution_counts,
)
from benchmarks.realtime.realtime_pipeline import (  # noqa: E402
    iter_video_frames,
    run_realtime_trial,
)
from benchmarks.realtime.realtime_records import (  # noqa: E402
    RealtimeFrameRecord,
    write_frame_records,
)
from image_process import (  # noqa: E402
    ImageProcess,
    ProcessingAlgorithm,
    ProcessingParameters,
)


FrameProcessor = Callable[
    [np.ndarray],
    np.ndarray,
]


ALGORITHM_MAP = {
    "original": ProcessingAlgorithm.ORIGINAL,
    "gamma_correction": (
        ProcessingAlgorithm.GAMMA
    ),
    "histogram_equalization": (
        ProcessingAlgorithm.HISTOGRAM
    ),
    "clahe": ProcessingAlgorithm.CLAHE,
}


def create_frame_processor(
    algorithm_config: dict[str, Any],
) -> FrameProcessor:
    algorithm_name = algorithm_config["name"]
    parameters = algorithm_config.get(
        "parameters",
        {},
    )

    if algorithm_name not in ALGORITHM_MAP:
        raise ValueError(
            "Unsupported algorithm: "
            f"{algorithm_name}"
        )

    if not isinstance(parameters, dict):
        raise ValueError(
            "Algorithm parameters must "
            "be a JSON object."
        )

    gamma_value = require_positive_number(
        parameters,
        "gamma_value",
        0.6,
    )
    clahe_clip_limit = (
        require_positive_number(
            parameters,
            "clip_limit",
            4.0,
        )
    )
    clahe_grid_size = (
        require_positive_integer(
            parameters,
            "grid_size",
            8,
        )
    )

    processing_parameters = (
        ProcessingParameters(
            gamma_value=gamma_value,
            clahe_clip_limit=(
                clahe_clip_limit
            ),
            clahe_grid_size=(
                clahe_grid_size
            ),
        )
    )
    processing_algorithm = ALGORITHM_MAP[
        algorithm_name
    ]
    image_processor = ImageProcess()

    def process_frame(
        frame: np.ndarray,
    ) -> np.ndarray:
        return image_processor.process(
            frame,
            processing_algorithm,
            processing_parameters,
        )

    return process_frame


def run_all_experiments(
    benchmark_config: dict[str, Any],
    realtime_config: RealtimeConfig,
) -> list[RealtimeFrameRecord]:
    validate_shared_execution_counts(
        benchmark_config,
        realtime_config,
    )

    video_path = resolve_video_path(
        benchmark_config,
        project_root=PROJECT_ROOT,
    )
    resolutions = load_resolutions(
        benchmark_config
    )
    algorithms = load_algorithms(
        benchmark_config
    )

    all_records = []

    for width, height in resolutions:
        resolution_name = (
            f"{width}x{height}"
        )

        for algorithm_config in algorithms:
            algorithm_name = (
                algorithm_config["name"]
            )

            for trial in range(
                1,
                realtime_config.trial_count + 1,
            ):
                print(
                    "Pure Python real-time | "
                    f"{algorithm_name} | "
                    f"{resolution_name} | "
                    f"trial {trial}/"
                    f"{realtime_config.trial_count}"
                )

                processor = (
                    create_frame_processor(
                        algorithm_config
                    )
                )

                trial_records = (
                    run_realtime_trial(
                        frame_source=(
                            iter_video_frames(
                                video_path
                            )
                        ),
                        processor=processor,
                        config=realtime_config,
                        architecture=(
                            "pure_python"
                        ),
                        algorithm=(
                            algorithm_name
                        ),
                        width=width,
                        height=height,
                        trial=trial,
                    )
                )

                all_records.extend(
                    trial_records
                )

    return all_records


def main() -> None:
    benchmark_config = (
        load_benchmark_config()
    )
    realtime_config = (
        load_realtime_config()
    )

    records = run_all_experiments(
        benchmark_config,
        realtime_config,
    )

    output_path = (
        realtime_config.output_directory
        / "pure_python"
        / realtime_config.frame_results_file
    )

    written_record_count = (
        write_frame_records(
            records,
            output_path,
        )
    )

    status_counts = Counter(
        record.frame_status.value
        for record in records
    )

    print(
        "Pure Python real-time evaluation "
        "completed."
    )
    print(
        f"Frame records: "
        f"{written_record_count}"
    )
    print(
        f"Processed: "
        f"{status_counts['processed']}"
    )
    print(
        f"Dropped: "
        f"{status_counts['dropped']}"
    )
    print(
        f"Skipped: "
        f"{status_counts['skipped']}"
    )
    print(
        f"Results: {output_path}"
    )


if __name__ == "__main__":
    main()