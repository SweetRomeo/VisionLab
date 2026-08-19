import importlib
import os
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HYBRID_PROJECT_DIRECTORY = (
    PROJECT_ROOT / "hybrid-python-cpp"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


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


FrameProcessor = Callable[
    [np.ndarray],
    np.ndarray,
]

DLL_DIRECTORY_HANDLES = []


def is_release_candidate(
    candidate: Path,
    build_directory: Path,
) -> bool:
    if "release" in str(
        candidate.parent
    ).lower():
        return True

    cache_path = (
        build_directory / "CMakeCache.txt"
    )

    if not cache_path.is_file():
        return False

    with cache_path.open(
        "r",
        encoding="utf-8",
    ) as cache_file:
        for line in cache_file:
            if line.startswith(
                "CMAKE_BUILD_TYPE:STRING="
            ):
                build_type = line.removeprefix(
                    "CMAKE_BUILD_TYPE:STRING="
                ).strip()

                return (
                    build_type.lower()
                    == "release"
                )

    return False


def find_hybrid_module_directory() -> Path:
    configured_directory = os.getenv(
        "VISIONLAB_CPP_MODULE_DIR"
    )

    if configured_directory:
        module_directory = Path(
            configured_directory
        ).resolve()

        if not module_directory.is_dir():
            raise FileNotFoundError(
                "VISIONLAB_CPP_MODULE_DIR is "
                f"invalid: {module_directory}"
            )

        configured_candidates = [
            candidate
            for pattern in (
                "visionlab_cpp*.pyd",
                "visionlab_cpp*.so",
            )
            for candidate in (
                module_directory.glob(pattern)
            )
            if candidate.is_file()
        ]

        if not configured_candidates:
            raise FileNotFoundError(
                "The configured directory does "
                "not contain a visionlab_cpp "
                f"module: {module_directory}"
            )

        if not any(
            is_release_candidate(
                candidate,
                module_directory,
            )
            for candidate
            in configured_candidates
        ):
            raise RuntimeError(
                "The configured visionlab_cpp "
                "module is not a Release build: "
                f"{module_directory}"
            )

        return module_directory

    build_directory = (
        HYBRID_PROJECT_DIRECTORY / "build"
    )

    candidates = []

    for pattern in (
        "visionlab_cpp*.pyd",
        "visionlab_cpp*.so",
    ):
        candidates.extend(
            candidate
            for candidate
            in build_directory.rglob(pattern)
            if candidate.is_file()
        )

    release_candidates = [
        candidate
        for candidate in candidates
        if is_release_candidate(
            candidate,
            build_directory,
        )
    ]

    if not release_candidates:
        raise FileNotFoundError(
            "The Release build of visionlab_cpp "
            "could not be found. Build the Hybrid "
            "project in Release mode or set "
            "VISIONLAB_CPP_MODULE_DIR."
        )

    newest_module = max(
        release_candidates,
        key=lambda path: path.stat().st_mtime,
    )

    return newest_module.parent


def load_hybrid_module():
    module_directory = (
        find_hybrid_module_directory()
    )

    module_directory_string = str(
        module_directory
    )

    if (
        module_directory_string
        not in sys.path
    ):
        sys.path.insert(
            0,
            module_directory_string,
        )

    if (
        os.name == "nt"
        and hasattr(
            os,
            "add_dll_directory",
        )
    ):
        DLL_DIRECTORY_HANDLES.append(
            os.add_dll_directory(
                module_directory_string
            )
        )

    hybrid_module = (
        importlib.import_module(
            "visionlab_cpp"
        )
    )

    module_file = getattr(
        hybrid_module,
        "__file__",
        None,
    )

    if module_file is None:
        raise RuntimeError(
            "The imported visionlab_cpp module "
            "does not expose its file path."
        )

    imported_directory = (
        Path(module_file)
        .resolve()
        .parent
    )

    if (
        imported_directory
        != module_directory.resolve()
    ):
        raise RuntimeError(
            "visionlab_cpp was imported from an "
            "unexpected directory. Expected: "
            f"{module_directory}; actual: "
            f"{imported_directory}."
        )

    if not hasattr(
        hybrid_module,
        "process_frame",
    ):
        raise RuntimeError(
            "visionlab_cpp does not expose "
            "process_frame."
        )

    if not hasattr(
        hybrid_module,
        "ProcessingAlgorithm",
    ):
        raise RuntimeError(
            "visionlab_cpp does not expose "
            "ProcessingAlgorithm."
        )

    return hybrid_module


def create_frame_processor(
    algorithm_config: dict[str, Any],
    hybrid_module,
) -> FrameProcessor:
    algorithm_name = algorithm_config["name"]
    parameters = algorithm_config.get(
        "parameters",
        {},
    )

    if not isinstance(parameters, dict):
        raise ValueError(
            "Algorithm parameters must "
            "be a JSON object."
        )

    algorithm_map = {
        "original": (
            hybrid_module
            .ProcessingAlgorithm.ORIGINAL
        ),
        "gamma_correction": (
            hybrid_module
            .ProcessingAlgorithm.GAMMA
        ),
        "histogram_equalization": (
            hybrid_module
            .ProcessingAlgorithm.HISTOGRAM
        ),
        "clahe": (
            hybrid_module
            .ProcessingAlgorithm.CLAHE
        ),
    }

    if algorithm_name not in algorithm_map:
        raise ValueError(
            "Unsupported algorithm: "
            f"{algorithm_name}"
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

    processing_algorithm = algorithm_map[
        algorithm_name
    ]

    def process_frame(
        frame: np.ndarray,
    ) -> np.ndarray:
        return hybrid_module.process_frame(
            frame,
            processing_algorithm,
            gamma_value=gamma_value,
            clahe_clip_limit=(
                clahe_clip_limit
            ),
            clahe_grid_size=(
                clahe_grid_size
            ),
        )

    return process_frame


def run_all_experiments(
    benchmark_config: dict[str, Any],
    realtime_config: RealtimeConfig,
    hybrid_module,
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
                    "Hybrid real-time | "
                    f"{algorithm_name} | "
                    f"{resolution_name} | "
                    f"trial {trial}/"
                    f"{realtime_config.trial_count}"
                )

                processor = (
                    create_frame_processor(
                        algorithm_config,
                        hybrid_module,
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
                        architecture="hybrid",
                        algorithm=algorithm_name,
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
    hybrid_module = load_hybrid_module()

    records = run_all_experiments(
        benchmark_config,
        realtime_config,
        hybrid_module,
    )

    output_path = (
        realtime_config.output_directory
        / "hybrid"
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
        "Hybrid real-time evaluation "
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