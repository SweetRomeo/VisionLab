import csv
import json
import os
import sys
from pathlib import Path
from time import perf_counter_ns

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYBRID_PROJECT_DIR = PROJECT_ROOT / "hybrid-python-cpp"
CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

DLL_DIRECTORY_HANDLES = []


def is_release_candidate(
    candidate: Path,
    build_directory: Path,
) -> bool:
    if "release" in str(candidate.parent).lower():
        return True

    cache_path = build_directory / "CMakeCache.txt"

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
                return (
                    line.removeprefix(
                        "CMAKE_BUILD_TYPE:STRING="
                    ).strip().lower()
                    == "release"
                )

    return False


def find_module_directory() -> Path:
    configured_directory = os.getenv(
        "VISIONLAB_CPP_MODULE_DIR"
    )

    if configured_directory:
        module_directory = Path(
            configured_directory
        ).resolve()

        if not module_directory.is_dir():
            raise FileNotFoundError(
                "VISIONLAB_CPP_MODULE_DIR geçerli değil: "
                f"{module_directory}"
            )

        module_candidates = []

        for pattern in (
                "visionlab_cpp*.pyd",
                "visionlab_cpp*.so",
        ):
            module_candidates.extend(
                candidate
                for candidate in module_directory.glob(
                    pattern
                )
                if candidate.is_file()
            )

        if not module_candidates:
            raise FileNotFoundError(
                "Belirtilen dizinde visionlab_cpp "
                "modülü bulunamadı: "
                f"{module_directory}"
            )

        if not any(
                is_release_candidate(
                    candidate,
                    module_directory,
                )
                for candidate in module_candidates
        ):
            raise RuntimeError(
                "VISIONLAB_CPP_MODULE_DIR içindeki "
                "visionlab_cpp modülü Release build değil: "
                f"{module_directory}"
            )

        return module_directory

    build_directory = HYBRID_PROJECT_DIR / "build"

    candidates = []

    for pattern in (
        "visionlab_cpp*.pyd",
        "visionlab_cpp*.so",
    ):
        candidates.extend(
            build_directory.rglob(pattern)
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
            "Release modundaki visionlab_cpp modülü "
            "bulunamadı. Önce hybrid projeyi Release "
            "modunda derleyin veya "
            "VISIONLAB_CPP_MODULE_DIR değişkenini ayarlayın."
        )

    newest_module = max(
        release_candidates,
        key=lambda path: path.stat().st_mtime,
    )

    return newest_module.parent


MODULE_DIRECTORY = find_module_directory()
sys.path.insert(0, str(MODULE_DIRECTORY))

if os.name == "nt" and hasattr(
    os,
    "add_dll_directory",
):
    DLL_DIRECTORY_HANDLES.append(
        os.add_dll_directory(
            str(MODULE_DIRECTORY)
        )
    )

import visionlab_cpp  # noqa: E402


ALGORITHM_MAP = {
    "original": (
        visionlab_cpp.ProcessingAlgorithm.ORIGINAL
    ),
    "gamma_correction": (
        visionlab_cpp.ProcessingAlgorithm.GAMMA
    ),
    "histogram_equalization": (
        visionlab_cpp.ProcessingAlgorithm.HISTOGRAM
    ),
    "clahe": (
        visionlab_cpp.ProcessingAlgorithm.CLAHE
    ),
}


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return json.load(config_file)


def read_frame(
    capture: cv2.VideoCapture,
):
    frame_received, frame = capture.read()

    if frame_received:
        return frame

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_received, frame = capture.read()

    if not frame_received:
        raise RuntimeError(
            "Benchmark videosundan kare okunamadı."
        )

    return frame


def run_test_case(
    writer: csv.DictWriter,
    video_path: Path,
    algorithm_name: str,
    algorithm,
    gamma_value: float,
    clahe_clip_limit: float,
    clahe_grid_size: int,
    width: int,
    height: int,
    trial: int,
    warmup_frames: int,
    measured_frames: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Benchmark videosu açılamadı: {video_path}"
        )

    try:
        for _ in range(warmup_frames):
            frame = read_frame(capture)
            resized_frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            visionlab_cpp.process_frame(
                resized_frame,
                algorithm,
                gamma_value=gamma_value,
                clahe_clip_limit=clahe_clip_limit,
                clahe_grid_size=clahe_grid_size,
            )

        for frame_index in range(1, measured_frames + 1):
            frame = read_frame(capture)
            resized_frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            processed_frame = None

            start_time = perf_counter_ns()

            processed_frame = visionlab_cpp.process_frame(
                resized_frame,
                algorithm,
                gamma_value=gamma_value,
                clahe_clip_limit=clahe_clip_limit,
                clahe_grid_size=clahe_grid_size,
            )

            process_time_ms = (
                perf_counter_ns() - start_time
            ) / 1_000_000.0

            if processed_frame.size == 0:
                raise RuntimeError(
                    "C++ modülü boş görüntü üretti."
                )

            writer.writerow(
                {
                    "architecture": "hybrid",
                    "algorithm": algorithm_name,
                    "resolution": f"{width}x{height}",
                    "trial": trial,
                    "frame_index": frame_index,
                    "processing_time_ms": (
                        f"{process_time_ms:.6f}"
                    ),
                }
            )
    finally:
        capture.release()


def main() -> None:
    config = load_config()

    video_path = (
        PROJECT_ROOT
        / config["input"]["video_path"]
    ).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            "Benchmark videosu bulunamadı: "
            f"{video_path}"
        )

    benchmark_config = config["benchmark"]
    warmup_frames = int(
        benchmark_config["warmup_frames"]
    )
    measured_frames = int(
        benchmark_config["measured_frames"]
    )
    trials = int(benchmark_config["trials"])

    output_directory = (
        PROJECT_ROOT
        / config["output"]["directory"]
    ).resolve()
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "hybrid_results.csv"
    )

    fieldnames = [
        "architecture",
        "algorithm",
        "resolution",
        "trial",
        "frame_index",
        "processing_time_ms",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for resolution in config["resolutions"]:
            width = int(resolution["width"])
            height = int(resolution["height"])

            for algorithm_config in config["algorithms"]:
                algorithm_name = algorithm_config["name"]

                if algorithm_name not in ALGORITHM_MAP:
                    raise ValueError(
                        "Desteklenmeyen algoritma: "
                        f"{algorithm_name}"
                    )

                parameters = algorithm_config.get(
                    "parameters",
                    {},
                )

                gamma_value = float(
                    parameters.get(
                        "gamma_value",
                        0.6,
                    )
                )
                clahe_clip_limit = float(
                    parameters.get(
                        "clip_limit",
                        4.0,
                    )
                )
                clahe_grid_size = int(
                    parameters.get(
                        "grid_size",
                        8,
                    )
                )

                for trial in range(1, trials + 1):
                    print(
                        f"Hybrid | {algorithm_name} | "
                        f"{width}x{height} | "
                        f"trial {trial}/{trials}"
                    )

                    run_test_case(
                        writer=writer,
                        video_path=video_path,
                        algorithm_name=algorithm_name,
                        algorithm=(
                            ALGORITHM_MAP[algorithm_name]
                        ),
                        gamma_value=gamma_value,
                        clahe_clip_limit=(
                            clahe_clip_limit
                        ),
                        clahe_grid_size=clahe_grid_size,
                        width=width,
                        height=height,
                        trial=trial,
                        warmup_frames=warmup_frames,
                        measured_frames=measured_frames,
                    )

                    output_file.flush()

    print(f"Benchmark tamamlandı: {output_path}")


if __name__ == "__main__":
    main()