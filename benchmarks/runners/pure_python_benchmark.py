import csv
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PURE_PYTHON_DIR = PROJECT_ROOT / "pure-python"
CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

sys.path.insert(0, str(PURE_PYTHON_DIR))

from image_process import (  # noqa: E402
    ImageProcess,
    ProcessingAlgorithm,
    ProcessingParameters,
)


ALGORITHM_MAP = {
    "original": ProcessingAlgorithm.ORIGINAL,
    "gamma_correction": ProcessingAlgorithm.GAMMA,
    "histogram_equalization": ProcessingAlgorithm.HISTOGRAM,
    "clahe": ProcessingAlgorithm.CLAHE,
}


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return json.load(config_file)


def create_parameters(
    algorithm_config: dict,
) -> ProcessingParameters:
    parameters = algorithm_config.get("parameters", {})

    return ProcessingParameters(
        gamma_value=float(
            parameters.get("gamma_value", 0.6)
        ),
        clahe_clip_limit=float(
            parameters.get("clip_limit", 4.0)
        ),
        clahe_grid_size=int(
            parameters.get("grid_size", 8)
        ),
    )


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
    processor: ImageProcess,
    video_path: Path,
    algorithm_name: str,
    algorithm: ProcessingAlgorithm,
    parameters: ProcessingParameters,
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

            processor.process(
                resized_frame,
                algorithm,
                parameters,
            )

        for frame_index in range(1, measured_frames + 1):
            frame = read_frame(capture)
            resized_frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            start_time = perf_counter_ns()

            processed_frame = processor.process(
                resized_frame,
                algorithm,
                parameters,
            )

            process_time_ms = (
                perf_counter_ns() - start_time
            ) / 1_000_000.0

            if processed_frame.size == 0:
                raise RuntimeError(
                    "Algoritma boş görüntü üretti."
                )

            writer.writerow(
                {
                    "architecture": "pure_python",
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
        / "pure_python_results.csv"
    )

    processor = ImageProcess()

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

                algorithm = ALGORITHM_MAP[algorithm_name]
                parameters = create_parameters(
                    algorithm_config
                )

                for trial in range(1, trials + 1):
                    print(
                        f"Pure Python | {algorithm_name} | "
                        f"{width}x{height} | "
                        f"trial {trial}/{trials}"
                    )

                    run_test_case(
                        writer=writer,
                        processor=processor,
                        video_path=video_path,
                        algorithm_name=algorithm_name,
                        algorithm=algorithm,
                        parameters=parameters,
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