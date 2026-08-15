import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, pstdev


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = PROJECT_ROOT / "benchmarks" / "results"

CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

RESULT_FILES = [
    "pure_python_results.csv",
    "hybrid_results.csv",
    "pure_cpp_results.csv",
]


@dataclass(frozen=True)
class Measurement:
    architecture: str
    algorithm: str
    resolution: str
    trial: int
    frame_index: int
    processing_time_ms: float

def load_benchmark_expectations() -> tuple[
    int,
    int,
    set[str],
    set[str],
]:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    measured_frames = int(
        config["benchmark"]["measured_frames"]
    )
    trials = int(config["benchmark"]["trials"])

    algorithms = {
        algorithm["name"]
        for algorithm in config["algorithms"]
    }

    resolutions = {
        f'{resolution["width"]}x{resolution["height"]}'
        for resolution in config["resolutions"]
    }

    if measured_frames <= 0 or trials <= 0:
        raise ValueError(
            "Measured frame and trial counts must be positive."
        )

    return (
        measured_frames,
        trials,
        algorithms,
        resolutions,
    )


def validate_measurements(
    measurements: list[Measurement],
) -> None:
    (
        measured_frames,
        trial_count,
        algorithms,
        resolutions,
    ) = load_benchmark_expectations()

    architectures = set(RESULT_FILES.values())
    expected_trials = set(range(1, trial_count + 1))
    expected_frames = set(
        range(1, measured_frames + 1)
    )

    groups = defaultdict(list)

    for measurement in measurements:
        if measurement.architecture not in architectures:
            raise ValueError(
                "Unexpected architecture: "
                f"{measurement.architecture}"
            )

        if measurement.algorithm not in algorithms:
            raise ValueError(
                "Unexpected algorithm: "
                f"{measurement.algorithm}"
            )

        if measurement.resolution not in resolutions:
            raise ValueError(
                "Unexpected resolution: "
                f"{measurement.resolution}"
            )

        if measurement.trial not in expected_trials:
            raise ValueError(
                "Unexpected trial number: "
                f"{measurement.trial}"
            )

        key = (
            measurement.architecture,
            measurement.algorithm,
            measurement.resolution,
            measurement.trial,
        )
        groups[key].append(measurement.frame_index)

    expected_groups = {
        (architecture, algorithm, resolution, trial)
        for architecture in architectures
        for algorithm in algorithms
        for resolution in resolutions
        for trial in expected_trials
    }

    missing_groups = expected_groups - set(groups)
    unexpected_groups = set(groups) - expected_groups

    if missing_groups:
        raise ValueError(
            "Missing benchmark groups: "
            f"{sorted(missing_groups)[:5]}"
        )

    if unexpected_groups:
        raise ValueError(
            "Unexpected benchmark groups: "
            f"{sorted(unexpected_groups)[:5]}"
        )

    for key, frame_indices in groups.items():
        unique_frames = set(frame_indices)

        if len(unique_frames) != len(frame_indices):
            raise ValueError(
                f"Duplicate frame indices in group {key}."
            )

        if unique_frames != expected_frames:
            missing_frames = (
                expected_frames - unique_frames
            )
            extra_frames = (
                unique_frames - expected_frames
            )

            raise ValueError(
                f"Invalid frames in group {key}. "
                f"Missing: {sorted(missing_frames)[:5]}, "
                f"extra: {sorted(extra_frames)[:5]}"
            )


def calculate_percentile(
    values: list[float],
    fraction: float,
) -> float:
    ordered_values = sorted(values)

    if len(ordered_values) == 1:
        return ordered_values[0]

    position = (len(ordered_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered_values[lower_index]

    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]
    weight = position - lower_index

    return lower_value + (
        upper_value - lower_value
    ) * weight


def calculate_statistics(
    values: list[float],
) -> dict[str, float]:
    mean_ms = fmean(values)

    return {
        "mean_ms": mean_ms,
        "median_ms": median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": pstdev(values),
        "p95_ms": calculate_percentile(
            values,
            0.95,
        ),
        "effective_fps": (
            1000.0 / mean_ms
            if mean_ms > 0.0
            else 0.0
        ),
    }


def load_measurements() -> list[Measurement]:
    measurements = []

    required_fields = {
        "architecture",
        "algorithm",
        "resolution",
        "trial",
        "frame_index",
        "processing_time_ms",
    }

    for file_name, expected_architecture in RESULT_FILES.items():
        result_path = RESULTS_DIRECTORY / file_name

        if not result_path.is_file():
            raise FileNotFoundError(
                f"Required result file not found: {result_path}"
            )

        with result_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as result_file:
            reader = csv.DictReader(result_file)

            if reader.fieldnames is None:
                raise ValueError(
                    f"CSV header missing: {result_path}"
                )

            missing_fields = (
                required_fields - set(reader.fieldnames)
            )

            if missing_fields:
                raise ValueError(
                    f"Missing CSV fields in {result_path}: "
                    f"{sorted(missing_fields)}"
                )

            for row in reader:
                architecture = row["architecture"].strip()

                if architecture != expected_architecture:
                    raise ValueError(
                        f"Unexpected architecture in {result_path}: "
                        f"expected {expected_architecture}, "
                        f"received {architecture}"
                    )
                processing_time_ms = float(
                    row["processing_time_ms"]
                )

                if (
                    not math.isfinite(processing_time_ms)
                    or processing_time_ms < 0.0
                ):
                    raise ValueError(
                        "Invalid processing time in "
                        f"{result_path}: "
                        f"{processing_time_ms}"
                    )

                measurements.append(
                    Measurement(
                        architecture=architecture,
                        algorithm=row["algorithm"],
                        resolution=row["resolution"],
                        trial=int(row["trial"]),
                        frame_index=int(row["frame_index"]),
                        processing_time_ms=(
                            processing_time_ms
                        ),
                    )
                )

    if not measurements:
        raise FileNotFoundError(
            "No benchmark result files were found."
        )

    return measurements


def write_trial_summary(
    measurements: list[Measurement],
) -> Path:
    groups = defaultdict(list)

    for measurement in measurements:
        key = (
            measurement.architecture,
            measurement.algorithm,
            measurement.resolution,
            measurement.trial,
        )

        groups[key].append(
            measurement.processing_time_ms
        )

    output_path = (
        RESULTS_DIRECTORY
        / "benchmark_trial_summary.csv"
    )

    fieldnames = [
        "architecture",
        "algorithm",
        "resolution",
        "trial",
        "frame_count",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "p95_ms",
        "effective_fps",
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

        for key in sorted(groups):
            architecture, algorithm, resolution, trial = key
            values = groups[key]
            statistics = calculate_statistics(values)

            writer.writerow(
                {
                    "architecture": architecture,
                    "algorithm": algorithm,
                    "resolution": resolution,
                    "trial": trial,
                    "frame_count": len(values),
                    **{
                        name: f"{value:.6f}"
                        for name, value
                        in statistics.items()
                    },
                }
            )

    return output_path


def write_overall_summary(
    measurements: list[Measurement],
) -> Path:
    groups = defaultdict(list)
    trial_numbers = defaultdict(set)

    for measurement in measurements:
        key = (
            measurement.architecture,
            measurement.algorithm,
            measurement.resolution,
        )

        groups[key].append(
            measurement.processing_time_ms
        )
        trial_numbers[key].add(
            measurement.trial
        )

    output_path = (
        RESULTS_DIRECTORY
        / "benchmark_summary.csv"
    )

    fieldnames = [
        "architecture",
        "algorithm",
        "resolution",
        "trial_count",
        "frame_count",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "p95_ms",
        "effective_fps",
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

        for key in sorted(groups):
            architecture, algorithm, resolution = key
            values = groups[key]
            statistics = calculate_statistics(values)

            writer.writerow(
                {
                    "architecture": architecture,
                    "algorithm": algorithm,
                    "resolution": resolution,
                    "trial_count": len(
                        trial_numbers[key]
                    ),
                    "frame_count": len(values),
                    **{
                        name: f"{value:.6f}"
                        for name, value
                        in statistics.items()
                    },
                }
            )

    return output_path


def main() -> None:
    measurements = load_measurements()
    validate_measurements(measurements)

    trial_summary_path = write_trial_summary(
        measurements
    )
    overall_summary_path = write_overall_summary(
        measurements
    )

    print(
        f"Trial summary created: {trial_summary_path}"
    )
    print(
        f"Overall summary created: "
        f"{overall_summary_path}"
    )


if __name__ == "__main__":
    main()