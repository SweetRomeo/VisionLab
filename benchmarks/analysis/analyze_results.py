import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, pstdev


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIRECTORY = PROJECT_ROOT / "benchmarks" / "results"

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
    processing_time_ms: float


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

    for file_name in RESULT_FILES:
        result_path = RESULTS_DIRECTORY / file_name

        if not result_path.is_file():
            print(
                f"Warning: result file not found: "
                f"{result_path}"
            )
            continue

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
                        architecture=row["architecture"],
                        algorithm=row["algorithm"],
                        resolution=row["resolution"],
                        trial=int(row["trial"]),
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