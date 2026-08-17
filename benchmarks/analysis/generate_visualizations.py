import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

from scipy.stats import t as student_t

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "config"
    / "benchmark_config.json"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
)

TRIAL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "benchmark_trial_summary.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "benchmark_summary.csv"
)

RESOURCE_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "benchmark_resource_summary.csv"
)

COMPARISON_OUTPUT_PATH = (
    RESULTS_DIRECTORY
    / "benchmark_comparison.csv"
)

EXPECTED_ARCHITECTURES = (
    "pure_python",
    "hybrid",
    "pure_cpp",
)


@dataclass(frozen=True)
class TrialSummary:
    architecture: str
    algorithm: str
    resolution: str
    trial: int
    frame_count: int
    mean_ms: float
    p95_ms: float
    effective_fps: float


@dataclass(frozen=True)
class OverallSummary:
    architecture: str
    algorithm: str
    resolution: str
    trial_count: int
    frame_count: int
    mean_ms: float
    p95_ms: float
    effective_fps: float


@dataclass(frozen=True)
class ResourceSummary:
    architecture: str
    wall_time_seconds: float
    cpu_time_seconds: float
    average_cpu_percent: float
    peak_rss_mib: float
    sample_count: int
    sampling_interval_seconds: float
    exit_code: int

@dataclass(frozen=True)
class ComparisonSummary:
    architecture: str
    algorithm: str
    resolution: str
    trial_count: int
    mean_processing_time_ms: float
    confidence_interval_95_lower_ms: float
    confidence_interval_95_upper_ms: float
    effective_fps: float
    speedup_vs_pure_python: float
    overhead_vs_pure_cpp_percent: float

def read_csv_rows(
    input_path: Path,
    required_fields: set[str],
) -> list[dict[str, str]]:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Required result file was not found: "
            f"{input_path}"
        )

    with input_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV header is missing: {input_path}"
            )

        missing_fields = (
            required_fields - set(reader.fieldnames)
        )

        if missing_fields:
            raise ValueError(
                f"Missing fields in {input_path}: "
                f"{sorted(missing_fields)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"CSV file contains no data rows: {input_path}"
        )

    return rows


def parse_non_negative_float(
    row: dict[str, str],
    field_name: str,
    input_path: Path,
) -> float:
    try:
        value = float(row[field_name])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {field_name} in {input_path}: "
            f"{row.get(field_name)!r}"
        ) from error

    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"Invalid {field_name} in {input_path}: "
            f"{value}"
        )

    return value


def parse_positive_integer(
    row: dict[str, str],
    field_name: str,
    input_path: Path,
) -> int:
    try:
        value = int(row[field_name])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {field_name} in {input_path}: "
            f"{row.get(field_name)!r}"
        ) from error

    if value <= 0:
        raise ValueError(
            f"{field_name} must be positive in "
            f"{input_path}: {value}"
        )

    return value


def validate_architecture(
    architecture: str,
    input_path: Path,
) -> None:
    if architecture not in EXPECTED_ARCHITECTURES:
        raise ValueError(
            f"Unexpected architecture in {input_path}: "
            f"{architecture!r}"
        )


def load_expected_experiment(
) -> tuple[set[tuple[str, str]], set[int]]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Benchmark configuration was not found: "
            f"{CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = json.load(config_file)

    algorithms = {
        str(algorithm["name"])
        for algorithm in config["algorithms"]
    }

    resolutions = {
        (
            f"{int(resolution['width'])}"
            f"x{int(resolution['height'])}"
        )
        for resolution in config["resolutions"]
    }

    trial_count = int(
        config["benchmark"]["trials"]
    )

    if not algorithms or not resolutions:
        raise ValueError(
            "Benchmark configuration contains no "
            "algorithms or resolutions."
        )

    if trial_count <= 0:
        raise ValueError(
            "Benchmark trial count must be positive."
        )

    expected_groups = {
        (algorithm, resolution)
        for algorithm in algorithms
        for resolution in resolutions
    }

    expected_trials = set(
        range(1, trial_count + 1)
    )

    return expected_groups, expected_trials


def load_trial_summaries() -> list[TrialSummary]:
    required_fields = {
        "architecture",
        "algorithm",
        "resolution",
        "trial",
        "frame_count",
        "mean_ms",
        "p95_ms",
        "effective_fps",
    }

    rows = read_csv_rows(
        TRIAL_SUMMARY_PATH,
        required_fields,
    )

    summaries = []
    observed_keys = set()

    for row in rows:
        architecture = row["architecture"]
        validate_architecture(
            architecture,
            TRIAL_SUMMARY_PATH,
        )

        summary = TrialSummary(
            architecture=architecture,
            algorithm=row["algorithm"],
            resolution=row["resolution"],
            trial=parse_positive_integer(
                row,
                "trial",
                TRIAL_SUMMARY_PATH,
            ),
            frame_count=parse_positive_integer(
                row,
                "frame_count",
                TRIAL_SUMMARY_PATH,
            ),
            mean_ms=parse_non_negative_float(
                row,
                "mean_ms",
                TRIAL_SUMMARY_PATH,
            ),
            p95_ms=parse_non_negative_float(
                row,
                "p95_ms",
                TRIAL_SUMMARY_PATH,
            ),
            effective_fps=parse_non_negative_float(
                row,
                "effective_fps",
                TRIAL_SUMMARY_PATH,
            ),
        )

        key = (
            summary.architecture,
            summary.algorithm,
            summary.resolution,
            summary.trial,
        )

        if key in observed_keys:
            raise ValueError(
                f"Duplicate trial summary group: {key}"
            )

        observed_keys.add(key)
        summaries.append(summary)

    return summaries


def load_overall_summaries(
) -> list[OverallSummary]:
    required_fields = {
        "architecture",
        "algorithm",
        "resolution",
        "trial_count",
        "frame_count",
        "mean_ms",
        "p95_ms",
        "effective_fps",
    }

    rows = read_csv_rows(
        OVERALL_SUMMARY_PATH,
        required_fields,
    )

    summaries = []
    observed_keys = set()

    for row in rows:
        architecture = row["architecture"]
        validate_architecture(
            architecture,
            OVERALL_SUMMARY_PATH,
        )

        summary = OverallSummary(
            architecture=architecture,
            algorithm=row["algorithm"],
            resolution=row["resolution"],
            trial_count=parse_positive_integer(
                row,
                "trial_count",
                OVERALL_SUMMARY_PATH,
            ),
            frame_count=parse_positive_integer(
                row,
                "frame_count",
                OVERALL_SUMMARY_PATH,
            ),
            mean_ms=parse_non_negative_float(
                row,
                "mean_ms",
                OVERALL_SUMMARY_PATH,
            ),
            p95_ms=parse_non_negative_float(
                row,
                "p95_ms",
                OVERALL_SUMMARY_PATH,
            ),
            effective_fps=parse_non_negative_float(
                row,
                "effective_fps",
                OVERALL_SUMMARY_PATH,
            ),
        )

        key = (
            summary.architecture,
            summary.algorithm,
            summary.resolution,
        )

        if key in observed_keys:
            raise ValueError(
                f"Duplicate overall summary group: {key}"
            )

        observed_keys.add(key)
        summaries.append(summary)

    return summaries


def load_resource_summaries(
) -> list[ResourceSummary]:
    required_fields = {
        "architecture",
        "wall_time_seconds",
        "cpu_time_seconds",
        "average_cpu_percent",
        "peak_rss_mib",
        "sample_count",
        "sampling_interval_seconds",
        "exit_code",
    }

    rows = read_csv_rows(
        RESOURCE_SUMMARY_PATH,
        required_fields,
    )

    summaries = []
    observed_architectures = set()

    for row in rows:
        architecture = row["architecture"]
        validate_architecture(
            architecture,
            RESOURCE_SUMMARY_PATH,
        )

        try:
            exit_code = int(row["exit_code"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Invalid exit_code in "
                f"{RESOURCE_SUMMARY_PATH}: "
                f"{row.get('exit_code')!r}"
            ) from error

        summary = ResourceSummary(
            architecture=architecture,
            wall_time_seconds=(
                parse_non_negative_float(
                    row,
                    "wall_time_seconds",
                    RESOURCE_SUMMARY_PATH,
                )
            ),
            cpu_time_seconds=parse_non_negative_float(
                row,
                "cpu_time_seconds",
                RESOURCE_SUMMARY_PATH,
            ),
            average_cpu_percent=(
                parse_non_negative_float(
                    row,
                    "average_cpu_percent",
                    RESOURCE_SUMMARY_PATH,
                )
            ),
            peak_rss_mib=parse_non_negative_float(
                row,
                "peak_rss_mib",
                RESOURCE_SUMMARY_PATH,
            ),
            sample_count=parse_positive_integer(
                row,
                "sample_count",
                RESOURCE_SUMMARY_PATH,
            ),
            sampling_interval_seconds=(
                parse_non_negative_float(
                    row,
                    "sampling_interval_seconds",
                    RESOURCE_SUMMARY_PATH,
                )
            ),
            exit_code=exit_code,
        )

        if summary.architecture in observed_architectures:
            raise ValueError(
                "Duplicate resource summary for: "
                f"{summary.architecture}"
            )

        if summary.exit_code != 0:
            raise ValueError(
                f"{summary.architecture} has a non-zero "
                f"exit code: {summary.exit_code}"
            )

        if summary.peak_rss_mib <= 0.0:
            raise ValueError(
                f"{summary.architecture} has an invalid "
                "peak RSS value."
            )

        observed_architectures.add(
            summary.architecture
        )
        summaries.append(summary)

    if observed_architectures != set(
        EXPECTED_ARCHITECTURES
    ):
        raise ValueError(
            "Resource summary architecture mismatch. "
            f"Expected {sorted(EXPECTED_ARCHITECTURES)}, "
            f"found {sorted(observed_architectures)}."
        )

    return summaries


def validate_experiment_coverage(
    trial_summaries: list[TrialSummary],
    overall_summaries: list[OverallSummary],
) -> None:
    (
        expected_groups,
        expected_trials,
    ) = load_expected_experiment()

    trials_by_group = defaultdict(set)

    for summary in trial_summaries:
        group_key = (
            summary.architecture,
            summary.algorithm,
            summary.resolution,
        )
        trials_by_group[group_key].add(
            summary.trial
        )

    overall_by_group = {
        (
            summary.architecture,
            summary.algorithm,
            summary.resolution,
        ): summary
        for summary in overall_summaries
    }

    for architecture in EXPECTED_ARCHITECTURES:
        for algorithm, resolution in expected_groups:
            group_key = (
                architecture,
                algorithm,
                resolution,
            )

            if group_key not in overall_by_group:
                raise ValueError(
                    "Missing overall summary group: "
                    f"{group_key}"
                )

            observed_trials = trials_by_group.get(
                group_key,
                set(),
            )

            if observed_trials != expected_trials:
                raise ValueError(
                    f"Trial mismatch for {group_key}. "
                    f"Expected {sorted(expected_trials)}, "
                    f"found {sorted(observed_trials)}."
                )

            overall = overall_by_group[group_key]

            if overall.trial_count != len(
                expected_trials
            ):
                raise ValueError(
                    f"Invalid trial_count for {group_key}: "
                    f"{overall.trial_count}"
                )

    expected_overall_count = (
        len(EXPECTED_ARCHITECTURES)
        * len(expected_groups)
    )

    if len(overall_summaries) != expected_overall_count:
        raise ValueError(
            "Unexpected overall summary group count. "
            f"Expected {expected_overall_count}, "
            f"found {len(overall_summaries)}."
        )

def calculate_confidence_interval(
    values: list[float],
) -> tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError(
            "At least two trial values are required "
            "for a confidence interval."
        )

    mean_value = fmean(values)
    standard_error = (
        stdev(values) / math.sqrt(len(values))
    )

    critical_value = float(
        student_t.ppf(
            0.975,
            df=len(values) - 1,
        )
    )

    if not math.isfinite(critical_value):
        raise ValueError(
            "The confidence-interval critical value "
            "is not finite."
        )

    margin = critical_value * standard_error

    return (
        mean_value,
        mean_value - margin,
        mean_value + margin,
    )


def resolution_sort_key(
    resolution: str,
) -> tuple[int, int]:
    try:
        width_text, height_text = resolution.split(
            "x",
            maxsplit=1,
        )
        return int(width_text), int(height_text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid resolution: {resolution!r}"
        ) from error


def build_comparison_summaries(
    trial_summaries: list[TrialSummary],
) -> list[ComparisonSummary]:
    values_by_group = defaultdict(list)

    for summary in trial_summaries:
        key = (
            summary.architecture,
            summary.algorithm,
            summary.resolution,
        )

        values_by_group[key].append(
            summary.mean_ms
        )

    group_means = {
        key: fmean(values)
        for key, values in values_by_group.items()
    }

    architecture_order = {
        architecture: index
        for index, architecture
        in enumerate(EXPECTED_ARCHITECTURES)
    }

    sorted_keys = sorted(
        values_by_group,
        key=lambda key: (
            key[1],
            resolution_sort_key(key[2]),
            architecture_order[key[0]],
        ),
    )

    comparisons = []

    for architecture, algorithm, resolution in sorted_keys:
        key = (
            architecture,
            algorithm,
            resolution,
        )
        trial_values = values_by_group[key]

        (
            mean_ms,
            confidence_lower_ms,
            confidence_upper_ms,
        ) = calculate_confidence_interval(
            trial_values
        )

        pure_python_key = (
            "pure_python",
            algorithm,
            resolution,
        )
        pure_cpp_key = (
            "pure_cpp",
            algorithm,
            resolution,
        )

        pure_python_mean = group_means[
            pure_python_key
        ]
        pure_cpp_mean = group_means[
            pure_cpp_key
        ]

        if (
            mean_ms <= 0.0
            or pure_python_mean <= 0.0
            or pure_cpp_mean <= 0.0
        ):
            raise ValueError(
                "Processing-time means must be "
                f"positive for {key}."
            )

        comparisons.append(
            ComparisonSummary(
                architecture=architecture,
                algorithm=algorithm,
                resolution=resolution,
                trial_count=len(trial_values),
                mean_processing_time_ms=mean_ms,
                confidence_interval_95_lower_ms=(
                    confidence_lower_ms
                ),
                confidence_interval_95_upper_ms=(
                    confidence_upper_ms
                ),
                effective_fps=1000.0 / mean_ms,
                speedup_vs_pure_python=(
                    pure_python_mean / mean_ms
                ),
                overhead_vs_pure_cpp_percent=(
                    (
                        mean_ms - pure_cpp_mean
                    )
                    / pure_cpp_mean
                    * 100.0
                ),
            )
        )

    return comparisons


def write_comparison_summaries(
    comparisons: list[ComparisonSummary],
) -> Path:
    if not comparisons:
        raise ValueError(
            "No benchmark comparisons were generated."
        )

    temporary_path = (
        COMPARISON_OUTPUT_PATH.with_suffix(
            ".csv.tmp"
        )
    )

    fieldnames = [
        "architecture",
        "algorithm",
        "resolution",
        "trial_count",
        "mean_processing_time_ms",
        "confidence_interval_95_lower_ms",
        "confidence_interval_95_upper_ms",
        "effective_fps",
        "speedup_vs_pure_python",
        "overhead_vs_pure_cpp_percent",
    ]

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for comparison in comparisons:
            writer.writerow(
                {
                    "architecture": (
                        comparison.architecture
                    ),
                    "algorithm": comparison.algorithm,
                    "resolution": (
                        comparison.resolution
                    ),
                    "trial_count": (
                        comparison.trial_count
                    ),
                    "mean_processing_time_ms": (
                        f"{comparison.mean_processing_time_ms:.6f}"
                    ),
                    "confidence_interval_95_lower_ms": (
                        f"{comparison.confidence_interval_95_lower_ms:.6f}"
                    ),
                    "confidence_interval_95_upper_ms": (
                        f"{comparison.confidence_interval_95_upper_ms:.6f}"
                    ),
                    "effective_fps": (
                        f"{comparison.effective_fps:.6f}"
                    ),
                    "speedup_vs_pure_python": (
                        f"{comparison.speedup_vs_pure_python:.6f}"
                    ),
                    "overhead_vs_pure_cpp_percent": (
                        f"{comparison.overhead_vs_pure_cpp_percent:.6f}"
                    ),
                }
            )

    temporary_path.replace(
        COMPARISON_OUTPUT_PATH
    )

    return COMPARISON_OUTPUT_PATH

def main() -> None:
    trial_summaries = load_trial_summaries()
    overall_summaries = load_overall_summaries()
    resource_summaries = load_resource_summaries()

    validate_experiment_coverage(
        trial_summaries,
        overall_summaries,
    )

    comparisons = build_comparison_summaries(
        trial_summaries
    )

    comparison_output_path = (
        write_comparison_summaries(
            comparisons
        )
    )

    print("Visualization inputs validated.")
    print(
        f"Trial summaries: {len(trial_summaries)}"
    )
    print(
        f"Overall summaries: {len(overall_summaries)}"
    )
    print(
        f"Resource summaries: {len(resource_summaries)}"
    )
    print(
        "Comparison summary created: "
        f"{comparison_output_path}"
    )

if __name__ == "__main__":
    main()