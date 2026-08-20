import csv
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.realtime import (
    analyze_realtime_results as analyzer,
)
from benchmarks.realtime.realtime_config import (
    RealtimeConfig,
)
from benchmarks.realtime.realtime_records import (
    RealtimeRunContext,
    create_dropped_record,
    create_processed_record,
    write_frame_records,
)


class AnalyzeRealtimeResultsTests(
    unittest.TestCase
):
    def create_config(
        self,
        output_directory: Path,
    ) -> RealtimeConfig:
        return RealtimeConfig(
            schema_version=1,
            target_fps=30.0,
            queue_capacity=1,
            warmup_frames=1,
            measured_frames=2,
            trial_count=1,
            drop_policy="latest_frame",
            deadline_multiplier=1.0,
            output_directory=output_directory,
            frame_results_file=(
                "realtime_frame_results.csv"
            ),
            summary_file="realtime_summary.csv",
        )

    def create_records(
        self,
        architecture: str,
        *,
        deadline_ms: float = 1000.0 / 30.0,
    ):
        context = RealtimeRunContext(
            architecture=architecture,
            algorithm="original",
            resolution="16x12",
            trial=1,
            origin_timestamp_ns=0,
            deadline_ms=deadline_ms,
        )

        processed_record = create_processed_record(
            context,
            frame_index=1,
            scheduled_timestamp_ns=0,
            enqueued_timestamp_ns=1_000_000,
            processing_start_timestamp_ns=(
                2_000_000
            ),
            processing_end_timestamp_ns=(
                10_000_000
            ),
        )
        dropped_record = create_dropped_record(
            context,
            frame_index=2,
            scheduled_timestamp_ns=33_000_000,
            enqueued_timestamp_ns=34_000_000,
            drop_timestamp_ns=35_000_000,
        )

        return [
            processed_record,
            dropped_record,
        ]

    def create_all_architecture_records(self):
        return {
            architecture: self.create_records(
                architecture
            )
            for architecture in analyzer.ARCHITECTURES
        }

    def test_loads_valid_frame_records(self) -> None:
        with TemporaryDirectory() as directory:
            result_path = (
                Path(directory) / "results.csv"
            )
            write_frame_records(
                self.create_records("pure_python"),
                result_path,
            )

            records = analyzer.load_frame_records(
                result_path,
                "pure_python",
            )

        self.assertEqual(len(records), 2)
        self.assertEqual(
            records[0].architecture,
            "pure_python",
        )

    def test_rejects_incorrect_architecture(self) -> None:
        with TemporaryDirectory() as directory:
            result_path = (
                Path(directory) / "results.csv"
            )
            write_frame_records(
                self.create_records("pure_python"),
                result_path,
            )

            with self.assertRaisesRegex(
                ValueError,
                "Incorrect architecture label",
            ):
                analyzer.load_frame_records(
                    result_path,
                    "hybrid",
                )

    def test_validates_complete_coverage(self) -> None:
        records_by_architecture = (
            self.create_all_architecture_records()
        )

        analyzer.validate_experiment_coverage(
            records_by_architecture,
            algorithm_names=["original"],
            resolution_names=["16x12"],
            trial_count=1,
            measured_frames=2,
        )

    def test_rejects_missing_frame(self) -> None:
        records_by_architecture = (
            self.create_all_architecture_records()
        )
        records_by_architecture["pure_cpp"] = (
            records_by_architecture["pure_cpp"][:1]
        )

        with self.assertRaisesRegex(
            ValueError,
            "coverage validation failed",
        ):
            analyzer.validate_experiment_coverage(
                records_by_architecture,
                algorithm_names=["original"],
                resolution_names=["16x12"],
                trial_count=1,
                measured_frames=2,
            )

    def test_rejects_mismatched_deadline(self) -> None:
        with TemporaryDirectory() as directory:
            config = self.create_config(
                Path(directory)
            )
            records_by_architecture = {
                architecture: self.create_records(
                    architecture,
                    deadline_ms=20.0,
                )
                for architecture
                in analyzer.ARCHITECTURES
            }

            with self.assertRaisesRegex(
                ValueError,
                "deadline does not match",
            ):
                analyzer.validate_deadlines(
                    records_by_architecture,
                    config,
                )

    def test_rejects_inconsistent_processing_time(
        self,
    ) -> None:
        records_by_architecture = (
            self.create_all_architecture_records()
        )
        records_by_architecture["pure_cpp"][0] = replace(
            records_by_architecture["pure_cpp"][0],
            processing_time_ms=999.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "processing_time_ms",
        ):
            analyzer.validate_measurement_consistency(
                records_by_architecture
            )

    def test_builds_expected_summary_metrics(self) -> None:
        summaries = analyzer.build_realtime_summaries(
            self.create_all_architecture_records(),
            algorithm_names=["original"],
            resolution_names=["16x12"],
            target_fps=30.0,
        )

        self.assertEqual(len(summaries), 3)

        pure_python_summary = summaries[0]
        self.assertEqual(
            pure_python_summary.processed_count,
            1,
        )
        self.assertEqual(
            pure_python_summary.dropped_count,
            1,
        )
        self.assertAlmostEqual(
            pure_python_summary.drop_rate_percent,
            50.0,
        )
        self.assertAlmostEqual(
            pure_python_summary.effective_fps,
            15.0,
        )
        self.assertAlmostEqual(
            pure_python_summary.mean_processing_time_ms,
            8.0,
        )

    def test_writes_realtime_summary(self) -> None:
        summaries = analyzer.build_realtime_summaries(
            self.create_all_architecture_records(),
            algorithm_names=["original"],
            resolution_names=["16x12"],
            target_fps=30.0,
        )

        with TemporaryDirectory() as directory:
            output_path = (
                Path(directory)
                / "realtime_summary.csv"
            )
            analyzer.write_realtime_summaries(
                summaries,
                output_path,
            )

            with output_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as output_file:
                rows = list(
                    csv.DictReader(output_file)
                )

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            rows[0]["architecture"],
            "pure_python",
        )
        self.assertEqual(
            rows[0]["drop_rate_percent"],
            "50.000000",
        )


if __name__ == "__main__":
    unittest.main()