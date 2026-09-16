from __future__ import annotations

import math
import unittest
import csv
import numpy as np
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchmarks.experiments.analyze_controlled_illumination_quality import (
    ControlledIlluminationQualityAnalysisError,
    QUALITY_SAMPLE_CSV_FIELDS,
    QUALITY_TRIAL_SUMMARY_CSV_FIELDS,
    QualitySampleAnalysis,
    QualityTrialSummary,
    ValidatedQualityRun,
    ValidatedQualitySample,
    analyze_quality_run,
    calculate_image_pair_quality_metrics,
    calculate_image_quality_metrics,
    convert_to_grayscale,
    discover_quality_capture_runs,
    load_and_validate_quality_capture_run,
    summarize_quality_trial,
    write_quality_sample_analysis_csv,
    write_quality_trial_summary_csv,
    OPTICAL_QUALITY_SAMPLES_FILE_NAME,
    OPTICAL_QUALITY_TRIAL_SUMMARY_FILE_NAME,
    analyze_quality_capture_results,
    build_argument_parser,
    main,
)

from benchmarks.experiments.controlled_illumination_quality_capture import (
    CapturedQualitySample,
    QualityCaptureConfig,
    write_quality_capture_artifacts_atomic,
)


class ImageQualityMetricTests(
    unittest.TestCase
):
    def test_constant_grayscale_image_metrics(
        self,
    ) -> None:
        image = np.full(
            (4, 4),
            100,
            dtype=np.uint8,
        )

        metrics = (
            calculate_image_quality_metrics(
                image
            )
        )

        self.assertEqual(
            metrics.mean_intensity,
            100.0,
        )
        self.assertEqual(
            metrics.intensity_std,
            0.0,
        )
        self.assertEqual(
            metrics.rms_contrast,
            0.0,
        )
        self.assertEqual(
            metrics.dark_clipping_pct,
            0.0,
        )
        self.assertEqual(
            metrics.bright_clipping_pct,
            0.0,
        )

    def test_dark_and_bright_clipping(
        self,
    ) -> None:
        image = np.array(
            [
                [0, 0],
                [100, 255],
            ],
            dtype=np.uint8,
        )

        metrics = (
            calculate_image_quality_metrics(
                image
            )
        )

        self.assertEqual(
            metrics.dark_clipping_pct,
            50.0,
        )
        self.assertEqual(
            metrics.bright_clipping_pct,
            25.0,
        )

    def test_bgr_image_is_supported(
        self,
    ) -> None:
        image = np.full(
            (4, 4, 3),
            100,
            dtype=np.uint8,
        )

        grayscale = convert_to_grayscale(
            image
        )

        self.assertEqual(
            grayscale.shape,
            (
                4,
                4,
            ),
        )

        self.assertTrue(
            np.all(
                grayscale == 100
            )
        )

    def test_identical_images_have_zero_error(
        self,
    ) -> None:
        image = np.full(
            (4, 4, 3),
            120,
            dtype=np.uint8,
        )

        metrics = (
            calculate_image_pair_quality_metrics(
                image,
                image.copy(),
            )
        )

        self.assertEqual(
            metrics.mae,
            0.0,
        )
        self.assertEqual(
            metrics.max_absolute_error,
            0.0,
        )
        self.assertEqual(
            metrics.mse,
            0.0,
        )
        self.assertTrue(
            math.isinf(
                metrics.psnr
            )
        )

    def test_known_image_difference_metrics(
        self,
    ) -> None:
        input_image = np.zeros(
            (2, 2),
            dtype=np.uint8,
        )

        processed_image = np.full(
            (2, 2),
            10,
            dtype=np.uint8,
        )

        metrics = (
            calculate_image_pair_quality_metrics(
                input_image,
                processed_image,
            )
        )

        self.assertEqual(
            metrics.mae,
            10.0,
        )
        self.assertEqual(
            metrics.max_absolute_error,
            10.0,
        )
        self.assertEqual(
            metrics.mse,
            100.0,
        )

        expected_psnr = (
            20.0
            * math.log10(
                255.0 / 10.0
            )
        )

        self.assertAlmostEqual(
            metrics.psnr,
            expected_psnr,
        )

    def test_shape_mismatch_is_rejected(
        self,
    ) -> None:
        input_image = np.zeros(
            (4, 4),
            dtype=np.uint8,
        )

        processed_image = np.zeros(
            (5, 4),
            dtype=np.uint8,
        )

        with self.assertRaisesRegex(
            ControlledIlluminationQualityAnalysisError,
            "identical shapes",
        ):
            calculate_image_pair_quality_metrics(
                input_image,
                processed_image,
            )

    def test_non_uint8_image_is_rejected(
        self,
    ) -> None:
        image = np.zeros(
            (4, 4),
            dtype=np.float32,
        )

        with self.assertRaisesRegex(
            ControlledIlluminationQualityAnalysisError,
            "uint8",
        ):
            calculate_image_quality_metrics(
                image
            )

class QualityCaptureManifestValidationTests(
    unittest.TestCase
):
    @staticmethod
    def create_frame(
        value: int,
    ) -> np.ndarray:
        return np.full(
            (4, 6, 3),
            value,
            dtype=np.uint8,
        )

    def create_artifacts(
        self,
        output_directory: Path,
    ) -> None:
        config = QualityCaptureConfig(
            enabled=True,
            measured_frame_indices=(0,),
            image_format="png",
        )

        samples = (
            CapturedQualitySample(
                measured_frame_index=0,
                input_frame=self.create_frame(10),
                processed_frame=self.create_frame(20),
            ),
        )

        write_quality_capture_artifacts_atomic(
            output_directory=output_directory,
            experiment_id="experiment-test",
            run_id="run-test",
            algorithm="clahe",
            width=6,
            height=4,
            config=config,
            samples=samples,
        )

    @staticmethod
    def load_manifest(
        output_directory: Path,
    ) -> dict:
        manifest_path = (
            output_directory
            / "quality_samples_manifest.json"
        )

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as manifest_file:
            return json.load(
                manifest_file
            )

    @staticmethod
    def write_manifest(
        output_directory: Path,
        manifest: dict,
    ) -> None:
        manifest_path = (
            output_directory
            / "quality_samples_manifest.json"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump(
                manifest,
                manifest_file,
                indent=2,
            )

    def test_valid_quality_capture_run_is_loaded(
            self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(
                temporary
            )

            self.create_artifacts(
                output_directory
            )

            run = (
                load_and_validate_quality_capture_run(
                    output_directory
                )
            )

            self.assertEqual(
                run.experiment_id,
                "experiment-test",
            )
            self.assertEqual(
                run.run_id,
                "run-test",
            )
            self.assertEqual(
                run.algorithm,
                "clahe",
            )
            self.assertEqual(
                (
                    run.width,
                    run.height,
                ),
                (
                    6,
                    4,
                ),
            )
            self.assertEqual(
                len(run.samples),
                1,
            )
            self.assertEqual(
                run.samples[
                    0
                ].measured_frame_index,
                0,
            )

            self.assertEqual(
                len(
                    run.samples[
                        0
                    ].input_sha256
                ),
                64,
            )

            self.assertEqual(
                len(
                    run.samples[
                        0
                    ].processed_sha256
                ),
                64,
            )

    def test_missing_manifest_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "manifest was not found",
            ):
                load_and_validate_quality_capture_run(
                    Path(temporary)
                )

    def test_modified_image_hash_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(
                temporary
            )

            self.create_artifacts(
                output_directory
            )

            image_path = (
                output_directory
                / "quality_samples"
                / "frame_000000_input.png"
            )

            image_path.write_bytes(
                image_path.read_bytes()
                + b"modified"
            )

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "SHA-256 mismatch",
            ):
                load_and_validate_quality_capture_run(
                    output_directory
                )

    def test_manifest_dimension_mismatch_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(
                temporary
            )

            self.create_artifacts(
                output_directory
            )

            manifest = self.load_manifest(
                output_directory
            )

            manifest["samples"][0][
                "input"
            ]["width"] = 999

            self.write_manifest(
                output_directory,
                manifest,
            )

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "dimensions",
            ):
                load_and_validate_quality_capture_run(
                    output_directory
                )

    def test_manifest_dtype_mismatch_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(
                temporary
            )

            self.create_artifacts(
                output_directory
            )

            manifest = self.load_manifest(
                output_directory
            )

            manifest["samples"][0][
                "input"
            ]["dtype"] = "float32"

            self.write_manifest(
                output_directory,
                manifest,
            )

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "dtype",
            ):
                load_and_validate_quality_capture_run(
                    output_directory
                )

    def test_missing_sample_image_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(
                temporary
            )

            self.create_artifacts(
                output_directory
            )

            (
                output_directory
                / "quality_samples"
                / "frame_000000_processed.png"
            ).unlink()

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "was not found",
            ):
                load_and_validate_quality_capture_run(
                    output_directory
                )

    def test_duplicate_sample_index_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(
                temporary
            )

            self.create_artifacts(
                output_directory
            )

            manifest = self.load_manifest(
                output_directory
            )

            manifest["samples"].append(
                dict(
                    manifest["samples"][0]
                )
            )

            self.write_manifest(
                output_directory,
                manifest,
            )

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "Duplicate quality sample",
            ):
                load_and_validate_quality_capture_run(
                    output_directory
                )

    def test_incomplete_sample_set_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(
                temporary
            )

            self.create_artifacts(
                output_directory
            )

            manifest = self.load_manifest(
                output_directory
            )

            manifest[
                "capture_configuration"
            ][
                "measured_frame_indices"
            ] = [
                0,
                1,
            ]

            self.write_manifest(
                output_directory,
                manifest,
            )

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "do not match configured",
            ):
                load_and_validate_quality_capture_run(
                    output_directory
                )

class QualityRunAnalysisTests(
    unittest.TestCase
):
    @staticmethod
    def create_frame(
        value: int,
    ) -> np.ndarray:
        return np.full(
            (4, 6, 3),
            value,
            dtype=np.uint8,
        )

    def test_quality_run_is_analyzed(
        self,
    ) -> None:
        sample = ValidatedQualitySample(
            measured_frame_index=0,
            input_path=Path("input.png"),
            processed_path=Path("processed.png"),
            input_sha256="a" * 64,
            processed_sha256="b" * 64,
            input_image=self.create_frame(10),
            processed_image=self.create_frame(20),
        )

        run = ValidatedQualityRun(
            experiment_id="experiment-test",
            run_id="run-test",
            algorithm="clahe",
            width=6,
            height=4,
            samples=(
                sample,
            ),
        )

        analyses = analyze_quality_run(
            run
        )

        self.assertEqual(
            len(analyses),
            1,
        )

        analysis = analyses[0]

        self.assertEqual(
            analysis.experiment_id,
            "experiment-test",
        )
        self.assertEqual(
            analysis.run_id,
            "run-test",
        )
        self.assertEqual(
            analysis.algorithm,
            "clahe",
        )
        self.assertEqual(
            analysis.resolution_width,
            6,
        )
        self.assertEqual(
            analysis.resolution_height,
            4,
        )
        self.assertEqual(
            analysis.measured_frame_index,
            0,
        )

        self.assertEqual(
            analysis.input_path,
            "input.png",
        )
        self.assertEqual(
            analysis.processed_path,
            "processed.png",
        )
        self.assertEqual(
            analysis.input_sha256,
            "a" * 64,
        )
        self.assertEqual(
            analysis.processed_sha256,
            "b" * 64,
        )

        self.assertEqual(
            analysis.input_mean_intensity,
            10.0,
        )
        self.assertEqual(
            analysis.processed_mean_intensity,
            20.0,
        )

        self.assertEqual(
            analysis.input_intensity_std,
            0.0,
        )
        self.assertEqual(
            analysis.processed_intensity_std,
            0.0,
        )

        self.assertEqual(
            analysis.input_rms_contrast,
            0.0,
        )
        self.assertEqual(
            analysis.processed_rms_contrast,
            0.0,
        )

        self.assertEqual(
            analysis.mae,
            10.0,
        )
        self.assertEqual(
            analysis.max_absolute_error,
            10.0,
        )
        self.assertEqual(
            analysis.mse,
            100.0,
        )

    def test_multiple_samples_preserve_order(
            self,
    ) -> None:
        run = ValidatedQualityRun(
            experiment_id="experiment-test",
            run_id="run-test",
            algorithm="gamma_correction",
            width=6,
            height=4,
            samples=(
                ValidatedQualitySample(
                    measured_frame_index=0,
                    input_path=Path(
                        "input0.png"
                    ),
                    processed_path=Path(
                        "processed0.png"
                    ),
                    input_sha256="a" * 64,
                    processed_sha256="b" * 64,
                    input_image=self.create_frame(
                        10
                    ),
                    processed_image=self.create_frame(
                        20
                    ),
                ),
                ValidatedQualitySample(
                    measured_frame_index=124,
                    input_path=Path(
                        "input124.png"
                    ),
                    processed_path=Path(
                        "processed124.png"
                    ),
                    input_sha256="c" * 64,
                    processed_sha256="d" * 64,
                    input_image=self.create_frame(
                        30
                    ),
                    processed_image=self.create_frame(
                        40
                    ),
                ),
            ),
        )

        analyses = analyze_quality_run(
            run
        )

        self.assertEqual(
            [
                analysis.measured_frame_index
                for analysis in analyses
            ],
            [
                0,
                124,
            ],
        )

    def test_empty_run_returns_empty_tuple(
        self,
    ) -> None:
        run = ValidatedQualityRun(
            experiment_id="experiment-test",
            run_id="run-test",
            algorithm="original",
            width=6,
            height=4,
            samples=(),
        )

        analyses = analyze_quality_run(
            run
        )

        self.assertEqual(
            analyses,
            (),
        )

class QualityTrialSummaryTests(
    unittest.TestCase
):
    @staticmethod
    def create_analysis(
        *,
        frame_index: int,
        input_intensity: float,
        processed_intensity: float,
        mae: float,
        mse: float,
        psnr: float,
        run_id: str = "run-test",
    ) -> QualitySampleAnalysis:
        return QualitySampleAnalysis(
            experiment_id="experiment-test",
            run_id=run_id,
            algorithm="clahe",
            resolution_width=640,
            resolution_height=480,
            measured_frame_index=frame_index,
            input_path=f"input_{frame_index}.png",
            processed_path=f"processed_{frame_index}.png",
            input_sha256="a" * 64,
            processed_sha256="b" * 64,
            input_mean_intensity=input_intensity,
            processed_mean_intensity=processed_intensity,
            input_intensity_std=10.0,
            processed_intensity_std=20.0,
            input_rms_contrast=0.10,
            processed_rms_contrast=0.20,
            input_dark_clipping_pct=1.0,
            processed_dark_clipping_pct=2.0,
            input_bright_clipping_pct=3.0,
            processed_bright_clipping_pct=4.0,
            mae=mae,
            max_absolute_error=25.0,
            mse=mse,
            psnr=psnr,
        )

    def test_trial_summary_averages_samples(
        self,
    ) -> None:
        analyses = (
            self.create_analysis(
                frame_index=0,
                input_intensity=10.0,
                processed_intensity=20.0,
                mae=5.0,
                mse=25.0,
                psnr=30.0,
            ),
            self.create_analysis(
                frame_index=124,
                input_intensity=30.0,
                processed_intensity=40.0,
                mae=15.0,
                mse=125.0,
                psnr=20.0,
            ),
        )

        summary = summarize_quality_trial(
            analyses
        )

        self.assertEqual(
            summary.experiment_id,
            "experiment-test",
        )
        self.assertEqual(
            summary.run_id,
            "run-test",
        )
        self.assertEqual(
            summary.sample_count,
            2,
        )

        self.assertEqual(
            summary.mean_input_intensity,
            20.0,
        )
        self.assertEqual(
            summary.mean_processed_intensity,
            30.0,
        )
        self.assertEqual(
            summary.mean_mae,
            10.0,
        )
        self.assertEqual(
            summary.mean_mse,
            75.0,
        )
        self.assertEqual(
            summary.mean_psnr,
            25.0,
        )

    def test_empty_trial_is_rejected(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ControlledIlluminationQualityAnalysisError,
            "must not be empty",
        ):
            summarize_quality_trial(
                ()
            )

    def test_samples_from_different_runs_are_rejected(
        self,
    ) -> None:
        analyses = (
            self.create_analysis(
                frame_index=0,
                input_intensity=10.0,
                processed_intensity=20.0,
                mae=5.0,
                mse=25.0,
                psnr=30.0,
                run_id="run-a",
            ),
            self.create_analysis(
                frame_index=124,
                input_intensity=30.0,
                processed_intensity=40.0,
                mae=15.0,
                mse=125.0,
                psnr=20.0,
                run_id="run-b",
            ),
        )

        with self.assertRaisesRegex(
            ControlledIlluminationQualityAnalysisError,
            "same run",
        ):
            summarize_quality_trial(
                analyses
            )

class QualityTrialSummaryCsvTests(
    unittest.TestCase
):
    @staticmethod
    def create_summary(
        *,
        run_id: str = "run-test",
    ) -> QualityTrialSummary:
        return QualityTrialSummary(
            experiment_id="experiment-test",
            run_id=run_id,
            algorithm="clahe",
            resolution_width=640,
            resolution_height=480,
            sample_count=5,
            mean_input_intensity=80.0,
            mean_processed_intensity=110.0,
            mean_input_rms_contrast=0.15,
            mean_processed_rms_contrast=0.22,
            mean_input_dark_clipping_pct=2.0,
            mean_processed_dark_clipping_pct=1.0,
            mean_input_bright_clipping_pct=1.0,
            mean_processed_bright_clipping_pct=3.0,
            mean_mae=20.0,
            mean_mse=500.0,
            mean_psnr=25.0,
        )

    def test_trial_summary_csv_is_written(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_path = (
                Path(temporary)
                / "optical_quality_trial_summary.csv"
            )

            write_quality_trial_summary_csv(
                output_path,
                (
                    self.create_summary(),
                ),
            )

            self.assertTrue(
                output_path.is_file()
            )

            with output_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as input_file:
                rows = list(
                    csv.DictReader(
                        input_file
                    )
                )

            self.assertEqual(
                len(rows),
                1,
            )

            self.assertEqual(
                rows[0]["run_id"],
                "run-test",
            )
            self.assertEqual(
                rows[0]["sample_count"],
                "5",
            )

    def test_trial_summary_csv_fields(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_path = (
                Path(temporary)
                / "summary.csv"
            )

            write_quality_trial_summary_csv(
                output_path,
                (
                    self.create_summary(),
                ),
            )

            with output_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as input_file:
                reader = csv.DictReader(
                    input_file
                )

                self.assertEqual(
                    tuple(
                        reader.fieldnames
                        or ()
                    ),
                    QUALITY_TRIAL_SUMMARY_CSV_FIELDS,
                )

    def test_empty_trial_summary_csv_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "must not be empty",
            ):
                write_quality_trial_summary_csv(
                    Path(temporary)
                    / "summary.csv",
                    (),
                )

    def test_duplicate_trial_summary_is_rejected(
        self,
    ) -> None:
        summary = self.create_summary()

        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "Duplicate",
            ):
                write_quality_trial_summary_csv(
                    Path(temporary)
                    / "summary.csv",
                    (
                        summary,
                        summary,
                    ),
                )

class QualityCaptureRunDiscoveryTests(
    unittest.TestCase
):
    @staticmethod
    def create_manifest(
        run_directory: Path,
    ) -> None:
        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = (
            run_directory
            / "quality_samples_manifest.json"
        )

        manifest_path.write_text(
            "{}",
            encoding="utf-8",
        )

    def test_quality_capture_runs_are_discovered(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            results_directory = Path(
                temporary
            )

            run_a = (
                results_directory
                / "experiment-a"
                / "run-a"
            )

            run_b = (
                results_directory
                / "experiment-a"
                / "run-b"
            )

            self.create_manifest(
                run_a
            )
            self.create_manifest(
                run_b
            )

            discovered = (
                discover_quality_capture_runs(
                    results_directory
                )
            )

            self.assertEqual(
                discovered,
                (
                    run_a.resolve(),
                    run_b.resolve(),
                ),
            )

    def test_nested_runs_are_discovered(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            results_directory = Path(
                temporary
            )

            run_directory = (
                results_directory
                / "profile"
                / "algorithm"
                / "resolution"
                / "trial-1"
            )

            self.create_manifest(
                run_directory
            )

            discovered = (
                discover_quality_capture_runs(
                    results_directory
                )
            )

            self.assertEqual(
                discovered,
                (
                    run_directory.resolve(),
                ),
            )

    def test_discovery_order_is_deterministic(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            results_directory = Path(
                temporary
            )

            run_z = (
                results_directory
                / "z-run"
            )

            run_a = (
                results_directory
                / "a-run"
            )

            run_m = (
                results_directory
                / "m-run"
            )

            self.create_manifest(
                run_z
            )
            self.create_manifest(
                run_a
            )
            self.create_manifest(
                run_m
            )

            discovered = (
                discover_quality_capture_runs(
                    results_directory
                )
            )

            self.assertEqual(
                discovered,
                (
                    run_a.resolve(),
                    run_m.resolve(),
                    run_z.resolve(),
                ),
            )

    def test_directory_without_runs_returns_empty_tuple(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            discovered = (
                discover_quality_capture_runs(
                    Path(temporary)
                )
            )

            self.assertEqual(
                discovered,
                (),
            )

    def test_missing_results_directory_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            missing_directory = (
                Path(temporary)
                / "missing"
            )

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "was not found",
            ):
                discover_quality_capture_runs(
                    missing_directory
                )

    def test_file_path_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            file_path = (
                Path(temporary)
                / "results.txt"
            )

            file_path.write_text(
                "not a directory",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "must be a directory",
            ):
                discover_quality_capture_runs(
                    file_path
                )

class QualityCaptureResultsAnalysisTests(
    unittest.TestCase
):
    @staticmethod
    def create_frame(
        value: int,
    ) -> np.ndarray:
        return np.full(
            (4, 6, 3),
            value,
            dtype=np.uint8,
        )

    def create_run(
        self,
        run_directory: Path,
        *,
        experiment_id: str,
        run_id: str,
        input_value: int,
        processed_value: int,
    ) -> None:
        config = QualityCaptureConfig(
            enabled=True,
            measured_frame_indices=(0,),
            image_format="png",
        )

        samples = (
            CapturedQualitySample(
                measured_frame_index=0,
                input_frame=self.create_frame(
                    input_value
                ),
                processed_frame=self.create_frame(
                    processed_value
                ),
            ),
        )

        write_quality_capture_artifacts_atomic(
            output_directory=run_directory,
            experiment_id=experiment_id,
            run_id=run_id,
            algorithm="clahe",
            width=6,
            height=4,
            config=config,
            samples=samples,
        )

    def test_results_are_analyzed_end_to_end(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(
                temporary
            )

            results_directory = (
                root
                / "results"
            )

            output_directory = (
                root
                / "analysis"
            )

            self.create_run(
                results_directory
                / "run-a",
                experiment_id="experiment-test",
                run_id="run-a",
                input_value=10,
                processed_value=20,
            )

            self.create_run(
                results_directory
                / "run-b",
                experiment_id="experiment-test",
                run_id="run-b",
                input_value=30,
                processed_value=40,
            )

            (
                sample_path,
                summary_path,
            ) = analyze_quality_capture_results(
                results_directory,
                output_directory,
            )

            self.assertEqual(
                sample_path.name,
                OPTICAL_QUALITY_SAMPLES_FILE_NAME,
            )

            self.assertEqual(
                summary_path.name,
                OPTICAL_QUALITY_TRIAL_SUMMARY_FILE_NAME,
            )

            self.assertTrue(
                sample_path.is_file()
            )

            self.assertTrue(
                summary_path.is_file()
            )

            with sample_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as input_file:
                sample_rows = list(
                    csv.DictReader(
                        input_file
                    )
                )

            with summary_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as input_file:
                summary_rows = list(
                    csv.DictReader(
                        input_file
                    )
                )

            self.assertEqual(
                len(sample_rows),
                2,
            )

            self.assertEqual(
                len(summary_rows),
                2,
            )

            self.assertEqual(
                [
                    row["run_id"]
                    for row in sample_rows
                ],
                [
                    "run-a",
                    "run-b",
                ],
            )

    def test_empty_results_are_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(
                temporary
            )

            results_directory = (
                root
                / "results"
            )

            results_directory.mkdir()

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "No completed quality-capture runs",
            ):
                analyze_quality_capture_results(
                    results_directory,
                    root / "analysis",
                )

    def test_duplicate_run_identity_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(
                temporary
            )

            results_directory = (
                root
                / "results"
            )

            self.create_run(
                results_directory
                / "physical-run-a",
                experiment_id="experiment-test",
                run_id="duplicate-run",
                input_value=10,
                processed_value=20,
            )

            self.create_run(
                results_directory
                / "physical-run-b",
                experiment_id="experiment-test",
                run_id="duplicate-run",
                input_value=30,
                processed_value=40,
            )

            with self.assertRaisesRegex(
                ControlledIlluminationQualityAnalysisError,
                "Duplicate quality-capture run",
            ):
                analyze_quality_capture_results(
                    results_directory,
                    root / "analysis",
                )

    def test_partial_outputs_are_cleaned_after_failure(
            self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(
                temporary
            )

            results_directory = (
                    root
                    / "results"
            )

            output_directory = (
                    root
                    / "analysis"
            )

            self.create_run(
                results_directory
                / "run-a",
                experiment_id="experiment-test",
                run_id="run-a",
                input_value=10,
                processed_value=20,
            )

            with patch(
                    "benchmarks.experiments."
                    "analyze_controlled_illumination_quality."
                    "write_quality_trial_summary_csv",
                    side_effect=RuntimeError(
                        "summary write failure"
                    ),
            ):
                with self.assertRaisesRegex(
                        RuntimeError,
                        "summary write failure",
                ):
                    analyze_quality_capture_results(
                        results_directory,
                        output_directory,
                    )

            self.assertFalse(
                (
                        output_directory
                        / OPTICAL_QUALITY_SAMPLES_FILE_NAME
                ).exists()
            )

            self.assertFalse(
                (
                        output_directory
                        / OPTICAL_QUALITY_TRIAL_SUMMARY_FILE_NAME
                ).exists()
            )

class QualityAnalysisCliTests(
    unittest.TestCase
):
    def test_required_arguments_are_parsed(
        self,
    ) -> None:
        parser = build_argument_parser()

        arguments = parser.parse_args(
            [
                "--results-directory",
                "results",
                "--output-directory",
                "analysis",
            ]
        )

        self.assertEqual(
            arguments.results_directory,
            Path("results"),
        )
        self.assertEqual(
            arguments.output_directory,
            Path("analysis"),
        )

    def test_main_reports_success(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(
                temporary
            )

            sample_path = (
                root
                / "optical_quality_samples.csv"
            )

            summary_path = (
                root
                / "optical_quality_trial_summary.csv"
            )

            with patch(
                "benchmarks.experiments."
                "analyze_controlled_illumination_quality."
                "analyze_quality_capture_results",
                return_value=(
                    sample_path,
                    summary_path,
                ),
            ):
                result = main(
                    [
                        "--results-directory",
                        str(root / "results"),
                        "--output-directory",
                        str(root / "analysis"),
                    ]
                )

        self.assertEqual(
            result,
            0,
        )

    def test_main_exits_on_analysis_error(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(
                temporary
            )

            with patch(
                "benchmarks.experiments."
                "analyze_controlled_illumination_quality."
                "analyze_quality_capture_results",
                side_effect=(
                    ControlledIlluminationQualityAnalysisError(
                        "analysis failed"
                    )
                ),
            ):
                with self.assertRaises(
                    SystemExit
                ) as context:
                    main(
                        [
                            "--results-directory",
                            str(root / "results"),
                            "--output-directory",
                            str(root / "analysis"),
                        ]
                    )

        self.assertEqual(
            context.exception.code,
            1,
        )

if __name__ == "__main__":
    unittest.main()