from __future__ import annotations

import math
import unittest

import numpy as np
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.experiments.analyze_controlled_illumination_quality import (
    ControlledIlluminationQualityAnalysisError,
    calculate_image_pair_quality_metrics,
    calculate_image_quality_metrics,
    convert_to_grayscale,
    load_and_validate_quality_capture_run,
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

if __name__ == "__main__":
    unittest.main()