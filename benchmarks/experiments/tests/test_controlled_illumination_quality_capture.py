from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from benchmarks.experiments.controlled_illumination_quality_capture import (
    CapturedQualitySample,
    QualityCaptureArtifactError,
    QualityCaptureBuffer,
    QualityCaptureConfig,
    QualityCaptureConfigError,
    calculate_bytes_sha256,
    load_quality_capture_config,
    write_quality_capture_artifacts_atomic,
)


class QualityCaptureConfigTests(
    unittest.TestCase
):
    def test_valid_configuration_is_loaded(
        self,
    ) -> None:
        config = {
            "quality_capture": {
                "enabled": True,
                "measured_frame_indices": [
                    0,
                    124,
                    249,
                    374,
                    499,
                ],
                "image_format": "png",
            }
        }

        quality_config = (
            load_quality_capture_config(
                config,
                measured_frames=500,
            )
        )

        self.assertTrue(
            quality_config.enabled
        )
        self.assertEqual(
            quality_config.measured_frame_indices,
            (
                0,
                124,
                249,
                374,
                499,
            ),
        )
        self.assertEqual(
            quality_config.image_format,
            "png",
        )

    def test_missing_configuration_is_disabled(
        self,
    ) -> None:
        quality_config = (
            load_quality_capture_config(
                {},
                measured_frames=500,
            )
        )

        self.assertFalse(
            quality_config.enabled
        )
        self.assertEqual(
            quality_config.measured_frame_indices,
            (),
        )

    def test_duplicate_indices_are_rejected(
        self,
    ) -> None:
        config = {
            "quality_capture": {
                "enabled": True,
                "measured_frame_indices": [
                    0,
                    124,
                    124,
                ],
                "image_format": "png",
            }
        }

        with self.assertRaises(
            QualityCaptureConfigError
        ):
            load_quality_capture_config(
                config,
                measured_frames=500,
            )

    def test_negative_index_is_rejected(
        self,
    ) -> None:
        config = {
            "quality_capture": {
                "enabled": True,
                "measured_frame_indices": [
                    -1,
                ],
                "image_format": "png",
            }
        }

        with self.assertRaises(
            QualityCaptureConfigError
        ):
            load_quality_capture_config(
                config,
                measured_frames=500,
            )

    def test_out_of_range_index_is_rejected(
        self,
    ) -> None:
        config = {
            "quality_capture": {
                "enabled": True,
                "measured_frame_indices": [
                    500,
                ],
                "image_format": "png",
            }
        }

        with self.assertRaises(
            QualityCaptureConfigError
        ):
            load_quality_capture_config(
                config,
                measured_frames=500,
            )

    def test_unsupported_format_is_rejected(
        self,
    ) -> None:
        config = {
            "quality_capture": {
                "enabled": True,
                "measured_frame_indices": [
                    0,
                ],
                "image_format": "jpg",
            }
        }

        with self.assertRaises(
            QualityCaptureConfigError
        ):
            load_quality_capture_config(
                config,
                measured_frames=500,
            )


class QualityCaptureBufferTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.config = QualityCaptureConfig(
            enabled=True,
            measured_frame_indices=(
                0,
                2,
                4,
            ),
            image_format="png",
        )

        self.buffer = QualityCaptureBuffer(
            self.config
        )

    @staticmethod
    def create_frame(
        value: int,
    ) -> np.ndarray:
        return np.full(
            (4, 6, 3),
            value,
            dtype=np.uint8,
        )

    def test_only_selected_frames_are_captured(
        self,
    ) -> None:
        for frame_index in range(5):
            input_frame = self.create_frame(
                frame_index
            )
            processed_frame = (
                input_frame + 10
            )

            self.buffer.capture(
                frame_index,
                input_frame,
                processed_frame,
            )

        self.assertEqual(
            [
                sample.measured_frame_index
                for sample in self.buffer.samples
            ],
            [
                0,
                2,
                4,
            ],
        )

    def test_captured_frames_are_copied(
        self,
    ) -> None:
        input_frame = self.create_frame(5)
        processed_frame = self.create_frame(10)

        self.buffer.capture(
            0,
            input_frame,
            processed_frame,
        )

        input_frame[:] = 100
        processed_frame[:] = 200

        sample = self.buffer.samples[0]

        self.assertTrue(
            np.all(
                sample.input_frame == 5
            )
        )
        self.assertTrue(
            np.all(
                sample.processed_frame == 10
            )
        )

    def test_missing_indices_are_reported(
        self,
    ) -> None:
        self.buffer.capture(
            0,
            self.create_frame(1),
            self.create_frame(2),
        )

        self.assertEqual(
            self.buffer.missing_indices,
            (
                2,
                4,
            ),
        )

    def test_duplicate_capture_is_rejected(
        self,
    ) -> None:
        self.buffer.capture(
            0,
            self.create_frame(1),
            self.create_frame(2),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "already captured",
        ):
            self.buffer.capture(
                0,
                self.create_frame(3),
                self.create_frame(4),
            )

    def test_disabled_buffer_captures_nothing(
        self,
    ) -> None:
        buffer = QualityCaptureBuffer(
            QualityCaptureConfig(
                enabled=False,
                measured_frame_indices=(),
                image_format="png",
            )
        )

        buffer.capture(
            0,
            self.create_frame(1),
            self.create_frame(2),
        )

        self.assertEqual(
            buffer.samples,
            (),
        )
        self.assertEqual(
            buffer.missing_indices,
            (),
        )


class QualityCaptureArtifactTests(
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

    def create_config(
        self,
    ) -> QualityCaptureConfig:
        return QualityCaptureConfig(
            enabled=True,
            measured_frame_indices=(
                0,
                2,
            ),
            image_format="png",
        )

    def create_samples(
        self,
    ) -> tuple[CapturedQualitySample, ...]:
        return (
            CapturedQualitySample(
                measured_frame_index=0,
                input_frame=self.create_frame(10),
                processed_frame=self.create_frame(20),
            ),
            CapturedQualitySample(
                measured_frame_index=2,
                input_frame=self.create_frame(30),
                processed_frame=self.create_frame(40),
            ),
        )

    def test_quality_capture_artifacts_are_written(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(temporary)

            samples_directory, manifest_path = (
                write_quality_capture_artifacts_atomic(
                    output_directory=output_directory,
                    experiment_id="experiment-test",
                    run_id="run-test",
                    algorithm="clahe",
                    width=6,
                    height=4,
                    config=self.create_config(),
                    samples=self.create_samples(),
                )
            )

            self.assertTrue(
                samples_directory.is_dir()
            )
            self.assertTrue(
                manifest_path.is_file()
            )

            self.assertTrue(
                (
                    samples_directory
                    / "frame_000000_input.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    samples_directory
                    / "frame_000002_processed.png"
                ).is_file()
            )

            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as manifest_file:
                manifest = json.load(
                    manifest_file
                )

            self.assertEqual(
                manifest["run_id"],
                "run-test",
            )
            self.assertEqual(
                manifest["algorithm"],
                "clahe",
            )
            self.assertEqual(
                len(manifest["samples"]),
                2,
            )

    def test_written_png_is_lossless(
        self,
    ) -> None:
        samples = self.create_samples()

        with TemporaryDirectory() as temporary:
            samples_directory, _ = (
                write_quality_capture_artifacts_atomic(
                    output_directory=Path(temporary),
                    experiment_id="experiment-test",
                    run_id="run-test",
                    algorithm="original",
                    width=6,
                    height=4,
                    config=self.create_config(),
                    samples=samples,
                )
            )

            loaded = cv2.imread(
                str(
                    samples_directory
                    / "frame_000000_input.png"
                ),
                cv2.IMREAD_UNCHANGED,
            )

            self.assertTrue(
                np.array_equal(
                    loaded,
                    samples[0].input_frame,
                )
            )

    def test_manifest_hash_matches_png(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(temporary)

            samples_directory, manifest_path = (
                write_quality_capture_artifacts_atomic(
                    output_directory=output_directory,
                    experiment_id="experiment-test",
                    run_id="run-test",
                    algorithm="original",
                    width=6,
                    height=4,
                    config=self.create_config(),
                    samples=self.create_samples(),
                )
            )

            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as manifest_file:
                manifest = json.load(
                    manifest_file
                )

            image_path = (
                samples_directory
                / "frame_000000_input.png"
            )

            expected_hash = (
                calculate_bytes_sha256(
                    image_path.read_bytes()
                )
            )

            self.assertEqual(
                manifest["samples"][0][
                    "input"
                ]["sha256"],
                expected_hash,
            )

    def test_existing_quality_artifacts_are_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            output_directory = Path(temporary)

            write_quality_capture_artifacts_atomic(
                output_directory=output_directory,
                experiment_id="experiment-test",
                run_id="run-test",
                algorithm="original",
                width=6,
                height=4,
                config=self.create_config(),
                samples=self.create_samples(),
            )

            with self.assertRaisesRegex(
                QualityCaptureArtifactError,
                "already exist",
            ):
                write_quality_capture_artifacts_atomic(
                    output_directory=output_directory,
                    experiment_id="experiment-test",
                    run_id="run-test",
                    algorithm="original",
                    width=6,
                    height=4,
                    config=self.create_config(),
                    samples=self.create_samples(),
                )

    def test_incomplete_sample_set_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                QualityCaptureArtifactError,
                "do not match",
            ):
                write_quality_capture_artifacts_atomic(
                    output_directory=Path(temporary),
                    experiment_id="experiment-test",
                    run_id="run-test",
                    algorithm="original",
                    width=6,
                    height=4,
                    config=self.create_config(),
                    samples=(
                        self.create_samples()[0],
                    ),
                )

    def test_write_failure_cleans_up_quality_artifacts(
        self,
    ) -> None:
        config = QualityCaptureConfig(
            enabled=True,
            measured_frame_indices=(0,),
            image_format="png",
        )

        frame = np.zeros(
            (4, 6, 3),
            dtype=np.uint8,
        )

        samples = (
            CapturedQualitySample(
                measured_frame_index=0,
                input_frame=frame,
                processed_frame=frame,
            ),
        )

        with TemporaryDirectory() as temporary:
            output_directory = Path(temporary)

            with patch(
                "benchmarks.experiments."
                "controlled_illumination_quality_capture."
                "write_bytes_durable",
                side_effect=OSError("disk failure"),
            ):
                with self.assertRaises(OSError):
                    write_quality_capture_artifacts_atomic(
                        output_directory=output_directory,
                        experiment_id="experiment-test",
                        run_id="run-test",
                        algorithm="original",
                        width=6,
                        height=4,
                        config=config,
                        samples=samples,
                    )

            self.assertFalse(
                (
                    output_directory
                    / "quality_samples"
                ).exists()
            )

            self.assertFalse(
                (
                    output_directory
                    / "quality_samples_manifest.json"
                ).exists()
            )

    def test_manifest_failure_cleans_up_quality_artifacts(
        self,
    ) -> None:
        config = QualityCaptureConfig(
            enabled=True,
            measured_frame_indices=(0,),
            image_format="png",
        )

        frame = np.zeros(
            (4, 6, 3),
            dtype=np.uint8,
        )

        samples = (
            CapturedQualitySample(
                measured_frame_index=0,
                input_frame=frame,
                processed_frame=frame,
            ),
        )

        with TemporaryDirectory() as temporary:
            output_directory = Path(temporary)

            with patch(
                "benchmarks.experiments."
                "controlled_illumination_quality_capture."
                "write_json_durable",
                side_effect=OSError("manifest failure"),
            ):
                with self.assertRaises(OSError):
                    write_quality_capture_artifacts_atomic(
                        output_directory=output_directory,
                        experiment_id="experiment-test",
                        run_id="run-test",
                        algorithm="original",
                        width=6,
                        height=4,
                        config=config,
                        samples=samples,
                    )

            self.assertFalse(
                (
                    output_directory
                    / "quality_samples"
                ).exists()
            )

            self.assertFalse(
                (
                    output_directory
                    / "quality_samples_manifest.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()