from __future__ import annotations

import unittest

from benchmarks.experiments.controlled_illumination_quality_capture import (
    QualityCaptureConfigError,
    load_quality_capture_config,
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


if __name__ == "__main__":
    unittest.main()