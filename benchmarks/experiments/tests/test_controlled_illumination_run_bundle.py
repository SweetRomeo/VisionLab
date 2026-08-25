from dataclasses import replace
import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from benchmarks.experiments.controlled_illumination_run_bundle import (
    ControlledIlluminationRunBundleError,
    ControlledIlluminationRunBundleManifest,
    EXECUTION_SUMMARY_FILE_NAME,
    FRAME_RESULTS_FILE_NAME,
    RUN_METADATA_FILE_NAME,
    RunBundleArtifact,
    REQUIRED_BUNDLE_ARTIFACT_ORDER,
    collect_bundle_artifacts,
    resolve_run_directory,
    ControlledIlluminationExecutionSummary,
    execution_summary_from_dict,
    load_execution_summary,
)


VALID_TIMESTAMP = "2026-08-26T10:00:00Z"
VALID_PLAN_SHA256 = "a" * 64
VALID_ARTIFACT_SHA256 = "b" * 64


class ControlledIlluminationRunBundleTests(
    unittest.TestCase
):
    def create_artifact(
        self,
        file_name: str,
    ) -> RunBundleArtifact:
        return RunBundleArtifact(
            file_name=file_name,
            sha256=VALID_ARTIFACT_SHA256,
            size_bytes=1024,
        )

    def create_artifacts(
        self,
    ) -> tuple[RunBundleArtifact, ...]:
        return (
            self.create_artifact(
                FRAME_RESULTS_FILE_NAME
            ),
            self.create_artifact(
                EXECUTION_SUMMARY_FILE_NAME
            ),
            self.create_artifact(
                RUN_METADATA_FILE_NAME
            ),
        )

    def create_manifest(
        self,
    ) -> ControlledIlluminationRunBundleManifest:
        return ControlledIlluminationRunBundleManifest(
            schema_version=1,
            finalized_at_utc=VALID_TIMESTAMP,
            experiment_id="experiment-test",
            run_id="run-test-0001",
            run_plan_sha256=VALID_PLAN_SHA256,
            metadata_dry_run=False, # type: ignore[arg-type]
            artifacts=self.create_artifacts(),
        )

    def create_bundle_files(
        self,
        directory: Path,
    ) -> dict[str, bytes]:
        file_contents = {
            FRAME_RESULTS_FILE_NAME: (
                b"architecture,frame_index\n"
                b"pure_python,1\n"
            ),
            EXECUTION_SUMMARY_FILE_NAME: (
                b'{"schema_version": 1}\n'
            ),
            RUN_METADATA_FILE_NAME: (
                b'{"dry_run": true}\n'
            ),
        }

        for file_name, content in (
            file_contents.items()
        ):
            (
                directory / file_name
            ).write_bytes(content)

        return file_contents

    def create_execution_summary_value(
        self,
    ) -> dict:
        return {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": "experiment-test",
            "run_id": "run-test-0001",
            "execution_order": 1,
            "phase": "constant_lux",
            "platform": "desktop",
            "architecture": "pure_python",
            "algorithm": "original",
            "resolution": {
                "width": 640,
                "height": 480,
            },
            "trial_number": 1,
            "started_at_utc": (
                "2026-08-26T10:00:00Z"
            ),
            "finished_at_utc": (
                "2026-08-26T10:01:00Z"
            ),
            "warmup_frame_count": 30,
            "measured_frame_count": 3,
            "processed_frame_count": 2,
            "dropped_frame_count": 1,
            "skipped_frame_count": 0,
            "deadline_met_count": 1,
            "deadline_miss_count": 1,
            "mean_processing_time_ms": 5.0,
            "mean_end_to_end_latency_ms": 7.0,
            "frame_results_file": (
                FRAME_RESULTS_FILE_NAME
            ),
            "frame_results_sha256": "c" * 64,
        }

    def write_execution_summary(
        self,
        directory: Path,
        value: dict,
    ) -> Path:
        output_path = (
            directory
            / EXECUTION_SUMMARY_FILE_NAME
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(
                value,
                output_file,
            )
            output_file.write("\n")

        return output_path

    def test_valid_manifest_is_serialized(
        self,
    ) -> None:
        manifest = self.create_manifest()
        serialized = manifest.to_dict()

        self.assertEqual(
            serialized["schema_version"],
            1,
        )
        self.assertEqual(
            serialized["experiment_id"],
            "experiment-test",
        )
        self.assertFalse(
            serialized["metadata_dry_run"]
        )
        self.assertEqual(
            len(serialized["artifacts"]),
            3,
        )

    def test_invalid_schema_version_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            replace(
                self.create_manifest(),
                schema_version=2,
            )

    def test_invalid_timestamp_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            replace(
                self.create_manifest(),
                finalized_at_utc="invalid",
            )

    def test_unsafe_experiment_id_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            replace(
                self.create_manifest(),
                experiment_id="../experiment",
            )

    def test_invalid_plan_hash_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            replace(
                self.create_manifest(),
                run_plan_sha256="invalid",
            )

    def test_invalid_dry_run_value_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            replace(
                self.create_manifest(),
                metadata_dry_run="false",
            )

    def test_missing_artifact_is_rejected(
        self,
    ) -> None:
        incomplete_artifacts = (
            self.create_artifact(
                FRAME_RESULTS_FILE_NAME
            ),
            self.create_artifact(
                EXECUTION_SUMMARY_FILE_NAME
            ),
        )

        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            replace(
                self.create_manifest(),
                artifacts=incomplete_artifacts,
            )

    def test_duplicate_artifact_is_rejected(
        self,
    ) -> None:
        duplicate_artifacts = (
            *self.create_artifacts(),
            self.create_artifact(
                FRAME_RESULTS_FILE_NAME
            ),
        )

        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            replace(
                self.create_manifest(),
                artifacts=duplicate_artifacts,
            )

    def test_unsupported_artifact_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            RunBundleArtifact(
                file_name="../unexpected.json",
                sha256=VALID_ARTIFACT_SHA256,
                size_bytes=100,
            )

    def test_invalid_artifact_hash_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            RunBundleArtifact(
                file_name=FRAME_RESULTS_FILE_NAME,
                sha256="INVALID",
                size_bytes=100,
            )

    def test_empty_artifact_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            RunBundleArtifact(
                file_name=FRAME_RESULTS_FILE_NAME,
                sha256=VALID_ARTIFACT_SHA256,
                size_bytes=0,
            )

    def test_bundle_artifacts_are_collected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            file_contents = self.create_bundle_files(
                run_directory
            )

            artifacts = collect_bundle_artifacts(
                run_directory
            )

        self.assertEqual(
            tuple(
                artifact.file_name
                for artifact in artifacts
            ),
            REQUIRED_BUNDLE_ARTIFACT_ORDER,
        )

        for artifact in artifacts:
            expected_content = file_contents[
                artifact.file_name
            ]
            expected_hash = hashlib.sha256(
                expected_content
            ).hexdigest()

            self.assertEqual(
                artifact.sha256,
                expected_hash,
            )
            self.assertEqual(
                artifact.size_bytes,
                len(expected_content),
            )

    def test_missing_run_directory_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            missing_directory = (
                Path(temporary) / "missing"
            )

            with self.assertRaises(
                ControlledIlluminationRunBundleError
            ):
                resolve_run_directory(
                    missing_directory
                )

    def test_missing_bundle_artifact_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)

            (
                run_directory
                / FRAME_RESULTS_FILE_NAME
            ).write_text(
                "frame_index\n1\n",
                encoding="utf-8",
            )
            (
                run_directory
                / EXECUTION_SUMMARY_FILE_NAME
            ).write_text(
                "{}\n",
                encoding="utf-8",
            )

            with self.assertRaises(
                ControlledIlluminationRunBundleError
            ):
                collect_bundle_artifacts(
                    run_directory
                )

    def test_empty_bundle_artifact_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            self.create_bundle_files(
                run_directory
            )

            (
                run_directory
                / RUN_METADATA_FILE_NAME
            ).write_bytes(b"")

            with self.assertRaises(
                ControlledIlluminationRunBundleError
            ):
                collect_bundle_artifacts(
                    run_directory
                )
    def test_valid_execution_summary_is_loaded(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            self.write_execution_summary(
                run_directory,
                self.create_execution_summary_value(),
            )

            summary = load_execution_summary(
                run_directory
            )

        self.assertIsInstance(
            summary,
            ControlledIlluminationExecutionSummary,
        )
        self.assertEqual(
            summary.experiment_id,
            "experiment-test",
        )
        self.assertEqual(
            summary.resolution.width,
            640,
        )
        self.assertEqual(
            summary.measured_frame_count,
            3,
        )

    def test_invalid_execution_summary_json_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            (
                run_directory
                / EXECUTION_SUMMARY_FILE_NAME
            ).write_text(
                "{invalid",
                encoding="utf-8",
            )

            with self.assertRaises(
                ControlledIlluminationRunBundleError
            ):
                load_execution_summary(
                    run_directory
                )

    def test_missing_summary_field_is_rejected(
        self,
    ) -> None:
        summary_value = (
            self.create_execution_summary_value()
        )
        del summary_value["run_id"]

        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            execution_summary_from_dict(
                summary_value
            )

    def test_unexpected_summary_field_is_rejected(
        self,
    ) -> None:
        summary_value = (
            self.create_execution_summary_value()
        )
        summary_value["unexpected"] = True

        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            execution_summary_from_dict(
                summary_value
            )

    def test_invalid_frame_counts_are_rejected(
        self,
    ) -> None:
        summary_value = (
            self.create_execution_summary_value()
        )
        summary_value[
            "processed_frame_count"
        ] = 3

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "measured_frame_count",
        ):
            execution_summary_from_dict(
                summary_value
            )

    def test_invalid_deadline_counts_are_rejected(
        self,
    ) -> None:
        summary_value = (
            self.create_execution_summary_value()
        )
        summary_value[
            "deadline_met_count"
        ] = 2
        summary_value[
            "deadline_miss_count"
        ] = 2

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "Deadline counts",
        ):
            execution_summary_from_dict(
                summary_value
            )

    def test_reversed_summary_timestamps_are_rejected(
        self,
    ) -> None:
        summary_value = (
            self.create_execution_summary_value()
        )
        summary_value["finished_at_utc"] = (
            "2026-08-26T09:59:00Z"
        )

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "finished_at_utc",
        ):
            execution_summary_from_dict(
                summary_value
            )

    def test_processed_frames_require_mean_values(
        self,
    ) -> None:
        summary_value = (
            self.create_execution_summary_value()
        )
        summary_value[
            "mean_processing_time_ms"
        ] = None

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "mean timing",
        ):
            execution_summary_from_dict(
                summary_value
            )

    def test_zero_processed_frames_reject_mean_values(
        self,
    ) -> None:
        summary_value = (
            self.create_execution_summary_value()
        )
        summary_value[
            "processed_frame_count"
        ] = 0
        summary_value[
            "dropped_frame_count"
        ] = 3
        summary_value[
            "deadline_met_count"
        ] = 0
        summary_value[
            "deadline_miss_count"
        ] = 0

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "must be null",
        ):
            execution_summary_from_dict(
                summary_value
            )

    def test_invalid_frame_result_hash_is_rejected(
        self,
    ) -> None:
        summary_value = (
            self.create_execution_summary_value()
        )
        summary_value[
            "frame_results_sha256"
        ] = "invalid"

        with self.assertRaises(
            ControlledIlluminationRunBundleError
        ):
            execution_summary_from_dict(
                summary_value
            )


if __name__ == "__main__":
    unittest.main()