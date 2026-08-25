from dataclasses import replace
import unittest

from benchmarks.experiments.controlled_illumination_run_bundle import (
    ControlledIlluminationRunBundleError,
    ControlledIlluminationRunBundleManifest,
    EXECUTION_SUMMARY_FILE_NAME,
    FRAME_RESULTS_FILE_NAME,
    RUN_METADATA_FILE_NAME,
    RunBundleArtifact,
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
            metadata_dry_run=False,
            artifacts=self.create_artifacts(),
        )

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


if __name__ == "__main__":
    unittest.main()