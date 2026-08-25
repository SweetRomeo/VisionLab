from dataclasses import replace
import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    unittest.main()