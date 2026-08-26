from dataclasses import replace
import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from dataclasses import asdict, replace

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
    validate_execution_summary_matches_run,
    validate_metadata_matches_run,
    validate_summary_counts_against_config,
    validate_summary_frame_hash,
    RUN_BUNDLE_MANIFEST_FILE_NAME,
    write_run_bundle_manifest_atomic,
    finalize_run_bundle_atomic,
)

from benchmarks.experiments.controlled_illumination_metadata import (
    load_controlled_illumination_config,
    save_run_metadata_atomic,
)
from benchmarks.experiments.controlled_illumination_run_planner import (
    PlannedRun,
)
from benchmarks.experiments.generate_dry_run_metadata import (
    build_dry_run_metadata,
)

from benchmarks.realtime.realtime_records import (
    RealtimeRunContext,
    create_processed_record,
    write_frame_records,
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

    def create_metadata_and_planned_run(
        self,
    ) -> tuple[dict, object, PlannedRun]:
        config = (
            load_controlled_illumination_config()
        )
        metadata = build_dry_run_metadata(
            config,
            "d" * 40,
        )

        planned_run = PlannedRun(
            execution_order=1,
            experiment_id=metadata.experiment_id,
            run_id=metadata.run_id,
            phase=metadata.phase,
            platform=metadata.platform,
            architecture=metadata.architecture,
            algorithm=metadata.algorithm,
            resolution=metadata.resolution,
            trial_number=metadata.trial_number,
            incidence_angle_degrees=(
                metadata.incidence_angle_degrees
            ),
            target_illuminance_lux=(
                metadata.target_illuminance_lux
            ),
            source_output_setting=None,
            target_fps=metadata.target_fps,
            frame_deadline_ms=(
                metadata.frame_deadline_ms
            ),
        )

        return config, metadata, planned_run

    def create_summary_for_run(
            self,
            planned_run: PlannedRun,
            config: dict,
            *,
            frame_hash: str = "c" * 64,
    ):
        summary_value = (
            self.create_execution_summary_value()
        )
        measured_frames = config["execution"][
            "measured_frames"
        ]

        summary_value.update(
            {
                "experiment_id": (
                    planned_run.experiment_id
                ),
                "run_id": planned_run.run_id,
                "execution_order": (
                    planned_run.execution_order
                ),
                "phase": planned_run.phase,
                "platform": planned_run.platform,
                "architecture": (
                    planned_run.architecture
                ),
                "algorithm": planned_run.algorithm,
                "resolution": {
                    "width": (
                        planned_run.resolution.width
                    ),
                    "height": (
                        planned_run.resolution.height
                    ),
                },
                "trial_number": (
                    planned_run.trial_number
                ),
                "warmup_frame_count": (
                    config["execution"][
                        "warmup_frames"
                    ]
                ),
                "measured_frame_count": (
                    measured_frames
                ),
                "processed_frame_count": (
                    measured_frames
                ),
                "dropped_frame_count": 0,
                "skipped_frame_count": 0,
                "deadline_met_count": (
                    measured_frames
                ),
                "deadline_miss_count": 0,
                "frame_results_sha256": (
                    frame_hash
                ),
            }
        )

        return execution_summary_from_dict(
            summary_value
        )

    def prepare_finalizable_run(
        self,
        run_directory: Path,
    ):
        config, metadata, planned_run = (
            self.create_metadata_and_planned_run()
        )

        frame_results_path = (
            run_directory
            / FRAME_RESULTS_FILE_NAME
        )

        measured_frame_count = (
            config["execution"]["measured_frames"]
        )
        origin_timestamp_ns = 1_000_000_000
        frame_period_ns = 33_333_333

        context = RealtimeRunContext(
            architecture=planned_run.architecture,
            algorithm=planned_run.algorithm,
            resolution=(
                f"{planned_run.resolution.width}"
                f"x{planned_run.resolution.height}"
            ),
            trial=planned_run.trial_number,
            origin_timestamp_ns=origin_timestamp_ns,
            deadline_ms=planned_run.frame_deadline_ms,
        )

        records = []

        for frame_index in range(
                1,
                measured_frame_count + 1,
        ):
            scheduled_timestamp_ns = (
                    origin_timestamp_ns
                    + (frame_index - 1) * frame_period_ns
            )

            records.append(
                create_processed_record(
                    context,
                    frame_index=frame_index,
                    scheduled_timestamp_ns=(
                        scheduled_timestamp_ns
                    ),
                    enqueued_timestamp_ns=(
                            scheduled_timestamp_ns
                            + 1_000_000
                    ),
                    processing_start_timestamp_ns=(
                            scheduled_timestamp_ns
                            + 2_000_000
                    ),
                    processing_end_timestamp_ns=(
                            scheduled_timestamp_ns
                            + 7_000_000
                    ),
                )
            )

        written_record_count = write_frame_records(
            records,
            frame_results_path,
        )

        self.assertEqual(
            written_record_count,
            measured_frame_count,
        )

        frame_results_content = (
            frame_results_path.read_bytes()
        )

        frame_results_sha256 = hashlib.sha256(
            frame_results_content
        ).hexdigest()

        summary = self.create_summary_for_run(
            planned_run,
            config,
            frame_hash=frame_results_sha256,
        )

        self.write_execution_summary(
            run_directory,
            asdict(summary),
        )

        save_run_metadata_atomic(
            metadata,
            config=config,
            output_path=(
                run_directory
                / RUN_METADATA_FILE_NAME
            ),
        )

        return (
            config,
            metadata,
            planned_run,
            frame_results_path,
        )

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

    def test_bundle_manifest_is_written_atomically(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            manifest = self.create_manifest()

            output_path = (
                write_run_bundle_manifest_atomic(
                    manifest,
                    run_directory,
                )
            )

            with output_path.open(
                "r",
                encoding="utf-8",
            ) as manifest_file:
                stored_value = json.load(
                    manifest_file
                )

        self.assertEqual(
            output_path.name,
            RUN_BUNDLE_MANIFEST_FILE_NAME,
        )
        self.assertEqual(
            stored_value,
            manifest.to_dict(),
        )

    def test_existing_bundle_manifest_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            manifest = self.create_manifest()

            output_path = (
                write_run_bundle_manifest_atomic(
                    manifest,
                    run_directory,
                )
            )
            original_content = (
                output_path.read_text(
                    encoding="utf-8",
                )
            )

            with self.assertRaisesRegex(
                ControlledIlluminationRunBundleError,
                "already been finalized",
            ):
                write_run_bundle_manifest_atomic(
                    manifest,
                    run_directory,
                )

            self.assertEqual(
                output_path.read_text(
                    encoding="utf-8",
                ),
                original_content,
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
                metadata_dry_run="false", # type: ignore[arg-type]
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

    def test_summary_matches_planned_run(
        self,
    ) -> None:
        config, _, planned_run = (
            self.create_metadata_and_planned_run()
        )
        summary = self.create_summary_for_run(
            planned_run,
            config,
        )

        validate_execution_summary_matches_run(
            summary,
            planned_run,
        )

    def test_summary_run_mismatch_is_rejected(
        self,
    ) -> None:
        config, _, planned_run = (
            self.create_metadata_and_planned_run()
        )
        summary = self.create_summary_for_run(
            planned_run,
            config,
        )
        mismatched_run = replace(
            planned_run,
            run_id="different-run",
        )

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "run_id",
        ):
            validate_execution_summary_matches_run(
                summary,
                mismatched_run,
            )

    def test_metadata_matches_planned_run(
        self,
    ) -> None:
        _, metadata, planned_run = (
            self.create_metadata_and_planned_run()
        )

        validate_metadata_matches_run(
            metadata,
            planned_run,
        )

    def test_metadata_angle_mismatch_is_rejected(
        self,
    ) -> None:
        _, metadata, planned_run = (
            self.create_metadata_and_planned_run()
        )
        mismatched_metadata = replace(
            metadata,
            incidence_angle_degrees=(
                metadata.incidence_angle_degrees
                + 1.0
            ),
        )

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "incidence_angle_degrees",
        ):
            validate_metadata_matches_run(
                mismatched_metadata,
                planned_run,
            )

    def test_constant_source_mismatch_is_rejected(
        self,
    ) -> None:
        _, metadata, planned_run = (
            self.create_metadata_and_planned_run()
        )
        source_run = replace(
            planned_run,
            phase="constant_source",
            target_illuminance_lux=None,
            source_output_setting="level-1",
        )
        source_metadata = replace(
            metadata,
            phase="constant_source",
            target_illuminance_lux=None,
            source_output_setting="level-2",
        )

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "source-output",
        ):
            validate_metadata_matches_run(
                source_metadata,
                source_run,
            )

    def test_summary_frame_hash_is_validated(
        self,
    ) -> None:
        config, _, planned_run = (
            self.create_metadata_and_planned_run()
        )
        summary = self.create_summary_for_run(
            planned_run,
            config,
            frame_hash="e" * 64,
        )
        frame_artifact = RunBundleArtifact(
            file_name=FRAME_RESULTS_FILE_NAME,
            sha256="e" * 64,
            size_bytes=100,
        )

        validate_summary_frame_hash(
            summary,
            (frame_artifact,),
        )

        mismatched_artifact = replace(
            frame_artifact,
            sha256="f" * 64,
        )

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "SHA-256",
        ):
            validate_summary_frame_hash(
                summary,
                (mismatched_artifact,),
            )

    def test_summary_counts_match_configuration(
        self,
    ) -> None:
        config, _, planned_run = (
            self.create_metadata_and_planned_run()
        )
        summary = self.create_summary_for_run(
            planned_run,
            config,
        )

        validate_summary_counts_against_config(
            summary,
            config,
        )

        mismatched_config = {
            **config,
            "execution": {
                **config["execution"],
                "measured_frames": (
                    summary.measured_frame_count
                    + 1
                ),
            },
        }

        with self.assertRaisesRegex(
            ControlledIlluminationRunBundleError,
            "measured-frame",
        ):
            validate_summary_counts_against_config(
                summary,
                mismatched_config,
            )
    def test_completed_run_bundle_is_finalized(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            (
                config,
                metadata,
                planned_run,
                _,
            ) = self.prepare_finalizable_run(
                run_directory
            )

            manifest, manifest_path = (
                finalize_run_bundle_atomic(
                    run_directory,
                    planned_run,
                    config,
                    VALID_PLAN_SHA256,
                    "2026-08-26T11:00:00Z",
                )
            )

            stored_manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8",
                )
            )

        self.assertEqual(
            stored_manifest,
            manifest.to_dict(),
        )
        self.assertEqual(
            manifest.experiment_id,
            planned_run.experiment_id,
        )
        self.assertEqual(
            manifest.run_id,
            planned_run.run_id,
        )
        self.assertEqual(
            manifest.metadata_dry_run,
            metadata.dry_run,
        )

    def test_modified_frame_results_are_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            (
                config,
                _,
                planned_run,
                frame_results_path,
            ) = self.prepare_finalizable_run(
                run_directory
            )

            frame_results_path.write_bytes(
                b"modified frame results\n"
            )

            with self.assertRaisesRegex(
                ControlledIlluminationRunBundleError,
                "SHA-256",
            ):
                finalize_run_bundle_atomic(
                    run_directory,
                    planned_run,
                    config,
                    VALID_PLAN_SHA256,
                    "2026-08-26T11:00:00Z",
                )

            self.assertFalse(
                (
                    run_directory
                    / RUN_BUNDLE_MANIFEST_FILE_NAME
                ).exists()
            )

    def test_early_finalization_time_is_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            (
                config,
                _,
                planned_run,
                _,
            ) = self.prepare_finalizable_run(
                run_directory
            )

            with self.assertRaisesRegex(
                ControlledIlluminationRunBundleError,
                "cannot be earlier",
            ):
                finalize_run_bundle_atomic(
                    run_directory,
                    planned_run,
                    config,
                    VALID_PLAN_SHA256,
                    "2026-08-26T09:59:00Z",
                )


if __name__ == "__main__":
    unittest.main()