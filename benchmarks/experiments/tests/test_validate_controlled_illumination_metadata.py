from contextlib import (
    redirect_stderr,
    redirect_stdout,
)
import io
from pathlib import Path
import tempfile
import unittest

from benchmarks.experiments.controlled_illumination_metadata import (
    load_controlled_illumination_config,
    save_run_metadata_atomic,
)
from benchmarks.experiments.generate_dry_run_metadata import (
    build_dry_run_metadata,
)
from benchmarks.experiments.validate_controlled_illumination_metadata import (
    main as validator_main,
)


class ControlledIlluminationValidatorTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.config = (
            load_controlled_illumination_config()
        )
        self.metadata = build_dry_run_metadata(
            self.config,
            git_commit_sha="c" * 40,
        )

    def test_valid_metadata_returns_success(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = (
                Path(directory)
                / "run_metadata.json"
            )

            save_run_metadata_atomic(
                self.metadata,
                config=self.config,
                output_path=metadata_path,
            )

            standard_output = io.StringIO()

            with redirect_stdout(standard_output):
                exit_code = validator_main(
                    [str(metadata_path)]
                )

            self.assertEqual(
                exit_code,
                0,
            )
            self.assertIn(
                "metadata validation passed",
                standard_output.getvalue(),
            )

    def test_invalid_metadata_returns_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = (
                Path(directory)
                / "invalid_metadata.json"
            )

            metadata_path.write_text(
                "{}\n",
                encoding="utf-8",
            )

            standard_error = io.StringIO()

            with redirect_stderr(standard_error):
                exit_code = validator_main(
                    [str(metadata_path)]
                )

            self.assertEqual(
                exit_code,
                1,
            )
            self.assertIn(
                "Metadata validation failed",
                standard_error.getvalue(),
            )


if __name__ == "__main__":
    unittest.main()