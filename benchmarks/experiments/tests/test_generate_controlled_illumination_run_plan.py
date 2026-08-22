from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.experiments.generate_controlled_illumination_run_plan import (
    create_argument_parser,
    run_cli,
)


class ControlledIlluminationRunPlanCliTests(
    unittest.TestCase
):
    def create_test_config(
        self,
        directory: Path,
    ) -> Path:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "controlled_illumination_config.json"
        )

        with source_path.open(
            "r",
            encoding="utf-8",
        ) as source_file:
            config = json.load(source_file)

        matrix = config["experiment_matrix"]
        matrix[
            "target_illuminance_levels_lux"
        ] = [50]
        matrix[
            "source_output_settings"
        ] = [25]
        matrix[
            "incidence_angles_degrees"
        ] = [0]
        matrix["algorithms"] = ["original"]
        matrix["architectures"] = [
            "pure_python"
        ]
        matrix["platforms"] = ["desktop"]
        matrix["resolutions"] = [
            {
                "width": 640,
                "height": 480,
            }
        ]
        matrix["trial_count"] = 1

        config_path = directory / "config.json"

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as config_file:
            json.dump(
                config,
                config_file,
                indent=2,
            )

        return config_path

    def test_dry_run_does_not_write_manifests(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config_path = self.create_test_config(
                temporary_path
            )
            output_path = (
                temporary_path / "output"
            )

            arguments = (
                create_argument_parser().parse_args(
                    [
                        "--config",
                        str(config_path),
                        "--experiment-id",
                        "experiment-cli-dry-run",
                        "--output-directory",
                        str(output_path),
                        "--dry-run",
                    ]
                )
            )

            captured_output = StringIO()

            with redirect_stdout(captured_output):
                exit_code = run_cli(arguments)

            self.assertEqual(exit_code, 0)
            self.assertFalse(output_path.exists())
            self.assertIn(
                "Run count: 2",
                captured_output.getvalue(),
            )
            self.assertIn(
                "no manifest files written",
                captured_output.getvalue(),
            )

    def test_cli_writes_json_and_csv_manifests(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            config_path = self.create_test_config(
                temporary_path
            )
            output_path = (
                temporary_path / "output"
            )

            arguments = (
                create_argument_parser().parse_args(
                    [
                        "--config",
                        str(config_path),
                        "--experiment-id",
                        "experiment-cli-write",
                        "--output-directory",
                        str(output_path),
                    ]
                )
            )

            with redirect_stdout(StringIO()):
                exit_code = run_cli(arguments)

            self.assertEqual(exit_code, 0)

            json_path = output_path / "run_plan.json"
            csv_path = output_path / "run_plan.csv"

            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())

            with json_path.open(
                "r",
                encoding="utf-8",
            ) as json_file:
                plan_data = json.load(json_file)

            self.assertEqual(
                plan_data["run_count"],
                2,
            )


if __name__ == "__main__":
    unittest.main()