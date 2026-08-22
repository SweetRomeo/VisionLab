from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from benchmarks.experiments.controlled_illumination_metadata import (
    ControlledIlluminationConfigError,
    ControlledIlluminationMetadataError,
    load_controlled_illumination_config,
    load_run_metadata,
)


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one or more controlled-illumination "
            "run metadata files."
        )
    )
    parser.add_argument(
        "metadata_paths",
        nargs="+",
        type=Path,
        help="Run metadata JSON files to validate.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional controlled-illumination "
            "configuration path."
        ),
    )

    return parser.parse_args(arguments)


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parsed_arguments = parse_arguments(arguments)

    try:
        config = load_controlled_illumination_config(
            parsed_arguments.config
        )

        for metadata_path in (
            parsed_arguments.metadata_paths
        ):
            metadata = load_run_metadata(
                metadata_path,
                config=config,
            )

            print(
                f"Valid metadata: {metadata_path}"
            )
            print(
                f"  Experiment ID: "
                f"{metadata.experiment_id}"
            )
            print(
                f"  Run ID: {metadata.run_id}"
            )
            print(
                f"  Phase: {metadata.phase}"
            )
            print(
                f"  Platform: {metadata.platform}"
            )
            print(
                f"  Architecture: "
                f"{metadata.architecture}"
            )
            print(
                f"  Dry run: {metadata.dry_run}"
            )

    except (
        FileNotFoundError,
        ControlledIlluminationConfigError,
        ControlledIlluminationMetadataError,
    ) as error:
        print(
            f"Metadata validation failed: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        "Controlled-illumination metadata "
        "validation passed."
    )
    print(
        "Validated files: "
        f"{len(parsed_arguments.metadata_paths)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())