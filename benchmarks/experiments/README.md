# Controlled-Illumination Experiment Infrastructure

This directory contains the configuration, metadata and validation infrastructure used to prepare and execute VisionLab controlled-illumination experiments.

The scientific experiment procedure is defined in:

```text
benchmarks/experiments/controlled_illumination_protocol.md
```

The infrastructure in this directory does not perform the physical experiment automatically. It ensures that each experiment run is configurable, traceable, reproducible and validated before its results are analyzed.

## Components

| File                                                      | Purpose                                                                                                          |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `config/controlled_illumination_config.json`              | Defines the experiment phases, lighting conditions, algorithms, architectures, platforms and execution settings. |
| `controlled_illumination_metadata.py`                     | Defines metadata models, configuration validation, run validation and atomic JSON storage.                       |
| `generate_dry_run_metadata.py`                            | Generates a validated metadata example without requiring experiment hardware.                                    |
| `validate_controlled_illumination_metadata.py`            | Validates one or more saved run metadata files.                                                                  |
| `tests/test_controlled_illumination_metadata.py`          | Tests configuration, measurement, metadata and storage behavior.                                                 |
| `tests/test_validate_controlled_illumination_metadata.py` | Tests the metadata-validation CLI.                                                                               |
| `controlled_illumination_run_planner.py` | Expands, validates, randomizes and serializes the experiment run plan. |
| `generate_controlled_illumination_run_plan.py` | Provides dry-run validation and JSON/CSV manifest generation through the CLI. |
| `tests/test_controlled_illumination_run_planner.py` | Tests matrix expansion, duplicate rejection, deterministic ordering and atomic manifests. |
| `tests/test_generate_controlled_illumination_run_plan.py` | Tests run-planner CLI dry-run and manifest-generation behavior. |
| `controlled_illumination_run_state.py`                    | Defines run states, validated transitions, progress persistence and resume integrity.                            |
| `manage_controlled_illumination_run.py`                   | Provides CLI commands for initializing, resuming and updating experiment execution progress.                    |
| `tests/test_controlled_illumination_run_state.py`         | Tests state transitions, progress integrity, persistence and timestamp chronology.                              |
| `tests/test_manage_controlled_illumination_run.py`        | Tests the controlled-illumination run-management CLI.                                                            |

## Experiment phases

Two experiment phases are represented separately.

### Constant-lux phase

The light-source output is adjusted until the required target-plane illuminance is reached. The incidence angle is then varied while the target illuminance is kept approximately constant.

Metadata field:

```text
phase = constant_lux
```

A `constant_lux` run must define `target_illuminance_lux`.

### Constant-source phase

The light-source output setting remains fixed while the incidence angle is changed. The resulting illuminance is measured rather than controlled.

Metadata field:

```text
phase = constant_source
```

A `constant_source` run must not define `target_illuminance_lux`. The measured centre and corner lux values are still required.

Results from the two phases must not be combined as though they represented the same experimental condition.

## Configuration

The shared configuration file is:

```text
benchmarks/experiments/config/controlled_illumination_config.json
```

It defines:

* Target illuminance levels
* Light-incidence angles
* Algorithms
* Software architectures
* Computing platforms
* Resolutions
* Trial count
* Target frame rate
* Warm-up and measured frame counts
* Queue capacity and drop policy
* Camera modes
* Illuminance-measurement positions
* Required run metadata
* Output directory

The algorithm names and parameters must remain compatible with:

```text
benchmarks/config/benchmark_config.json
```

Real-time execution settings must remain compatible with:

```text
benchmarks/config/realtime_config.json
```

Validate the configuration from the repository root:

```bash
python benchmarks/experiments/controlled_illumination_metadata.py
```

Expected output:

```text
Controlled-illumination configuration is valid.
```

## Dry-run metadata

A dry run validates the metadata workflow without a lux meter, camera, Raspberry Pi or NVIDIA Jetson device.

Run it from the repository root:

```bash
python -m benchmarks.experiments.generate_dry_run_metadata
```

The command:

1. Loads and validates the experiment configuration.
2. Reads the current Git commit SHA.
3. Generates unique experiment and run identifiers.
4. Creates placeholder camera, lighting and platform measurements.
5. Validates the completed metadata record.
6. Writes the metadata file atomically.

Expected output begins with:

```text
Controlled-illumination dry run passed.
Experiment ID:
Run ID:
Metadata created:
```

Dry-run records contain:

```text
dry_run = true
```

Dry-run records are infrastructure tests and must never be interpreted as physical experiment results.

## Output structure

By default, metadata files are written under:

```text
benchmarks/results/controlled_illumination/
```

The directory structure is:

```text
benchmarks/results/controlled_illumination/
└── <platform>/
    └── <experiment_id>/
        └── <run_id>/
            └── run_metadata.json
```

Generated results must not be committed to Git.

After generating a dry run, verify this with:

```bash
git status --short
```

The generated `run_metadata.json` file should not appear.

## Metadata validation

Validate a single metadata file:

```bash
python -m \
benchmarks.experiments.validate_controlled_illumination_metadata \
path/to/run_metadata.json
```

Validate multiple metadata files:

```bash
python -m \
benchmarks.experiments.validate_controlled_illumination_metadata \
path/to/first/run_metadata.json \
path/to/second/run_metadata.json
```

Use a custom configuration when required:

```bash
python -m \
benchmarks.experiments.validate_controlled_illumination_metadata \
--config path/to/controlled_illumination_config.json \
path/to/run_metadata.json
```

Successful validation ends with:

```text
Controlled-illumination metadata validation passed.
```

A metadata record is rejected when it contains conditions such as:

* Missing required fields
* Invalid experiment phases
* Unsupported algorithms, architectures or platforms
* Unsupported resolutions
* Invalid trial numbers
* Unsupported incidence angles
* Invalid constant-lux or constant-source fields
* Missing camera settings
* Invalid timestamps
* Invalid Git commit SHAs
* Inconsistent illuminance summaries
* Invalid temperature measurements
* Invalid frame deadlines
* Unexpected metadata fields

## Illuminance measurements

Lux must be recorded at five target-plane positions:

* Centre
* Top left
* Top right
* Bottom left
* Bottom right

The metadata infrastructure calculates:

* Mean lux
* Minimum lux
* Maximum lux

When metadata is loaded, the stored summary values are recalculated and compared with the five original measurements. A modified or inconsistent summary is rejected.

## Atomic storage

Metadata is first written to a temporary file in the destination directory. The temporary file is flushed to disk and then atomically moved to the final `run_metadata.json` path.

This prevents an interrupted write from leaving a partially written official metadata file.

## Running the tests

Run all controlled-illumination tests from the repository root:

```bash
python -m unittest discover \
-s benchmarks/experiments/tests \
-p "test_*.py" \
-v
```

The tests cover:

* Valid configuration loading
* Lux summary calculation
* Invalid lux rejection
* Experiment-condition validation
* Trial and incidence-angle validation
* Camera-setting validation
* Unique identifier generation
* Atomic metadata storage
* JSON metadata loading
* Dry-run metadata generation
* CLI success behavior
* CLI failure behavior

All tests must pass before publishing changes or collecting official experiment results.

## Official experiment workflow

For each official experiment run:

1. Confirm that the working tree is clean.
2. Record the full Git commit SHA.
3. Select the experiment phase.
4. Configure the platform, architecture, algorithm and resolution.
5. Fix and record the camera settings.
6. Configure the light source and incidence angle.
7. Measure lux at all five target positions.
8. Record device temperature and power settings.
9. Execute the warm-up and measured frames.
10. Complete and validate the run metadata.
11. Save the metadata atomically.
12. Validate the generated file with the CLI.
13. Archive the raw results, configuration and metadata together.

Official results must not be used when metadata validation fails.

## Current scope

The current infrastructure supports configuration and metadata preparation on the desktop reference platform.

Physical measurements, Raspberry Pi deployment, NVIDIA Jetson deployment, hardware-specific acceleration and final architecture selection remain separate future stages.

## Controlled-illumination run planning

The controlled-illumination run planner expands the experiment
configuration into an ordered and reproducible execution manifest.

The planner:

- Expands the complete experimental matrix.
- Supports `constant_lux` and `constant_source` phases.
- Randomizes execution order using the configured deterministic seed.
- Assigns sequential execution numbers and unique run identifiers.
- Rejects duplicate experimental conditions.
- Writes each JSON and CSV manifest using atomic file replacement.

### Constant-source configuration

The `constant_source` phase requires explicit source-output settings
under `experiment_matrix`:

```json
"source_output_settings": [
  "device_setting_low",
  "device_setting_medium",
  "device_setting_high"
]
```

These values must represent real, reproducible controls supported by
the selected illumination device. Do not add placeholder values to
official experiment configurations.

Until these settings are defined, the planner intentionally rejects
the configuration instead of inventing hardware-specific values.

Both dry-run validation and manifest generation will fail until
`source_output_settings` contains the real settings supported by the
selected illumination device.

### Validate the plan without writing files

After configuring the real source-output settings, run from the
repository root:

```bash
python -m \
benchmarks.experiments.generate_controlled_illumination_run_plan \
--experiment-id controlled-illumination-pilot \
--dry-run
```

A dry run expands, randomizes and validates the complete execution
plan without writing manifest files.

### Generate manifests

After the source-output settings have been measured and added to the
configuration, generate the manifests with:

```bash
python -m \
benchmarks.experiments.generate_controlled_illumination_run_plan \
--experiment-id controlled-illumination-pilot
```

By default, manifests are written under:

```text
benchmarks/results/controlled_illumination/<experiment-id>/
```

The generated files are:

```text
run_plan.json
run_plan.csv
```

An alternative output directory can be selected with:

```bash
python -m \
benchmarks.experiments.generate_controlled_illumination_run_plan \
--experiment-id controlled-illumination-pilot \
--output-directory path/to/output
```

Generated run plans are experimental outputs and must not be committed
to the repository.

## Resumable run execution tracking

The run-tracking CLI records the execution state of every run in a controlled-illumination manifest. It supports interruption recovery without losing completed, failed or skipped run information.

By default, progress is stored beside the selected run plan:

```text
<run-plan-directory>/run_progress.json
```

The progress file is written atomically and bound to the run-plan manifest by its SHA-256 hash. A progress file cannot be resumed with a modified or different run plan.

### Run states

Each run uses one of the following states:

* `planned`
* `running`
* `completed`
* `failed`
* `skipped`

Only one run may be in the `running` state at a time. Invalid state transitions and inconsistent timestamps are rejected.

### Initialize or resume progress

From the repository root:

```bash
python -m \
benchmarks.experiments.manage_controlled_illumination_run \
--plan path/to/run_plan.json \
init
```

If a compatible progress file already exists, it is loaded instead of overwritten.

### Display experiment status

```bash
python -m \
benchmarks.experiments.manage_controlled_illumination_run \
--plan path/to/run_plan.json \
status
```

### Start the next planned run

```bash
python -m \
benchmarks.experiments.manage_controlled_illumination_run \
--plan path/to/run_plan.json \
start-next
```

The command selects the next run according to `execution_order`.

### Complete a running run

```bash
python -m \
benchmarks.experiments.manage_controlled_illumination_run \
--plan path/to/run_plan.json \
complete \
--run-id RUN_ID
```

### Mark a running run as failed

```bash
python -m \
benchmarks.experiments.manage_controlled_illumination_run \
--plan path/to/run_plan.json \
fail \
--run-id RUN_ID \
--reason "Failure reason"
```

### Skip a planned run

```bash
python -m \
benchmarks.experiments.manage_controlled_illumination_run \
--plan path/to/run_plan.json \
skip \
--run-id RUN_ID \
--reason "Skip reason"
```

### Return a failed or skipped run to planned status

```bash
python -m \
benchmarks.experiments.manage_controlled_illumination_run \
--plan path/to/run_plan.json \
replan \
--run-id RUN_ID
```

Replanning preserves the previous attempt count. Starting the run again increments the attempt count.

### Custom progress path

Use `--progress` before the subcommand when the progress file must be stored elsewhere:

```bash
python -m \
benchmarks.experiments.manage_controlled_illumination_run \
--plan path/to/run_plan.json \
--progress path/to/run_progress.json \
status
```

Generated `run_progress.json` files are experiment artifacts and must not be committed to Git.

## Controlled-illumination execution orchestrator

The execution orchestrator connects a generated run-plan manifest to
the architecture-specific experiment runners.

Each invocation:

1. Loads and validates `run_plan.json`.
2. Loads or creates `run_progress.json`.
3. Selects the next run with `planned` status.
4. Atomically marks the selected run as `running`.
5. Executes the command configured for its architecture.
6. Atomically records the run as `completed` or `failed`.

Only one planned run is executed per invocation. Repeated invocations
continue according to the manifest's `execution_order`.

### Runner configuration

The committed example configuration is:

```text
benchmarks/experiments/config/controlled_illumination_runner_config.example.json
```

