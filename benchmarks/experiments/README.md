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
| `controlled_illumination_run_bundle.py` | Validates completed run artifacts and creates immutable run-bundle manifests. |
| `finalize_controlled_illumination_run_bundle.py` | Provides the completed-run bundle finalization CLI. |
| `tests/test_controlled_illumination_run_bundle.py` | Tests bundle models, artifact integrity, cross-file consistency and atomic finalization. |
| `tests/test_finalize_controlled_illumination_run_bundle.py` | Tests finalization CLI selection and orchestration behavior. |

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

A bundle with `metadata_dry_run = true` is an infrastructure-validation artifact. It must not be interpreted as an official physical experiment
or included in official controlled-illumination analysis.

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

Copy the example configuration into the experiment output directory:

```bash
VISIONLAB_EXPERIMENT_DIRECTORY="benchmarks/results/controlled_illumination/controlled-illumination-pilot"

mkdir -p "$VISIONLAB_EXPERIMENT_DIRECTORY"

cp \
benchmarks/experiments/config/controlled_illumination_runner_config.example.json \
"$VISIONLAB_EXPERIMENT_DIRECTORY/runner_config.json"
```

Replace every placeholder command with the real runner paths for the
selected platform.

The runner configuration must contain exactly these architectures:

```text
pure_python
hybrid
pure_cpp
```

Each architecture defines:

* `arguments`: command and arguments passed directly to the process
* `working_directory`: process working directory
* `environment`: optional base environment values
* `timeout_seconds`: optional positive execution timeout

Relative working directories are resolved from the repository root.

The example paths are placeholders and must not be used for official
experiments. The selected commands must execute exactly one planned
condition. Do not configure a general benchmark runner that expands
and executes the complete benchmark matrix.

### Architecture-specific runners

The repository provides one controlled-illumination entry point for
each software architecture:

| Architecture      | Runner module                                                       |
| ----------------- | ------------------------------------------------------------------- |
| Pure Python       | `benchmarks.experiments.controlled_illumination_pure_python_runner` |
| Hybrid Python+C++ | `benchmarks.experiments.controlled_illumination_hybrid_runner`      |
| Pure C++          | `benchmarks.experiments.controlled_illumination_pure_cpp_runner`    |

Each entry point executes exactly one condition received from the
orchestrator. The runners validate the selected architecture,
algorithm, resolution, trial number, target frame rate and frame
deadline before processing begins.

#### Pure Python runner

Use the Python interpreter from the Pure Python virtual environment:

```json
{
  "arguments": [
    "C:/path/to/VisionLab/pure-python/.venv/Scripts/python.exe",
    "-m",
    "benchmarks.experiments.controlled_illumination_pure_python_runner"
  ],
  "working_directory": ".",
  "environment": {},
  "timeout_seconds": 3600
}
```

On Linux and macOS, use the corresponding virtual-environment
interpreter:

```text
/path/to/VisionLab/pure-python/.venv/bin/python
```

#### Hybrid Python+C++ runner

Use the Hybrid virtual environment and provide the directory containing
the Release `visionlab_cpp` module:

```json
{
  "arguments": [
    "C:/path/to/VisionLab/hybrid-python-cpp/.venv/Scripts/python.exe",
    "-m",
    "benchmarks.experiments.controlled_illumination_hybrid_runner"
  ],
  "working_directory": ".",
  "environment": {
    "VISIONLAB_CPP_MODULE_DIR": "C:/path/to/VisionLab/hybrid-python-cpp/build/RELEASE_DIRECTORY"
  },
  "timeout_seconds": 3600
}
```

On Linux and macOS, use the corresponding Hybrid virtual-environment
interpreter and the directory containing the Release `visionlab_cpp`
shared module.

The Hybrid runner rejects a missing or unsuitable compiled module. All
official Hybrid measurements must use a Release build.

#### Pure C++ runner

The Pure C++ entry point is a Python adapter around the Release
`VisionLabCppRealtime` executable:

```json
{
  "arguments": [
    "C:/path/to/VisionLab/pure-python/.venv/Scripts/python.exe",
    "-m",
    "benchmarks.experiments.controlled_illumination_pure_cpp_runner"
  ],
  "working_directory": ".",
  "environment": {
    "VISIONLAB_CPP_REALTIME_EXECUTABLE": "C:/path/to/VisionLab/cpp-opencv-core/build/RELEASE_DIRECTORY/VisionLabCppRealtime.exe"
  },
  "timeout_seconds": 3600
}
```

On Linux and macOS, configure the path to the Release executable
without the `.exe` extension.

The Pure C++ adapter:

1. Rejects Debug executable paths.
2. Starts `VisionLabCppRealtime` in single-condition mode.
3. Validates the generated C++ frame-result CSV.
4. Converts the results into the shared artifact schema.
5. Writes the official CSV and execution summary atomically.

On Windows, `Qt6Core.dll` and the required OpenCV runtime DLLs must be
available beside `VisionLabCppRealtime.exe` or through `PATH`. The CMake
Release build copies these runtime files into the executable directory.

#### Output artifacts

A successful architecture runner creates:

```text
<results-root>/
└── <platform>/
    └── <experiment-id>/
        └── <run-id>/
            ├── realtime_frame_results.csv
            └── execution_summary.json
```

The CSV contains the shared per-frame timing, deadline and frame-status
schema.

The JSON summary is written last and acts as the completed-run marker.
Existing completed artifacts are not silently overwritten.

#### Current input-source scope

The current architecture runners use the deterministic benchmark video
to validate orchestration, timing, processing and artifact generation
without experiment hardware.

These software smoke-test results must not be interpreted as physical
controlled-illumination measurements. Official lux- and angle-dependent
experiments require the live-camera input adapter and calibrated
illumination hardware.

### Run environment

The orchestrator passes the selected condition to the architecture
runner through environment variables:

```text
VISIONLAB_EXPERIMENT_ID
VISIONLAB_RUN_ID
VISIONLAB_EXECUTION_ORDER
VISIONLAB_PHASE
VISIONLAB_PLATFORM
VISIONLAB_ARCHITECTURE
VISIONLAB_ALGORITHM
VISIONLAB_RESOLUTION_WIDTH
VISIONLAB_RESOLUTION_HEIGHT
VISIONLAB_TRIAL_NUMBER
VISIONLAB_INCIDENCE_ANGLE_DEGREES
VISIONLAB_TARGET_ILLUMINANCE_LUX
VISIONLAB_SOURCE_OUTPUT_SETTING
VISIONLAB_TARGET_FPS
VISIONLAB_FRAME_DEADLINE_MS
```

Condition-specific values override matching values from the base
runner environment.

For `constant_lux` runs,
`VISIONLAB_SOURCE_OUTPUT_SETTING` is empty. For `constant_source`
runs, `VISIONLAB_TARGET_ILLUMINANCE_LUX` is empty.

Architecture runners must validate these values before starting image
capture or processing.

Runner-specific optional environment values include:

```text
VISIONLAB_RESULTS_ROOT
VISIONLAB_CPP_MODULE_DIR
VISIONLAB_CPP_REALTIME_EXECUTABLE
```

`VISIONLAB_RESULTS_ROOT` selects the root directory for generated run
artifacts.

`VISIONLAB_CPP_MODULE_DIR` identifies the directory containing the
Release Hybrid module.

`VISIONLAB_CPP_REALTIME_EXECUTABLE` identifies the Release Pure C++
real-time executable.

`VISIONLAB_CPP_FRAME_RESULTS_PATH` is internal to the Pure C++ adapter
and must not be configured manually.

### Execute the next planned run

Run from the repository root:

```bash
python -m \
benchmarks.experiments.execute_controlled_illumination_run \
--plan path/to/run_plan.json \
--runner-config path/to/runner_config.json
```

By default, progress is stored beside the run plan as:

```text
run_progress.json
```

A custom progress path can be selected with:

```bash
python -m \
benchmarks.experiments.execute_controlled_illumination_run \
--plan path/to/run_plan.json \
--progress path/to/run_progress.json \
--runner-config path/to/runner_config.json
```

### Exit behavior

The command uses the following exit codes:

* `0`: the selected run completed successfully, or no planned runs remain
* `1`: configuration, validation or architecture-runner failure
* `130`: execution was interrupted by the user

A successful runner result transitions the selected run from
`running` to `completed`.

A non-zero runner result or execution exception transitions the run
to `failed` and records the failure reason.

When the user interrupts execution with `Ctrl+C`, the run is atomically
marked as `failed` before the command exits with code `130`. The run can
then be returned to `planned` status with the run-tracking CLI before
being attempted again.

Generated progress files and local runner configurations are
experiment artifacts. Archive them together with the run plan,
environment metadata and result files, but do not commit them to Git.

## Finalizing completed run bundles

After an architecture runner finishes successfully, finalize the run
bundle before using its results in scientific analysis.

A finalizable run directory must contain these required artifacts:

```text
realtime_frame_results.csv
execution_summary.json
run_metadata.json
```

The finalization process:

1. Loads and validates the controlled-illumination configuration.
2. Loads the selected run-plan manifest.
3. Locates the requested planned run.
4. Calculates the SHA-256 hash of the complete run plan.
5. Validates the execution summary against the planned condition.
6. Validates the run metadata against the planned condition.
7. Validates warm-up and measured-frame counts.
8. Validates the complete frame-results CSV, including row identities,
   sequential frame indices, frame statuses and deadline statistics.
9. Verifies the frame-results SHA-256 hash.
10. Records each required artifact's SHA-256 hash and file size.
11. Atomically publishes an immutable bundle manifest without
    overwriting an existing manifest.

Run the finalizer from the repository root:

```bash
python -m \
benchmarks.experiments.finalize_controlled_illumination_run_bundle \
--plan path/to/run_plan.json \
--run-directory path/to/completed/run-directory \
--run-id RUN_ID
```

Use a non-default experiment configuration when required:

```bash
python -m \
benchmarks.experiments.finalize_controlled_illumination_run_bundle \
--plan path/to/run_plan.json \
--run-directory path/to/completed/run-directory \
--run-id RUN_ID \
--config path/to/controlled_illumination_config.json
```

### Validate without finalizing

A completed run bundle can be validated without creating or modifying
`run_bundle_manifest.json`:

```bash
python -m \
benchmarks.experiments.finalize_controlled_illumination_run_bundle \
--plan path/to/run_plan.json \
--run-directory path/to/completed/run-directory \
--run-id RUN_ID \
--validate-only
```

A custom configuration can also be used in validation-only mode:

```bash
python -m \
benchmarks.experiments.finalize_controlled_illumination_run_bundle \
--plan path/to/run_plan.json \
--run-directory path/to/completed/run-directory \
--run-id RUN_ID \
--config path/to/controlled_illumination_config.json \
--validate-only
```

Validation-only mode performs the complete bundle validation but does
not create, replace or modify any experiment artifact.

A successful finalization creates:

```text
run_bundle_manifest.json
```

The manifest records:

* Experiment and run identifiers
* Finalization timestamp
* Run-plan SHA-256 hash
* Dry-run status
* Required artifact names
* Artifact SHA-256 hashes
* Artifact sizes in bytes

Finalization is rejected when:

* A required artifact is missing or empty.
* Metadata does not match the planned run.
* The execution summary does not match the planned run.
* Frame counts do not match the experiment configuration.
* The frame-results CSV contains missing, duplicate or invalid frame records.
* The frame-results hash does not match the execution summary.
* The finalization timestamp precedes the execution finish time.
* A bundle manifest already exists.

A finalized bundle must not be modified. If an artifact must be
recreated, create a new run with a new run identifier instead of
overwriting the finalized bundle.

Generated bundle manifests and run artifacts must not be committed to
Git. Archive them together with the run plan, progress file,
configuration and environment metadata.
