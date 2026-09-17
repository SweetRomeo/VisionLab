# Controlled-Illumination Physical Pilot Checklist

This checklist must be completed before collecting the full
controlled-illumination optical-screening dataset.

The physical pilot is a hardware-validation step. It is not part of
the final 300-run dataset and must not be used for scientific
conclusions.

## 1. Repository state

Before connecting the camera:

* Confirm that the working tree is clean.
* Record the full Git commit SHA.
* Confirm that all controlled-illumination tests pass.
* Confirm that generated experiment artifacts are ignored by Git.

Commands:

```bash
git status --short
git rev-parse HEAD

python -m unittest discover \
-s benchmarks/experiments/tests \
-p "test_*.py" \
-v
```

The working tree must be clean before the pilot begins.

## 2. Camera-control profile

The physical pilot must use explicit camera controls.

Do not invent camera-specific values in the repository.

Before execution, define real values supported by the selected camera
for the required controlled settings:

* Auto exposure state
* Exposure
* Gain
* Auto white balance state
* White-balance temperature
* Autofocus state
* Focus

Automatic exposure, white balance or focus must not remain enabled
silently during a controlled-illumination run.

The requested values must be based on the real camera/backend being
used for the pilot.

## 3. Camera preflight

Run the camera preflight before creating any completed experiment
artifacts.

First inspect the available command-line controls:

```bash
python -m \
benchmarks.experiments.controlled_illumination_camera_preflight \
--help
```

Then run the preflight with:

* The physical camera index
* Width `1280`
* Height `720`
* Target FPS `30`
* The intended camera-control values
* A finite positive sample-frame count

The preflight must report:

* Camera index
* Effective resolution
* Effective FPS
* Requested camera controls
* Applied state
* Effective/read-back values when available
* Verification state
* Sampled frame count

The preflight must not write completed experiment artifacts.

## 4. Preflight acceptance criteria

Do not proceed to the pilot run unless all of the following are true:

* The camera opens successfully.
* Effective resolution is suitable for the requested `1280x720` mode.
* Effective FPS is suitable for the requested `30 FPS` mode.
* The requested number of frames is sampled successfully.
* Every required camera control is successfully applied.
* Every control that supports read-back is verified.
* Requested and effective values are consistent within the configured
  tolerance.
* Controls that cannot be verified are explicitly reported as
  unverifiable rather than silently treated as verified.
* The camera is released after preflight completion.
* No completed run artifacts are produced by preflight.

If a required control cannot be established, stop the pilot.

## 5. Physical setup

Before the pilot run:

* Fix the camera position.
* Fix the camera-to-target distance.
* Fix the light-to-target distance.
* Fix the target surface and scene.
* Set the required incidence angle.
* Stabilize the illumination source.
* Measure illuminance at:

    * Centre
    * Top left
    * Top right
    * Bottom left
    * Bottom right
* Record the real lux-meter information.
* Record starting temperature and platform state.

Do not change camera placement, focus, exposure, gain, white balance or
scene geometry between pilot trials unless the pilot is intentionally
testing that variable.

## 6. Pilot execution

Run only a small physical pilot before the full 300-run dataset.

Use:

* Platform: `desktop`
* Architecture: `pure_python`
* Resolution: `1280x720`
* Target FPS: `30`
* The controlled camera profile validated by preflight

The pilot must exercise the real camera path rather than the video-file
path.

The full 300-run matrix must not be started yet.

## 7. Failure behaviour

A failed required camera control must stop execution before completed
artifacts are written.

The following must not be treated as successful completed runs:

* Required control application failure
* Camera open failure
* Required frame acquisition failure
* Missing required quality samples
* Invalid runtime metadata
* Invalid execution summary
* Failed run-bundle validation

The camera must be released on both success and failure paths.

## 8. Execution-summary verification

After a successful pilot run, inspect:

```text
execution_summary.json
```

Confirm that `camera_controls` contains the controls used by the
runtime.

For each control, verify the recorded fields:

```text
property_id
requested
effective
applied
verified
matches_requested
```

The values must reflect the actual runtime camera-control result.

## 9. Run-metadata verification

Finalize the pilot run bundle using the normal controlled-illumination
finalization workflow.

Then inspect:

```text
run_metadata.json
```

Confirm that:

```text
camera_settings.controls
```

contains the same requested/effective camera-control information from
the execution summary.

The synchronization must occur before the final run-bundle artifact
hashes are calculated.

## 10. Bundle verification

Validate the finalized pilot bundle.

Confirm that:

* Frame-result counts match the configuration.
* Execution-summary counts match the frame results.
* Frame-result SHA-256 matches the execution summary.
* Run metadata validates successfully.
* Camera controls are present in run metadata for camera runs.
* The run-bundle manifest hashes the final synchronized metadata file.
* No temporary artifact files remain.

## 11. Video-path regression

The existing video-input benchmark path must remain valid.

Camera-control enforcement must not require camera controls when the
input source is a video file.

Run the complete experiment test suite after any physical-pilot code
changes:

```bash
python -m unittest discover \
-s benchmarks/experiments/tests \
-p "test_*.py" \
-v
```

## 12. Pilot completion gate

The full optical-screening dataset may begin only after the physical
pilot demonstrates all of the following:

* Camera mode is repeatable.
* Required controls are explicitly applied.
* Effective controls are recorded when readable.
* Unverifiable controls are explicitly identified.
* Required-control failure prevents completed artifacts.
* Camera resources are released on all paths.
* Requested/effective controls propagate into run metadata.
* Finalized artifact hashes remain valid.
* The video-input path remains unchanged.
* The complete automated test suite passes.

Only after these conditions are satisfied should the planned 300-run
optical-screening dataset be collected.
