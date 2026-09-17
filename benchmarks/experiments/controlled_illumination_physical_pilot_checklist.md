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

## 2. Camera backend and control profile

The physical pilot must use explicit camera controls appropriate to the selected camera backend.

For Raspberry Pi 5 CSI-camera execution, use:

```text
camera backend = picamera2
platform = raspberry_pi
architecture = pure_python
```

Picamera2/libcamera control units must not be treated as equivalent to OpenCV `VideoCapture` property values.

The Raspberry Pi control profile may define:

```text
AeEnable
ExposureTime
AnalogueGain
AwbEnable
ColourGains
```

VisionLab represents these values through:

```text
ae_enable
exposure_time_us
analogue_gain
awb_enable
colour_gains
```

Manual exposure or analogue gain requires:

```text
ae_enable = false
```

Manual colour gains require:

```text
awb_enable = false
```

Do not invent hardware-specific values in the repository.

Determine the real values during physical preflight and record both requested and effective/read-back values.

Controls that cannot be read back must be reported as unverifiable rather than silently treated as verified.

OpenCV control behavior remains valid for the desktop/USB-camera backend and must not be translated directly into Picamera2 values.

## 3. Raspberry Pi camera setup and preflight

Use a current Raspberry Pi OS installation with the libcamera/rpicam camera stack.

Update the operating system before the hardware pilot:

```bash
sudo apt update
sudo apt full-upgrade
```

If Picamera2 is not already installed, install the headless-compatible package with:

```bash
sudo apt install -y python3-picamera2 --no-install-recommends
```

Confirm that the Raspberry Pi detects the attached camera:

```bash
rpicam-hello --list-cameras
```

Record the camera index shown by this command.

Before running VisionLab, confirm that Picamera2 can be imported:

```bash
python3 -c "from picamera2 import Picamera2; print('Picamera2 available')"
```

Inspect the VisionLab preflight options:

```bash
python -m \
benchmarks.experiments.controlled_illumination_camera_preflight \
--help
```

A basic Raspberry Pi acquisition preflight is:

```bash
python -m \
benchmarks.experiments.controlled_illumination_camera_preflight \
--camera-backend picamera2 \
--camera-index 0 \
--width 1280 \
--height 720 \
--fps 30 \
--sample-frames 30
```

Replace camera index `0` when `rpicam-hello --list-cameras` reports a different index.

For controlled manual exposure and white balance, run the preflight using real values supported by the attached camera:

```bash
python -m \
benchmarks.experiments.controlled_illumination_camera_preflight \
--camera-backend picamera2 \
--camera-index 0 \
--width 1280 \
--height 720 \
--fps 30 \
--sample-frames 30 \
--ae-enable false \
--exposure-time-us <EXPOSURE_TIME_US> \
--analogue-gain <ANALOGUE_GAIN> \
--awb-enable false \
--colour-gains <RED_GAIN> <BLUE_GAIN>
```

Do not replace the placeholders until the actual camera has been inspected.

The preflight must report:

* Camera backend
* Camera index
* Effective resolution
* Effective FPS
* Requested camera controls
* Effective/read-back camera controls when available
* Applied state
* Verification state
* Sampled frame count

The preflight must create no completed experiment artifacts.

A successful preflight verifies camera acquisition and camera-control behavior only. It does not replace physical geometry, illuminance or scene verification.

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

Allow the lighting and camera conditions to stabilize before starting
measured execution.

## 6. Pilot execution

Run only a small physical pilot before the full 300-run dataset.

Use:

* Platform: `raspberry_pi`
* Camera backend: picamera2
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

Requested values must not be treated as equivalent to effective values
unless verification confirms the requested camera state.

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

Do not accept a pilot run whose finalized bundle fails integrity or
cross-file validation.

## 11. Quality-capture verification

After the physical pilot run:

* Confirm that the configured quality-sample PNG files exist.
* Confirm that `quality_samples_manifest.json` exists.
* Verify that the recorded SHA-256 values match the captured files.
* Confirm that all configured measured-frame sample indices are present.

Do not accept the pilot if any required sample or hash is missing or
inconsistent.

## 12. Optical-quality analysis

Run the existing optical-quality analysis pipeline:

```bash
python -m benchmarks.experiments.analyze_controlled_illumination_quality \
  --results-directory <pilot-results-directory> \
  --output-directory <analysis-directory>
```

Confirm that both outputs are created:

```text
optical_quality_samples.csv
optical_quality_trial_summary.csv
```

Inspect the captured sample images manually for:

* framing changes
* clipping
* unexpected blur
* focus changes
* unexpected exposure adjustment
* unexpected white-balance adjustment

The pilot must not be accepted when these checks indicate unintended
automatic camera behaviour.

The optical-quality analysis results are pilot-validation artifacts.
They must not be interpreted as final thesis conclusions from the
300-run optical-screening dataset.

## 13. Video-path regression

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

Also run the real-time regression suite:

```bash
python -m unittest discover \
-s benchmarks/realtime/tests \
-p "test_*.py" \
-v
```

## 14. Pilot completion gate

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
* Quality-sample PNGs and manifest hashes are valid.
* Optical-quality analysis completes successfully.
* `optical_quality_samples.csv` is generated and inspected.
* `optical_quality_trial_summary.csv` is generated and inspected.
* Captured samples show no unexpected automatic camera adjustment.
* The video-input path remains unchanged.
* The complete controlled-illumination experiment test suite passes.
* The complete real-time regression test suite passes.

Only after these conditions are satisfied should the planned 300-run
optical-screening dataset be collected.
