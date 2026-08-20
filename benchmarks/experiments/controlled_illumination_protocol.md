# VisionLab Controlled-Illumination Evaluation Protocol

## 1. Purpose

## Experimental protocols

This protocol defines a reproducible experimental method for evaluating VisionLab image-enhancement algorithms and software architectures under controlled illumination conditions relevant to an autonomous-vehicle vision system.

The experiment separates two questions:

1. How do illumination conditions and enhancement algorithms affect image and perception quality?
2. How do software architecture and computing platform affect real-time performance?

The desktop real-time reference baseline was generated using commit:

```text
78886b4eaaed059d88d7d34adc7a244f6d17a207
```

The desktop baseline is a preliminary reference and must not be interpreted as the final Raspberry Pi, NVIDIA Jetson or autonomous-vehicle result.

## 2. Research questions

### RQ1

How do illuminance and light-incidence angle affect the quality of Original, Gamma Correction, Histogram Equalization and CLAHE outputs?

### RQ2

Which enhancement algorithm provides the best trade-off between image quality, perception accuracy and real-time latency at each illumination level?

### RQ3

How do Pure Python, Hybrid Python+C++ and Pure C++ architectures differ in processing latency, deadline compliance and frame loss under identical visual conditions?

### RQ4

Does the relative performance of the architectures change between the desktop, Raspberry Pi and NVIDIA Jetson platforms?

### RQ5

Which architecture and enhancement configuration provides the most suitable latency–accuracy–energy trade-off for the autonomous-vehicle prototype?

## 3. Photometric definitions

### 3.1. Luminous intensity

Luminous intensity describes the amount of visible light emitted by a source in a particular direction. Its unit is candela (`cd`).

Candela characterizes the light source. It does not directly describe how much light reaches the experimental target.

### 3.2. Illuminance

Illuminance describes the luminous flux reaching a surface per unit area. Its unit is lux (`lx`).

Illuminance at the target plane is the primary controlled lighting variable in this experiment.

### 3.3. Luminance

Luminance describes the light reflected or emitted by a surface toward the camera. Its unit is candela per square metre (`cd/m²`).

The camera observes reflected luminance rather than illuminance directly. Target material, colour and reflectance must therefore remain constant between trials.

### 3.4. Approximate relationship

For an ideal point source, illuminance can be approximated by:

```text
E = I × cos(θ) / r²
```

where:

* `E` is illuminance in lux,
* `I` is luminous intensity in candela,
* `θ` is the light-incidence angle relative to the target-surface normal,
* `r` is the light-to-target distance in metres.

This relationship is an approximation. Actual lux values must be measured with a lux meter.

If calibrated candela measurement equipment is unavailable, the experiment must record the manufacturer-specified luminous intensity or source power setting without treating it as a directly measured candela value.

## 4. Experimental phases

### 4.1. Constant-lux angle experiment

This experiment isolates the effect of light-incidence angle.

For each selected angle:

1. Position the light source at the required angle.
2. Keep the light-to-target distance constant.
3. Adjust the source output until the required target-plane lux level is reached.
4. Record the measured lux values.
5. Capture the experiment input.

Lux is held approximately constant while the incidence angle changes.

### 4.2. Constant-source experiment

This experiment represents a more natural lighting change.

For each selected angle:

1. Keep the source output setting constant.
2. Keep the light-to-target distance constant.
3. Change only the light-incidence angle.
4. Measure and record the resulting lux values.
5. Capture the experiment input.

In this experiment, lux is expected to change naturally with the angle.

Results from the constant-lux and constant-source experiments must be reported separately.

## 5. Candidate experimental factors

### 5.1. Illuminance levels

Initial candidate target-plane illuminance levels are:

| Level                            | Target illuminance |
| -------------------------------- | -----------------: |
| Very low                         |              5 lux |
| Low                              |             50 lux |
| Moderate                         |            200 lux |
| Bright                           |            500 lux |
| Very bright laboratory condition |           1000 lux |

These values are provisional. A pilot experiment must confirm that the light source can produce each level consistently and that the lux meter can measure them reliably.

### 5.2. Light-incidence angles

Initial candidate incidence angles are:

```text
0°, 30° and 60°
```

The angle is defined relative to the target-surface normal:

* `0°`: light directed along the target normal,
* `30°`: moderate side illumination,
* `60°`: strong oblique illumination.

The camera viewing angle must not be changed while the light angle is being evaluated.

Backlighting and direct headlight glare are not included in the initial matrix. They may be added later as separate conditions because they require a different geometric definition.

### 5.3. Algorithms

The following processing algorithms will be evaluated:

* Original
* Gamma Correction
* Histogram Equalization
* CLAHE

Algorithm parameters must remain identical between architectures and platforms.

### 5.4. Resolutions

The initial resolutions are:

```text
640×480
1280×720
1920×1080
```

The optical-quality screening stage may use one selected resolution. All three resolutions will be retained for real-time platform comparisons.

### 5.5. Architectures

* Pure Python
* Hybrid Python+C++
* Pure C++

### 5.6. Computing platforms

* Desktop reference system
* Raspberry Pi
* NVIDIA Jetson

CPU-only comparisons and hardware-accelerated comparisons must be treated as separate experiments.

## 6. Physical experiment setup

The experimental setup must include:

* A controllable and repeatable light source.
* A calibrated or traceable lux meter.
* A fixed camera mount.
* A fixed target surface.
* Distance and angle measurement equipment.
* Ambient-light isolation where possible.
* A grayscale or colour reference chart.
* Lane-marking and obstacle-detection targets.
* A global shutter camera for the later dynamic experiment.
* Platform power and temperature monitoring.
* An external power meter when available.

The following distances must be measured and recorded:

* Camera-to-target distance.
* Light-to-target distance.
* Camera height.
* Light-source height.
* Lateral offset between camera and light source.

The position of every component must be marked so that the setup can be reconstructed.

## 7. Illuminance measurement procedure

Lux must be measured at five locations on the target plane:

1. Centre
2. Top left
3. Top right
4. Bottom left
5. Bottom right

The following values must be recorded:

* Centre lux
* Four corner lux values
* Mean lux
* Minimum lux
* Maximum lux
* Measurement time
* Lux-meter model
* Lux-meter range and resolution

The uniformity of the illuminated area must be evaluated during the pilot experiment. Conditions with excessive spatial variation must be corrected or documented.

Lux measurements must be taken immediately before or after each trial block without changing the light geometry.

## 8. Camera controls

Two camera modes will be evaluated separately.

### 8.1. Controlled camera mode

The following settings must be fixed:

* Exposure time
* Sensor gain or ISO
* White balance
* Focus
* Frame rate
* Resolution
* Lens
* Aperture, when adjustable

This mode isolates the effect of lighting and enhancement algorithms.

### 8.2. Operational camera mode

Automatic exposure or automatic white balance may be enabled only in a separate operational experiment.

The selected automatic settings and their reported values must be logged. Controlled-mode and automatic-mode results must not be combined.

## 9. Target scene

The target scene must remain unchanged between trials and must contain:

* A consistent background surface.
* Lane-like high-contrast markings.
* Low-contrast markings.
* Bright and dark regions.
* A grayscale or colour reference chart.
* At least one obstacle-detection target.
* Materials with fixed and documented surface properties.

Target position, orientation and reflectance must not be changed during a test block.

A reference image must be captured under a predefined reference illumination condition. Reference-based metrics may only be used when the reference image is geometrically aligned with the evaluated image.

## 10. Experiment matrix

### 10.1. Optical-quality screening

The initial controlled-lighting screening matrix consists of:

```text
4 algorithms
× 5 illuminance levels
× 3 incidence angles
× 5 independent trials
```

The architecture dimension does not need to be repeated for every optical-quality measurement when cross-architecture output equivalence has already been validated.

One canonical implementation may be used for optical-quality screening, while representative lighting conditions are retained for architecture-performance measurements.

### 10.2. Embedded-platform comparison

After the screening stage, representative lighting profiles will be selected, including at least:

* A difficult low-light condition.
* A nominal condition.
* A bright or oblique-light condition.

All architectures will be evaluated under these selected profiles on each supported platform.

## 11. Trial procedure

For every experimental condition:

1. Verify the source, camera and target positions.
2. Configure the required light condition.
3. Measure and record target-plane lux.
4. Configure and verify the camera settings.
5. Record the initial device temperature and power mode.
6. Run the configured warm-up frames.
7. Run the measured frames.
8. Record the final and maximum temperature.
9. Record thermal-throttling status.
10. Save result and metadata files atomically.
11. Verify experiment coverage and data integrity.
12. Allow the device to cool when required before the next trial.

The existing real-time defaults are:

```text
Target frame rate: 30 FPS
Frame deadline: 33.333 ms
Warm-up frames: 30
Measured frames per trial: 500
Trials per condition: 5
Queue capacity: 1
Drop policy: latest frame
```

## 12. Randomization and independence

Run order must be randomized or counterbalanced within each device.

The randomization seed must be recorded.

Repeated trials must be treated as the primary independent statistical units. Individual frames inside the same trial must not be reported as 500 independent experimental replications.

Architectures must not run simultaneously.

Background applications and unnecessary services must be minimized. Power mode, cooling configuration and clock settings must remain unchanged within a comparison block.

## 13. Recorded metadata

Every experiment run must record:

* Experiment identifier
* UTC timestamp
* Git commit SHA
* Device identifier
* Operating system
* Architecture
* Algorithm and parameters
* Resolution
* Trial number
* Target FPS and deadline
* Target lux
* Measured centre and corner lux values
* Incidence angle
* Source output setting
* Camera settings
* Camera and light distances
* Input video or scene identifier
* Power mode
* Clock configuration
* Starting, ending and maximum temperature
* Thermal-throttling status
* Software dependency versions

Desktop, Raspberry Pi and Jetson results must be stored separately to prevent accidental overwriting.

## 14. Evaluation metrics

### 14.1. Real-time metrics

* Mean processing time
* Median processing time
* p95 and p99 processing time
* Mean end-to-end latency
* p95 and p99 end-to-end latency
* Deadline-miss count and rate
* On-time completion rate
* Dropped-frame count and rate
* Skipped-frame count and rate
* Effective FPS

### 14.2. Image-quality metrics

* Contrast
* Signal-to-noise ratio
* SSIM when a valid reference exists
* Maximum pixel difference
* Mean absolute error
* Clipping in dark and bright regions
* Colour or grayscale reproduction error when measurable

No single image-quality metric will be interpreted as sufficient by itself.

### 14.3. Perception metrics

* Lane-detection precision and recall
* Lane-position or segmentation error
* Obstacle-detection precision and recall
* False-positive count
* Missed-detection count
* Detection confidence
* Detection stability between frames

### 14.4. Resource metrics

* Average and peak CPU utilization
* GPU utilization when applicable
* Peak memory usage
* Average and peak power
* Energy per processed frame
* Maximum temperature
* Thermal-throttling duration

## 15. Statistical reporting

Results will be summarized at the trial level.

The report must include:

* Trial count
* Mean
* Median
* Standard deviation
* p95 and p99
* 95% confidence interval
* Effect size where applicable

Architecture, platform, algorithm, resolution, lux and angle interactions may be evaluated after the screening stage.

Statistical conclusions must account for repeated measurements and trial-level dependence.

## 16. Data integrity and storage

Generated experiment outputs must not be committed to the Git repository.

Each official run must be archived with:

* Raw frame-level result files
* Summary files
* Experiment configuration
* Environment metadata
* Lighting and camera metadata
* Git commit SHA
* Validation status
* Experiment notes

A result set must not be used for comparison when validation fails or required metadata is missing.

## 17. Safety

* High-intensity light must not be directed toward a person’s eyes.
* Light-source temperature and electrical safety must be monitored.
* Cables and mounts must be secured.
* Initial experiments must be performed with the vehicle stationary.
* Dynamic tests must be conducted only in a controlled, closed area.
* An independent emergency-stop mechanism must be available during vehicle tests.
* Critical motor and braking control must not depend solely on a non-real-time user-space process.

## 18. Limitations

The initial protocol does not include:

* Direct sunlight reproduction.
* Rain, fog or dust.
* Headlight glare.
* Rapidly changing illumination.
* Motion blur.
* Camera vibration.
* Full sensor-to-actuator latency.

These factors may be introduced in later experiment phases.

## 19. Acceptance criteria

The protocol is considered complete when:

* Candela, lux and luminance are correctly distinguished.
* The light-incidence angle has an unambiguous definition.
* Constant-lux and constant-source experiments are separated.
* Camera and lighting geometry are reproducible.
* Camera settings are controlled and recorded.
* Trial order and replication rules are defined.
* Performance and perception metrics are separated.
* Desktop, Raspberry Pi and Jetson extensions are documented.
* CPU-only and hardware-optimized experiments are separated.
* Generated data and metadata storage requirements are defined.
* Experimental limitations and safety requirements are documented.
