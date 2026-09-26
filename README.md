# VisionLab

**VisionLab** is a research-oriented computer vision benchmarking platform for evaluating real-time image enhancement pipelines across different software architectures and computing platforms.

The project focuses on the reproducible comparison of image enhancement algorithms used in camera-based perception systems, with particular emphasis on:

- computational performance,
- output equivalence,
- real-time execution behavior,
- image-quality characteristics,
- reproducible experiment design,
- and embedded-platform deployment.

VisionLab was developed as part of master's thesis research on image enhancement performance for autonomous vehicle vision systems.

---

## Overview

Camera-based perception systems can be strongly affected by environmental conditions such as low illumination, glare, poor contrast, fog, and uneven lighting.

VisionLab provides a controlled environment for investigating how image enhancement algorithms behave under these conditions while also measuring their computational cost.

Instead of evaluating only one implementation, the project compares the same processing workload across three software architectures:

| Architecture | Description |
| --- | --- |
| **Pure Python** | Python pipeline using the OpenCV Python API |
| **Hybrid Python + C++** | Python orchestration with C++ image-processing components |
| **Pure C++** | Native C++ / OpenCV implementation |

The benchmark infrastructure is designed so that the same input data, algorithm parameters, resolutions, frame counts, and experimental conditions can be used across implementations.

---

## Supported Image Enhancement Algorithms

VisionLab currently evaluates four processing modes:

- **Original** — unmodified reference pipeline
- **Gamma Correction**
- **Histogram Equalization**
- **CLAHE — Contrast Limited Adaptive Histogram Equalization**

The benchmark configuration currently uses:

```text
Gamma value:        0.6
CLAHE clip limit:   4.0
CLAHE grid size:    8 × 8
```

Gamma Correction, Histogram Equalization, and CLAHE operate on image luminance/lightness information while preserving the color structure of the input frame.

---

## Core Research Goals

VisionLab is designed to investigate several related questions:

1. How much computational overhead do different image enhancement algorithms introduce?
2. How does implementation architecture affect processing latency and throughput?
3. Do the Python, Hybrid, and C++ implementations produce equivalent outputs?
4. Can the processing pipelines satisfy real-time frame deadlines?
5. How do lighting conditions affect enhanced image characteristics?
6. Can the same experimental workflow be reproduced across desktop and embedded platforms?

---

## Cross-Architecture Benchmarking

The main benchmark evaluates the same workload across:

```text
Pure Python
Hybrid Python + C++
Pure C++
```

The default benchmark configuration contains:

```text
3 resolutions
× 4 processing modes
× 5 trials
× 500 measured frames
```

Supported benchmark resolutions are:

```text
640 × 480
1280 × 720
1920 × 1080
```

Official C++ and Hybrid performance measurements are performed using Release builds.

---

## Output Equivalence Validation

Performance measurements are meaningful only when the different implementations perform equivalent image transformations.

VisionLab therefore includes a dedicated cross-architecture validation stage.

The default validation configuration compares:

```text
4 deterministic frames
× 3 resolutions
× 4 algorithms
× 3 architecture pairs
= 144 comparisons
```

Architecture pairs:

```text
Pure Python vs Hybrid Python+C++
Pure Python vs Pure C++
Hybrid Python+C++ vs Pure C++
```

Validation includes:

- output dimensions,
- channel count,
- data type,
- Mean Absolute Error,
- maximum absolute error,
- Mean Squared Error,
- PSNR,
- SSIM.

This stage is intended to verify implementation consistency before architecture-level performance results are compared.

---

## Real-Time Performance Evaluation

VisionLab includes a real-time benchmarking pipeline for measuring processing behavior beyond simple average execution time.

The infrastructure supports measurements and validation related to:

- per-frame execution time,
- processing FPS,
- frame timing,
- configured frame deadlines,
- deadline misses,
- dropped frames,
- skipped frames,
- trial-level execution summaries.

Timing and summary statistics are cross-validated against recorded frame-level results before completed runs are accepted.

---

## Controlled-Illumination Experiments

VisionLab also includes infrastructure for reproducible physical experiments under controlled lighting conditions.

The experiment framework separates two lighting protocols:

### Constant-Lux

The target-plane illuminance is maintained approximately constant while the light incidence angle is changed.

```text
phase = constant_lux
```

### Constant-Source

The light-source output remains fixed while incidence angle changes and resulting illuminance is measured.

```text
phase = constant_source
```

The infrastructure records and validates experimental metadata including:

- illuminance,
- incidence angle,
- platform,
- architecture,
- algorithm,
- resolution,
- camera configuration,
- environmental measurements,
- trial identifiers,
- Git revision,
- execution timing.

---

## 300-Run Optical Screening Profile

A dedicated controlled-illumination optical-screening configuration is included.

The current profile expands to:

```text
1 platform
× 1 architecture
× 4 algorithms
× 1 resolution
× 5 illuminance levels
× 3 incidence angles
× 5 trials
= 300 runs
```

The experiment planner provides deterministic matrix expansion, validation, run ordering, and manifest generation before physical data collection begins.

---

## Reproducible Run Artifacts

Completed controlled-illumination runs can be finalized into validated run bundles.

VisionLab verifies consistency between:

- experiment configuration,
- run plan,
- run metadata,
- execution summary,
- frame-level results,
- captured quality samples.

Artifacts are protected using SHA-256 hashes and an immutable run-bundle manifest.

A successful finalization produces:

```text
run_bundle_manifest.json
```

This makes it possible to verify that experimental artifacts have not been modified after the run was completed.

---

## Optical Quality Analysis

VisionLab includes a reproducible analysis pipeline for captured optical-quality samples.

The pipeline calculates descriptive image statistics such as:

- mean grayscale intensity,
- intensity standard deviation,
- RMS contrast,
- dark clipping percentage,
- bright clipping percentage.

Input-versus-processed transformation metrics include:

- MAE,
- maximum absolute error,
- MSE,
- PSNR.

Analysis results are generated at both sample and trial levels.

```text
optical_quality_samples.csv
optical_quality_trial_summary.csv
```

Transformation metrics such as MAE, MSE, and PSNR between input and processed images are treated as measurements of transformation magnitude rather than standalone indicators of perceptual image quality.

---

## Physical Camera Support

The controlled-illumination execution pipeline supports multiple camera backends.

### Desktop / USB Camera

OpenCV `VideoCapture` backend.

### Raspberry Pi 5

CSI-camera execution using:

```text
Picamera2
libcamera
```

The backend supports requested and effective reporting for camera parameters such as:

- exposure,
- analogue gain,
- automatic exposure,
- automatic white balance,
- color gains,
- capture resolution,
- frame rate.

### NVIDIA Jetson

CSI-camera execution using:

```text
NVIDIA Argus
GStreamer
nvarguscamerasrc
```

The Jetson backend supports controlled-illumination execution, camera preflight, capture metadata, and platform-specific camera-control profiles.

---

## Camera Preflight

Physical experiments should not begin before the camera configuration passes a preflight stage.

The preflight workflow verifies information such as:

- camera availability,
- camera backend,
- effective resolution,
- effective FPS,
- sampled frame acquisition,
- requested camera controls,
- effective camera controls where available,
- camera resource cleanup.

The full experiment dataset should only be collected after the physical pilot and preflight checks have passed.

---

## Project Structure

```text
VisionLab/
│
├── benchmarks/
│   ├── analysis/
│   ├── config/
│   ├── data/
│   ├── environment/
│   ├── experiments/
│   ├── orchestration/
│   ├── realtime/
│   ├── results/
│   ├── runners/
│   └── validation/
│
├── cpp-opencv-core/
│   └── Pure C++ / OpenCV implementation
│
├── hybrid-python-cpp/
│   └── Python + C++ hybrid implementation
│
├── pure-python/
│   └── Pure Python / OpenCV implementation
│
└── results/
```

Detailed benchmark reproduction instructions are available in:

```text
benchmarks/README.md
```

Controlled-illumination infrastructure documentation is available in:

```text
benchmarks/experiments/README.md
```

---

## Technologies

### Core

- C++17
- Python 3
- OpenCV
- NumPy
- CMake

### Desktop

- Qt 6

### Embedded / Camera

- Raspberry Pi 5
- Picamera2
- libcamera
- NVIDIA Jetson
- NVIDIA Argus
- GStreamer

### Validation & Experiment Infrastructure

- JSON-based experiment configuration
- CSV result generation
- SHA-256 artifact verification
- automated regression testing

---

## Benchmark Reproduction

The detailed benchmark setup, build, validation, and execution procedure is documented in:

[`benchmarks/README.md`](benchmarks/README.md)

A typical workflow is:

```text
1. Prepare benchmark input
        ↓
2. Build / configure all architectures
        ↓
3. Validate cross-architecture output equivalence
        ↓
4. Execute performance benchmarks
        ↓
5. Validate generated results
        ↓
6. Run controlled-illumination experiments
        ↓
7. Finalize run bundles
        ↓
8. Analyze optical-quality results
```

---

## Project Status

The main software infrastructure is largely implemented.

Completed areas include:

- Pure Python implementation
- Hybrid Python+C++ implementation
- Pure C++ implementation
- shared benchmark configuration
- cross-architecture output validation
- real-time benchmark infrastructure
- controlled-illumination experiment planning
- experiment metadata validation
- run-state management
- run-bundle integrity validation
- optical-quality capture and analysis
- live-camera execution
- Raspberry Pi 5 camera backend
- NVIDIA Jetson camera backend

Current work is focused primarily on **physical hardware validation, pilot execution, experimental data collection, and final scientific analysis**.

Hardware-specific results should not be interpreted as validated physical experiment results until the corresponding physical pilot and acceptance checks have been completed.

---

## Research Context

VisionLab was developed to support master's thesis research investigating image enhancement algorithms for autonomous vehicle vision systems.

The broader objective is not only to compare enhancement algorithms, but also to study the relationship between:

```text
Image Enhancement
        +
Software Architecture
        +
Real-Time Performance
        +
Lighting Conditions
        +
Embedded Hardware
```

The project is therefore structured as a reproducible experimental platform rather than a collection of isolated image-processing examples.

---

## Documentation

- [Benchmark Reproducibility Guide](benchmarks/README.md)
- [Controlled-Illumination Experiment Infrastructure](benchmarks/experiments/README.md)
- [Controlled-Illumination Evaluation Protocol](benchmarks/experiments/controlled_illumination_protocol.md)

---

## Repository

GitHub:

https://github.com/SweetRomeo/VisionLab
