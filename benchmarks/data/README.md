# Benchmark Input Data

This directory contains the recorded input video used by all VisionLab benchmark implementations.

## Required File

Place the benchmark video at:

```text
benchmarks/data/benchmark_input.mp4
```

The video file is excluded from Git because of its size. It must not be committed to the repository.

## Recommended Video Properties

* Resolution: 1920×1080
* Frame rate: 30 FPS
* Duration: at least 20 seconds
* Format: MP4
* Codec: H.264 or another codec supported by OpenCV
* Color format: standard BGR-compatible video
* Content: a scene containing shadows, highlights, textures, movement and different contrast levels

The same unchanged video must be used for Pure C++, Hybrid Python+C++ and Pure Python benchmarks.

## Reproducibility

Record the following information before running the final experiments:

* Video resolution
* Frame rate
* Duration
* Codec
* File size
* Recording device
* Recording conditions
* SHA-256 hash

On Windows, the SHA-256 hash can be calculated with:

```powershell
certutil -hashfile benchmarks\data\benchmark_input.mp4 SHA256
```

On Linux:

```bash
sha256sum benchmarks/data/benchmark_input.mp4
```

Store the original benchmark video in a safe external location so the experiments can be repeated later.
