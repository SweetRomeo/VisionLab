# VisionLab Benchmark Reproducibility Guide

Follow this guide from the repository root (`VisionLab/`) to reproduce the benchmark results across all three implementations.

## 1) Prepare input video

Place the benchmark video at:

```text
benchmarks/data/benchmark_input.mp4
```

Detailed input requirements are documented in `benchmarks/data/README.md`.

## 2) Install dependencies

### Pure Python runner

```bash
cd pure-python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd ..
```

### Hybrid Python+C++ runner

```bash
cd hybrid-python-cpp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd ..
```

### C++ benchmark executable

Install system dependencies required by CMake for:

- C++17 compiler toolchain
- OpenCV development package
- Qt6 Core/Gui/Widgets/OpenGLWidgets

## 3) Build Release binaries

### Hybrid module (Release)

```bash
cmake -S hybrid-python-cpp -B hybrid-python-cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build hybrid-python-cpp/build --config Release
```

### Pure C++ benchmark executable (Release)

```bash
cmake -S cpp-opencv-core -B cpp-opencv-core/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp-opencv-core/build --config Release --target VisionLabCppBenchmark
```

## 4) Run benchmark runners

From repository root:

```bash
python benchmarks/runners/pure_python_benchmark.py
python benchmarks/runners/hybrid_benchmark.py
```

Run pure C++ benchmark:

```bash
./cpp-opencv-core/build/VisionLabCppBenchmark
# or (multi-config generators)
./cpp-opencv-core/build/Release/VisionLabCppBenchmark
```

If your generator places the hybrid module in a non-default location, set:

```bash
export VISIONLAB_CPP_MODULE_DIR=/absolute/path/to/hybrid-python-cpp/build[/Release]
```

## 5) Run analysis

```bash
python benchmarks/analysis/analyze_results.py
```

## 6) Expected output files

After a successful run, `benchmarks/results/` must contain:

- `pure_python_results.csv`
- `hybrid_results.csv`
- `pure_cpp_results.csv`
- `benchmark_trial_summary.csv`
- `benchmark_summary.csv`
