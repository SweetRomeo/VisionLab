# VisionLab Benchmark Reproducibility Guide

This guide explains how to reproduce the VisionLab benchmarks across the following implementations:

* Pure Python
* Hybrid Python+C++
* Pure C++

Run all commands from the repository root (`VisionLab/`) unless otherwise specified.

## 1. Prepare the input video

Place the benchmark video at:

```text
benchmarks/data/benchmark_input.mp4
```

Detailed input requirements are documented in:

```text
benchmarks/data/README.md
```

All implementations must use the same input video and benchmark configuration.

The shared configuration file is:

```text
benchmarks/config/benchmark_config.json
```

It defines:

* Warm-up frame count
* Measured frame count
* Trial count
* Resolutions
* Algorithms and their parameters

Do not change the configuration between architecture runs.

## 2. Requirements

The project requires:

* Python 3
* CMake 3.15 or newer
* A C++17-compatible compiler
* OpenCV
* Qt6 Core, Gui, Widgets and OpenGLWidgets
* Python packages listed in the project requirement files

The C++ and Hybrid implementations must be built in Release mode for official measurements.

## 3. Linux and macOS setup

### 3.1. Pure Python environment

```bash
python3 -m venv pure-python/.venv
source pure-python/.venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r pure-python/requirements.txt

deactivate
```

### 3.2. Hybrid Python+C++ environment

```bash
python3 -m venv hybrid-python-cpp/.venv
source hybrid-python-cpp/.venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r hybrid-python-cpp/requirements.txt

deactivate
```

## 4. Build Release binaries

### 4.1. Hybrid module

Activate the Hybrid environment before configuring CMake:

```bash
source hybrid-python-cpp/.venv/bin/activate

cmake \
    -S hybrid-python-cpp \
    -B hybrid-python-cpp/build \
    -DCMAKE_BUILD_TYPE=Release

cmake \
    --build hybrid-python-cpp/build \
    --config Release

deactivate
```

The build must create a Python module named similar to:

```text
visionlab_cpp.so
```

On Windows, the module uses the `.pyd` extension.

### 4.2. Pure C++ benchmark

```bash
cmake \
    -S cpp-opencv-core \
    -B cpp-opencv-core/build \
    -DCMAKE_BUILD_TYPE=Release

cmake \
    --build cpp-opencv-core/build \
    --config Release \
    --target VisionLabCppBenchmark
```

The benchmark executable must not be run from a Debug build. Debug execution is intentionally rejected by the application.

## 5. Validate output equivalence

Output equivalence must be validated before running the official performance benchmarks.

The validation script compares the Pure Python implementation against the shared C++ `ImageProcess` core used by both the Hybrid and Pure C++ implementations.

It tests:

* Five deterministic frames distributed across the benchmark video
* Every algorithm defined in `benchmark_config.json`
* Every configured resolution
* Output shape and data type
* Mean absolute error
* Maximum absolute error
* PSNR
* Exact pixel equality

The default validation requires exact pixel-level equality:

```text
mean absolute error = 0
maximum absolute error = 0
```

The validation script also verifies that the active environment
and `pure-python/.venv` use identical Python, NumPy, and OpenCV
versions before running comparisons.

### Linux and macOS

```bash
source hybrid-python-cpp/.venv/bin/activate

python benchmarks/validation/validate_output_equivalence.py

deactivate
```

### Windows PowerShell

```powershell
.\hybrid-python-cpp\.venv\Scripts\Activate.ps1

python benchmarks\validation\validate_output_equivalence.py

deactivate
```

### Windows Git Bash

```bash
source hybrid-python-cpp/.venv/Scripts/activate

python benchmarks/validation/validate_output_equivalence.py

deactivate
```

If the compiled C++ module is not discovered automatically, set `VISIONLAB_CPP_MODULE_DIR` to the directory containing `visionlab_cpp` before running the validation.

A successful default validation finishes with:

```text
Output equivalence passed for 60 comparison(s).
```

The number of comparisons is calculated as:

```text
5 frames × 3 resolutions × 4 algorithms
= 60 comparisons
```

Detailed results are written to:

```text
benchmarks/results/output_equivalence_results.csv
```

This generated CSV is ignored by Git and should not be committed.

Do not increase the error tolerances unless the numerical differences have been investigated and scientifically justified.


## 6. Run benchmarks on Linux and macOS

Run the implementations in the following order.

### 6.1. Pure Python

```bash
source pure-python/.venv/bin/activate

python benchmarks/runners/pure_python_benchmark.py

deactivate
```

### 6.2. Hybrid Python+C++

Activate the Hybrid environment:

```bash
source hybrid-python-cpp/.venv/bin/activate
```

If the `visionlab_cpp` module is not located automatically, set the directory containing the compiled module:

```bash
export VISIONLAB_CPP_MODULE_DIR="/absolute/path/to/hybrid-python-cpp/build"
```

For multi-config generators, the module may be inside a configuration subdirectory:

```bash
export VISIONLAB_CPP_MODULE_DIR="/absolute/path/to/hybrid-python-cpp/build/Release"
```

Run the benchmark:

```bash
python benchmarks/runners/hybrid_benchmark.py

deactivate
```

### 6.3. Pure C++

For a single-config build:

```bash
./cpp-opencv-core/build/VisionLabCppBenchmark
```

For a multi-config build:

```bash
./cpp-opencv-core/build/Release/VisionLabCppBenchmark
```

## 7. Windows setup and execution

The recommended Windows configuration is a 64-bit MSVC kit with Release selected in Qt Creator.

Qt Creator may place binaries in kit-specific directories such as:

```text
build/Desktop_Qt_6_11_1_MSVC2022_64bit-Release
```

Use the actual Release directory generated on your computer.

### 7.1. PowerShell environments

If PowerShell prevents virtual-environment activation, allow scripts for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Create the Pure Python environment:

```powershell
py -3 -m venv pure-python\.venv
.\pure-python\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r pure-python\requirements.txt

deactivate
```

Create the Hybrid environment:

```powershell
py -3 -m venv hybrid-python-cpp\.venv
.\hybrid-python-cpp\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r hybrid-python-cpp\requirements.txt

deactivate
```

### 7.2. Build with Qt Creator

For both `hybrid-python-cpp` and `cpp-opencv-core`:

1. Open the corresponding `CMakeLists.txt` file in Qt Creator.
2. Select a 64-bit MSVC desktop kit.
3. Select the Release build configuration.
4. Build the project.
5. For the Pure C++ project, build the `VisionLabCppBenchmark` target.

Do not use a Debug binary for official measurements.

### 7.3. Run Pure Python in PowerShell

```powershell
.\pure-python\.venv\Scripts\Activate.ps1

python benchmarks\runners\pure_python_benchmark.py

deactivate
```

### 7.4. Locate and run the Hybrid module

Locate the compiled module:

```powershell
Get-ChildItem hybrid-python-cpp\build -Recurse -Filter "visionlab_cpp*.pyd"
```

Activate the Hybrid environment and set the directory containing the `.pyd` file:

```powershell
.\hybrid-python-cpp\.venv\Scripts\Activate.ps1

$env:VISIONLAB_CPP_MODULE_DIR="C:\path\to\VisionLab\hybrid-python-cpp\build\YOUR_RELEASE_DIRECTORY"

python benchmarks\runners\hybrid_benchmark.py

deactivate
```

Replace `YOUR_RELEASE_DIRECTORY` with the directory reported by Qt Creator or the preceding search command.

### 7.5. Locate and run the Pure C++ benchmark

Locate the executable:

```powershell
Get-ChildItem cpp-opencv-core\build -Recurse -Filter "VisionLabCppBenchmark.exe"
```

Run the Release executable:

```powershell
.\cpp-opencv-core\build\YOUR_RELEASE_DIRECTORY\VisionLabCppBenchmark.exe
```

If the following message appears, a Debug executable was selected:

```text
VisionLabCppBenchmark must be run in Release mode.
```

Switch to the Release build before continuing.

### 7.6. Git Bash on Windows

Run Pure Python:

```bash
source pure-python/.venv/Scripts/activate

python benchmarks/runners/pure_python_benchmark.py

deactivate
```

Run Hybrid Python+C++:

```bash
source hybrid-python-cpp/.venv/Scripts/activate

export VISIONLAB_CPP_MODULE_DIR="C:/path/to/VisionLab/hybrid-python-cpp/build/YOUR_RELEASE_DIRECTORY"

python benchmarks/runners/hybrid_benchmark.py

deactivate
```

Run Pure C++:

```bash
./cpp-opencv-core/build/YOUR_RELEASE_DIRECTORY/VisionLabCppBenchmark.exe
```

## 8. Analyze the results

Run the analyzer after all three benchmark runners finish successfully:

```bash
python benchmarks/analysis/analyze_results.py
```

The analyzer validates the result files against the current benchmark configuration. It rejects:

* Missing architecture, algorithm, resolution or trial groups
* Missing frame indices
* Duplicate frame indices
* Unexpected frame indices
* Incorrect architecture labels
* Invalid processing-time values
* Partial results generated with a different configuration

If the configuration changes, rerun all three implementations before analyzing the results.

## 9. Expected output files

After a successful benchmark and analysis run, `benchmarks/results/` must contain:

```text
pure_python_results.csv
hybrid_results.csv
pure_cpp_results.csv
benchmark_trial_summary.csv
benchmark_summary.csv
```

The first three files contain per-frame measurements. The final two files contain trial-level and overall statistics.

With the default configuration, each architecture produces:

```text
4 algorithms × 3 resolutions × 5 trials × 500 measured frames
= 30,000 measurement rows
```

Warm-up frames are not written to the result files.

## 10. Fair-comparison requirements

For scientifically meaningful results:

* Use the same input video for all architectures.
* Use the same benchmark configuration.
* Build the Hybrid and Pure C++ implementations in Release mode.
* Run all benchmarks on the same computer.
* Keep power and performance settings unchanged.
* Close unnecessary background applications.
* Avoid running multiple benchmarks simultaneously.
* Do not compare partial, smoke-test or Debug results with official results.
* Record the hardware, operating system, compiler, Python, OpenCV and Qt versions used for the experiment.

Generated result files should only be compared after the analyzer completes without an integrity error.