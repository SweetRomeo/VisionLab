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

## 6. Collect environment metadata

Collect the benchmark environment metadata before running the official performance measurements.

The metadata collector records:

* Operating system and system architecture
* CPU model, logical CPU count and total memory
* Pure Python and Hybrid Python environment versions
* NumPy and OpenCV versions
* Release build type and CMake generator
* CMake, C++ compiler, OpenCV and Qt versions
* Git commit, branch and working-tree state
* SHA-256 hashes of the benchmark configuration and input video

Run the collector from the repository root:

```bash
python benchmarks/environment/collect_environment.py
```

The generated metadata is written to:

```text
benchmarks/results/environment_metadata.json
```

If multiple Release build directories exist, select the exact
build directories used for the experiment:

```bash
export VISIONLAB_CPP_BUILD_DIR="/absolute/path/to/cpp-build"
export VISIONLAB_HYBRID_BUILD_DIR="/absolute/path/to/hybrid-build"
```

The powershell version of those commands are:
```powershell
$env:VISIONLAB_CPP_BUILD_DIR="C:\path\to\cpp-build"
$env:VISIONLAB_HYBRID_BUILD_DIR="C:\path\to\hybrid-build"
```

For an official experiment, run the collector from the exact Git commit used by all benchmark implementations. The working tree should be clean before collecting the final metadata.

A successful run prints:

```text
Environment metadata created: benchmarks/results/environment_metadata.json
```

The generated JSON file contains no username, hostname or user-specific absolute paths. It is ignored by Git and should not be included in normal source commits.

## 7. Resource-monitored benchmark orchestration

The recommended way to run the complete benchmark suite is through the resource-monitoring orchestrator.

The orchestrator:

* Runs the Pure Python benchmark
* Runs the Hybrid Python+C++ benchmark
* Runs the Pure C++ Release benchmark
* Samples process CPU and memory usage every 0.1 seconds
* Includes recursively discovered child processes
* Rejects benchmark processes that return a non-zero exit code
* Runs the existing result-integrity analyzer
* Writes an architecture-level resource summary

The monitoring dependency is defined in:

```text
benchmarks/requirements.txt
```

### Linux and macOS

```bash
source hybrid-python-cpp/.venv/bin/activate

python -m pip install -r benchmarks/requirements.txt

export VISIONLAB_CPP_BUILD_DIR="/absolute/path/to/cpp-release-build"
export VISIONLAB_CPP_MODULE_DIR="/absolute/path/to/hybrid-release-module"

python benchmarks/orchestration/run_benchmarks.py

deactivate
```

### Windows PowerShell

```powershell
.\hybrid-python-cpp\.venv\Scripts\Activate.ps1

python -m pip install -r benchmarks\requirements.txt

$env:VISIONLAB_CPP_BUILD_DIR="C:\path\to\cpp-release-build"
$env:VISIONLAB_CPP_MODULE_DIR="C:\path\to\hybrid-release-module"

python benchmarks\orchestration\run_benchmarks.py

deactivate
```

### Windows Git Bash

```bash
source hybrid-python-cpp/.venv/Scripts/activate

python -m pip install -r benchmarks/requirements.txt

export VISIONLAB_CPP_BUILD_DIR="C:/path/to/cpp-release-build"
export VISIONLAB_CPP_MODULE_DIR="C:/path/to/hybrid-release-module"

python benchmarks/orchestration/run_benchmarks.py

deactivate
```

The Pure C++ build directory must contain the Release `VisionLabCppBenchmark` executable. The Hybrid module directory must contain the compiled `visionlab_cpp` `.pyd` or `.so` module.

The generated resource summary is written to:

```text
benchmarks/results/benchmark_resource_summary.csv
```

It contains:

```text
architecture
wall_time_seconds
cpu_time_seconds
average_cpu_percent
peak_rss_mib
sample_count
sampling_interval_seconds
exit_code
```

`average_cpu_percent` uses the process-level `psutil` convention, where approximately 100 percent represents one fully utilized logical CPU core. Multi-threaded processing may therefore report values greater than 100 percent.

`peak_rss_mib` is the maximum combined resident memory observed for the benchmark process and its discovered child processes.

Resource monitoring is performed outside the existing per-frame timed processing regions. The original processing-time and effective-FPS measurements therefore remain unchanged.


## 8. Run benchmarks on Linux and macOS

Run the implementations in the following order.

### 8.1. Pure Python

```bash
source pure-python/.venv/bin/activate

python benchmarks/runners/pure_python_benchmark.py

deactivate
```

### 8.2. Hybrid Python+C++

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

### 8.3. Pure C++

For a single-config build:

```bash
./cpp-opencv-core/build/VisionLabCppBenchmark
```

For a multi-config build:

```bash
./cpp-opencv-core/build/Release/VisionLabCppBenchmark
```

## 9. Windows setup and execution

The recommended Windows configuration is a 64-bit MSVC kit with Release selected in Qt Creator.

Qt Creator may place binaries in kit-specific directories such as:

```text
build/Desktop_Qt_6_11_1_MSVC2022_64bit-Release
```

Use the actual Release directory generated on your computer.

### 9.1. PowerShell environments

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

### 9.2. Build with Qt Creator

For both `hybrid-python-cpp` and `cpp-opencv-core`:

1. Open the corresponding `CMakeLists.txt` file in Qt Creator.
2. Select a 64-bit MSVC desktop kit.
3. Select the Release build configuration.
4. Build the project.
5. For the Pure C++ project, build the `VisionLabCppBenchmark` target.

Do not use a Debug binary for official measurements.

### 9.3. Run Pure Python in PowerShell

```powershell
.\pure-python\.venv\Scripts\Activate.ps1

python benchmarks\runners\pure_python_benchmark.py

deactivate
```

### 9.4. Locate and run the Hybrid module

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

### 9.5. Locate and run the Pure C++ benchmark

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

### 9.6. Git Bash on Windows

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

## 10. Analyze the results

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

### 10.1. Generate comparative metrics and figures

After the analyzer and resource-monitored benchmark run complete successfully, generate the comparative metrics and figures:

```bash
python benchmarks/analysis/generate_visualizations.py
```

## 11. Expected output files

After a successful benchmark and analysis run, `benchmarks/results/` must contain:

```text
pure_python_results.csv
hybrid_results.csv
pure_cpp_results.csv
benchmark_trial_summary.csv
benchmark_summary.csv
benchmark_resource_summary.csv
environment_metadata.json
```

The first three files contain per-frame measurements. `benchmark_trial_summary.csv` and `benchmark_summary.csv` contain trial-level and overall statistics. `benchmark_resource_summary.csv` contains architecture-level wall-clock time, CPU time, average CPU utilization, peak resident memory, sampling information, and process exit status. `environment_metadata.json` contains the experimental environment metadata recorded by `collect_environment.py`.

With the default configuration, each architecture produces:

```text
4 algorithms × 3 resolutions × 5 trials × 500 measured frames
= 30,000 measurement rows
```

Warm-up frames are not written to the result files.

## 12. Fair-comparison requirements

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