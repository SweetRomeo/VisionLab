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

## 5. Validate cross-architecture output equivalence

Output equivalence must be validated before comparing the performance of the three architectures:

* Pure Python
* Hybrid Python+C++
* Pure C++

The validation uses deterministic frames selected by:

```text
benchmarks/config/output_validation_config.json
```

The same input frames, resolutions, algorithms and parameters are supplied to all three implementations.

### 5.1. Install validation dependencies

Activate the Hybrid virtual environment and install the validation requirements.

Linux and macOS:

```bash
source hybrid-python-cpp/.venv/bin/activate
python -m pip install -r benchmarks/validation/requirements.txt
```

Windows PowerShell:

```powershell
.\hybrid-python-cpp\.venv\Scripts\Activate.ps1
python -m pip install -r benchmarks\validation\requirements.txt
```

Windows Git Bash:

```bash
source hybrid-python-cpp/.venv/Scripts/activate
python -m pip install -r benchmarks/validation/requirements.txt
```

### 5.2. Build the validation targets

Build the Hybrid module and the Pure C++ validation executable in Release mode.

Hybrid module:

```bash
cmake \
    --build hybrid-python-cpp/build \
    --config Release
```

Pure C++ validation executable:

```bash
cmake \
    --build cpp-opencv-core/build \
    --config Release \
    --target VisionLabCppValidation
```

The validation script automatically searches the build directories for:

```text
visionlab_cpp
VisionLabCppValidation
```

If automatic discovery fails, set the following environment variables to the corresponding Release build locations:

```text
VISIONLAB_CPP_MODULE_DIR
VISIONLAB_CPP_VALIDATION_EXE
```

Example for Windows PowerShell:

```powershell
$env:VISIONLAB_CPP_MODULE_DIR="C:\path\to\VisionLab\hybrid-python-cpp\build\YOUR_RELEASE_DIRECTORY"
$env:VISIONLAB_CPP_VALIDATION_EXE="C:\path\to\VisionLab\cpp-opencv-core\build\YOUR_RELEASE_DIRECTORY\VisionLabCppValidation.exe"
```

Example for Windows Git Bash:

```bash
export VISIONLAB_CPP_MODULE_DIR="C:/path/to/VisionLab/hybrid-python-cpp/build/YOUR_RELEASE_DIRECTORY"
export VISIONLAB_CPP_VALIDATION_EXE="C:/path/to/VisionLab/cpp-opencv-core/build/YOUR_RELEASE_DIRECTORY/VisionLabCppValidation.exe"
```

### 5.3. Run the validation

Run the following command from the repository root while the Hybrid virtual environment is active:

```bash
python benchmarks/validation/validate_output_equivalence.py
```

The default validation configuration evaluates:

```text
4 deterministic frames
× 3 resolutions
× 4 algorithms
× 3 architecture pairs
= 144 comparisons
```

The architecture pairs are:

```text
Pure Python vs Hybrid Python+C++
Pure Python vs Pure C++
Hybrid Python+C++ vs Pure C++
```

For each comparison, the validator checks:

* Output dimensions
* Channel count
* NumPy data type
* Mean absolute error
* Maximum absolute error
* Mean squared error
* Peak signal-to-noise ratio (PSNR)
* Structural similarity index (SSIM)

The configurable tolerance values are stored in:

```text
benchmarks/config/output_validation_config.json
```

The Original algorithm must produce an exact pixel match. Hybrid Python+C++ and Pure C++ must also match exactly because they use the same C++ image-processing core.

Small platform-dependent numerical differences between Pure Python and the C++ implementations may be accepted only when all configured MAE, maximum-difference, PSNR and SSIM thresholds are satisfied.

A successful run finishes with:

```text
Equivalence comparisons: 144
Failed comparisons: 0
Output equivalence validation passed.
```

If any comparison violates the configured rules, the report is still written and the validator terminates with a non-zero exit code.

### 5.4. Validation outputs

Generated validation artifacts are written under:

```text
benchmarks/results/output_equivalence/
```

The consolidated machine-readable report is:

```text
benchmarks/results/output_equivalence/output_equivalence_report.csv
```

Additional pair-specific reports are:

```text
benchmarks/results/output_equivalence/python_hybrid_comparison.csv
benchmarks/results/output_equivalence/python_cpp_comparison.csv
```

The consolidated report contains 144 data rows and records the calculated metrics, exact-match requirement, pass/fail status and failure reason for every comparison.

Generated input images, processed images and CSV reports are ignored by Git and must not be committed.

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
## 11. Run the real-time pipeline evaluation

The real-time experiment replays the benchmark video at a fixed target frame rate and evaluates whether each architecture can process frames before their deadlines.

The shared configuration is located at:

```text
benchmarks/config/realtime_config.json
```

The default configuration uses:

* 30 FPS target frame rate
* 33.333 ms frame deadline
* 30 warm-up frames
* 500 measured frames per trial
* 5 trials
* A capacity-one latest-frame queue
* Latest-frame replacement when the consumer falls behind

Run the implementations separately. Do not run multiple architectures simultaneously.

### 11.1. Pure Python

Activate the Pure Python environment and run:

```bash
python benchmarks/realtime/pure_python_realtime.py
```

### 11.2. Hybrid Python+C++

Activate the Hybrid environment, configure `VISIONLAB_CPP_MODULE_DIR` when necessary, and run:

```bash
python benchmarks/realtime/hybrid_realtime.py
```

### 11.3. Pure C++

Build and run `VisionLabCppRealtime` in Release mode:

```bash
cmake \
    --build cpp-opencv-core/build \
    --config Release \
    --target VisionLabCppRealtime

./cpp-opencv-core/build/Release/VisionLabCppRealtime
```

On Windows with a Qt Creator kit-specific directory:

```powershell
.\cpp-opencv-core\build\YOUR_RELEASE_DIRECTORY\VisionLabCppRealtime.exe
```

Each runner writes its result file only after all experiment cases finish successfully.

### 11.4. Analyze real-time results

After all three runners complete, run:

```bash
python benchmarks/realtime/analyze_realtime_results.py
```

The analyzer validates experiment coverage and reports processed frames, dropped frames, deadline misses, processing latency, end-to-end latency and achieved throughput for every architecture, algorithm and resolution.

The generated files are:

```text
benchmarks/results/realtime/
├── pure_python/realtime_frame_results.csv
├── hybrid/realtime_frame_results.csv
├── pure_cpp/realtime_frame_results.csv
└── realtime_summary.csv
```

With the default configuration, each architecture must produce 30,000 frame records:

```text
4 algorithms × 3 resolutions × 5 trials × 500 measured frames
= 30,000 records
```

Processed and dropped record counts must add up to the expected total.

## 12. Expected output files

After a successful benchmark and analysis run, `benchmarks/results/` must contain:

```text
pure_python_results.csv
hybrid_results.csv
pure_cpp_results.csv
benchmark_trial_summary.csv
benchmark_summary.csv
benchmark_resource_summary.csv
environment_metadata.json
benchmark_comparison.csv
figures/
```

The first three files contain per-frame measurements. `benchmark_trial_summary.csv` and `benchmark_summary.csv` contain trial-level and overall statistics. `benchmark_resource_summary.csv` contains architecture-level wall-clock time, CPU time, average CPU utilization, peak resident memory, sampling information, and process exit status. `environment_metadata.json` contains the experimental environment metadata recorded by `collect_environment.py`.

With the default configuration, each architecture produces:

```text
4 algorithms × 3 resolutions × 5 trials × 500 measured frames
= 30,000 measurement rows
```

Warm-up frames are not written to the result files.

## 13. Fair-comparison requirements

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