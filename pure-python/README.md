# VisionLab Pure Python

Pure Python baseline implementation of the VisionLab real-time image-processing pipeline.

This project is designed to be compared with the other VisionLab implementations:

* Pure C++
* Python–C++ hybrid
* Pure Python

Camera capture, application control, performance measurement, and algorithm orchestration are implemented in Python. Image-processing operations are performed through the OpenCV Python API without using the project-specific C++ core or the `visionlab_cpp` extension.

## Supported Algorithms

* Original
* Gamma Correction
* Histogram Equalization
* CLAHE

Gamma Correction, Histogram Equalization, and CLAHE operate only on the lightness channel of the Lab color space to preserve the image's color information.

## Default Parameters

The same default parameters are used across the VisionLab implementations:

* Gamma value: `0.6`
* CLAHE clip limit: `4.0`
* CLAHE grid size: `8 × 8`

## Requirements

* Python 3.10 or newer
* A working camera
* NumPy
* OpenCV for Python

## Project Structure

```text
pure-python/
├── image_process.py
├── main.py
├── requirements.txt
└── README.md
```

* `image_process.py`: Algorithms, parameters, input validation, and processing dispatch
* `main.py`: Camera capture, runtime controls, visualization, and performance measurement
* `requirements.txt`: Python dependencies
* `README.md`: Setup and usage documentation

## Setup

Open a terminal in the `pure-python` directory:

```bash
cd pure-python
```

Create a virtual environment.

### Windows

```powershell
py -m venv .venv
```

PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt activation:

```bat
.venv\Scripts\activate.bat
```

Git Bash activation:

```bash
source .venv/Scripts/activate
```

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Application

Run the application from the `pure-python` directory:

```bash
python main.py
```

The application opens the default camera using:

```python
cv2.VideoCapture(0)
```

If the required camera has a different index, update the value in `main.py`.

## Runtime Controls

| Key                | Processing mode        |
| ------------------ | ---------------------- |
| `1`                | Original               |
| `2`                | Gamma Correction       |
| `3`                | Histogram Equalization |
| `4`                | CLAHE                  |
| `Q`, `q`, or `Esc` | Exit                   |

The selected algorithm, processing time, and calculated processing FPS are displayed on the output frame.

## Performance Measurement

Processing time is measured using `time.perf_counter()` only around the image-processing operation:

```python
processed_frame = processor.process(
    frame,
    algorithm,
    parameters,
)
```

The measurement excludes:

* Camera capture
* Text overlay
* Window rendering
* Keyboard input handling

Processing FPS is calculated as:

```text
Processing FPS = 1000 / processing time in milliseconds
```

This value represents algorithm throughput and is not necessarily equal to the camera's actual display frame rate.

## Input and Output Format

The processing core expects:

* NumPy array input
* `uint8` data type
* `H × W × 3` dimensions
* BGR channel order

Every processing mode preserves the input frame dimensions, data type, and channel structure.

## Validation

The implementation should be validated using the following checks:

* Camera opens successfully
* Original mode returns a copy of the input frame
* Gamma Correction changes image brightness
* Histogram Equalization improves global contrast
* CLAHE improves local contrast
* Runtime algorithm switching works
* Processing time and FPS are displayed
* `Q`, `q`, and `Esc` close the application
* Camera and OpenCV resources are released correctly

Syntax can be checked without opening the camera:

```bash
python -m py_compile image_process.py
python -m py_compile main.py
```

## Architectural Note

The term **Pure Python** indicates that VisionLab's application and image-processing pipeline are written using Python APIs and do not use the project's custom C++ implementation.

OpenCV's Python package internally calls native compiled OpenCV code. Therefore, the benchmark represents a Python/OpenCV implementation rather than image-processing operations written entirely with Python loops.

Detailed scientific benchmarking will be implemented separately after all three VisionLab architectures are finalized.
