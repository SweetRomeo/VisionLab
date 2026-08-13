# VisionLab Hybrid

Python and C++ hybrid image-processing implementation for VisionLab.

## Architecture

- Python: camera capture, application control and visualization
- C++: image-processing algorithms
- pybind11: communication between Python and C++
- OpenCV: image capture and processing

## Supported Algorithms

- Original
- Gamma Correction
- Histogram Equalization
- CLAHE

## Controls

- `1`: Original
- `2`: Gamma Correction
- `3`: Histogram Equalization
- `4`: CLAHE
- `Q` or `Esc`: Exit

## Setup

```bash
py -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt