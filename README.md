# VisionLab

VisionLab is a research-oriented computer vision project focused on image enhancement benchmarking for autonomous vehicle vision systems.

The project aims to evaluate different image enhancement algorithms by comparing their visual quality, computational performance, and real-time applicability. It combines C++, OpenCV, and Qt to build a modular benchmark environment that can support academic research and thesis experiments.

## Project Motivation

Autonomous vehicles rely heavily on camera-based perception systems. However, environmental conditions such as low light, glare, fog, rain, and poor contrast can reduce image quality and affect the reliability of vision-based algorithms.

VisionLab is developed to test and compare image enhancement techniques under different visual conditions. The project is also intended to support my master's thesis research on performance testing of image enhancement algorithms for autonomous vehicle vision systems.

## Project Goals

* Implement image enhancement algorithms using C++ and OpenCV
* Test both static image input and real-time camera input
* Measure computational performance using execution time and FPS
* Compare visual quality using metrics such as PSNR and SSIM
* Build a Qt-based desktop application for visual comparison
* Export benchmark results for thesis-related analysis
* Extend the system with a backend API for storing experiment results

## Current Progress

* Added OpenCV-based gamma correction implementation
* Added adaptive gamma selection based on average frame brightness
* Added real-time camera input support
* Added execution time and FPS measurement
* Planned Qt integration for live visual comparison

## Technologies

* C++
* OpenCV
* Qt
* CMake
* Java
* Spring Boot
* PostgreSQL

## Planned Modules

```text
cpp-opencv-core     -> Image processing algorithms and benchmark logic
qt-desktop-app      -> Desktop GUI application for visual comparison
spring-boot-api     -> Backend API for storing benchmark results
docs                -> Architecture, roadmap, and thesis-related documentation
results             -> Benchmark outputs and experiment results
```

## Planned Image Enhancement Algorithms

* Gamma Correction
* Adaptive Gamma Correction
* Histogram Equalization
* CLAHE
* Sharpening
* Denoising
* Low-light Enhancement
* Fog / Rain preprocessing experiments

## Benchmark Metrics

* Execution Time
* FPS
* PSNR
* SSIM
* Brightness
* Contrast

## Roadmap

### Phase 1: OpenCV Core

* Implement static image enhancement examples
* Add real-time camera enhancement demos
* Measure execution time and FPS
* Save output images and benchmark results

### Phase 2: Qt Desktop Application

* Add image loading from the user interface
* Display original and enhanced images side by side
* Add algorithm selection controls
* Show benchmark metrics in the interface

### Phase 3: Dataset Benchmarking

* Add batch processing for image datasets
* Export benchmark results as CSV
* Compare algorithms across different visual conditions

### Phase 4: Backend Integration

* Develop a Spring Boot REST API
* Store benchmark results in PostgreSQL
* Query and compare previous experiment results

## Thesis Relation

This project supports my thesis work on image enhancement performance testing for autonomous vehicle vision systems. The main objective is to evaluate how different enhancement algorithms affect image quality and computational performance, especially in real-time or near real-time scenarios.

## Project Status

In development.
