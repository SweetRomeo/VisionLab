#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/core.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "imageprocess.h"

namespace py = pybind11;

namespace
{

using ImageArray = py::array_t<
    std::uint8_t,
    py::array::c_style | py::array::forcecast
>;

py::array_t<std::uint8_t> processFrame(
    ImageArray sourceArray,
    ProcessingAlgorithm algorithm,
    double gammaValue,
    double claheClipLimit,
    int claheGridSize
)
{
    const py::buffer_info sourceBuffer =
        sourceArray.request();

    if (sourceBuffer.ndim != 3 ||
        sourceBuffer.shape[2] != 3)
    {
        throw std::invalid_argument(
            "Girdi H x W x 3 biciminde bir BGR goruntu olmalidir."
        );
    }

    cv::Mat source(
        static_cast<int>(sourceBuffer.shape[0]),
        static_cast<int>(sourceBuffer.shape[1]),
        CV_8UC3,
        sourceBuffer.ptr
    );

    ProcessingParameters parameters;
    parameters.gammaValue = gammaValue;
    parameters.claheClipLimit = claheClipLimit;
    parameters.claheGridSize = claheGridSize;

    cv::Mat destination;
    const ImageProcess processor;

    {
        py::gil_scoped_release release;

        processor.process(
            source,
            destination,
            algorithm,
            parameters
        );
    }

    if (destination.empty())
    {
        return {};
    }

    if (!destination.isContinuous())
    {
        destination = destination.clone();
    }

    const auto rows =
        static_cast<py::ssize_t>(destination.rows);

    const auto columns =
        static_cast<py::ssize_t>(destination.cols);

    const auto channels =
        static_cast<py::ssize_t>(destination.channels());

    py::array_t<std::uint8_t> result(
        py::array::ShapeContainer{
            rows,
            columns,
            channels
        }
    );

    const py::buffer_info resultBuffer =
        result.request();

    std::memcpy(
        resultBuffer.ptr,
        destination.data,
        destination.total() * destination.elemSize()
    );

    return result;
}

} // namespace

PYBIND11_MODULE(visionlab_cpp, module)
{
    module.doc() =
        "VisionLab C++ image-processing module";

    py::enum_<ProcessingAlgorithm>(
        module,
        "ProcessingAlgorithm"
    )
        .value(
            "ORIGINAL",
            ProcessingAlgorithm::Original
        )
        .value(
            "GAMMA",
            ProcessingAlgorithm::GammaCorrection
        )
        .value(
            "HISTOGRAM",
            ProcessingAlgorithm::HistogramEqualization
        )
        .value(
            "CLAHE",
            ProcessingAlgorithm::Clahe
        );

    module.def(
        "process_frame",
        &processFrame,
        py::arg("frame"),
        py::arg("algorithm") =
            ProcessingAlgorithm::Original,
        py::arg("gamma_value") = 1.0,
        py::arg("clahe_clip_limit") = 4.0,
        py::arg("clahe_grid_size") = 8,
        "Process a BGR NumPy image using the C++ core."
    );
}