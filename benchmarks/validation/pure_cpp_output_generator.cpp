#include "imageprocess.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
namespace fs = std::filesystem;

ProcessingAlgorithm algorithmFromName(
    const std::string &name
)
{
    if (name == "original")
    {
        return ProcessingAlgorithm::Original;
    }

    if (name == "gamma_correction")
    {
        return ProcessingAlgorithm::GammaCorrection;
    }

    if (name == "histogram_equalization")
    {
        return ProcessingAlgorithm::HistogramEqualization;
    }

    if (name == "clahe")
    {
        return ProcessingAlgorithm::Clahe;
    }

    throw std::invalid_argument(
        "Unsupported algorithm: " + name
    );
}

double parsePositiveDouble(
    const std::string &text,
    const std::string &parameterName
)
{
    std::size_t processedCharacters = 0;

    const double value = std::stod(
        text,
        &processedCharacters
    );

    if (
        processedCharacters != text.size()
        || !std::isfinite(value)
        || value <= 0.0
    )
    {
        throw std::invalid_argument(
            "Invalid " + parameterName + ": " + text
        );
    }

    return value;
}

int parsePositiveInteger(
    const std::string &text,
    const std::string &parameterName
)
{
    std::size_t processedCharacters = 0;

    const int value = std::stoi(
        text,
        &processedCharacters
    );

    if (
        processedCharacters != text.size()
        || value <= 0
    )
    {
        throw std::invalid_argument(
            "Invalid " + parameterName + ": " + text
        );
    }

    return value;
}

void printUsage(
    const std::string &executableName
)
{
    std::cerr
        << "Usage:\n"
        << executableName
        << " <input.png> <output.png> <algorithm>"
        << " <gamma> <clahe_clip_limit>"
        << " <clahe_grid_size>\n";
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        printUsage(argv[0]);
        return 2;
    }

    try
    {
        const fs::path inputPath = argv[1];
        const fs::path outputPath = argv[2];
        const std::string algorithmName = argv[3];

        ProcessingParameters parameters;

        parameters.gammaValue = parsePositiveDouble(
            argv[4],
            "gamma value"
        );

        parameters.claheClipLimit =
            parsePositiveDouble(
                argv[5],
                "CLAHE clip limit"
            );

        parameters.claheGridSize =
            parsePositiveInteger(
                argv[6],
                "CLAHE grid size"
            );

        if (!fs::is_regular_file(inputPath))
        {
            throw std::runtime_error(
                "Input image was not found: "
                + inputPath.string()
            );
        }

        const cv::Mat source = cv::imread(
            inputPath.string(),
            cv::IMREAD_COLOR
        );

        if (source.empty())
        {
            throw std::runtime_error(
                "Input image could not be read: "
                + inputPath.string()
            );
        }

        const cv::Mat sourceBeforeProcessing =
            source.clone();

        cv::Mat destination;

        const ImageProcess processor;

        processor.process(
            source,
            destination,
            algorithmFromName(algorithmName),
            parameters
        );

        if (destination.empty())
        {
            throw std::runtime_error(
                "The C++ implementation produced "
                "an empty image."
            );
        }

        if (
            destination.size() != source.size()
            || destination.type() != source.type()
        )
        {
            throw std::runtime_error(
                "The C++ output shape or type "
                "does not match the input."
            );
        }

        if (
            cv::norm(
                source,
                sourceBeforeProcessing,
                cv::NORM_INF
            ) != 0.0
        )
        {
            throw std::runtime_error(
                "The C++ implementation modified "
                "the source image."
            );
        }

        if (
            algorithmName == "original"
            && destination.data == source.data
        )
        {
            throw std::runtime_error(
                "The Original algorithm returned "
                "shared image memory."
            );
        }

        const fs::path parentDirectory =
            outputPath.parent_path();

        if (!parentDirectory.empty())
        {
            fs::create_directories(
                parentDirectory
            );
        }

        if (!cv::imwrite(
            outputPath.string(),
            destination
        ))
        {
            throw std::runtime_error(
                "Output image could not be written: "
                + outputPath.string()
            );
        }

        std::cout
            << "C++ validation output created: "
            << outputPath
            << '\n';

        return 0;
    }
    catch (const std::exception &exception)
    {
        std::cerr
            << "C++ validation failed: "
            << exception.what()
            << '\n';

        return 1;
    }
}