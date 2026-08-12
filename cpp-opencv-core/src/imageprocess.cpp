#include "imageprocess.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

void ImageProcess::process(
    const cv::Mat &source,
    cv::Mat &destination,
    ProcessingAlgorithm algorithm,
    const ProcessingParameters &parameters
    ) const
{
    if (source.empty())
    {
        destination.release();
        return;
    }

    // Şimdilik görüntü işleme çekirdeğinin giriş formatı:
    // 8-bit, 3 kanallı BGR görüntü.
    if (source.type() != CV_8UC3)
    {
        throw std::invalid_argument(
            "ImageProcess yalnızca CV_8UC3 BGR görüntü kabul ediyor."
            );
    }

    switch (algorithm)
    {
    case ProcessingAlgorithm::Original:
        destination = source.clone();
        break;

    case ProcessingAlgorithm::GammaCorrection:
        applyGamma(
            source,
            destination,
            parameters.gammaValue
            );
        break;

    case ProcessingAlgorithm::HistogramEqualization:
        applyHistogramEqualization(
            source,
            destination
            );
        break;

    case ProcessingAlgorithm::Clahe:
        applyClahe(
            source,
            destination,
            parameters.claheClipLimit,
            parameters.claheGridSize
            );
        break;

    default:
        destination = source.clone();
        break;
    }
}

void ImageProcess::applyGamma(
    const cv::Mat &source,
    cv::Mat &destination,
    double gamma
    )
{
    if (!std::isfinite(gamma) || gamma <= 0.0)
    {
        throw std::invalid_argument(
            "Gamma değeri sıfırdan büyük olmalıdır."
            );
    }

    cv::Mat lookupTable(1, 256, CV_8U);
    auto *table = lookupTable.ptr<uchar>();

    for (int i = 0; i < 256; ++i)
    {
        const double normalizedValue =
            static_cast<double>(i) / 255.0;

        table[i] = cv::saturate_cast<uchar>(
            std::pow(normalizedValue, gamma) * 255.0
            );
    }

    cv::LUT(
        source,
        lookupTable,
        destination
        );
}

void ImageProcess::applyHistogramEqualization(
    const cv::Mat &source,
    cv::Mat &destination
    )
{
    cv::Mat labImage;

    cv::cvtColor(
        source,
        labImage,
        cv::COLOR_BGR2Lab
        );

    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);

    // Yalnızca L, yani parlaklık kanalı işleniyor.
    cv::equalizeHist(
        labChannels[0],
        labChannels[0]
        );

    cv::merge(
        labChannels,
        labImage
        );

    cv::cvtColor(
        labImage,
        destination,
        cv::COLOR_Lab2BGR
        );
}

void ImageProcess::applyClahe(
    const cv::Mat &source,
    cv::Mat &destination,
    double clipLimit,
    int gridSize
    )
{
    if (!std::isfinite(clipLimit) || clipLimit <= 0.0)
    {
        throw std::invalid_argument(
            "CLAHE clip limit sıfırdan büyük olmalıdır."
            );
    }

    if (gridSize <= 0)
    {
        throw std::invalid_argument(
            "CLAHE grid size sıfırdan büyük olmalıdır."
            );
    }

    cv::Mat labImage;

    cv::cvtColor(
        source,
        labImage,
        cv::COLOR_BGR2Lab
        );

    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);

    const cv::Ptr<cv::CLAHE> clahe =
        cv::createCLAHE(
            clipLimit,
            cv::Size(gridSize, gridSize)
            );

    clahe->apply(
        labChannels[0],
        labChannels[0]
        );

    cv::merge(
        labChannels,
        labImage
        );

    cv::cvtColor(
        labImage,
        destination,
        cv::COLOR_Lab2BGR
        );
}
