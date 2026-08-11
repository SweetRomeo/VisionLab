#include "imageprocess.h"

void ImageProcess::process(const cv::Mat &source, cv::Mat &destination, ProcessingAlgorithm algorithm, const ProcessingParameters &parameters) const
{
    if (source.empty())
    {
        destination.release();
        return;
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
        applyHistogramEqualization(source, destination);
        break;

    case ProcessingAlgorithm::Clahe:
        applyClahe(
            source,
            destination,
            parameters.claheClipLimit,
            parameters.claheGridSize
            );
        break;
    }
}
