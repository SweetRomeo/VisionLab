#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include <opencv2/core.hpp>

enum class ProcessingAlgorithm
{
    Original,
    GammaCorrection,
    HistogramEqualization,
    Clahe
};

struct ProcessingParameters
{
    double gammaValue{1.0};
    double claheClipLimit{4.0};
    int claheGridSize{8};
};

class ImageProcess final
{
public:
    void process(
        const cv::Mat &source,
        cv::Mat &destination,
        ProcessingAlgorithm algorithm,
        const ProcessingParameters &parameters
        ) const;

private:
    static void applyGamma(
        const cv::Mat &source,
        cv::Mat &destination,
        double gamma
        );

    static void applyHistogramEqualization(
        const cv::Mat &source,
        cv::Mat &destination
        );

    static void applyClahe(
        const cv::Mat &source,
        cv::Mat &destination,
        double clipLimit,
        int gridSize
        );
};

#endif // IMAGEPROCESS_H
