#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <chrono>
#include <cmath>

cv::Mat applyGammaCorrection(const cv::Mat& image, double gamma)
{
    cv::Mat lookupTable(1, 256, CV_8U);
    uchar* table = lookupTable.ptr();

    for (int i = 0; i < 256; ++i)
    {
        table[i] = cv::saturate_cast<uchar>(
            std::pow(i / 255.0, gamma) * 255.0
        );
    }

    cv::Mat result;
    cv::LUT(image, lookupTable, result);

    return result;
}

int main()
{
    const std::string inputPath = "../images/test.png";
    const std::string outputPath = "../output/gamma_result.png";

    cv::Mat image = cv::imread(inputPath);

    if (image.empty())
    {
        std::cerr << "Error: Could not read image from " << inputPath << std::endl;
        std::cerr << "Please add a test image as images/test.png" << std::endl;
        return 1;
    }

    const double gamma = 1.5;

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat enhancedImage = applyGammaCorrection(image, gamma);

    auto end = std::chrono::high_resolution_clock::now();

    const double durationMs =
        std::chrono::duration<double, std::milli>(end - start).count();

    bool saved = cv::imwrite(outputPath, enhancedImage);

    if (!saved)
    {
        std::cerr << "Error: Could not save output image to " << outputPath << std::endl;
        return 1;
    }

    std::cout << "Gamma Correction completed successfully." << std::endl;
    std::cout << "Input image: " << inputPath << std::endl;
    std::cout << "Output image: " << outputPath << std::endl;
    std::cout << "Gamma value: " << gamma << std::endl;
    std::cout << "Execution time: " << durationMs << " ms" << std::endl;

    return 0;
}