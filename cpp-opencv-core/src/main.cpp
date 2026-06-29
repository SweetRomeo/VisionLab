#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <sstream>
#include <iomanip>

double calculateAverageBrightness(const cv::Mat& image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Scalar meanValue = cv::mean(gray);

    return meanValue[0];
}

double determineAdaptiveGamma(double brightness)
{
    if (brightness < 70.0)
    {
        return 0.5;
    }
    else if (brightness < 120.0)
    {
        return 0.8;
    }
    else if (brightness < 180.0)
    {
        return 1.0;
    }
    else
    {
        return 1.3;
    }
}

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

std::string formatDouble(double value, int precision = 2)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera.\n";
        return 1;
    }

    cv::Mat frame;
    cv::Mat enhancedFrame;

    while (true)
    {
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Error: Empty frame captured.\n";
            break;
        }

        double brightness = calculateAverageBrightness(frame);
        double gamma = determineAdaptiveGamma(brightness);

        auto start = std::chrono::high_resolution_clock::now();

        enhancedFrame = applyGammaCorrection(frame, gamma);

        auto end = std::chrono::high_resolution_clock::now();

        const double durationMs =
            std::chrono::duration<double, std::milli>(end - start).count();

        const double fps = durationMs > 0.0 ? 1000.0 / durationMs : 0.0;

        cv::putText(
            enhancedFrame,
            "Brightness: " + formatDouble(brightness),
            cv::Point(20, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            2
            );

        cv::putText(
            enhancedFrame,
            "Adaptive Gamma: " + formatDouble(gamma),
            cv::Point(20, 60),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            2
            );

        cv::putText(
            enhancedFrame,
            "Processing Time: " + formatDouble(durationMs) + " ms",
            cv::Point(20, 90),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            2
            );

        cv::putText(
            enhancedFrame,
            "FPS: " + formatDouble(fps),
            cv::Point(20, 120),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            2
            );

        cv::imshow("Original Camera Feed", frame);
        cv::imshow("Adaptive Gamma Corrected Feed", enhancedFrame);

        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
