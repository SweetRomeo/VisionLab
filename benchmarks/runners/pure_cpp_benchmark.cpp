#include "imageprocess.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QString>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef VISIONLAB_PROJECT_ROOT
#error "VISIONLAB_PROJECT_ROOT is not defined."
#endif

#ifndef VISIONLAB_BENCHMARK_RELEASE
#define VISIONLAB_BENCHMARK_RELEASE 0
#endif

namespace {
    namespace fs = std::filesystem;

    struct Resolution {
        int width;
        int height;
    };

    struct AlgorithmConfiguration {
        std::string name;
        ProcessingAlgorithm algorithm;
        ProcessingParameters parameters;
    };

    struct BenchmarkConfiguration {
        fs::path videoPath;
        fs::path outputDirectory;
        int warmupFrames;
        int measuredFrames;
        int trials;
        std::vector<Resolution> resolutions;
        std::vector<AlgorithmConfiguration> algorithms;
    };

    ProcessingAlgorithm algorithmFromName(
        const std::string &name
    ) {
        if (name == "original") {
            return ProcessingAlgorithm::Original;
        }

        if (name == "gamma_correction") {
            return ProcessingAlgorithm::GammaCorrection;
        }

        if (name == "histogram_equalization") {
            return ProcessingAlgorithm::HistogramEqualization;
        }

        if (name == "clahe") {
            return ProcessingAlgorithm::Clahe;
        }

        throw std::invalid_argument(
            "Unsupported algorithm: " + name
        );
    }

    BenchmarkConfiguration loadConfiguration(
        const fs::path &projectRoot
    ) {
        const fs::path configPath =
                projectRoot
                / "benchmarks"
                / "config"
                / "benchmark_config.json";

        QFile configFile(
            QString::fromStdString(
                configPath.generic_string()
            )
        );

        if (!configFile.open(QIODevice::ReadOnly)) {
            throw std::runtime_error(
                "Could not open benchmark configuration: "
                + configPath.string()
            );
        }

        QJsonParseError parseError;

        const QJsonDocument document =
                QJsonDocument::fromJson(
                    configFile.readAll(),
                    &parseError
                );

        if (parseError.error !=
            QJsonParseError::NoError) {
            throw std::runtime_error(
                "Invalid benchmark JSON: "
                + parseError.errorString().toStdString()
            );
        }

        if (!document.isObject()) {
            throw std::runtime_error(
                "Benchmark configuration must be an object."
            );
        }

        const QJsonObject root = document.object();
        const QJsonObject input =
                root.value("input").toObject();
        const QJsonObject benchmark =
                root.value("benchmark").toObject();
        const QJsonObject output =
                root.value("output").toObject();

        BenchmarkConfiguration configuration;

        configuration.videoPath =
                projectRoot
                / input.value("video_path")
                .toString()
                .toStdString();

        configuration.outputDirectory =
                projectRoot
                / output.value("directory")
                .toString()
                .toStdString();

        configuration.warmupFrames =
                benchmark.value("warmup_frames").toInt(-1);

        configuration.measuredFrames =
                benchmark.value("measured_frames").toInt(-1);

        configuration.trials =
                benchmark.value("trials").toInt(-1);

        if (configuration.warmupFrames < 0 ||
            configuration.measuredFrames <= 0 ||
            configuration.trials <= 0) {
            throw std::runtime_error(
                "Invalid benchmark frame or trial counts."
            );
        }

        const QJsonArray resolutionArray =
                root.value("resolutions").toArray();

        for (const QJsonValue &value: resolutionArray) {
            const QJsonObject object = value.toObject();

            Resolution resolution{
                object.value("width").toInt(-1),
                object.value("height").toInt(-1)
            };

            if (resolution.width <= 0 ||
                resolution.height <= 0) {
                throw std::runtime_error(
                    "Invalid benchmark resolution."
                );
            }

            configuration.resolutions.push_back(
                resolution
            );
        }

        const QJsonArray algorithmArray =
                root.value("algorithms").toArray();

        for (const QJsonValue &value: algorithmArray) {
            const QJsonObject object = value.toObject();
            const QJsonObject parameters =
                    object.value("parameters").toObject();

            AlgorithmConfiguration algorithmConfig;

            algorithmConfig.name =
                    object.value("name")
                    .toString()
                    .toStdString();

            algorithmConfig.algorithm =
                    algorithmFromName(
                        algorithmConfig.name
                    );

            algorithmConfig.parameters.gammaValue =
                    parameters.value("gamma_value")
                    .toDouble(0.6);

            algorithmConfig.parameters.claheClipLimit =
                    parameters.value("clip_limit")
                    .toDouble(4.0);

            algorithmConfig.parameters.claheGridSize =
                    parameters.value("grid_size")
                    .toInt(8);

            configuration.algorithms.push_back(
                algorithmConfig
            );
        }

        if (configuration.resolutions.empty() ||
            configuration.algorithms.empty()) {
            throw std::runtime_error(
                "Benchmark configuration is incomplete."
            );
        }

        return configuration;
    }

    cv::Mat readFrame(cv::VideoCapture &capture) {
        cv::Mat frame;

        if (capture.read(frame) && !frame.empty()) {
            return frame;
        }

        capture.set(cv::CAP_PROP_POS_FRAMES, 0);

        if (!capture.read(frame) || frame.empty()) {
            throw std::runtime_error(
                "Could not read a frame from the video."
            );
        }

        return frame;
    }

    void runTestCase(
        std::ofstream &output,
        const ImageProcess &processor,
        const BenchmarkConfiguration &configuration,
        const Resolution &resolution,
        const AlgorithmConfiguration &algorithm,
        int trial
    ) {
        cv::VideoCapture capture(
            configuration.videoPath.string()
        );

        if (!capture.isOpened()) {
            throw std::runtime_error(
                "Could not open benchmark video: "
                + configuration.videoPath.string()
            );
        }

        for (int index = 0;
             index < configuration.warmupFrames;
             ++index) {
            const cv::Mat frame = readFrame(capture);

            cv::Mat resizedFrame;
            cv::resize(
                frame,
                resizedFrame,
                cv::Size(
                    resolution.width,
                    resolution.height
                ),
                0.0,
                0.0,
                cv::INTER_LINEAR
            );

            cv::Mat processedFrame;

            processor.process(
                resizedFrame,
                processedFrame,
                algorithm.algorithm,
                algorithm.parameters
            );
        }

        for (int frameIndex = 1;
             frameIndex <= configuration.measuredFrames;
             ++frameIndex) {
            const cv::Mat frame = readFrame(capture);

            cv::Mat resizedFrame;
            cv::resize(
                frame,
                resizedFrame,
                cv::Size(
                    resolution.width,
                    resolution.height
                ),
                0.0,
                0.0,
                cv::INTER_LINEAR
            );

            cv::Mat processedFrame;

            const auto startTime =
                    std::chrono::steady_clock::now();

            processor.process(
                resizedFrame,
                processedFrame,
                algorithm.algorithm,
                algorithm.parameters
            );

            const auto endTime =
                    std::chrono::steady_clock::now();

            if (processedFrame.empty()) {
                throw std::runtime_error(
                    "The algorithm produced an empty frame."
                );
            }

            const double processTimeMs =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(endTime - startTime).count();

            output
                    << "pure_cpp,"
                    << algorithm.name << ','
                    << resolution.width << 'x'
                    << resolution.height << ','
                    << trial << ','
                    << frameIndex << ','
                    << processTimeMs
                    << '\n';
        }
    }
} // namespace

int main() {
#if !VISIONLAB_BENCHMARK_RELEASE
    std::cerr
            << "VisionLabCppBenchmark must be run in Release mode.\n";
    return 1;
#endif
    try {
        const fs::path projectRoot =
                fs::path(VISIONLAB_PROJECT_ROOT)
                .lexically_normal();

        const BenchmarkConfiguration configuration =
                loadConfiguration(projectRoot);

        if (!fs::is_regular_file(
            configuration.videoPath
        )) {
            throw std::runtime_error(
                "Benchmark video was not found: "
                + configuration.videoPath.string()
            );
        }

        fs::create_directories(
            configuration.outputDirectory
        );

        const fs::path outputPath =
                configuration.outputDirectory
                / "pure_cpp_results.csv";

        std::ofstream output(outputPath);

        if (!output.is_open()) {
            throw std::runtime_error(
                "Could not create result file: "
                + outputPath.string()
            );
        }

        output
                << "architecture,algorithm,resolution,"
                << "trial,frame_index,processing_time_ms\n";

        output
                << std::fixed
                << std::setprecision(6);

        const ImageProcess processor;

        for (const Resolution &resolution:
             configuration.resolutions) {
            for (const AlgorithmConfiguration &algorithm:
                 configuration.algorithms) {
                for (int trial = 1;
                     trial <= configuration.trials;
                     ++trial) {
                    std::cout
                            << "Pure C++ | "
                            << algorithm.name
                            << " | "
                            << resolution.width
                            << 'x'
                            << resolution.height
                            << " | trial "
                            << trial
                            << '/'
                            << configuration.trials
                            << '\n';

                    runTestCase(
                        output,
                        processor,
                        configuration,
                        resolution,
                        algorithm,
                        trial
                    );

                    output.flush();
                }
            }
        }

        std::cout
                << "Benchmark completed: "
                << outputPath
                << '\n';

        return 0;
    } catch (const std::exception &exception) {
        std::cerr
                << "Benchmark failed: "
                << exception.what()
                << '\n';

        return 1;
    }
}
