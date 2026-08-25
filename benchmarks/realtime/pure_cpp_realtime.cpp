#include "imageprocess.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QString>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <locale>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#ifndef VISIONLAB_PROJECT_ROOT
#error "VISIONLAB_PROJECT_ROOT is not defined."
#endif

#ifndef VISIONLAB_REALTIME_RELEASE
#define VISIONLAB_REALTIME_RELEASE 0
#endif

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Resolution {
    int width{};
    int height{};

    std::string name() const
    {
        return std::to_string(width)
               + "x"
               + std::to_string(height);
    }
};

struct AlgorithmConfiguration {
    std::string name;
    ProcessingAlgorithm algorithm{
        ProcessingAlgorithm::Original
    };
    ProcessingParameters parameters;
};

struct RealtimeConfiguration {
    double targetFps{};
    int queueCapacity{};
    int warmupFrames{};
    int measuredFrames{};
    int trialCount{};
    std::string dropPolicy;
    double deadlineMultiplier{};
    fs::path outputDirectory;
    std::string frameResultsFile;
    std::string summaryFile;

    double deadlineMilliseconds() const
    {
        return (
            1000.0 / targetFps
        ) * deadlineMultiplier;
    }
};

struct ExperimentPlan {
    fs::path videoPath;
    std::vector<Resolution> resolutions;
    std::vector<AlgorithmConfiguration> algorithms;
    RealtimeConfiguration realtime;
};

std::optional<std::string> readOptionalEnvironmentVariable(
    const char *variableName
)
{
    const char *value = std::getenv(variableName);

    if (value == nullptr || std::string(value).empty()) {
        return std::nullopt;
    }

    return std::string(value);
}

std::string requireEnvironmentVariable(
    const char *variableName
)
{
    const std::optional<std::string> value =
        readOptionalEnvironmentVariable(
            variableName
        );

    if (!value.has_value()) {
        throw std::runtime_error(
            std::string(
                "Required environment variable is missing: "
            )
            + variableName
        );
    }

    return *value;
}

int parsePositiveEnvironmentInteger(
    const char *variableName
)
{
    const std::string text =
        requireEnvironmentVariable(
            variableName
        );

    try {
        std::size_t parsedCharacterCount = 0;
        const int value = std::stoi(
            text,
            &parsedCharacterCount
        );

        if (
            parsedCharacterCount != text.size()
            || value <= 0
        ) {
            throw std::runtime_error(
                std::string(
                    "Environment variable must be a "
                    "positive integer: "
                )
                + variableName
            );
        }

        return value;
    } catch (
        const std::invalid_argument &
    ) {
        throw std::runtime_error(
            std::string(
                "Environment variable must be a "
                "positive integer: "
            )
            + variableName
        );
    } catch (
        const std::out_of_range &
    ) {
        throw std::runtime_error(
            std::string(
                "Environment variable is outside the "
                "supported integer range: "
            )
            + variableName
        );
    }
}

bool controlledIlluminationRunRequested()
{
    return readOptionalEnvironmentVariable(
        "VISIONLAB_RUN_ID"
    ).has_value();
}

double parsePositiveEnvironmentNumber(
    const char *variableName
)
{
    const std::string text =
        requireEnvironmentVariable(
            variableName
        );

    try {
        std::size_t parsedCharacterCount = 0;
        const double value = std::stod(
            text,
            &parsedCharacterCount
        );

        if (
            parsedCharacterCount != text.size()
            || !std::isfinite(value)
            || value <= 0.0
        ) {
            throw std::runtime_error(
                std::string(
                    "Environment variable must be a "
                    "positive finite number: "
                )
                + variableName
            );
        }

        return value;
    } catch (
        const std::invalid_argument &
    ) {
        throw std::runtime_error(
            std::string(
                "Environment variable must be a "
                "positive finite number: "
            )
            + variableName
        );
    } catch (
        const std::out_of_range &
    ) {
        throw std::runtime_error(
            std::string(
                "Environment variable is outside the "
                "supported numeric range: "
            )
            + variableName
        );
    }
}

const Resolution &findPlannedResolution(
    const ExperimentPlan &plan,
    int width,
    int height
)
{
    const auto resolutionIterator = std::find_if(
        plan.resolutions.begin(),
        plan.resolutions.end(),
        [width, height](
            const Resolution &resolution
        ) {
            return (
                resolution.width == width
                && resolution.height == height
            );
        }
    );

    if (resolutionIterator == plan.resolutions.end()) {
        throw std::runtime_error(
            "The controlled-illumination resolution "
            "is not present in the experiment plan: "
            + std::to_string(width)
            + "x"
            + std::to_string(height)
        );
    }

    return *resolutionIterator;
}

const AlgorithmConfiguration &findPlannedAlgorithm(
    const ExperimentPlan &plan,
    const std::string &algorithmName
)
{
    const auto algorithmIterator = std::find_if(
        plan.algorithms.begin(),
        plan.algorithms.end(),
        [&algorithmName](
            const AlgorithmConfiguration &algorithm
        ) {
            return algorithm.name == algorithmName;
        }
    );

    if (algorithmIterator == plan.algorithms.end()) {
        throw std::runtime_error(
            "The controlled-illumination algorithm "
            "is not present in the experiment plan: "
            + algorithmName
        );
    }

    return *algorithmIterator;
}

struct ScheduledFrame {
    int sequenceIndex{};
    TimePoint scheduledTimestamp;
    TimePoint enqueuedTimestamp;
    cv::Mat payload;
};

enum class FrameStatus {
    Processed,
    Dropped
};

struct FrameRecord {
    std::string architecture;
    std::string algorithm;
    std::string resolution;
    int trial{};
    int frameIndex{};
    double scheduledTimestampMs{};
    std::optional<double> enqueuedTimestampMs;
    std::optional<double> processingStartTimestampMs;
    std::optional<double> processingEndTimestampMs;
    std::optional<double> dropTimestampMs;
    std::optional<double> sourceDelayMs;
    std::optional<double> queueWaitTimeMs;
    std::optional<double> processingTimeMs;
    std::optional<double> endToEndLatencyMs;
    double deadlineMs{};
    std::optional<bool> deadlineMissed;
    FrameStatus frameStatus{FrameStatus::Processed};
};

class LatestFrameQueue final {
public:
    std::optional<ScheduledFrame> put(
        ScheduledFrame frame
    )
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (closed_) {
            throw std::runtime_error(
                "Cannot enqueue a frame after queue closure."
            );
        }

        std::optional<ScheduledFrame> droppedFrame;

        if (frame_.has_value()) {
            droppedFrame = std::move(frame_);
        }

        frame_ = std::move(frame);
        condition_.notify_one();
        return droppedFrame;
    }

    std::optional<ScheduledFrame> get()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(
            lock,
            [this] {
                return closed_ || frame_.has_value();
            }
        );

        if (!frame_.has_value()) {
            return std::nullopt;
        }

        std::optional<ScheduledFrame> result(
            std::move(frame_)
        );
        frame_.reset();
        return result;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<ScheduledFrame> frame_;
    bool closed_{false};
};

QJsonObject loadJsonObject(const fs::path &path)
{
    QFile file(QString::fromStdString(path.string()));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error(
            "Configuration file could not be opened: "
            + path.string()
        );
    }

    QJsonParseError parseError;
    const QJsonDocument document =
        QJsonDocument::fromJson(
            file.readAll(),
            &parseError
        );

    if (parseError.error != QJsonParseError::NoError) {
        throw std::runtime_error(
            "Configuration contains invalid JSON: "
            + path.string()
        );
    }

    if (!document.isObject()) {
        throw std::runtime_error(
            "Configuration root must be a JSON object: "
            + path.string()
        );
    }

    return document.object();
}

int requireInteger(
    const QJsonObject &object,
    const char *fieldName,
    int minimum
)
{
    const QJsonValue value = object.value(fieldName);

    if (!value.isDouble()) {
        throw std::runtime_error(
            std::string(fieldName)
            + " must be an integer."
        );
    }

    const double numericValue = value.toDouble();
    const int integerValue = value.toInt(minimum - 1);

    if (
        numericValue != static_cast<double>(integerValue)
        || integerValue < minimum
    ) {
        throw std::runtime_error(
            std::string(fieldName)
            + " must be an integer greater than or equal to "
            + std::to_string(minimum)
            + "."
        );
    }

    return integerValue;
}

double requirePositiveNumber(
    const QJsonObject &object,
    const char *fieldName,
    std::optional<double> defaultValue = std::nullopt
)
{
    const QJsonValue value = object.value(fieldName);

    if (value.isUndefined() && defaultValue.has_value()) {
        return *defaultValue;
    }

    if (!value.isDouble()) {
        throw std::runtime_error(
            std::string(fieldName)
            + " must be a number greater than zero."
        );
    }

    const double number = value.toDouble();

    if (!std::isfinite(number) || number <= 0.0) {
        throw std::runtime_error(
            std::string(fieldName)
            + " must be a finite number greater than zero."
        );
    }

    return number;
}

std::string requireString(
    const QJsonObject &object,
    const char *fieldName
)
{
    const QJsonValue value = object.value(fieldName);

    if (!value.isString()) {
        throw std::runtime_error(
            std::string(fieldName)
            + " must be a non-empty string."
        );
    }

    const QString text = value.toString().trimmed();

    if (text.isEmpty()) {
        throw std::runtime_error(
            std::string(fieldName)
            + " must be a non-empty string."
        );
    }

    return text.toStdString();
}

ProcessingAlgorithm algorithmFromName(
    const std::string &name
)
{
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

    throw std::runtime_error(
        "Unsupported algorithm: " + name
    );
}

RealtimeConfiguration loadRealtimeConfiguration(
    const fs::path &projectRoot
)
{
    const fs::path configPath =
        projectRoot
        / "benchmarks"
        / "config"
        / "realtime_config.json";

    const QJsonObject root = loadJsonObject(configPath);
    const std::set<std::string> requiredFields{
        "schema_version",
        "target_fps",
        "queue_capacity",
        "warmup_frames",
        "measured_frames",
        "trial_count",
        "drop_policy",
        "deadline_multiplier",
        "output_directory",
        "frame_results_file",
        "summary_file"
    };
    std::set<std::string> actualFields;

    for (const QString &key : root.keys()) {
        actualFields.insert(key.toStdString());
    }

    if (actualFields != requiredFields) {
        throw std::runtime_error(
            "Real-time configuration fields do not match "
            "the supported schema."
        );
    }

    const int schemaVersion = requireInteger(
        root,
        "schema_version",
        1
    );

    if (schemaVersion != 1) {
        throw std::runtime_error(
            "Unsupported real-time configuration schema version."
        );
    }

    RealtimeConfiguration configuration;
    configuration.targetFps = requirePositiveNumber(
        root,
        "target_fps"
    );
    configuration.queueCapacity = requireInteger(
        root,
        "queue_capacity",
        1
    );
    configuration.warmupFrames = requireInteger(
        root,
        "warmup_frames",
        0
    );
    configuration.measuredFrames = requireInteger(
        root,
        "measured_frames",
        1
    );
    configuration.trialCount = requireInteger(
        root,
        "trial_count",
        1
    );
    configuration.dropPolicy = requireString(
        root,
        "drop_policy"
    );
    configuration.deadlineMultiplier =
        requirePositiveNumber(
            root,
            "deadline_multiplier"
        );
    configuration.frameResultsFile = requireString(
        root,
        "frame_results_file"
    );
    configuration.summaryFile = requireString(
        root,
        "summary_file"
    );

    if (
        configuration.dropPolicy != "latest_frame"
        || configuration.queueCapacity != 1
    ) {
        throw std::runtime_error(
            "Pure C++ real-time evaluation requires "
            "drop_policy=latest_frame and queue_capacity=1."
        );
    }

    const fs::path resultsRoot =
        (projectRoot / "benchmarks" / "results")
            .lexically_normal();
    const fs::path relativeOutputDirectory(
        requireString(root, "output_directory")
    );

    if (relativeOutputDirectory.is_absolute()) {
        throw std::runtime_error(
            "output_directory must be relative to the project root."
        );
    }

    configuration.outputDirectory =
        (projectRoot / relativeOutputDirectory)
            .lexically_normal();

    const fs::path relativeToResults =
        configuration.outputDirectory
            .lexically_relative(resultsRoot);

    if (
        relativeToResults.empty()
        || *relativeToResults.begin() == ".."
    ) {
        throw std::runtime_error(
            "output_directory must be located under benchmarks/results."
        );
    }

    for (const auto &[fieldName, fileName] : {
             std::pair<const char *, std::string>{
                 "frame_results_file",
                 configuration.frameResultsFile
             },
             std::pair<const char *, std::string>{
                 "summary_file",
                 configuration.summaryFile
             }
         }) {
        const fs::path filePath(fileName);

        if (
            filePath.has_parent_path()
            || filePath.extension() != ".csv"
        ) {
            throw std::runtime_error(
                std::string(fieldName)
                + " must be a CSV file name without "
                  "a directory component."
            );
        }
    }

    if (
        configuration.frameResultsFile
        == configuration.summaryFile
    ) {
        throw std::runtime_error(
            "frame_results_file and summary_file "
            "must be different."
        );
    }

    return configuration;
}

ExperimentPlan loadExperimentPlan(
    const fs::path &projectRoot
)
{
    ExperimentPlan plan;
    plan.realtime = loadRealtimeConfiguration(projectRoot);

    const fs::path configPath =
        projectRoot
        / "benchmarks"
        / "config"
        / "benchmark_config.json";
    const QJsonObject root = loadJsonObject(configPath);
    const QJsonObject input = root.value("input").toObject();
    const QJsonObject benchmark =
        root.value("benchmark").toObject();

    if (input.isEmpty() || benchmark.isEmpty()) {
        throw std::runtime_error(
            "Benchmark configuration is missing input or benchmark."
        );
    }

    const fs::path relativeVideoPath(
        requireString(input, "video_path")
    );

    if (relativeVideoPath.is_absolute()) {
        throw std::runtime_error(
            "input.video_path must be relative to the project root."
        );
    }

    plan.videoPath =
        (projectRoot / relativeVideoPath)
            .lexically_normal();

    const fs::path videoRelativeToRoot =
        plan.videoPath.lexically_relative(projectRoot);

    if (
        videoRelativeToRoot.empty()
        || *videoRelativeToRoot.begin() == ".."
        || !fs::is_regular_file(plan.videoPath)
    ) {
        throw std::runtime_error(
            "Benchmark video was not found or points outside "
            "the project root: " + plan.videoPath.string()
        );
    }

    const int benchmarkWarmup = requireInteger(
        benchmark,
        "warmup_frames",
        0
    );
    const int benchmarkMeasured = requireInteger(
        benchmark,
        "measured_frames",
        1
    );
    const int benchmarkTrials = requireInteger(
        benchmark,
        "trials",
        1
    );

    if (
        benchmarkWarmup != plan.realtime.warmupFrames
        || benchmarkMeasured != plan.realtime.measuredFrames
        || benchmarkTrials != plan.realtime.trialCount
    ) {
        throw std::runtime_error(
            "Offline and real-time experiment counts must match."
        );
    }

    const QJsonArray resolutionArray =
        root.value("resolutions").toArray();
    std::set<std::pair<int, int>> resolutionKeys;

    for (const QJsonValue &value : resolutionArray) {
        if (!value.isObject()) {
            throw std::runtime_error(
                "Each resolution must be a JSON object."
            );
        }

        const QJsonObject object = value.toObject();
        Resolution resolution{
            requireInteger(object, "width", 1),
            requireInteger(object, "height", 1)
        };

        if (!resolutionKeys.insert(
                {resolution.width, resolution.height}
            ).second) {
            throw std::runtime_error(
                "Duplicate resolutions were found."
            );
        }

        plan.resolutions.push_back(resolution);
    }

    const QJsonArray algorithmArray =
        root.value("algorithms").toArray();
    std::set<std::string> algorithmNames;

    for (const QJsonValue &value : algorithmArray) {
        if (!value.isObject()) {
            throw std::runtime_error(
                "Each algorithm must be a JSON object."
            );
        }

        const QJsonObject object = value.toObject();
        const std::string name = requireString(
            object,
            "name"
        );
        const QJsonObject parameters =
            object.value("parameters").toObject();

        if (!algorithmNames.insert(name).second) {
            throw std::runtime_error(
                "Duplicate algorithms were found."
            );
        }

        ProcessingParameters processingParameters;
        processingParameters.gammaValue =
            requirePositiveNumber(
                parameters,
                "gamma_value",
                0.6
            );
        processingParameters.claheClipLimit =
            requirePositiveNumber(
                parameters,
                "clip_limit",
                4.0
            );
        processingParameters.claheGridSize =
            parameters.contains("grid_size")
                ? requireInteger(
                      parameters,
                      "grid_size",
                      1
                  )
                : 8;

        plan.algorithms.push_back(
            AlgorithmConfiguration{
                name,
                algorithmFromName(name),
                processingParameters
            }
        );
    }

    if (
        plan.resolutions.empty()
        || plan.algorithms.empty()
    ) {
        throw std::runtime_error(
            "At least one resolution and algorithm are required."
        );
    }

    return plan;
}

double relativeMilliseconds(
    TimePoint timestamp,
    TimePoint origin
)
{
    return std::chrono::duration<double, std::milli>(
        timestamp - origin
    ).count();
}

FrameRecord createProcessedRecord(
    const std::string &algorithm,
    const std::string &resolution,
    int trial,
    int frameIndex,
    TimePoint origin,
    double deadlineMs,
    const ScheduledFrame &frame,
    TimePoint processingStart,
    TimePoint processingEnd
)
{
    const double scheduledMs = relativeMilliseconds(
        frame.scheduledTimestamp,
        origin
    );
    const double enqueuedMs = relativeMilliseconds(
        frame.enqueuedTimestamp,
        origin
    );
    const double processingStartMs = relativeMilliseconds(
        processingStart,
        origin
    );
    const double processingEndMs = relativeMilliseconds(
        processingEnd,
        origin
    );
    const double endToEndLatencyMs =
        processingEndMs - scheduledMs;

    return FrameRecord{
        "pure_cpp",
        algorithm,
        resolution,
        trial,
        frameIndex,
        scheduledMs,
        enqueuedMs,
        processingStartMs,
        processingEndMs,
        std::nullopt,
        enqueuedMs - scheduledMs,
        processingStartMs - enqueuedMs,
        processingEndMs - processingStartMs,
        endToEndLatencyMs,
        deadlineMs,
        endToEndLatencyMs > deadlineMs,
        FrameStatus::Processed
    };
}

FrameRecord createDroppedRecord(
    const std::string &algorithm,
    const std::string &resolution,
    int trial,
    int frameIndex,
    TimePoint origin,
    double deadlineMs,
    const ScheduledFrame &frame,
    TimePoint dropTimestamp
)
{
    const double scheduledMs = relativeMilliseconds(
        frame.scheduledTimestamp,
        origin
    );
    const double enqueuedMs = relativeMilliseconds(
        frame.enqueuedTimestamp,
        origin
    );

    return FrameRecord{
        "pure_cpp",
        algorithm,
        resolution,
        trial,
        frameIndex,
        scheduledMs,
        enqueuedMs,
        std::nullopt,
        std::nullopt,
        relativeMilliseconds(dropTimestamp, origin),
        enqueuedMs - scheduledMs,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        deadlineMs,
        std::nullopt,
        FrameStatus::Dropped
    };
}

std::vector<FrameRecord> runRealtimeTrial(
    const ExperimentPlan &plan,
    const Resolution &resolution,
    const AlgorithmConfiguration &algorithm,
    int trial
)
{
    const auto framePeriod = std::chrono::nanoseconds(
        static_cast<long long>(std::llround(
            1'000'000'000.0
            / plan.realtime.targetFps
        ))
    );
    const int totalFrameCount =
        plan.realtime.warmupFrames
        + plan.realtime.measuredFrames;
    const double deadlineMs =
        plan.realtime.deadlineMilliseconds();

    LatestFrameQueue frameQueue;
    std::atomic<bool> stopRequested{false};
    std::mutex waitMutex;
    std::condition_variable waitCondition;
    std::mutex recordMutex;
    std::mutex errorMutex;
    std::vector<FrameRecord> records;
    std::exception_ptr firstError;
    const TimePoint origin = Clock::now();

    const auto requestStop = [&] {
        stopRequested.store(true);
        waitCondition.notify_all();
        frameQueue.close();
    };

    const auto registerError = [&](std::exception_ptr error) {
        {
            std::lock_guard<std::mutex> lock(errorMutex);

            if (!firstError) {
                firstError = error;
            }
        }

        requestStop();
    };

    const auto appendRecord = [&](FrameRecord record) {
        std::lock_guard<std::mutex> lock(recordMutex);
        records.push_back(std::move(record));
    };

    const auto measuredFrameIndex = [&](int sequenceIndex) {
        if (sequenceIndex <= plan.realtime.warmupFrames) {
            return 0;
        }

        return sequenceIndex - plan.realtime.warmupFrames;
    };

    const auto waitUntil = [&](TimePoint target) {
        std::unique_lock<std::mutex> lock(waitMutex);
        return !waitCondition.wait_until(
            lock,
            target,
            [&] {
                return stopRequested.load();
            }
        );
    };

    std::thread consumer([&] {
        try {
            ImageProcess processor;

            while (true) {
                std::optional<ScheduledFrame> scheduledFrame =
                    frameQueue.get();

                if (!scheduledFrame.has_value()) {
                    break;
                }

                const TimePoint processingStart = Clock::now();
                cv::Mat processedFrame;
                processor.process(
                    scheduledFrame->payload,
                    processedFrame,
                    algorithm.algorithm,
                    algorithm.parameters
                );
                const TimePoint processingEnd = Clock::now();

                if (
                    processedFrame.empty()
                    || processedFrame.size()
                       != scheduledFrame->payload.size()
                    || processedFrame.type()
                       != scheduledFrame->payload.type()
                ) {
                    throw std::runtime_error(
                        "Processed frame does not match the source frame."
                    );
                }

                const int resultFrameIndex =
                    measuredFrameIndex(
                        scheduledFrame->sequenceIndex
                    );

                if (resultFrameIndex > 0) {
                    appendRecord(
                        createProcessedRecord(
                            algorithm.name,
                            resolution.name(),
                            trial,
                            resultFrameIndex,
                            origin,
                            deadlineMs,
                            *scheduledFrame,
                            processingStart,
                            processingEnd
                        )
                    );
                }
            }
        } catch (...) {
            registerError(std::current_exception());
        }
    });

    std::thread producer([&] {
        try {
            cv::VideoCapture capture(plan.videoPath.string());

            if (!capture.isOpened()) {
                throw std::runtime_error(
                    "Input video could not be opened: "
                    + plan.videoPath.string()
                );
            }

            for (
                int sequenceIndex = 1;
                sequenceIndex <= totalFrameCount;
                ++sequenceIndex
            ) {
                if (stopRequested.load()) {
                    break;
                }

                const TimePoint scheduledTimestamp =
                    origin
                    + framePeriod * (sequenceIndex - 1);
                cv::Mat sourceFrame;

                if (
                    !capture.read(sourceFrame)
                    || sourceFrame.empty()
                ) {
                    throw std::runtime_error(
                        "Input video ended before the real-time "
                        "trial completed. Required frames: "
                        + std::to_string(totalFrameCount)
                        + "; received: "
                        + std::to_string(sequenceIndex - 1)
                        + "."
                    );
                }

                if (
                    sourceFrame.type() != CV_8UC3
                ) {
                    throw std::runtime_error(
                        "Input frame must be an 8-bit, "
                        "three-channel BGR image."
                    );
                }

                cv::Mat resizedFrame;
                cv::resize(
                    sourceFrame,
                    resizedFrame,
                    cv::Size(
                        resolution.width,
                        resolution.height
                    ),
                    0.0,
                    0.0,
                    cv::INTER_LINEAR
                );

                if (!waitUntil(scheduledTimestamp)) {
                    break;
                }

                if (stopRequested.load()) {
                    break;
                }

                const TimePoint enqueuedTimestamp = Clock::now();
                std::optional<ScheduledFrame> droppedFrame =
                    frameQueue.put(
                        ScheduledFrame{
                            sequenceIndex,
                            scheduledTimestamp,
                            enqueuedTimestamp,
                            std::move(resizedFrame)
                        }
                    );

                if (droppedFrame.has_value()) {
                    const int droppedIndex =
                        measuredFrameIndex(
                            droppedFrame->sequenceIndex
                        );

                    if (droppedIndex > 0) {
                        appendRecord(
                            createDroppedRecord(
                                algorithm.name,
                                resolution.name(),
                                trial,
                                droppedIndex,
                                origin,
                                deadlineMs,
                                *droppedFrame,
                                enqueuedTimestamp
                            )
                        );
                    }
                }
            }
        } catch (...) {
            registerError(std::current_exception());
        }

        frameQueue.close();
    });

    producer.join();
    consumer.join();

    if (firstError) {
        std::rethrow_exception(firstError);
    }

    std::sort(
        records.begin(),
        records.end(),
        [](const FrameRecord &left, const FrameRecord &right) {
            return left.frameIndex < right.frameIndex;
        }
    );

    if (
        static_cast<int>(records.size())
        != plan.realtime.measuredFrames
    ) {
        throw std::runtime_error(
            "Real-time frame-result integrity validation failed: "
            "incorrect record count."
        );
    }

    for (
        int index = 0;
        index < plan.realtime.measuredFrames;
        ++index
    ) {
        if (records[index].frameIndex != index + 1) {
            throw std::runtime_error(
                "Real-time frame-result integrity validation failed: "
                "missing, duplicate or unexpected frame index."
            );
        }
    }

    return records;
}

std::string formatOptionalDouble(
    const std::optional<double> &value
)
{
    if (!value.has_value()) {
        return {};
    }

    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::fixed << std::setprecision(6) << *value;
    return stream.str();
}

std::string formatDouble(double value)
{
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

const char *frameStatusName(FrameStatus status)
{
    return status == FrameStatus::Processed
        ? "processed"
        : "dropped";
}

void writeFrameRecords(
    std::vector<FrameRecord> records,
    const fs::path &outputPath
)
{
    if (records.empty()) {
        throw std::runtime_error(
            "At least one real-time frame record is required."
        );
    }

    std::sort(
        records.begin(),
        records.end(),
        [](const FrameRecord &left, const FrameRecord &right) {
            return std::tie(
                left.architecture,
                left.algorithm,
                left.resolution,
                left.trial,
                left.frameIndex
            ) < std::tie(
                right.architecture,
                right.algorithm,
                right.resolution,
                right.trial,
                right.frameIndex
            );
        }
    );

    for (std::size_t index = 1; index < records.size(); ++index) {
        const FrameRecord &previous = records[index - 1];
        const FrameRecord &current = records[index];

        if (
            std::tie(
                previous.architecture,
                previous.algorithm,
                previous.resolution,
                previous.trial,
                previous.frameIndex
            ) == std::tie(
                current.architecture,
                current.algorithm,
                current.resolution,
                current.trial,
                current.frameIndex
            )
        ) {
            throw std::runtime_error(
                "Duplicate real-time frame records were detected."
            );
        }
    }

    fs::create_directories(outputPath.parent_path());
    const fs::path temporaryPath =
        outputPath.string() + ".tmp";
    std::ofstream output(temporaryPath);

    if (!output.is_open()) {
        throw std::runtime_error(
            "Real-time result file could not be opened: "
            + temporaryPath.string()
        );
    }

    output
        << "architecture,algorithm,resolution,trial,frame_index,"
        << "scheduled_timestamp_ms,enqueued_timestamp_ms,"
        << "processing_start_timestamp_ms,"
        << "processing_end_timestamp_ms,drop_timestamp_ms,"
        << "source_delay_ms,queue_wait_time_ms,"
        << "processing_time_ms,end_to_end_latency_ms,"
        << "deadline_ms,deadline_missed,frame_status\n";

    for (const FrameRecord &record : records) {
        output
            << record.architecture << ','
            << record.algorithm << ','
            << record.resolution << ','
            << record.trial << ','
            << record.frameIndex << ','
            << formatDouble(record.scheduledTimestampMs) << ','
            << formatOptionalDouble(record.enqueuedTimestampMs) << ','
            << formatOptionalDouble(
                   record.processingStartTimestampMs
               ) << ','
            << formatOptionalDouble(
                   record.processingEndTimestampMs
               ) << ','
            << formatOptionalDouble(record.dropTimestampMs) << ','
            << formatOptionalDouble(record.sourceDelayMs) << ','
            << formatOptionalDouble(record.queueWaitTimeMs) << ','
            << formatOptionalDouble(record.processingTimeMs) << ','
            << formatOptionalDouble(record.endToEndLatencyMs) << ','
            << formatDouble(record.deadlineMs) << ',';

        if (record.deadlineMissed.has_value()) {
            output << (*record.deadlineMissed ? "true" : "false");
        }

        output << ',' << frameStatusName(record.frameStatus) << '\n';
    }

    output.close();

    if (!output) {
        throw std::runtime_error(
            "Real-time result file could not be written completely."
        );
    }

    std::error_code error;
    fs::remove(outputPath, error);
    error.clear();
    fs::rename(temporaryPath, outputPath, error);

    if (error) {
        throw std::runtime_error(
            "Temporary result file could not replace the output: "
            + error.message()
        );
    }
}

} // namespace

void runControlledIlluminationTrial(
    const ExperimentPlan &plan
)
{
    const std::string architecture =
        requireEnvironmentVariable(
            "VISIONLAB_ARCHITECTURE"
        );

    if (architecture != "pure_cpp") {
        throw std::runtime_error(
            "VisionLabCppRealtime controlled mode "
            "requires architecture=pure_cpp."
        );
    }

    const std::string algorithmName =
        requireEnvironmentVariable(
            "VISIONLAB_ALGORITHM"
        );
    const int width =
        parsePositiveEnvironmentInteger(
            "VISIONLAB_RESOLUTION_WIDTH"
        );
    const int height =
        parsePositiveEnvironmentInteger(
            "VISIONLAB_RESOLUTION_HEIGHT"
        );
    const int trial =
        parsePositiveEnvironmentInteger(
            "VISIONLAB_TRIAL_NUMBER"
        );
    const double targetFps =
        parsePositiveEnvironmentNumber(
            "VISIONLAB_TARGET_FPS"
        );
    const double frameDeadlineMs =
        parsePositiveEnvironmentNumber(
            "VISIONLAB_FRAME_DEADLINE_MS"
        );

    if (trial > plan.realtime.trialCount) {
        throw std::runtime_error(
            "The controlled-illumination trial "
            "number exceeds the configured trial count."
        );
    }

    if (
        std::abs(
            targetFps
            - plan.realtime.targetFps
        ) > 1e-9
    ) {
        throw std::runtime_error(
            "The controlled-illumination target FPS "
            "does not match the real-time configuration."
        );
    }

    if (
        std::abs(
            frameDeadlineMs
            - plan.realtime.deadlineMilliseconds()
        ) > 1e-6
    ) {
        throw std::runtime_error(
            "The controlled-illumination frame deadline "
            "does not match the real-time configuration."
        );
    }

    const Resolution &resolution =
        findPlannedResolution(
            plan,
            width,
            height
        );
    const AlgorithmConfiguration &algorithm =
        findPlannedAlgorithm(
            plan,
            algorithmName
        );

    const fs::path outputPath =
        fs::path(
            requireEnvironmentVariable(
                "VISIONLAB_CPP_FRAME_RESULTS_PATH"
            )
        ).lexically_normal();

    if (!outputPath.is_absolute()) {
        throw std::runtime_error(
            "VISIONLAB_CPP_FRAME_RESULTS_PATH "
            "must be an absolute path."
        );
    }

    if (outputPath.extension() != ".csv") {
        throw std::runtime_error(
            "VISIONLAB_CPP_FRAME_RESULTS_PATH "
            "must use the .csv extension."
        );
    }

    std::cout
        << "Pure C++ controlled illumination | "
        << algorithm.name << " | "
        << resolution.name() << " | trial "
        << trial << '/'
        << plan.realtime.trialCount
        << std::endl;

    std::vector<FrameRecord> records =
        runRealtimeTrial(
            plan,
            resolution,
            algorithm,
            trial
        );

    const std::size_t expectedRecordCount =
        static_cast<std::size_t>(
            plan.realtime.measuredFrames
        );

    if (records.size() != expectedRecordCount) {
        throw std::runtime_error(
            "The controlled-illumination run produced "
            "an unexpected frame-record count."
        );
    }

    writeFrameRecords(
        records,
        outputPath
    );

    const std::size_t processedCount =
        static_cast<std::size_t>(
            std::count_if(
                records.begin(),
                records.end(),
                [](const FrameRecord &record) {
                    return (
                        record.frameStatus
                        == FrameStatus::Processed
                    );
                }
            )
        );
    const std::size_t droppedCount =
        records.size() - processedCount;

    std::cout
        << "Pure C++ controlled-illumination "
        << "run completed.\n"
        << "Frame records: "
        << records.size() << '\n'
        << "Processed: "
        << processedCount << '\n'
        << "Dropped: "
        << droppedCount << '\n'
        << "Results: "
        << outputPath.string()
        << std::endl;
}

int main()
{
#if !VISIONLAB_REALTIME_RELEASE
    std::cerr
        << "VisionLabCppRealtime must be run in Release mode."
        << std::endl;
    return 1;
#else
    try {
        const fs::path projectRoot =
            fs::path(VISIONLAB_PROJECT_ROOT)
                .lexically_normal();
        const ExperimentPlan plan =
            loadExperimentPlan(projectRoot);

        if (controlledIlluminationRunRequested()) {
            runControlledIlluminationTrial(
                plan
            );
            return 0;
        }

        std::vector<FrameRecord> allRecords;

        const std::size_t expectedRecordCount =
            plan.resolutions.size()
            * plan.algorithms.size()
            * static_cast<std::size_t>(
                plan.realtime.trialCount
            )
            * static_cast<std::size_t>(
                plan.realtime.measuredFrames
            );
        allRecords.reserve(expectedRecordCount);

        for (const Resolution &resolution : plan.resolutions) {
            for (
                const AlgorithmConfiguration &algorithm
                : plan.algorithms
            ) {
                for (
                    int trial = 1;
                    trial <= plan.realtime.trialCount;
                    ++trial
                ) {
                    std::cout
                        << "Pure C++ real-time | "
                        << algorithm.name << " | "
                        << resolution.name() << " | trial "
                        << trial << '/'
                        << plan.realtime.trialCount
                        << std::endl;

                    std::vector<FrameRecord> trialRecords =
                        runRealtimeTrial(
                            plan,
                            resolution,
                            algorithm,
                            trial
                        );

                    allRecords.insert(
                        allRecords.end(),
                        std::make_move_iterator(
                            trialRecords.begin()
                        ),
                        std::make_move_iterator(
                            trialRecords.end()
                        )
                    );
                }
            }
        }

        const fs::path outputPath =
            plan.realtime.outputDirectory
            / "pure_cpp"
            / plan.realtime.frameResultsFile;
        writeFrameRecords(allRecords, outputPath);

        const std::size_t processedCount =
            static_cast<std::size_t>(std::count_if(
                allRecords.begin(),
                allRecords.end(),
                [](const FrameRecord &record) {
                    return record.frameStatus
                           == FrameStatus::Processed;
                }
            ));
        const std::size_t droppedCount =
            allRecords.size() - processedCount;

        std::cout
            << "Pure C++ real-time evaluation completed.\n"
            << "Frame records: " << allRecords.size() << '\n'
            << "Processed: " << processedCount << '\n'
            << "Dropped: " << droppedCount << '\n'
            << "Skipped: 0\n"
            << "Results: " << outputPath.string()
            << std::endl;

        return 0;
    } catch (const std::exception &error) {
        std::cerr
            << "Pure C++ real-time evaluation failed: "
            << error.what()
            << std::endl;
        return 1;
    }
#endif
}