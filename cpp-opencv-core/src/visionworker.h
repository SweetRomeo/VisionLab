#ifndef VISIONWORKER_H
#define VISIONWORKER_H
#include "imageprocess.h"

#include <QImage>
#include <QMutex>
#include <QObject>
#include <QString>
#include <QElapsedTimer>


#include <atomic>
#include <opencv2/opencv.hpp>

class VisionWorker final : public QObject
{
    Q_OBJECT

public:
    explicit VisionWorker(QObject *parent = nullptr);
    ~VisionWorker() override;

public slots:
    void startProcessing();
    void stopProcessing();
    void setAlgorithm(const QString &algorithmName);
    void setGammaValue(double gamma);
    void setClaheClipLimit(double clipLimit);
    void setClaheGridSize(int gridSize);

signals:
    void framesReady(const QImage &readyFrame);
    void metricsUpdated(double fps, double processTimeMs);
    void statusChanged(const QString &status, bool isError);

private:
    void processNextFrame();

    void applyAlgorithm(
        const cv::Mat &source,
        cv::Mat &destination
        );

    cv::VideoCapture camera;
    QTimer *frameTimer = nullptr;

    QElapsedTimer fpsTimer;
    int processedFrameCount = 0;
    double accumulatedProcessingTimeMs = 0.0;

    std::atomic<bool> isRunning{false};
    QMutex parameterMutex;
    ProcessingAlgorithm currentAlgorithm = ProcessingAlgorithm::Original;
    double gammaValue{1.0};
    double claheClipLimit{4.0};
    int claheGridSize{8};
};

#endif // VISIONWORKER_H
