#include "visionworker.h"

#include <QDebug>
#include <QElapsedTimer>
#include <QMutexLocker>
#include <QTimer>

#include <cmath>
#include <algorithm>

VisionWorker::VisionWorker(QObject *parent)
    : QObject(parent)
{
}

VisionWorker::~VisionWorker()
{
    stopProcessing();
}

void VisionWorker::startProcessing()
{
    // Aynı worker'ın ikinci kez başlatılmasını engelle
    if (frameTimer != nullptr && frameTimer->isActive())
    {
        return;
    }

    if (!camera.open(0))
    {
        emit statusChanged("Kamera açılamadı", true);
        qWarning() << "Kamera açılamadı.";
        return;
    }

    // Timer, VisionWorker worker thread'e taşındıktan sonra
    // burada oluşturuluyor.
    if (frameTimer == nullptr)
    {
        frameTimer = new QTimer(this);
        frameTimer->setTimerType(Qt::PreciseTimer);

        connect(
            frameTimer,
            &QTimer::timeout,
            this,
            &VisionWorker::processNextFrame
            );
    }

    processedFrameCount = 0;
    accumulatedProcessingTimeMs = 0.0;
    fpsTimer.start();

    // Kamera FPS bilgisini desteklemiyorsa 30 FPS kullan.
    const double cameraFps = camera.get(cv::CAP_PROP_FPS);

    int frameIntervalMs = 33;

    if (std::isfinite(cameraFps) &&
        cameraFps >= 1.0 &&
        cameraFps <= 240.0)
    {
        frameIntervalMs = std::clamp(
            static_cast<int>(std::lround(1000.0 / cameraFps)),
            1,
            1000
            );
    }

    frameTimer->start(frameIntervalMs);

    emit statusChanged("Kamera çalışıyor", false);
}

void VisionWorker::stopProcessing()
{
    if (frameTimer != nullptr)
    {
        frameTimer->stop();
    }

    if (camera.isOpened())
    {
        camera.release();
    }

    emit statusChanged("Kamera durduruldu", false);
}

void VisionWorker::setAlgorithm(const QString &algorithmName)
{
    ProcessingAlgorithm selectedAlgorithm =
        ProcessingAlgorithm::Original;

    if (algorithmName == "Gamma Correction")
    {
        selectedAlgorithm =
            ProcessingAlgorithm::GammaCorrection;
    }
    else if (algorithmName == "Histogram Equalization")
    {
        selectedAlgorithm =
            ProcessingAlgorithm::HistogramEqualization;
    }
    else if (algorithmName == "CLAHE")
    {
        selectedAlgorithm =
            ProcessingAlgorithm::Clahe;
    }

    QMutexLocker locker(&parameterMutex);
    currentAlgorithm = selectedAlgorithm;
}

void VisionWorker::setGammaValue(double gamma)
{
    QMutexLocker locker(&parameterMutex);
    processingParameters.gammaValue = gamma;
}

void VisionWorker::setClaheClipLimit(double clipLimit)
{
    QMutexLocker locker(&parameterMutex);
    processingParameters.claheClipLimit = clipLimit;
}

void VisionWorker::setClaheGridSize(int gridSize)
{
    QMutexLocker locker(&parameterMutex);
    processingParameters.claheGridSize = gridSize;
}

void VisionWorker::processNextFrame()
{
    if (!camera.isOpened())
    {
        return;
    }

    cv::Mat frame;

    if (!camera.read(frame) || frame.empty())
    {
        qWarning() << "Kameradan geçerli kare alınamadı.";
        return;
    }

    ProcessingAlgorithm algorithm;
    ProcessingParameters parameters;

    {
        QMutexLocker locker(&parameterMutex);
        algorithm = currentAlgorithm;
        parameters = processingParameters;
    }

    QElapsedTimer processingTimer;
    processingTimer.start();

    cv::Mat processedFrame;

    imageProcessor.process(frame, processedFrame, algorithm, parameters);

    const double processingTimeMs =
        processingTimer.nsecsElapsed() / 1'000'000.0;

    cv::Mat rgbFrame;
    cv::cvtColor(
        processedFrame,
        rgbFrame,
        cv::COLOR_BGR2RGB
        );

    const QImage image(
        rgbFrame.data,
        rgbFrame.cols,
        rgbFrame.rows,
        static_cast<qsizetype>(rgbFrame.step),
        QImage::Format_RGB888
        );

    emit framesReady(image.copy());

    ++processedFrameCount;
    accumulatedProcessingTimeMs += processingTimeMs;

    const qint64 elapsedMs = fpsTimer.elapsed();

    // Metrikleri her kare yerine saniyede yaklaşık bir kez gönder
    if (elapsedMs >= 1000)
    {
        const double actualFps =
            processedFrameCount * 1000.0 /
            static_cast<double>(elapsedMs);

        const double averageProcessingTimeMs =
            accumulatedProcessingTimeMs /
            static_cast<double>(processedFrameCount);

        emit metricsUpdated(
            actualFps,
            averageProcessingTimeMs
            );

        processedFrameCount = 0;
        accumulatedProcessingTimeMs = 0.0;
        fpsTimer.restart();
    }
}
