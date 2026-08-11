#include "visionworker.h"

#include <QCoreApplication>
#include <QDebug>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QMutexLocker>
#include <QTimer>

#include <cmath>

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
    isRunning = false;
}

void VisionWorker::setAlgorithm(const QString &algorithmName)
{
    QMutexLocker locker(&parameterMutex);
    currentAlgorithm = algorithmName;
}

void VisionWorker::setGammaValue(double gamma)
{
    QMutexLocker locker(&parameterMutex);
    gammaValue = gamma;
}

void VisionWorker::setClaheClipLimit(double clipLimit)
{
    QMutexLocker locker(&parameterMutex);
    claheClipLimit = clipLimit;
}

void VisionWorker::setClaheGridSize(int gridSize)
{
    QMutexLocker locker(&parameterMutex);
    claheGridSize = gridSize;
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

    QElapsedTimer processingTimer;
    processingTimer.start();

    cv::Mat processedFrame;
    applyAlgorithm(frame, processedFrame);

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


void VisionWorker::applyAlgorithm(
    const cv::Mat &source,
    cv::Mat &destination
)
{
    QString algorithm;
    double gamma = 1.0;

    double clipLimit = 4.0;
    int gridSize = 8;

    {
        QMutexLocker locker(&parameterMutex);

        algorithm = currentAlgorithm;
        gamma = gammaValue;

        clipLimit = claheClipLimit;
        gridSize = claheGridSize;
    }

    if (algorithm == "Gamma Correction")
    {
        cv::Mat lookupTable(1, 256, CV_8U);
        auto *table = lookupTable.ptr<uchar>();

        for (int i = 0; i < 256; ++i)
        {
            table[i] = cv::saturate_cast<uchar>(
                std::pow(i / 255.0, gamma) * 255.0
            );
        }

        cv::LUT(source, lookupTable, destination);
        return;
    }

    if (algorithm == "CLAHE")
    {
        cv::Mat grayFrame;
        cv::Mat claheFrame;

        cv::cvtColor(
            source,
            grayFrame,
            cv::COLOR_BGR2GRAY
        );

        const cv::Ptr<cv::CLAHE> clahe =
            cv::createCLAHE(
                clipLimit,
                cv::Size(gridSize, gridSize)
                );
        clahe->apply(grayFrame, claheFrame);

        cv::cvtColor(
            claheFrame,
            destination,
            cv::COLOR_GRAY2BGR
        );
        return;
    }

    destination = source.clone();
}
