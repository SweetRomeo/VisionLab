#ifndef VISIONWORKER_H
#define VISIONWORKER_H

#pragma once
#include <QObject>
#include <QImage>
#include <QString>
#include <QMutex>
#include <QMutexLocker>
#include <atomic>
#include <opencv2/opencv.hpp>

class VisionWorker : public QObject
{
    Q_OBJECT
public:
    explicit VisionWorker(QObject *parent = nullptr);
    ~VisionWorker();

public slots:
    // Arayüzden (UI) tetiklenecek fonksiyonlar
    void startProcessing();
    void stopProcessing();
    void setAlgorithm(const QString& algoName);
    void setGammaValue(double gamma);

signals:
    // Arayüze fırlatılacak veriler
    void framesReady(const QImage& readyFrame);
    void metricsUpdated(double fps, double processTimeMs);

private:
    // Thread güvenliği (Thread-safety) için değişkenler
    std::atomic<bool> isRunning; // Döngüyü güvenle kırmak için

    QMutex paramMutex;           // Algoritma parametrelerini koruyan kilit
    QString currentAlgorithm;
    double gammaValue;

    // Arka plan iş mantığı
    void applyAlgorithm(const cv::Mat& src, cv::Mat& dst);
};

#endif // VISIONWORKER_H
