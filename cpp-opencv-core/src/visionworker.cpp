#include "VisionWorker.h"
#include <QElapsedTimer>
#include <QCoreApplication>
#include <QThread>

VisionWorker::VisionWorker(QObject *parent)
    : QObject(parent), isRunning(false), currentAlgorithm("Orijinal"), gammaValue(1.0) {
}

VisionWorker::~VisionWorker() {
    stopProcessing();
}

void VisionWorker::startProcessing() {
    cv::VideoCapture cap(0); // 0: Varsayılan kamera. (Video dosyası için yol verilebilir)

    if (!cap.isOpened()) {
        qWarning() << "Kamera açılamadı!";
        return;
    }

    isRunning = true;
    QElapsedTimer timer; // Süre ölçümü için Qt'nin yüksek hassasiyetli sayacı

    while (isRunning) {
        cv::Mat frame;
        cap >> frame; // Kameradan kareyi al

        if (frame.empty()) continue;

        // 1. İŞLEM SÜRESİNİ BAŞLAT
        timer.restart();

        // 2. ALGORİTMAYI UYGULA
        cv::Mat processedFrame;
        applyAlgorithm(frame, processedFrame);

        // 3. METRİKLERİ HESAPLA
        double processTime = timer.elapsed();
        double fps = (processTime > 0) ? (1000.0 / processTime) : 0.0;

        // 4. RENK UZAYINI DÖNÜŞTÜR (OpenCV BGR -> Qt RGB)
        cv::Mat rgbFrame;
        cv::cvtColor(processedFrame, rgbFrame, cv::COLOR_BGR2RGB);

        // 5. QIMAGE OLUŞTUR VE KOPYALA (Deadlock ve Memory Leak'i önlemek için)
        QImage qimg((const unsigned char*) rgbFrame.data,
                    rgbFrame.cols,
                    rgbFrame.rows,
                    rgbFrame.step,
                    QImage::Format_RGB888);

        QImage readyFrame = qimg.copy(); // Derin kopya (Deep Copy)

        // 6. VERİLERİ ARAYÜZE (UI) FIRLAT
        emit framesReady(readyFrame);
        emit metricsUpdated(fps, processTime);

        // ÖNEMLİ: Qt'nin slotları (setAlgorithm vb.) bu while döngüsü içinde
        // yakalayabilmesi için olay döngüsüne kısa bir nefes aldırıyoruz.
        QCoreApplication::processEvents();
    }

    cap.release(); // Döngü kırıldığında kamerayı serbest bırak
}

void VisionWorker::stopProcessing() {
    isRunning = false; // While döngüsünü kırar
}

void VisionWorker::setAlgorithm(const QString& algoName) {
    // UI thread'i bu değişkeni değiştirirken, Worker okumasın diye kilitliyoruz.
    QMutexLocker locker(&paramMutex);
    currentAlgorithm = algoName;
}

void VisionWorker::setGammaValue(double gamma) {
    QMutexLocker locker(&paramMutex);
    gammaValue = gamma;
}

void VisionWorker::applyAlgorithm(const cv::Mat& src, cv::Mat& dst) {
    // İşleme başlamadan önce güncel parametrelerin kopyasını alıyoruz.
    // Böylece tüm algoritma boyunca kilitli kalıp arayüzü bekletmiyoruz.
    paramMutex.lock();
    QString algo = currentAlgorithm;
    double gamma = gammaValue;
    paramMutex.unlock();

    if (algo == "Gamma Correction") {
        // Hızlı Gamma işlemi için Look-Up Table (LUT) kullanımı
        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar* p = lookUpTable.ptr();
        for (int i = 0; i < 256; ++i) {
            p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        }
        cv::LUT(src, lookUpTable, dst);

    } else if (algo == "CLAHE") {
        // CLAHE genellikle Gri seviye görüntülerde çalışır
        cv::Mat gray, claheDst;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4.0);
        clahe->apply(gray, claheDst);

        // Arayüze RGB formatında vermek için tekrar renklendiriyoruz
        cv::cvtColor(claheDst, dst, cv::COLOR_GRAY2BGR);

    } else {
        // "Orijinal" seçiliyse veya bilinmeyen bir algoritmaysa doğrudan kopyala
        dst = src.clone();
    }
}
