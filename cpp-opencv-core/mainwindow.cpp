#include "MainWindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);

    // 1. Thread ve Worker nesnelerini oluştur (Heap üzerinde)
    workerThread = new QThread(this);
    visionWorker = new VisionWorker(); // Parent vermiyoruz! (moveToThread için şart)

    // 2. İşçiyi yeni thread'e taşı
    visionWorker->moveToThread(workerThread);

    // ==========================================
    // 3. SİNYAL VE SLOT BAĞLANTILARI (KONEKSİYONLAR)
    // ==========================================

    // A. Görüntü Akışı: İşçi yeni kareyi bitirdiğinde OpenGL ekranına yolla
    connect(visionWorker, &VisionWorker::framesReady,
            ui->myVideoWidget, &VideoGLWidget::updateFrame);

    // B. Performans Metrikleri: İşçi yeni metrik hesapladığında arayüzü güncelle
    // Not: C++11 Lambda fonksiyonları küçük UI güncellemeleri için çok zariftir.
    connect(visionWorker, &VisionWorker::metricsUpdated, this, [=](double fps, double processTime) {
        ui->lcdFPS->display(fps);
        ui->lcdTime->display(processTime);
    });

    // C. Kullanıcı Kontrolleri: UI'daki değişiklikleri İşçiye bildir
    connect(ui->comboAlgorithm, &QComboBox::currentTextChanged,
            visionWorker, &VisionWorker::setAlgorithm);

    connect(ui->sliderGamma, &QSlider::valueChanged, this, [=](int value) {
        // Slider tam sayı döndürür, Worker'a ondalıklı (double) göndermek için
        double gamma = value / 10.0;
        visionWorker->setGammaValue(gamma);
    });

    // D. Temizlik (Memory Management): Thread kapandığında nesneleri RAM'den sil
    connect(workerThread, &QThread::finished, visionWorker, &QObject::deleteLater);
    connect(workerThread, &QThread::finished, workerThread, &QObject::deleteLater);

    // ==========================================
    // 4. SİSTEMİ BAŞLAT
    // ==========================================

    // Thread başladığında Worker'ın içindeki iş döngüsünü (process) tetikle
    connect(workerThread, &QThread::started, visionWorker, &VisionWorker::startProcessing);

    workerThread->start(); // Motoru ateşle!
}

MainWindow::~MainWindow() {
    // Uygulama kapanırken önce thread'i güvenlice durdur
    if(workerThread->isRunning()) {
        visionWorker->stopProcessing(); // Döngüyü kıracak flag'i set et
        workerThread->quit();           // Thread'e bitmesini söyle
        workerThread->wait();           // Kapanana kadar bekle
    }
    delete ui;
}
