#include "mainwindow.h"

#include "src/videoglwidget.h"
#include "src/visionworker.h"

#include <QAction>
#include <QDataStream>
#include <QFile>
#include <QFileDialog>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSaveFile>
#include <QStringList>
#include <QtMath>
#include <QComboBox>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>
#include <QSaveFile>
#include <QXmlStreamWriter>

namespace {
constexpr quint32 MagicNumber = 0x564C5031; // "VLP1"
constexpr quint16 FileVersion = 1;

struct ProjectData
{
    QString algorithm{"Orijinal"};
    double gamma{1.0};
    double claheClipLimit{4.0};
    int claheGridSize{8};
};

bool saveProjectToFile(
    const QString &fileName,
    const ProjectData &project,
    QString &errorMessage
    )
{
    QSaveFile file(fileName);

    if (!file.open(QIODevice::WriteOnly))
    {
        errorMessage = file.errorString();
        return false;
    }

    QDataStream stream(&file);
    stream.setVersion(QDataStream::Qt_6_0);

    stream << MagicNumber
           << FileVersion
           << project.algorithm
           << project.gamma;

    if (stream.status() != QDataStream::Ok)
    {
        file.cancelWriting();
        errorMessage = "Proje verileri yazılamadı.";
        return false;
    }

    if (!file.commit())
    {
        errorMessage = file.errorString();
        return false;
    }

    return true;
}

bool loadProjectFromFile(
    const QString &fileName,
    ProjectData &project,
    QString &errorMessage
    )
{
    QFile file(fileName);

    if (!file.open(QIODevice::ReadOnly))
    {
        errorMessage = file.errorString();
        return false;
    }

    QDataStream stream(&file);
    stream.setVersion(QDataStream::Qt_6_0);

    quint32 magicNumber = 0;
    quint16 fileVersion = 0;

    stream >> magicNumber >> fileVersion;

    if (stream.status() != QDataStream::Ok)
    {
        errorMessage = "Proje dosyasının başlığı okunamadı.";
        return false;
    }

    if (magicNumber != MagicNumber)
    {
        errorMessage = "Bu dosya geçerli bir VisionLab projesi değil.";
        return false;
    }

    if (fileVersion != FileVersion)
    {
        errorMessage = "Bu proje daha yeni bir VisionLab sürümüyle oluşturulmuş.";
        return false;
    }

    ProjectData loadedProject;

    stream >> loadedProject.algorithm
        >> loadedProject.gamma;

    if (stream.status() != QDataStream::Ok)
    {
        errorMessage = "Proje verileri eksik veya bozuk.";
        return false;
    }

    const QStringList validAlgorithms{
        "Orijinal",
        "Gamma Correction",
        "Histogram Equalization",
        "CLAHE",
    };

    if (!validAlgorithms.contains(loadedProject.algorithm))
    {
        errorMessage = "Projede bilinmeyen bir algoritma bulunuyor.";
        return false;
    }

    if (loadedProject.gamma < 0.1 || loadedProject.gamma > 3.0)
    {
        errorMessage = "Projede geçersiz bir gamma değeri bulunuyor.";
        return false;
    }

    project = loadedProject;
    return true;
}
}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    createInterface();
    createWorker();
}

MainWindow::~MainWindow()
{
    if (visionWorker != nullptr)
    {
        visionWorker->stopProcessing();
    }

    if (workerThread != nullptr && workerThread->isRunning())
    {
        workerThread->quit();
        workerThread->wait();
    }
}

void MainWindow::createInterface()
{
    setWindowTitle("VisionLab - Gerçek Zamanlı Görüntü İyileştirme");
    resize(1200, 760);
    setMinimumSize(960, 620);

    auto *centralWidget = new QWidget(this);
    auto *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setContentsMargins(24, 20, 24, 24);
    mainLayout->setSpacing(18);

    auto *titleLabel = new QLabel(
        "VisionLab  |  Gerçek Zamanlı Görüntü İyileştirme",
        centralWidget
    );
    titleLabel->setObjectName("titleLabel");
    mainLayout->addWidget(titleLabel);

    auto *contentLayout = new QHBoxLayout();
    contentLayout->setSpacing(18);

    auto *videoPanel = new QFrame(centralWidget);
    videoPanel->setObjectName("panel");
    auto *videoLayout = new QVBoxLayout(videoPanel);
    videoLayout->setContentsMargins(12, 12, 12, 12);

    videoWidget = new VideoGLWidget(videoPanel);
    videoWidget->setMinimumSize(640, 480);
    videoLayout->addWidget(videoWidget);

    contentLayout->addWidget(videoPanel, 1);

    auto *controlPanel = new QFrame(centralWidget);
    controlPanel->setObjectName("panel");
    controlPanel->setFixedWidth(310);

    auto *controlLayout = new QVBoxLayout(controlPanel);
    controlLayout->setContentsMargins(22, 22, 22, 22);
    controlLayout->setSpacing(12);

    auto *controlsTitle = new QLabel("Görüntü ayarları", controlPanel);
    controlsTitle->setObjectName("sectionTitle");
    controlLayout->addWidget(controlsTitle);

    auto *algorithmLabel = new QLabel("Algoritma", controlPanel);
    algorithmLabel->setObjectName("fieldLabel");
    controlLayout->addWidget(algorithmLabel);

    algorithmCombo = new QComboBox(controlPanel);
    algorithmCombo->addItems({
        "Orijinal",
        "Gamma Correction",
        "Histogram Equalization",
        "CLAHE",
    });
    controlLayout->addWidget(algorithmCombo);

    auto *gammaHeaderLayout = new QHBoxLayout();
    auto *gammaLabel = new QLabel("Gamma", controlPanel);
    gammaLabel->setObjectName("fieldLabel");

    gammaValueLabel = new QLabel("1.0", controlPanel);
    gammaValueLabel->setObjectName("accentValue");

    gammaHeaderLayout->addWidget(gammaLabel);
    gammaHeaderLayout->addStretch();
    gammaHeaderLayout->addWidget(gammaValueLabel);
    controlLayout->addLayout(gammaHeaderLayout);

    gammaSlider = new QSlider(Qt::Horizontal, controlPanel);
    gammaSlider->setRange(1, 30);
    gammaSlider->setValue(10);
    gammaSlider->setEnabled(false);
    controlLayout->addWidget(gammaSlider);

    auto *claheClipHeaderLayout = new QHBoxLayout();

    auto *claheClipLabel =
        new QLabel("CLAHE Clip Limit", controlPanel);
    claheClipLabel->setObjectName("fieldLabel");

    claheClipLimitValueLabel =
        new QLabel("4.0", controlPanel);
    claheClipLimitValueLabel->setObjectName("accentValue");

    claheClipHeaderLayout->addWidget(claheClipLabel);
    claheClipHeaderLayout->addStretch();
    claheClipHeaderLayout->addWidget(claheClipLimitValueLabel);

    controlLayout->addLayout(claheClipHeaderLayout);

    claheClipLimitSlider =
        new QSlider(Qt::Horizontal, controlPanel);

    // 10 → 1.0, 40 → 4.0, 100 → 10.0
    claheClipLimitSlider->setRange(10, 100);
    claheClipLimitSlider->setValue(40);
    claheClipLimitSlider->setEnabled(false);

    controlLayout->addWidget(claheClipLimitSlider);

    // CLAHE Grid Size

    auto *claheGridLabel =
        new QLabel("CLAHE Grid Size", controlPanel);
    claheGridLabel->setObjectName("fieldLabel");

    controlLayout->addWidget(claheGridLabel);

    claheGridSizeCombo = new QComboBox(controlPanel);

    claheGridSizeCombo->addItem("4 × 4", 4);
    claheGridSizeCombo->addItem("8 × 8", 8);
    claheGridSizeCombo->addItem("16 × 16", 16);

    claheGridSizeCombo->setCurrentIndex(1);
    claheGridSizeCombo->setEnabled(false);

    controlLayout->addWidget(claheGridSizeCombo);

    auto *separator = new QFrame(controlPanel);
    separator->setFrameShape(QFrame::HLine);
    separator->setObjectName("separator");
    controlLayout->addWidget(separator);

    auto *metricsTitle = new QLabel("Performans", controlPanel);
    metricsTitle->setObjectName("sectionTitle");
    controlLayout->addWidget(metricsTitle);

    auto *fpsLabel = new QLabel("FPS", controlPanel);
    fpsLabel->setObjectName("fieldLabel");
    controlLayout->addWidget(fpsLabel);

    fpsValueLabel = new QLabel("0.00", controlPanel);
    fpsValueLabel->setObjectName("metricValue");
    controlLayout->addWidget(fpsValueLabel);

    auto *timeLabel = new QLabel("İşlem süresi", controlPanel);
    timeLabel->setObjectName("fieldLabel");
    controlLayout->addWidget(timeLabel);

    processTimeValueLabel = new QLabel("0.00 ms", controlPanel);
    processTimeValueLabel->setObjectName("metricValue");
    controlLayout->addWidget(processTimeValueLabel);

    controlLayout->addStretch();

    statusLabel = new QLabel("● Kamera başlatılıyor", controlPanel);
    statusLabel->setObjectName("statusLabel");
    statusLabel->setWordWrap(true);
    controlLayout->addWidget(statusLabel);

    contentLayout->addWidget(controlPanel);
    mainLayout->addLayout(contentLayout, 1);

    setCentralWidget(centralWidget);

    auto *fileMenu = menuBar()->addMenu("Dosya");

    auto *saveProjectAction =
        fileMenu->addAction("Projeyi Kaydet...");

    auto *loadProjectAction =
        fileMenu->addAction("Projeyi Aç...");

    connect(
        saveProjectAction,
        &QAction::triggered,
        this,
        &MainWindow::saveProject
        );

    connect(
        loadProjectAction,
        &QAction::triggered,
        this,
        &MainWindow::loadProject
        );

    setStyleSheet(R"(
        QMainWindow, QWidget {
            background-color: #0f141a;
            color: #e7edf3;
            font-family: "Segoe UI";
            font-size: 14px;
        }

        #titleLabel {
            color: #f4f8fb;
            font-size: 22px;
            font-weight: 600;
        }

        #panel {
            background-color: #171e26;
            border: 1px solid #27313c;
            border-radius: 12px;
        }

        #sectionTitle {
            color: #f4f8fb;
            font-size: 17px;
            font-weight: 600;
            padding-bottom: 4px;
        }

        #fieldLabel {
            color: #9eacba;
            font-size: 13px;
        }

        #accentValue {
            color: #45d6a5;
            font-weight: 600;
        }

        #metricValue {
            color: #45d6a5;
            background-color: #10161c;
            border: 1px solid #27313c;
            border-radius: 8px;
            font-size: 22px;
            font-weight: 600;
            padding: 8px 12px;
        }

        #statusLabel {
            color: #45d6a5;
            background-color: #10231e;
            border: 1px solid #1d5747;
            border-radius: 8px;
            padding: 10px;
        }

        #separator {
            color: #27313c;
            margin-top: 8px;
            margin-bottom: 8px;
        }

        QComboBox {
            background-color: #10161c;
            border: 1px solid #34414e;
            border-radius: 7px;
            padding: 8px 10px;
        }

        QComboBox:hover {
            border-color: #45d6a5;
        }

        QComboBox QAbstractItemView {
            background-color: #171e26;
            selection-background-color: #236c58;
        }

        QSlider::groove:horizontal {
            height: 5px;
            background: #303b47;
            border-radius: 2px;
        }

        QSlider::handle:horizontal {
            width: 16px;
            margin: -6px 0;
            background: #45d6a5;
            border-radius: 8px;
        }

        QSlider::sub-page:horizontal {
            background: #45d6a5;
            border-radius: 2px;
        }

        QSlider:disabled {
            color: #59636d;
        }
    )");

    connect(
        algorithmCombo,
        &QComboBox::currentTextChanged,
        this,
        [this](const QString &algorithm)
        {
            const bool isGamma =
                algorithm == "Gamma Correction";

            const bool isClahe =
                algorithm == "CLAHE";

            gammaSlider->setEnabled(isGamma);

            claheClipLimitSlider->setEnabled(isClahe);
            claheGridSizeCombo->setEnabled(isClahe);
        }
    );

    connect(
        gammaSlider,
        &QSlider::valueChanged,
        this,
        [this](int value)
        {
            gammaValueLabel->setText(
                QString::number(value / 10.0, 'f', 1)
            );
        }
    );

    connect(
        claheClipLimitSlider,
        &QSlider::valueChanged,
        this,
        [this](int value)
        {
            claheClipLimitValueLabel->setText(
                QString::number(value / 10.0, 'f', 1)
            );
        }
    );
}

void MainWindow::createWorker()
{
    workerThread = new QThread(this);
    visionWorker = new VisionWorker();
    visionWorker->moveToThread(workerThread);

    connect(
        workerThread,
        &QThread::started,
        visionWorker,
        &VisionWorker::startProcessing
    );

    connect(
        workerThread,
        &QThread::finished,
        visionWorker,
        &QObject::deleteLater
    );

    connect(
        visionWorker,
        &VisionWorker::framesReady,
        videoWidget,
        &VideoGLWidget::updateFrame
    );

    connect(
        visionWorker,
        &VisionWorker::metricsUpdated,
        this,
        [this](double fps, double processTime)
        {
            fpsValueLabel->setText(QString::number(fps, 'f', 2));
            processTimeValueLabel->setText(
                QString("%1 ms").arg(processTime, 0, 'f', 2)
            );
        }
    );

    connect(
        visionWorker,
        &VisionWorker::statusChanged,
        this,
        [this](const QString &status, bool isError)
        {
            statusLabel->setText(QString("● %1").arg(status));

            statusLabel->setStyleSheet(
                isError
                    ? "color:#ff7b72; background:#2a1718;"
                      "border:1px solid #71383b; border-radius:8px;"
                      "padding:10px;"
                    : "color:#45d6a5; background:#10231e;"
                      "border:1px solid #1d5747; border-radius:8px;"
                      "padding:10px;"
            );
        }
    );

    connect(
        algorithmCombo,
        &QComboBox::currentTextChanged,
        visionWorker,
        &VisionWorker::setAlgorithm
    );

    connect(
        gammaSlider,
        &QSlider::valueChanged,
        visionWorker,
        [this](int value)
        {
            visionWorker->setGammaValue(value / 10.0);
        }
    );

    connect(
        algorithmCombo,
        &QComboBox::currentTextChanged,
        this,
        [this](const QString &algorithm)
        {
            const bool isGamma =
                algorithm == "Gamma Correction";

            const bool isClahe =
                algorithm == "CLAHE";

            gammaSlider->setEnabled(isGamma);

            claheClipLimitSlider->setEnabled(isClahe);
            claheGridSizeCombo->setEnabled(isClahe);
        }
    );

    connect(
        claheClipLimitSlider,
        &QSlider::valueChanged,
        visionWorker,
        [worker = visionWorker](int value)
        {
            worker->setClaheClipLimit(value / 10.0);
        }
    );

    connect(
        claheGridSizeCombo,
        QOverload<int>::of(&QComboBox::currentIndexChanged),
        visionWorker,
        [worker = visionWorker](int index)
        {
            constexpr int gridSizes[] = {4, 8, 16};

            if (index >= 0 && index < 3)
            {
                worker->setClaheGridSize(gridSizes[index]);
            }
        }
    );

    workerThread->start();
}

void MainWindow::saveProject()
{
    QString fileName = QFileDialog::getSaveFileName(
        this,
        "VisionLab Projesini Kaydet",
        QString(),
        "VisionLab Projesi (*.vlp)"
    );

    if (fileName.isEmpty())
        return;

    if (!fileName.endsWith(".vlp", Qt::CaseInsensitive))
    {
        fileName += ".vlp";
    }

    const ProjectData project{
        algorithmCombo->currentText(),
        gammaSlider->value() / 10.0
    };

    QString errorMessage;

    if (!saveProjectToFile(fileName, project, errorMessage))
    {
        QMessageBox::critical(
            this,
            "Kaydetme Hatası",
            errorMessage
        );
        return;
    }
    statusLabel->setText("● Proje başarıyla kaydedildi");
}

void MainWindow::loadProject()
{
    const QString fileName = QFileDialog::getOpenFileName(
        this,
        "VisionLab Projesini Aç",
        QString(),
        "VisionLab Projesi (*.vlp)"
        );

    if (fileName.isEmpty())
        return;

    ProjectData project;
    QString errorMessage;

    if (!loadProjectFromFile(fileName, project, errorMessage))
    {
        QMessageBox::critical(
            this,
            "Proje Açma Hatası",
            errorMessage
            );
        return;
    }

    algorithmCombo->setCurrentText(project.algorithm);
    gammaSlider->setValue(qRound(project.gamma * 10.0));

    statusLabel->setText("● Proje başarıyla yüklendi");
}

bool MainWindow::savePresetToXml(const QString &filePath)
{
    QSaveFile file(filePath);

    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QXmlStreamWriter writer(&file);
    writer.setAutoFormatting(true);

    writer.writeStartDocument();

    writer.writeStartElement("VisionLabPreset");
    writer.writeAttribute("version", "1.0");

    writer.writeStartElement("Algorithm");
    writer.writeAttribute("type", "CLAHE");

    writer.writeStartElement("Parameter");
    writer.writeAttribute("name", "clipLimit");
    writer.writeAttribute("value", "2.0");
    writer.writeEndElement();

    writer.writeStartElement("Parameter");
    writer.writeAttribute("name", "gridSize");
    writer.writeAttribute("value", "8");
    writer.writeEndElement();

    writer.writeEndElement(); // Algorithm
    writer.writeEndElement(); // VisionLabPreset

    writer.writeEndDocument();

    return file.commit();
}
