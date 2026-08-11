#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class QLabel;
class QComboBox;
class QSlider;
class QThread;
class VideoGLWidget;
class VisionWorker;

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private:
    void createInterface();
    void createWorker();
    void saveProject();
    void loadProject();
    bool savePresetToXml(const QString& filepath);

    VideoGLWidget *videoWidget = nullptr;
    QComboBox *algorithmCombo = nullptr;
    QSlider *gammaSlider = nullptr;
    QLabel *gammaValueLabel = nullptr;

    QSlider *claheClipLimitSlider = nullptr;
    QLabel *claheClipLimitValueLabel = nullptr;
    QComboBox *claheGridSizeCombo = nullptr;

    QLabel *fpsValueLabel = nullptr;
    QLabel *processTimeValueLabel = nullptr;
    QLabel *statusLabel = nullptr;

    QThread *workerThread = nullptr;
    VisionWorker *visionWorker = nullptr;
};

#endif // MAINWINDOW_H
