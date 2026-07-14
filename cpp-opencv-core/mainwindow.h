#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QObject>
#include <QThread>

class MainWindow: public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
signals:

private:
    Ui::MainWindow *ui;
    QThread *workerThread;
    VisionWorker *visionWorker;
};

#endif // MAINWINDOW_H
