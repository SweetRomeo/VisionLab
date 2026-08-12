#include "mainwindow.h"

#include <QApplication>
#include <QIcon>

int main(int argc, char *argv[])
{
    QApplication application(argc, argv);

    application.setWindowIcon(
        QIcon(":/assets/visionlab-logo.png")
        );

    MainWindow window;
    window.show();

    return application.exec();
}
