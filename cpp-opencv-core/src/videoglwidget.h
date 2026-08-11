#ifndef VIDEOGLWIDGET_H
#define VIDEOGLWIDGET_H

#include <QImage>
#include <QOpenGLWidget>

class VideoGLWidget final : public QOpenGLWidget
{
    Q_OBJECT

public:
    explicit VideoGLWidget(QWidget *parent = nullptr);

public slots:
    void updateFrame(const QImage &frame);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QImage currentFrame;
};

#endif // VIDEOGLWIDGET_H
