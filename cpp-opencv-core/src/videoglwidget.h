#ifndef VIDEOGLWIDGET_H
#define VIDEOGLWIDGET_H
#include <QOpenGLWidget>
#include <QImage>

class VideoGLWidget: public QOpenGLWidget
{
    Q_OBJECT
    QImage currentFrame;
public:
    explicit VideoGLWidget(QOpenGLWidget *parent = nullptr);
public slots:
    void updateFrame(const QImage& frame);
protected:
    void paintEvent(QPaintEvent *event) override;
signals:
};

#endif // VIDEOGLWIDGET_H
