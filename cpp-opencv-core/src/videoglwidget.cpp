#include "videoglwidget.h"

#include <QPainter>

VideoGLWidget::VideoGLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
}

void VideoGLWidget::updateFrame(const QImage &frame)
{
    currentFrame = frame;
    update();
}

void VideoGLWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.fillRect(rect(), QColor("#080b0f"));
    painter.setRenderHint(QPainter::SmoothPixmapTransform, false);

    if (currentFrame.isNull())
    {
        painter.setPen(QColor("#758391"));
        painter.drawText(
            rect(),
            Qt::AlignCenter,
            "Kamera görüntüsü bekleniyor..."
            );
        return;
    }

    const QImage scaledFrame = currentFrame.scaled(
        size(),
        Qt::KeepAspectRatio,
        Qt::FastTransformation
        );

    const int x = (width() - scaledFrame.width()) / 2;
    const int y = (height() - scaledFrame.height()) / 2;
    painter.drawImage(x, y, scaledFrame);
}
