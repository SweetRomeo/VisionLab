#include "videoglwidget.h"
#include <QPainter>

VideoGLWidget::VideoGLWidget(QOpenGLWidget *parent)
    : QOpenGLWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
}

void VideoGLWidget::updateFrame(const QImage& frame)
{
    currentFrame = frame;
    update();
}

void VideoGLWidget::paintEvent(QPaintEvent *event) {
    QPainter painter(this);

    // Yüksek FPS için pürüzsüzleştirme (smooth transformation) gibi CPU yoran
    // İşlemleri kapalı tutmak uç nokta donanımlarda faydalıdır.
    painter.setRenderHint(QPainter::SmoothPixmapTransform, false);

    if (!currentFrame.isNull()) {
        // Gelen kareyi widget'ın mevcut boyutuna, en-boy oranını koruyarak hızlıca ölçekle
        QImage scaledFrame = currentFrame.scaled(this->size(), Qt::KeepAspectRatio, Qt::FastTransformation);

        // Resmi tam ortaya hizalamak için X ve Y ofsetlerini hesapla
        int x = (this->width() - scaledFrame.width()) / 2;
        int y = (this->height() - scaledFrame.height()) / 2;

        painter.drawImage(x, y, scaledFrame);
    } else {
        // Kamera kapalıysa veya henüz veri gelmediyse ekranı siyah tut
        painter.fillRect(this->rect(), Qt::black);
    }
}


