cmake_minimum_required(VERSION 3.15)

project(VisionLabOpenCVCore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt'nin sinyal-slot ve arayüz dosyalarını otomatik derlemesi için ŞART
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# OpenCV Bağlantısı
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# HATA BURADAYDI: Sadece Core değil, ihtiyacımız olan tüm modülleri ekledik
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGLWidgets)

# main.cpp tekrarı silindi
add_executable(VisionLabCore
    src/main.cpp
    src/videoglwidget.h
    src/videoglwidget.cpp
    mainwindow.h mainwindow.cpp
    src/visionworker.h src/visionworker.cpp
)

# Kütüphaneleri Bağlama
target_link_libraries(VisionLabCore PRIVATE ${OpenCV_LIBS})

target_link_libraries(VisionLabCore PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGLWidgets
)
