cmake_minimum_required(VERSION 3.15)

project(VisionLabOpenCVCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    OpenGLWidgets
)

qt_add_executable(VisionLabCore
    src/main.cpp
    src/videoglwidget.h
    src/videoglwidget.cpp
    src/visionworker.h
    src/visionworker.cpp
    mainwindow.h
    mainwindow.cpp
    src/imageprocess.h src/imageprocess.cpp
    resources.qrc
)

target_include_directories(VisionLabCore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(VisionLabCore PRIVATE
    ${OpenCV_LIBS}
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGLWidgets
)

if(WIN32)
    file(GLOB OPENCV_RUNTIME_DLLS
        "C:/opencv/build/x64/vc16/bin/*.dll"
    )

    add_custom_command(TARGET VisionLabCore POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:VisionLabCore>
        COMMENT "Copying OpenCV runtime DLL files"
    )
endif()
