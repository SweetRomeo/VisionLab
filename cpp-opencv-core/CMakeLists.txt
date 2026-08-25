cmake_minimum_required(VERSION 3.15)

project(VisionLabOpenCVCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(
    OpenCV_DIR
    "C:/opencv/build"
    CACHE PATH
    "OpenCV installation directory"
)
find_package(OpenCV REQUIRED)

find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    OpenGLWidgets
)
find_package(Threads REQUIRED)

qt_add_executable(VisionLabCore
    src/main.cpp
    src/videoglwidget.h
    src/videoglwidget.cpp
    src/visionworker.h
    src/visionworker.cpp
    mainwindow.h
    mainwindow.cpp
    src/imageprocess.h
    src/imageprocess.cpp
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

add_executable(
    VisionLabCppBenchmark

    ../benchmarks/runners/pure_cpp_benchmark.cpp
    src/imageprocess.h
    src/imageprocess.cpp
)

add_executable(
    VisionLabCppValidation

    ../benchmarks/validation/pure_cpp_output_generator.cpp
    src/imageprocess.h
    src/imageprocess.cpp
)

add_executable(
    VisionLabCppRealtime

    ../benchmarks/realtime/pure_cpp_realtime.cpp
    src/imageprocess.h
    src/imageprocess.cpp
)

target_include_directories(
    VisionLabCppBenchmark
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(
    VisionLabCppValidation
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(
    VisionLabCppRealtime
    PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    VisionLabCppValidation
    PRIVATE

    ${OpenCV_LIBS}
)

target_link_libraries(
    VisionLabCppBenchmark
    PRIVATE

    ${OpenCV_LIBS}
    Qt6::Core
)

target_link_libraries(
    VisionLabCppRealtime
    PRIVATE

    ${OpenCV_LIBS}
    Qt6::Core
    Threads::Threads
)

target_compile_definitions(
    VisionLabCppBenchmark
    PRIVATE

    VISIONLAB_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    VISIONLAB_BENCHMARK_RELEASE=$<IF:$<CONFIG:Release>,1,0>
)

target_compile_definitions(
    VisionLabCppRealtime
    PRIVATE

    VISIONLAB_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
    VISIONLAB_REALTIME_RELEASE=$<IF:$<CONFIG:Release>,1,0>
)

if(WIN32)
    set(
        OPENCV_RUNTIME_DIR
        "${OpenCV_DIR}/x64/vc16/bin"
        CACHE PATH
        "OpenCV runtime DLL directory"
    )

    file(
        GLOB OPENCV_RUNTIME_DLLS
        "${OPENCV_RUNTIME_DIR}/*.dll"
    )

    # Do not copy Debug OpenCV DLLs into Release output.

    if(OPENCV_RUNTIME_DLLS)
        add_custom_command(
            TARGET VisionLabCore
            POST_BUILD

            COMMAND
                ${CMAKE_COMMAND}
                -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:VisionLabCore>

            COMMENT
                "Copying OpenCV runtime DLL files"
        )

        add_custom_command(
            TARGET VisionLabCppBenchmark
            POST_BUILD

            COMMAND
                ${CMAKE_COMMAND}
                -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:VisionLabCppBenchmark>

            COMMENT
                "Copying OpenCV runtime DLL files for benchmark"
        )

        add_custom_command(
            TARGET VisionLabCppValidation
            POST_BUILD

            COMMAND
                ${CMAKE_COMMAND}
                -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:VisionLabCppValidation>

            COMMENT
                "Copying OpenCV runtime DLL files for validation"
        )

        add_custom_command(
            TARGET VisionLabCppRealtime
            POST_BUILD

            COMMAND
                ${CMAKE_COMMAND}
                -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:VisionLabCppRealtime>

            COMMENT
                "Copying OpenCV runtime DLL files for real-time evaluation"
        )
    else()
        message(
            WARNING
            "OpenCV runtime DLL files were not found."
        )
    endif()

    add_custom_command(
        TARGET VisionLabCppRealtime
        POST_BUILD

        COMMAND
            ${CMAKE_COMMAND}
            -E copy_if_different
            $<TARGET_FILE:Qt6::Core>
            $<TARGET_FILE_DIR:VisionLabCppRealtime>

        COMMENT
            "Copying Qt Core runtime DLL for real-time runner"
    )
endif()
