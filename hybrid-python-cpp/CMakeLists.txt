cmake_minimum_required(VERSION 3.15)

project(VisionLabHybrid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# Python configuration
# ------------------------------------------------------------

if(WIN32)
    set(
        VISIONLAB_VENV_PYTHON
        "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe"
    )
else()
    set(
        VISIONLAB_VENV_PYTHON
        "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
    )
endif()

# Prefer the project's virtual environment when it exists.
if(EXISTS "${VISIONLAB_VENV_PYTHON}" AND
   NOT Python_EXECUTABLE)
    set(
        Python_EXECUTABLE
        "${VISIONLAB_VENV_PYTHON}"
        CACHE FILEPATH
        "Python interpreter"
    )
endif()

find_package(
    Python
    COMPONENTS
        Interpreter
        Development
    REQUIRED
)

# Find the pybind11 CMake package installed in the selected Python
# environment.
execute_process(
    COMMAND
        "${Python_EXECUTABLE}"
        -m
        pybind11
        --cmakedir

    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE

    ERROR_VARIABLE PYBIND11_ERROR
    RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "pybind11 bulunamadı. Önce şu komutu çalıştırın: "
        "python -m pip install -r requirements.txt\n"
        "${PYBIND11_ERROR}"
    )
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# ------------------------------------------------------------
# OpenCV configuration
# ------------------------------------------------------------

if(WIN32 AND NOT DEFINED OpenCV_DIR)
    set(
        OpenCV_DIR
        "C:/opencv/build"
        CACHE PATH
        "OpenCV installation directory"
    )
endif()

find_package(OpenCV REQUIRED)

# ------------------------------------------------------------
# VisionLab C++ core
# ------------------------------------------------------------

set(
    VISIONLAB_CORE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpp-opencv-core/src"
)

if(NOT EXISTS "${VISIONLAB_CORE_DIR}/imageprocess.cpp")
    message(
        FATAL_ERROR
        "VisionLab C++ core bulunamadı: ${VISIONLAB_CORE_DIR}"
    )
endif()

# ------------------------------------------------------------
# Python extension module
# ------------------------------------------------------------

pybind11_add_module(
    visionlab_cpp

    cpp/bindings.cpp
    ${VISIONLAB_CORE_DIR}/imageprocess.h
    ${VISIONLAB_CORE_DIR}/imageprocess.cpp
)

target_include_directories(
    visionlab_cpp
    PRIVATE

    ${VISIONLAB_CORE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    visionlab_cpp
    PRIVATE

    ${OpenCV_LIBS}
)

# ------------------------------------------------------------
# OpenCV runtime DLL files
# ------------------------------------------------------------

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

    if(OPENCV_RUNTIME_DLLS)
        add_custom_command(
            TARGET visionlab_cpp
            POST_BUILD

            COMMAND
                ${CMAKE_COMMAND}
                -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                "$<TARGET_FILE_DIR:visionlab_cpp>"

            COMMENT
                "Copying OpenCV runtime DLL files"
        )
    else()
        message(
            WARNING
            "OpenCV runtime DLL files were not found in: "
            "${OPENCV_RUNTIME_DIR}"
        )
    endif()
endif()