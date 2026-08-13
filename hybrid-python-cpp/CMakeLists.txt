if(WIN32 AND NOT DEFINED OpenCV_DIR)
    set(
        OpenCV_DIR
        "C:/opencv/build"
        CACHE PATH
        "OpenCV installation directory"
    )
endif()

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
    COMPONENTS Interpreter Development
    REQUIRED
)

execute_process(
    COMMAND
        "${Python_EXECUTABLE}"
        -m
        pybind11
        --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
    message(
        FATAL_ERROR
        "pybind11 bulunamadi. Once requirements.txt dosyasini kur."
    )
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

if(WIN32 AND NOT DEFINED OpenCV_DIR)
    set(
        OpenCV_DIR
        "C:/opencv/build"
        CACHE PATH
        "OpenCV installation directory"
    )
endif()

>>>>>>> dfa3f1ce7a788a88e2613797bd69b8f78348e122
find_package(OpenCV REQUIRED)

set(
    VISIONLAB_CORE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpp-opencv-core/src"
)

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

if(WIN32)
    set(
        OPENCV_RUNTIME_DIR
        "${OpenCV_DIR}/x64/vc16/bin"
        CACHE PATH
        "OpenCV runtime DLL directory"
    )

    file(
        GLOB OPENCV_RUNTIME_DLLS
<<<<<<< HEAD
        "${OPENCV_RUNTIME_DIR}/*.dll"
=======
        "${OpenCV_DIR}/x64/vc16/bin/*.dll"
>>>>>>> dfa3f1ce7a788a88e2613797bd69b8f78348e122
    )

    if(OPENCV_RUNTIME_DLLS)
        add_custom_command(
            TARGET visionlab_cpp
            POST_BUILD

            COMMAND
                ${CMAKE_COMMAND}
                -E copy_if_different
                ${OPENCV_RUNTIME_DLLS}
                $<TARGET_FILE_DIR:visionlab_cpp>

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