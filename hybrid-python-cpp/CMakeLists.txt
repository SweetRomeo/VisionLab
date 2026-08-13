if(WIN32 AND NOT DEFINED OpenCV_DIR)
    set(
        OpenCV_DIR
        "C:/opencv/build"
        CACHE PATH
        "OpenCV installation directory"
    )
endif()

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