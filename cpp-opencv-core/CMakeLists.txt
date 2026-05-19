cmake_minimum_required(VERSION 3.15)

project(VisionLabOpenCVCore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_executable(VisionLabCore
    src/main.cpp
        src/main.cpp
)

target_link_libraries(VisionLabCore ${OpenCV_LIBS})