cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)
project(myTrain)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES train.cpp)
add_executable(myTrain ${SOURCE_FILES})
target_link_libraries(myTrain ${OpenCV_LIBS})