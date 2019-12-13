cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 11)
project(myTest)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES test.cpp)
add_executable(myTest ${SOURCE_FILES})
target_link_libraries(myTest ${OpenCV_LIBS})