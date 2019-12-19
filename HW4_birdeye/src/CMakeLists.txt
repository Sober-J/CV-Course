cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)
project(transform)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp example_18-01.cpp example_18-01_from_disk.cpp example_19-01.cpp hw4.h)
add_executable(CB ${SOURCE_FILES})
target_link_libraries (CB ${OpenCV_LIBS})