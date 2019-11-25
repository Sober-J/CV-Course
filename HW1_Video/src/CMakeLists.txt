cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)
project(PersonalVideo)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp transition.cpp transition.h)
add_executable(PersonalVideo ${SOURCE_FILES})
target_link_libraries (PersonalVideo ${OpenCV_LIBS})