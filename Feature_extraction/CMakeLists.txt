cmake_minimum_required(VERSION 3.0)
project(VO)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(FE Feature_extraction.cpp)
target_link_libraries(PE ${OpenCV_LIBS})
