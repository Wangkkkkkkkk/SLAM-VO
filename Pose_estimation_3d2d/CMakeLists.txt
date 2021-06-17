cmake_minimum_required(VERSION 3.0)
project(VO)

set( CMAKE_BUILD_TYPE "Release" )
set(CMAKE_CXX_FLAGS "-std=c++11")

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

link_libraries("/usr/local/lib/libg2o_ext_csparse.so")

find_package( OpenCV REQUIRED )
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
)

add_executable(PE pose_estimation_3d2d.cpp)
target_link_libraries(PE 
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)