cmake_minimum_required( VERSION 2.8 )
project( sfm )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(OpenCV_DIR "/usr/local/opencv-3.1.0/share/OpenCV")
include_directories("/usr/local/opencv-3.1.0/include")

# 添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# 寻找Ceres库并添加它的头文件
find_package(Ceres REQUIRED COMPONENTS SuiteSparse EigenSparse)
include_directories( ${CERES_INCLUDE_DIRS} )

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen 
include_directories( "/usr/include/eigen3/" )

# pcl
find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#add_executable( sfm main.cpp )
#target_link_libraries( sfm ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable( sfm main.cpp )
target_link_libraries( sfm ${OpenCV_LIBS} ${PCL_LIBRARIES} )


