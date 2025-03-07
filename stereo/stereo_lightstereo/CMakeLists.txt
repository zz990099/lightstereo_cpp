cmake_minimum_required(VERSION 3.8)
project(stereo_lightstereo)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)



include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

set(source_file src/lightstereo.cpp)

add_library(${PROJECT_NAME} SHARED ${source_file})   

target_link_libraries(${PROJECT_NAME} PUBLIC 
  glog::glog
  ${OpenCV_LIBS}
  deploy_core
)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)