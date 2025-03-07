cmake_minimum_required(VERSION 3.8)
project(simple_tests)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

set(easy_deploy_packages
  deploy_core
  image_processing_utils  
)

set(algorithm_packages
  stereo_lightstereo
)

if(ENABLE_TENSORRT)
  list(APPEND platform_core_packages trt_core)
endif()

if(ENABLE_RKNN)
  list(APPEND platform_core_packages rknn_core)
endif()

if(ENABLE_ORT)
  list(APPEND platform_core_packages ort_core)
endif()



find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(OpenCV REQUIRED)

set(source_file 
  # src/gtest_main.cpp
  src/fs_util.cpp
)

if(ENABLE_TENSORRT)
  list(APPEND source_file src/test_jetson_devkit.cpp)
endif()

if(ENABLE_RKNN)
  list(APPEND source_file src/test_rk_devkit.cpp)
endif()

if(ENABLE_ORT)
  list(APPEND source_file src/test_cpu_devkit.cpp)
endif()

enable_testing()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} ${source_file})   

target_link_libraries(${PROJECT_NAME} PUBLIC 
  GTest::gtest_main
  glog::glog
  ${OpenCV_LIBS}
  ${algorithm_packages}
  ${easy_deploy_packages}
  ${platform_core_packages}
)



target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)