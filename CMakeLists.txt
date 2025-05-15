cmake_minimum_required(VERSION 3.8)
project(lightstereo_cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) 

if (BUILD_TESTING)
  enable_testing()
endif()

add_subdirectory(easy_deploy_tool)
add_subdirectory(stereo)
