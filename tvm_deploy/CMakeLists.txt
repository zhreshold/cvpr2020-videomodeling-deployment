cmake_minimum_required(VERSION 3.6)
project(gluoncv_tvm_deploy_example)
add_subdirectory(decord)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")
#SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR})
SET(TVM_HOME ${CMAKE_CURRENT_SOURCE_DIR}/tvm)

INCLUDE_DIRECTORIES(${TVM_HOME}/include)
INCLUDE_DIRECTORIES(${TVM_HOME}/3rdparty/dmlc-core/include)
INCLUDE_DIRECTORIES(${TVM_HOME}/3rdparty/dlpack/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/decord/include)

add_executable(video_classification  tvm_runtime_pack.cc src/classification.cpp src/video.cpp)
target_link_libraries(video_classification decord)
