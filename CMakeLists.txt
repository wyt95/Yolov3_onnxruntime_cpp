cmake_minimum_required(VERSION 3.15.3)

PROJECT(DEMO)
set(CMAKE_CXX_FLAGS "-std=c++14")
message(${PROJECT_SOURCE_DIR})

option(USE_CUDA "project built by cuda" OFF)

if (USE_CUDA)
    add_definitions(-DUSE_CUDA)
    message(status " USE_CUDA")
else()
    add_definitions(-DUSE_CPU)
    message(status " USE_CPU")
endif()

find_package(OpenCV REQUIRED)
message(status " ======opencv library status======")
message(status " version: ${opencv_version}")
message(status " libraries: ${opencv_libs}")
message(status " include path: ${opencv_include_dirs}")
message(status " ======opencv library status======")

FILE(GLOB SRC_LIST_CPP ${PROJECT_SOURCE_DIR}/src/*.cpp)

FILE(GLOB SRC_LIST "src/*.cpp")
message(STATUS "src_list: ${SRC_LIST}")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/session)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/providers/cpu)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/framework)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/providers/cuda)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

ADD_EXECUTABLE( demo ${SRC_LIST})

TARGET_LINK_LIBRARIES( demo ${PROJECT_SOURCE_DIR}/lib/libonnxruntime.so ${OpenCV_LIBS} -lstdc++fs -lboost_filesystem -lboost_system)
