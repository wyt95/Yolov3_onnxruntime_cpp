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
message(STATUS "OpenCV library status:")
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

FILE(GLOB SRC_LIST_CPP ${PROJECT_SOURCE_DIR}/src/*.cpp)

#FILE(GLOB SRC_LIST "src/*.cpp")
message(STATUS "SRC_LIST_CPP: ${SRC_LIST_CPP}")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/session)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/providers/cpu)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/framework)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/providers/cuda)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

ADD_EXECUTABLE( demo ${SRC_LIST_CPP})

TARGET_LINK_LIBRARIES( demo ${PROJECT_SOURCE_DIR}/lib/libonnxruntime.so ${OpenCV_LIBS} -lstdc++fs -lboost_filesystem -lboost_system)
