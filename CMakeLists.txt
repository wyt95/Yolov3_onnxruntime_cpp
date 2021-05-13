cmake_minimum_required(VERSION 3.15.3)

PROJECT(DEMO)
set(CMAKE_CXX_FLAGS "-std=c++14")
message(${PROJECT_SOURCE_DIR})

find_package(OpenCV REQUIRED)
message(status " opencv library status:")
message(status " version: ${opencv_version}")
message(status " libraries: ${opencv_libs}")
message(status " include path: ${opencv_include_dirs}")

find_package(NumCpp 2.1.0 REQUIRED)
message(STATUS "include path: ${NumCpp_INCLUDE_DIRS}")

FILE(GLOB SRC_LIST_CPP ${PROJECT_SOURCE_DIR}/src/*.cpp)

#NumCpp
INCLUDE_DIRECTORIES(${NumCpp_INCLUDE_DIRS})

file(GLOB SRC_LIST "src/*.cpp")
message(STATUS "src_list: ${SRC_LIST}")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/session)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/onnxruntime/core/providers/cpu)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

ADD_EXECUTABLE( demo ${SRC_LIST})

TARGET_LINK_LIBRARIES( demo onnxruntime ${OpenCV_LIBS} -lstdc++fs)
