cmake_minimum_required(VERSION 3.16)
project(a3_lingbot_depth_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT DEFINED TRT_ROOT)
  if(DEFINED ENV{TRT_ROOT})
    set(TRT_ROOT "$ENV{TRT_ROOT}")
  else()
    set(TRT_ROOT "$ENV{TRT_ROOT}")
  endif()
endif()

set(TRT_INCLUDE_DIR "${TRT_ROOT}/usr/include/aarch64-linux-gnu")
set(CUDA_INCLUDE_DIR "${TRT_ROOT}/usr/local/cuda-12.8/targets/sbsa-linux/include")

find_library(NVINFER_LIB NAMES libnvinfer.so.10 nvinfer PATHS /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu NO_DEFAULT_PATH)
find_library(NVONNXPARSER_LIB NAMES libnvonnxparser.so.10 nvonnxparser PATHS /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu NO_DEFAULT_PATH)
find_library(CUDART_LIB NAMES libcudart.so.12 cudart PATHS /usr/local/cuda-12.8/targets/aarch64-linux/lib /usr/local/cuda/targets/aarch64-linux/lib NO_DEFAULT_PATH)

if(NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB OR NOT CUDART_LIB)
  message(FATAL_ERROR "TensorRT/CUDA runtime libraries not found")
endif()

include_directories(
  ${TRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(a3_lingbot_depth_trt_node src/a3_lingbot_depth_trt_node.cpp)
ament_target_dependencies(a3_lingbot_depth_trt_node rclcpp sensor_msgs)
target_link_libraries(a3_lingbot_depth_trt_node ${OpenCV_LIBS} ${NVINFER_LIB} ${CUDART_LIB})

add_executable(build_lingbot_trt_engine src/build_lingbot_trt_engine.cpp)
target_link_libraries(build_lingbot_trt_engine ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${CUDART_LIB})

install(TARGETS
  a3_lingbot_depth_trt_node
  build_lingbot_trt_engine
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
