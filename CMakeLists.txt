
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(t5_engine VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog_include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/json_include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/http)

set(SENTENCEPIECE_DIR /workspace/sentencepiece-0.2.0-Linux)
link_directories(${SENTENCEPIECE_DIR}/lib)
include_directories(${SENTENCEPIECE_DIR}/include)

find_package(CUDA)
set(TRT_INC_DIR /home/wzp/TensorRT-10.0.1.6/include)
set(TRT_LIB_DIR /home/wzp/TensorRT-10.0.1.6/lib)
set(TRT_OUT_DIR /home/wzp/TensorRT-10.0.1.6/bin)
# set(TRT_INC_DIR /usr/local/tensorrt/include)
# set(TRT_LIB_DIR /usr/local/tensorrt/lib)
# set(TRT_OUT_DIR /usr/local/tensorrt/bin)

include_directories(
    src
    ${CUDA_INCLUDE_DIRS}
    ${TRT_INC_DIR}
)

link_directories(
    ${TRT_OUT_DIR}
    ${TRT_LIB_DIR}
)


add_executable(t5_engine
    # src/perf.cpp
    main.cpp
    src/logging.cpp
    src/encoderdecoder_infer.cpp
    src/sentencepiece_tokenizer.cpp
)

target_link_libraries(t5_engine
    sentencepiece
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
)
