#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(t5_engine VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
## debug版本
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
find_package(CUDA)

set(TRT_INC_DIR /root/autodl-tmp/TensorRT-10.0.1.6/include)
set(TRT_LIB_DIR /root/autodl-tmp/TensorRT-10.0.1.6/lib)
set(TRT_OUT_DIR /root/autodl-tmp/TensorRT-10.0.1.6/bin)

include_directories(
    src
    ${CUDA_INCLUDE_DIRS}
    ${TRT_INC_DIR}
)

link_directories(
    ${TRT_OUT_DIR}
    ${TRT_LIB_DIR}
)

file(GLOB FRONT_SOURCES
    # logging.h
    # logging.cpp
    *.h
    *.cpp
)

add_executable(t5_engine
    # src/perf.cpp
    main.cpp
    src/logging.cpp
    src/encoderdecoder_infer.cpp
)

target_link_libraries(t5_engine
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
)
