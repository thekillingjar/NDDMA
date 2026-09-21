# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
PROJECT(DataFlow)

if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    message(STATUS "Read ASCEND_HOME_PATH from environment variable")
else ()
    if (EXISTS "/usr/local/Ascend/cann")
        set(ASCEND_HOME_PATH "/usr/local/Ascend/cann")
        message(STATUS "ASCEND_HOME_PATH is not set, use default path: ${ASCEND_HOME_PATH}")
    elseif (EXISTS "/usr/local/Ascend/latest")
        set(ASCEND_HOME_PATH "/usr/local/Ascend/latest")
        message(STATUS "ASCEND_HOME_PATH is not set, use default path: ${ASCEND_HOME_PATH}")
    else ()
        message(FATAL_ERROR "ASCEND_HOME_PATH is not set, please export ASCEND_HOME_PATH based on actual installation path.")
    endif ()
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../output)

set(INC_DIR
        "${ASCEND_HOME_PATH}/include/"
        "${ASCEND_HOME_PATH}/include/external"
        "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc/"
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_executable(sample_base sample_base.cpp)
target_include_directories(sample_base PRIVATE ${INC_DIR})
target_compile_options(sample_base PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
        -fPIC
)
target_link_libraries(sample_base PRIVATE
        -Wl,--whole-archive
        ${ASCEND_HOME_PATH}/lib64/libgraph.so
        ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
        ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
        ${ASCEND_HOME_PATH}/lib64/libge_runner.so
        ${ASCEND_HOME_PATH}/lib64/libdflow_runner.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
        -Wl,--no-whole-archive
)

add_executable(sample_timebatch sample_timebatch.cpp)
target_include_directories(sample_timebatch PRIVATE ${INC_DIR})
target_compile_options(sample_timebatch PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
        -fPIC
)
target_link_libraries(sample_timebatch PRIVATE
        -Wl,--whole-archive
        ${ASCEND_HOME_PATH}/lib64/libgraph.so
        ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
        ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
        ${ASCEND_HOME_PATH}/lib64/libdflow_runner.so
        ${ASCEND_HOME_PATH}/lib64/libge_runner.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
        -Wl,--no-whole-archive
)

add_executable(sample_countbatch sample_countbatch.cpp)
target_include_directories(sample_countbatch PRIVATE ${INC_DIR})
target_compile_options(sample_countbatch PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
        -fPIC
)
target_link_libraries(sample_countbatch PRIVATE
        -Wl,--whole-archive
        ${ASCEND_HOME_PATH}/lib64/libgraph.so
        ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
        ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
        ${ASCEND_HOME_PATH}/lib64/libdflow_runner.so
        ${ASCEND_HOME_PATH}/lib64/libge_runner.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
        -Wl,--no-whole-archive
)

add_executable(sample_tensorflow sample_tensorflow.cpp)
target_include_directories(sample_tensorflow PRIVATE ${INC_DIR})
target_compile_options(sample_tensorflow PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
        -fPIC
)
target_link_libraries(sample_tensorflow PRIVATE
        -Wl,--whole-archive
        ${ASCEND_HOME_PATH}/lib64/libgraph.so
        ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
        ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
        ${ASCEND_HOME_PATH}/lib64/libdflow_runner.so
        ${ASCEND_HOME_PATH}/lib64/libge_runner.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
        -Wl,--no-whole-archive
)

add_executable(sample_multifunc sample_multifunc.cpp)
target_include_directories(sample_multifunc PRIVATE ${INC_DIR})
target_compile_options(sample_multifunc PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
        -fPIC
)
target_link_libraries(sample_multifunc PRIVATE
        -Wl,--whole-archive
        ${ASCEND_HOME_PATH}/lib64/libgraph.so
        ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
        ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
        ${ASCEND_HOME_PATH}/lib64/libdflow_runner.so
        ${ASCEND_HOME_PATH}/lib64/libge_runner.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
        -Wl,--no-whole-archive
)

add_executable(sample_exception sample_exception.cpp)
target_include_directories(sample_exception PRIVATE ${INC_DIR})
target_compile_options(sample_exception PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
        -fPIC
)
target_link_libraries(sample_exception PRIVATE
        -Wl,--whole-archive
        ${ASCEND_HOME_PATH}/lib64/libgraph.so
        ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
        ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
        ${ASCEND_HOME_PATH}/lib64/libdflow_runner.so
        ${ASCEND_HOME_PATH}/lib64/libge_runner.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
        -Wl,--no-whole-archive
)

add_executable(sample_perf sample_perf.cpp)
target_include_directories(sample_perf PRIVATE ${INC_DIR})
target_compile_options(sample_perf PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
        -fPIC
)
target_link_libraries(sample_perf PRIVATE
        -Wl,--whole-archive
        ${ASCEND_HOME_PATH}/lib64/libgraph.so
        ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
        ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
        ${ASCEND_HOME_PATH}/lib64/libdflow_runner.so
        ${ASCEND_HOME_PATH}/lib64/libge_runner.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
        ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
        -Wl,--no-whole-archive
)