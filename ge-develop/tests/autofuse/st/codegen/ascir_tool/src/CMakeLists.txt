# Copyright (c) 2025 Huawei Technologies Co., Ltd.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.14)

project(ACLNN_EXAMPLE)

add_compile_options(-std=c++11 -g)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
set(CMAKE_CXX_FLAGS_DEBUG  "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_executable(fused_graph_test
               fused_graph_test.cpp)

if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/latest")
endif()

set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
set(JSON_DIR "${ASCEND_PATH}/opensdk/opensdk/json/include")
message("include dir: ${INCLUDE_BASE_DIR}")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${INCLUDE_BASE_DIR}/aclnn
    ${JSON_DIR}
)

target_link_libraries(fused_graph_test PRIVATE
                      ${ASCEND_PATH}/lib64/libascendcl.so
                      ${ASCEND_PATH}/lib64/libmsprofiler.so
                      ${ASCEND_PATH}/lib64/libprofapi.so
                      ${ASCEND_PATH}/lib64/libmmpa.so)

install(TARGETS fused_graph_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
