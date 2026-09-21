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
PROJECT(UDF)
if ("x${RESOURCE_TYPE}" STREQUAL "xAscend")
    message(STATUS "ascend compiler enter.")
    # if unsupport current resource type, please uncomment the next line.
    # message(FATAL_ERROR "unsupport compile Ascend target!")
elseif ("x${RESOURCE_TYPE}" STREQUAL "xAarch")
    message(STATUS "Aarch64 compiler enter.")
    # if unsupport current resource type, please uncomment the next line.
    # message(FATAL_ERROR "Unsupport compile aarch64 target!")
else ()
    message(STATUS "x86 compiler enter.")
    # if unsupport current resource type, please uncomment the next line.
    # message(FATAL_ERROR "Unsupport compile x86 target!")
endif ()

if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    message(STATUS "Read ASCEND_HOME_PATH from ENV: ${ASCEND_HOME_PATH}")
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

# set dynamic library output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
# set static library output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})

message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")

set(INC_DIR ${ASCEND_HOME_PATH}/include)
file(GLOB SRC_LIST "*.cpp")

# Specify cross compiler
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Set C++ compiler
set(CMAKE_CXX_COMPILER ${TOOLCHAIN})

#========================UDF so compile========================================
# check if SRC_LIST is exist
if ("x${SRC_LIST}" STREQUAL "x")
    message(UDF "=================no resource file")
    add_custom_target(${UDF_TARGET_LIB}
            COMMAND echo "no resource file to make lib${UDF_TARGET_LIB}.so")
endif ()

# message(UDF "=================SRC_LIST=${SRC_LIST}=========================")
add_library(${UDF_TARGET_LIB} SHARED ${SRC_LIST})

target_include_directories(${UDF_TARGET_LIB} PRIVATE ${INC_DIR})

target_compile_options(${UDF_TARGET_LIB} PRIVATE
        -O2
        -std=c++11
        -ftrapv
        -fstack-protector-all
        -fPIC
)

if ("x${RESOURCE_TYPE}" STREQUAL "xAscend")
    target_link_libraries(${UDF_TARGET_LIB} PRIVATE
            -Wl,--whole-archive
            ${ASCEND_HOME_PATH}/devlib/device/libflow_func.so
            -Wl,--no-whole-archive
    )
elseif ("x${RESOURCE_TYPE}" STREQUAL "xAarch")
    target_link_libraries(${UDF_TARGET_LIB} PRIVATE
            -Wl,--whole-archive
            ${ASCEND_HOME_PATH}/devlib/linux/aarch64/libflow_func.so
            -Wl,--no-whole-archive
    )
else ()
    target_link_libraries(${UDF_TARGET_LIB} PRIVATE
            -Wl,--whole-archive
            ${ASCEND_HOME_PATH}/devlib/linux/x86_64/libflow_func.so
            -Wl,--no-whole-archive
    )
endif ()