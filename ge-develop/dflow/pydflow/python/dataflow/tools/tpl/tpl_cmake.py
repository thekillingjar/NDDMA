#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
from jinja2 import Template
import dataflow.dflow_wrapper as dwrapper

CONTENT = """
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.5)

project({{prj_name}})

# set ASCEND_HOME_PATH
if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    message(STATUS "Read ASCEND_HOME_PATH=${ASCEND_HOME_PATH} from environment variable")
    if (NOT EXISTS "${ASCEND_HOME_PATH}")
        message(FATAL_ERROR "ASCEND_HOME_PATH=${ASCEND_HOME_PATH} does not exist. \
Please check ASCEND_HOME_PATH environment variable.")
    endif ()
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

# set compiler
if ("x${RESOURCE_TYPE}" STREQUAL "xAscend")
    message(STATUS "ascend compiler enter")
    # if unsupport current resource type, please uncomment the next line.
    message(FATAL_ERROR "Unsupport compile Ascend target!")
elseif ("x${RESOURCE_TYPE}" STREQUAL "xAarch")
    message(STATUS "Aarch compiler enter")
    set(LIB_FLOW_FUNC ${ASCEND_HOME_PATH}/devlib/linux/aarch64/libflow_func.so)
    # if unsupport current resource type, please uncomment the next line.
    #message(FATAL_ERROR "Unsupport compile Aarch64 target!")
else ()
    message(STATUS "x86 compiler enter")
    set(LIB_FLOW_FUNC ${ASCEND_HOME_PATH}/devlib/linux/x86_64/libflow_func.so)
    # if unsupport current resource type, please uncomment the next line.
    # message(FATAL_ERROR "Unsupport compile X86 target!")
endif ()

set(CMAKE_CXX_COMPILER ${TOOLCHAIN})

find_package(Python3 {{running_python_version}} EXACT REQUIRED Interpreter COMPONENTS Development)

# set dynamic library output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
# set static library output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})

execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
        ${ASCEND_HOME_PATH}/include/flow_func
        ${pybind11_INCLUDE_DIRS}
)

#=========================UDF so compile============================
file(GLOB SRC_LIST "{{src_dir}}/*.cpp")

# check if SRC_LIST is exist
if ("x${SRC_LIST}" STREQUAL "x")
    message(UDF "=========no source file=============")
    add_custom_target(${UDF_TARGET_LIB}
            COMMAND echo "no source to make lib${UDF_TARGET_LIB}.so")
    return(0)
endif ()

#message(UDF "=========SRC_LIST: ${SRC_LIST}=============")
add_library(${UDF_TARGET_LIB} SHARED
        ${SRC_LIST}
)

target_compile_definitions(${UDF_TARGET_LIB} PRIVATE
        PYBIND11_BUILD_ABI="{{dflow_pybind11_build_abi}}"
)

target_compile_options(${UDF_TARGET_LIB} PRIVATE
        -O2
        -std=c++17
        -ftrapv
        -fstack-protector-all
        -fPIC
)

target_link_options(${UDF_TARGET_LIB} PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        -s
)

target_link_libraries(${UDF_TARGET_LIB} PRIVATE
        -Wl,--whole-archive
        ${LIB_FLOW_FUNC}
        ${Python3_LIBRARIES}
        pybind11::embed
        -Wl,--no-whole-archive
)
file(COPY {{py_src_dir}}/ DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

"""

TPL = Template(CONTENT)


def gen_func_cmake(prj_name, src_dir, py_src_dir):
    global TPL
    running_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    dflow_pybind11_build_abi = dwrapper.get_dflow_pybind11_build_abi()
    return TPL.render(
        prj_name=prj_name,
        src_dir=src_dir,
        py_src_dir=py_src_dir,
        running_python_version=running_python_version,
        dflow_pybind11_build_abi=dflow_pybind11_build_abi,
    )
