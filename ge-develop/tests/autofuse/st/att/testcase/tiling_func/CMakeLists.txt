# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
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
project (TestProject)
add_compile_options(-std=c++17)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
include(CMakePrintHelpers)
cmake_print_variables(TOP_DIR)
cmake_print_variables(CMAKE_CURRENT_SOURCE_DIR)

message(STATUS "Using environment variable ASCEND_INSTALL_PATH: ${ASCEND_INSTALL_PATH}")
message(STATUS "Using environment variable ASCEND_INSTALL_LIB_PATH: ${ASCEND_INSTALL_LIB_PATH}")
message(STATUS "Using environment variable LD_LIBRARY_PATH: $ENV{LD_LIBRARY_PATH}")

set(CMAKE_VERBOSE_MAKEFILE ON)

set(COMMON_COMPILE_OPTIONS
        -fPIC
        -Werror
        -lc
        -fno-common
        -Wextra
        -Wfloat-equal
        -O2
        -DSTUB_LOG
        )
set(COMMON_LINK_OPTIONS
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,-rpath
        )

file(GLOB TILING_CODES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${TOP_DIR}/sample/depends/fake/*.cpp
    )

add_executable(tiling_func
        ${TILING_CODES}
        )

foreach(dir $ENV{LD_LIBRARY_PATH})
    file(TO_CMAKE_PATH "${dir}" dir)
    message(STATUS "Using environment variable enum: ${dir}")
    target_link_directories(tiling_func PRIVATE ${dir})
endforeach()

target_link_directories(tiling_func PRIVATE ${ASCEND_INSTALL_LIB_PATH})
target_compile_options(tiling_func PRIVATE ${COMMON_COMPILE_OPTIONS})
target_link_options(tiling_func PRIVATE ${COMMON_LINK_OPTIONS})
target_include_directories(tiling_func PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TOP_DIR}/sample/depends/fake
        ${TOP_DIR}/sample/gen_data_st/stub
        ${GE_METADEF_DIR}
        ${GE_METADEF_DIR}/inc
        ${GE_METADEF_DIR}/inc/external
        ${GE_METADEF_DIR}/inc/graph
        ${GE_METADEF_DIR}/inc/graph/ascendc_ir
        ${GE_METADEF_DIR}/inc/graph/utils
        ${GE_METADEF_DIR}/graph
        ${ASCEND_INSTALL_PATH}/include
        )

target_link_libraries(tiling_func PRIVATE
        graph_base
        graph
        metadef
        exe_graph
        lowering
        register
        c_sec
        )
