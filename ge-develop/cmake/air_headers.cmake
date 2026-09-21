# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

add_library(air_headers INTERFACE)

# INSTALL_INTERFACE影响sdk包导出cmake，不要随意增加，增加时只能包含BUILD_INTERFACE
target_include_directories(air_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inc/external>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inc/framework>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../compiler/engines/nn_engine/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../compiler/engines/nn_engine/inc/common>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inc/graph_metadef>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inc/graph_metadef/graph>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inc/graph_metadef/external>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inc/graph_metadef/register>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../graph_metadef>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../graph_metadef/third_party/transformer/inc>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include/graph>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/include/external>
    $<BUILD_INTERFACE:${ASCEND_INSTALL_PATH}/pkg_inc>
    $<INSTALL_INTERFACE:include/air>
    $<INSTALL_INTERFACE:include/air/external>
    $<INSTALL_INTERFACE:include/air/framework>
    $<INSTALL_INTERFACE:include/air/nn_engine>
    $<INSTALL_INTERFACE:include/air/nn_engine/common>
)

if (NOT EXISTS "${ASCEND_INSTALL_PATH}/include")
    target_include_directories(air_headers INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../metadef/inc>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../metadef/inc/graph>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../metadef/inc/external>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../metadef/inc/external/base>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../metadef/inc/external/exe_graph>
            )
endif ()

target_include_directories(air_headers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../base/metadef/inc>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../base/metadef/pkg_inc>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inc/graph_metadef>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../base/metadef/pkg_inc/graph>
        )

get_target_property(INCLUDE_PATHS air_headers INTERFACE_INCLUDE_DIRECTORIES)
foreach(line IN LISTS INCLUDE_PATHS)
    message(STATUS "air_headers ${line}")
endforeach()