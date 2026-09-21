# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.7)
project(llm_engine)
set(INC_DIR
        "${AIR_CODE_DIR}/api/session/"
        "${AIR_CODE_DIR}/inc/external"
        "${METADEF_DIR}/inc/external"
        "${CMAKE_CURRENT_LIST_DIR}/"
        "${CMAKE_CURRENT_LIST_DIR}/common"
        "${CMAKE_BINARY_DIR}/proto/graphengine_protos/"
        )

if(NOT MDC_COMPILE_RUNTIME)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
else()
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI})
endif()

file(GLOB LLM_SRC_LIST
        "${CMAKE_CURRENT_LIST_DIR}/api/*.cc"
        "${CMAKE_CURRENT_LIST_DIR}/common/*.cc"
        "${CMAKE_CURRENT_LIST_DIR}/*.cc"
        )

message("llm_engine utils: ${CMAKE_CURRENT_LIST_DIR}")
message("llm_engine CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message("llm_engine AIR_CODE_DIR: ${AIR_CODE_DIR}")

add_library(llm_engine SHARED ${LLM_SRC_LIST})

target_include_directories(llm_engine PRIVATE ${INC_DIR})

target_compile_options(llm_engine PRIVATE
        -ftrapv
        -O2
        -fno-common
        -Wfloat-equal
        -Wall
        -Werror
        -Wextra)

target_link_libraries(llm_engine PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:hccl_headers>
        PUBLIC c_sec
        ascendcl_headers
        acl_rt
        alog
        graph
        graph_base
        gert
        ge_compiler
        ge_runner
        flow_graph
        json
        runtime_headers
        runtime
        msprof_headers
        ascend_protobuf
        -O2 -fPIC -Wextra -Wfloat-equal
        -Wl,--no-as-needed
        unified_dlog
        )

set(INSTALL_LIBRARY_DIR lib)
install(TARGETS llm_engine OPTIONAL
        EXPORT llm_engine-targets
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR})

