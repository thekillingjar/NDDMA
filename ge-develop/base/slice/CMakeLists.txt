# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(AIR_CODE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

set(SRC_LIST
    "data_slice_adapter.cc"
    "data_slice_helper.cc"
    "data_slice_factory.cc"
    "data_slice_default_impl.cc"
    "data_slice_elementwise_impl.cc")

#if (NOT ENABLE_D AND NOT ENABLE_ACL)
############ libslice.so ############
add_library(slice SHARED ${SRC_LIST})

target_compile_definitions(slice PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    FUNC_VISIBILITY
)

target_compile_options(slice PRIVATE
    -fvisibility=default
    -O2
    -Werror
    -Wextra
    -Wfloat-equal
    -fno-common
)

target_include_directories(slice PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/graph_metadef
    ${AIR_CODE_DIR}/inc/external
    ${AIR_CODE_DIR}/base
    ${CMAKE_BINARY_DIR}
)

target_link_options(slice PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(slice
    PRIVATE
        intf_pub
        -Wl,--no-as-needed
        graph
        graph_base
        ascend_protobuf
        c_sec
        error_manager
        unified_dlog
        -Wl,--as-needed
        json
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        air_headers
)
