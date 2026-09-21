# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#########
include_directories(${ASCEND_INSTALL_PATH}/pkg_inc)
include_directories(${ASCEND_INSTALL_PATH}/pkg_inc/base)

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(aihac_autofusion SHARED
        ${SOURCES}
)

add_dependencies(aihac_autofusion metadef_graph_protos_obj ascir_ops_headers)

target_compile_definitions(aihac_autofusion PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)
if (ENABLE_OPEN_SRC)
        target_include_directories(aihac_autofusion PUBLIC
               ${CMAKE_CURRENT_SOURCE_DIR}
               ${TOP_DIR}/runtime/include
               ${TOP_DIR}/runtime/include/external
        )
else ()
        target_include_directories(aihac_autofusion PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}
               ${TOP_DIR}/air/inc
               ${TOP_DIR}/air/inc/graph_metadef
               ${TOP_DIR}/runtime/include
               ${TOP_DIR}/runtime/include/external
        )
endif ()

target_compile_options(aihac_autofusion PRIVATE
        -fvisibility=default
        -O2
        -Werror
        -fno-common
        -Wextra
        -Wfloat-equal
        -Wall
)

target_link_options(aihac_autofusion PRIVATE
        $<$<CONFIG:Release>:-s>
)

if (ENABLE_OPEN_SRC)
        target_link_libraries(aihac_autofusion PUBLIC
               graph
               unified_dlog
               atrace_share
               graph_base
               mmpa_headers
               metadef_headers
               ascir_ops_headers
               error_manager
               ascend_protobuf
               c_sec
               json
               aihac_symbolizer
               aihac_codegen
        )
else ()
        target_link_libraries(aihac_autofusion PRIVATE
               graph
               unified_dlog
               atrace_share
               graph_base
               mmpa_headers
               metadef_headers
               ascir_ops_headers
               atrace_headers
               error_manager
               ascend_protobuf
               c_sec
               json
               aihac_symbolizer
               aihac_codegen
        )
endif ()
