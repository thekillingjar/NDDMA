# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(DATA_FLOW_PROTO_LIST
        "${METADEF_PROTO_DIR}/ge_ir.proto"
        "dflow.proto"
)
protobuf_generate(data_flow_protos DATA_FLOW_PROTO_SRCS DATA_FLOW_PROTO_HDRS ${DATA_FLOW_PROTO_LIST} "--proto_path=${METADEF_PROTO_DIR}" TARGET)
set(FLOW_GRAPH_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/data_flow_protos/proto/dflow.pb.cc"
)
add_library(flow_graph_protos_obj OBJECT ${FLOW_GRAPH_PROTO_SRCS})
add_dependencies(flow_graph_protos_obj data_flow_protos)
target_compile_definitions(flow_graph_protos_obj PRIVATE
        google=ascend_private
)
target_link_libraries(flow_graph_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(flow_graph_protos_obj PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)