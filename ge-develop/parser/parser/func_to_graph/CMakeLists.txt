# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(PROTO_LIST
    "${METADEF_PROTO_DIR}/tensorflow/graph.proto"
    "${METADEF_PROTO_DIR}/tensorflow/node_def.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto"
    "${METADEF_PROTO_DIR}/tensorflow/attr_value.proto"
    "${METADEF_PROTO_DIR}/tensorflow/function.proto"
    "${METADEF_PROTO_DIR}/tensorflow/op_def.proto"
    "${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor.proto"
    "${METADEF_PROTO_DIR}/tensorflow/types.proto"
    "${METADEF_PROTO_DIR}/tensorflow/versions.proto"
    "${METADEF_PROTO_DIR}/tensorflow/graph_library.proto"
)

protobuf_generate_py(ge PROTO_SRCS ${PROTO_LIST})

include_directories(${CMAKE_CURRENT_LIST_DIR})

############ func2graph/util ############
add_custom_target(util ALL
        DEPENDS ${PROTO_SRCS}
        COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/util
                && cp -r ${PROTO_SRCS} ${CMAKE_CURRENT_BINARY_DIR}/util
        )

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/util OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}/func2graph
)

