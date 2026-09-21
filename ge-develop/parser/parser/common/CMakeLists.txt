# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(SRC_LIST
    "parser_factory.cc"
    "data_op_parser.cc"
    "op_parser_factory.cc"
    "pre_checker.cc"
    "op_registration_tbe.cc"
    "parser_api.cc"
    "parser_inner_ctx.cc"
    "proto_file_parser.cc"
    "acl_graph_parser_util.cc"
    "tbe_plugin_loader.cc"
    "model_saver.cc"
    "../tensorflow/tensorflow_custom_parser_adapter.cc"
    "../tensorflow/tensorflow_fusion_custom_parser_adapter.cc"
    "../tensorflow/tensorflow_fusion_op_parser.cc"
    "../tensorflow/tensorflow_util.cc"
    "convert/pb2json.cc"
    "convert/message2operator.cc"
    "op_def/ir_pb_converter.cc"
    "op_def/operator.cc"
    "op_map.cc"
    "parser_types.cc"
    "pass_manager.cc"
    "parser_fp16_t.cc"
    "thread_pool.cc"
    "parser_utils.cc"
    "auto_mapping_subgraph_io_index_func.cc"
    "prototype_pass_manager.cc"
)

############ libparser_common.so ############
add_library(parser_common SHARED ${SRC_LIST})

add_dependencies(parser_common
    parser_protos
)

target_compile_options(parser_common PRIVATE
    -Werror
    -fno-common
    -Wextra
    -Wfloat-equal
)

target_compile_definitions(parser_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)
message(${AIR_CODE_DIR}/parser/parser)
target_include_directories(parser_common PRIVATE
    ${AIR_CODE_DIR}/parser
    ${AIR_CODE_DIR}/parser/parser
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/parser_protos
    ${ASCEND_INSTALL_PATH}/include/graph
)

target_link_options(parser_common PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(parser_common
    PRIVATE
        intf_pub
        air_headers
        static_mmpa
        -Wl,--no-as-needed
        graph
        ascend_protobuf
        register
        c_sec
        unified_dlog
        error_manager
        -Wl,--as-needed
        json
        -lrt
        -ldl
    PUBLIC
        parser_headers
)

