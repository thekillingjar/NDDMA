# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(proto_src_files
    ${METADEF_PROTO_DIR}/tensorflow/tensor.proto
    ${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto
    ${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto
    ${METADEF_PROTO_DIR}/tensorflow/types.proto
    ${METADEF_PROTO_DIR}/tensorflow/function.proto
    ${METADEF_PROTO_DIR}/tensorflow/op_def.proto
    ${METADEF_PROTO_DIR}/tensorflow/versions.proto
    ${METADEF_PROTO_DIR}/task.proto
    ${METADEF_PROTO_DIR}/tensorflow/graph.proto
    ${METADEF_PROTO_DIR}/tensorflow/node_def.proto
    ${METADEF_PROTO_DIR}/tensorflow/attr_value.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/fwk_adapter.proto
)
protobuf_generate(aicpu PROTO_SRCS PROTO_HDRS ${proto_src_files})

set (local_lib_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/tf_engine.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tf_kernel_info/tf_kernel_info.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tf_optimizer/tf_optimizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tf_optimizer/tensorflow_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tf_optimizer/tf_optimizer_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tf_optimizer/tf_function_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/util/tf_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ir2tf/ir2tf_parser_factory.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ir2tf/ir2tf_base_parser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/config/ir2tf_json_file.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tf_kernel_builder/tf_kernel_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tf_kernel_builder/tf_ops_kernel_builder.cc
    ${PROTO_HDRS}
)

set (local_builder_acl_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/tf_kernel_builder/tf_kernel_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tf_kernel_builder/tf_ops_kernel_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/util/tf_util.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ir2tf/ir2tf_parser_factory.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ir2tf/ir2tf_base_parser.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/config/ir2tf_json_file.cc
    ${PROTO_HDRS}
)

set(local_lib_inc_path
    ${ASCEND_INSTALL_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/common
    ${CMAKE_BINARY_DIR}/proto/aicpu
    ${CMAKE_BINARY_DIR}/proto/aicpu/proto
)

add_library(aicpu_tf_engine SHARED
    ${local_lib_src_files}
)
###aicpu_ms_engine
target_link_libraries(aicpu_tf_engine PRIVATE
    intf_pub
    msprof_headers
    cce_headers
    air_headers
    datagw_headers
    c_sec_headers
    mmpa_headers
    -Wl,--no-as-needed
    ascend_protobuf
    aicpu_engine_common
    json
    slog_headers
    graph
#    graph_base
    error_manager
    register
    platform
    runtime
    json
    -Wl,--as-needed
    -lrt
    -ldl
)

target_link_options(aicpu_tf_engine PRIVATE
    -Wl,-Bsymbolic
)

target_include_directories(aicpu_tf_engine PRIVATE
    ${local_lib_inc_path}
    ${OP_PROTO_DIR}
)

target_compile_definitions(aicpu_tf_engine PRIVATE
    AICPU_PLUGIN
    google=ascend_private
)

#------------atclib/aicpu_tf_engine

target_compile_options(aicpu_tf_engine PRIVATE
    -Wno-deprecated-declarations
    -fno-common
    -fno-strict-aliasing
    -Werror
)

install(TARGETS aicpu_tf_engine OPTIONAL
    EXPORT aicpu_tf_engine-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
)

set(json_src_path   ${CMAKE_CURRENT_SOURCE_DIR}/config/ir2tf/ir2tf_op_mapping_lib.json)

install(FILES ${json_src_path}
    DESTINATION ${INSTALL_LIBRARY_DIR}
    OPTIONAL
)
