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
    ${METADEF_PROTO_DIR}/task.proto
)

set(proto_dvpp_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/dvpp_attr.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/dvpp_node_def.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/dvpp_tensor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/proto/dvpp_tensor_shape.proto
)

protobuf_generate(dvpp PROTO_SRCS PROTO_HDRS ${proto_src_files})
protobuf_generate(dvpp PROTO_DVPP_SRCS PROTO_DVPP_HDRS ${proto_dvpp_src_files})

set(dvpp_engine_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter/dvpp_engine.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter/dvpp_graph_optimizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter/dvpp_ops_kernel_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter/dvpp_ops_kernel_info_store.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend910b/dvpp_builder_910b.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend910b/dvpp_info_store_910b.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend910b/dvpp_optimizer_910b.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/dvpp_builder.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/dvpp_chip_capability.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/dvpp_info_store.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/dvpp_ops_checker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/dvpp_ops_lib.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/dvpp_optimizer.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/util/json_file_operator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/util/util.cc
    ${PROTO_HDRS}
    ${PROTO_DVPP_SRCS}
)

add_library(dvpp_engine SHARED
    ${dvpp_engine_src_files}
)

set(local_lib_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/adapter
    ${CMAKE_CURRENT_SOURCE_DIR}/ascend910b
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/util
    ${CMAKE_CURRENT_SOURCE_DIR}/proto
    ${CMAKE_BINARY_DIR}/proto/dvpp
    ${CMAKE_BINARY_DIR}/proto/dvpp/proto
    ${AIR_CODE_DIR}/inc
    ${AIR_CODE_DIR}/inc/external
)

target_include_directories(dvpp_engine PRIVATE
    ${local_lib_inc_path}
)

target_compile_options(dvpp_engine PRIVATE
    -Wno-deprecated-declarations
    -fno-common
    -fno-strict-aliasing
    -Werror
    -Wextra
    -Wfloat-equal
)

target_compile_definitions(dvpp_engine PRIVATE
    google=ascend_private
)

target_link_options(dvpp_engine PRIVATE
    -rdynamic
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_libraries(dvpp_engine PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    air_headers
    -Wl,--no-as-needed
    unified_dlog
    error_manager
    graph
    register
    platform
    ascend_protobuf
    c_sec
    mmpa
    -Wl,--as-needed
    -lrt
    -ldl
    json
)

install(TARGETS dvpp_engine OPTIONAL
    EXPORT dvpp_engine-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
