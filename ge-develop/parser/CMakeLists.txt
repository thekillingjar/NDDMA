# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(PARSER_PROTO_LIST
    "${METADEF_PROTO_DIR}/onnx/ge_onnx.proto"
    "${METADEF_PROTO_DIR}/om.proto"
    "${METADEF_PROTO_DIR}/caffe/caffe.proto"
    "${METADEF_PROTO_DIR}/tensorflow/graph_library.proto"
    "${METADEF_PROTO_DIR}/tensorflow/graph.proto"
    "${METADEF_PROTO_DIR}/tensorflow/node_def.proto"
    "${METADEF_PROTO_DIR}/tensorflow/function.proto"
    "${METADEF_PROTO_DIR}/tensorflow/versions.proto"
    "${METADEF_PROTO_DIR}/tensorflow/attr_value.proto"
    "${METADEF_PROTO_DIR}/tensorflow/op_def.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor.proto"
    "${METADEF_PROTO_DIR}/tensorflow/tensor_shape.proto"
    "${METADEF_PROTO_DIR}/tensorflow/types.proto"
    "${METADEF_PROTO_DIR}/tensorflow/resource_handle.proto"
)

protobuf_generate(parser_protos PARSER_PROTO_SRCS PARSER_PROTO_HDRS ${PARSER_PROTO_LIST} TARGET)

add_library(parser_headers INTERFACE)
target_include_directories(parser_headers INTERFACE
    $<BUILD_INTERFACE:${AIR_CODE_DIR}/inc/parser/>
    $<BUILD_INTERFACE:${AIR_CODE_DIR}/inc/parser/external>
    $<BUILD_INTERFACE:${AIR_CODE_DIR}/inc/parser/external/parser>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/parser>
    $<INSTALL_INTERFACE:include/parser/external>
    $<INSTALL_INTERFACE:include/parser/external/parser>
)

set(PARSER_GRAPH_LIBRARY_PROTO_SRC
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/graph_library.pb.cc"
)

set(PARSER_TENSORFLOW_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/graph.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/node_def.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/function.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/versions.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/attr_value.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/op_def.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/tensor.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/tensor_shape.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/types.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/tensorflow/resource_handle.pb.cc"
)

set(PARSER_ONNX_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/onnx/ge_onnx.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/om.pb.cc"
)

set(PARSER_CAFFE_PROTO_SRC
    "${CMAKE_BINARY_DIR}/proto/parser_protos/proto/caffe/caffe.pb.cc"
)

add_library(parser_graph_library_proto_obj OBJECT ${PARSER_GRAPH_LIBRARY_PROTO_SRC})
add_dependencies(parser_graph_library_proto_obj parser_protos)
target_compile_definitions(parser_graph_library_proto_obj PRIVATE
    google=ascend_private
)
target_link_libraries(parser_graph_library_proto_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(parser_graph_library_proto_obj PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)


add_library(parser_tensorflow_protos_obj OBJECT ${PARSER_TENSORFLOW_PROTO_SRCS})
add_dependencies(parser_tensorflow_protos_obj parser_protos)
target_compile_definitions(parser_tensorflow_protos_obj PRIVATE
    google=ascend_private
)
target_link_libraries(parser_tensorflow_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(parser_tensorflow_protos_obj PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)


add_library(parser_onnx_protos_obj OBJECT ${PARSER_ONNX_PROTO_SRCS})
add_dependencies(parser_onnx_protos_obj parser_protos)
target_compile_definitions(parser_onnx_protos_obj PRIVATE
    google=ascend_private
)
target_link_libraries(parser_onnx_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(parser_onnx_protos_obj PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)


add_library(parser_caffe_proto_obj OBJECT ${PARSER_CAFFE_PROTO_SRC})
add_dependencies(parser_caffe_proto_obj parser_protos)
target_compile_definitions(parser_caffe_proto_obj PRIVATE
    google=ascend_private
)
target_link_libraries(parser_caffe_proto_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(parser_caffe_proto_obj PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

############ lib_caffe_parser.so ############
add_library(_caffe_parser SHARED
    $<TARGET_OBJECTS:parser_caffe_proto_obj>
)

target_compile_definitions(_caffe_parser PRIVATE
    google=ascend_private
)

target_include_directories(_caffe_parser PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_options(_caffe_parser PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(_caffe_parser
    PRIVATE
        intf_pub
        -Wl,--no-as-needed
        ascend_protobuf
        -Wl,--as-needed
    PUBLIC
        parser_headers
)

add_subdirectory(parser)
add_subdirectory(parser/common)
add_subdirectory(parser/func_to_graph)
add_subdirectory(parser/onnx)
if (BUILD_METADEF AND BUILD_WITHOUT_AIR)
    add_subdirectory(metadef)
endif()

install(TARGETS _caffe_parser parser_common fmk_onnx_parser fmk_parser parser_headers
    EXPORT parser-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL COMPONENT opensdk
)
install(FILES   ${AIR_CODE_DIR}/inc/parser/external/parser/onnx_parser.h
                ${AIR_CODE_DIR}/inc/parser/external/parser/caffe_parser.h
                ${AIR_CODE_DIR}/inc/parser/external/parser/tensorflow_parser.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/parser/external/parser COMPONENT opensdk EXCLUDE_FROM_ALL
)
# 下列头文件发布是非法的，需要在后续整改中删掉
# --------------------start------------------------
install(FILES   ${PARSER_DIR}/parser/common/pre_checker.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/parser/parser/common COMPONENT opensdk EXCLUDE_FROM_ALL
)
install(FILES   ${PARSER_DIR}/parser/common/convert/pb2json.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/parser/parser/common/convert COMPONENT opensdk EXCLUDE_FROM_ALL
)
# ---------------------end-------------------------
if (PACKAGE STREQUAL "opensdk")
    install(EXPORT parser-targets DESTINATION ${INSTALL_CONFIG_DIR}
        FILE parser-targets.cmake COMPONENT opensdk EXCLUDE_FROM_ALL
    )
    set(PKG_NAME parser)
    configure_package_config_file(${TOP_DIR}/cmake/config/pkg_config_template.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/parser-config.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
        PATH_VARS INSTALL_BASE_DIR INSTALL_INCLUDE_DIR INSTALL_LIBRARY_DIR INSTALL_RUNTIME_DIR INSTALL_CONFIG_DIR
        INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    )
    unset(PKG_NAME)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parser-config.cmake
        DESTINATION ${INSTALL_CONFIG_DIR} COMPONENT opensdk EXCLUDE_FROM_ALL
    )
endif()
