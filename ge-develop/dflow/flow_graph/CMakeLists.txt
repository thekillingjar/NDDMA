# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if (NOT TARGET flow_graph)
    set(DATA_FLOW_GRAPH_SOURCE_LIST
            "flow_attr_util.cc"
            "flow_graph.cc"
            "process_point.cc"
            "data_flow_attr_define.cc"
            "inner_pp.cc"
            "model_pp.cc"
    )
    ######### libflow_graph.so #############
    add_library(flow_graph SHARED
            ${DATA_FLOW_GRAPH_SOURCE_LIST}
            $<TARGET_OBJECTS:flow_graph_protos_obj>
    )

    target_compile_options(flow_graph PRIVATE
            $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
            $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal -Wno-array-bounds>
            $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
            $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

    target_compile_definitions(flow_graph PRIVATE
            google=ascend_private
            $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    )

    target_include_directories(flow_graph PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}
            ${AIR_CODE_DIR}/inc/external
            ${AIR_CODE_DIR}/inc/graph_metadef
            ${AIR_CODE_DIR}/dflow/inc
            ${AIR_CODE_DIR}/dflow/inc/data_flow
            ${CMAKE_BINARY_DIR}
            ${CMAKE_BINARY_DIR}/proto/data_flow_protos
            ${METADEF_DIR}/pkg_inc
    )

    target_link_options(flow_graph PRIVATE
            -Wl,-Bsymbolic
    )

    target_link_libraries(flow_graph
            PRIVATE
            intf_pub
            static_mmpa
            -Wl,--no-as-needed
            graph_base
            graph
            c_sec
            unified_dlog
            json
            -Wl,--as-needed
            $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
            -ldl
            PUBLIC
            metadef_headers
    )

    target_link_libraries(flow_graph PRIVATE ascend_protobuf error_manager)
    target_compile_options(flow_graph PRIVATE -O2)

    install(TARGETS flow_graph
            EXPORT dflow-targets
            LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
            ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
            RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL COMPONENT opensdk
    )

    if (PACKAGE STREQUAL "opensdk")
        install(EXPORT dflow-targets DESTINATION ${INSTALL_CONFIG_DIR}
                FILE dflow-targets.cmake COMPONENT opensdk EXCLUDE_FROM_ALL
        )
        set(PKG_NAME dflow)
        configure_package_config_file(${TOP_DIR}/cmake/config/pkg_config_template.cmake.in
                ${CMAKE_CURRENT_BINARY_DIR}/dflow-config.cmake
                INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
                PATH_VARS INSTALL_BASE_DIR INSTALL_INCLUDE_DIR INSTALL_LIBRARY_DIR INSTALL_RUNTIME_DIR INSTALL_CONFIG_DIR
                INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
        )
        unset(PKG_NAME)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dflow-config.cmake
                DESTINATION ${INSTALL_CONFIG_DIR} COMPONENT opensdk EXCLUDE_FROM_ALL
        )
    endif()
endif ()