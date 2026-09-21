# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

protobuf_generate(hcom_graph_adaptor_proto  hcom_graph_adaptor_proto_c hcom_graph_adaptor_proto_h
                      ${CMAKE_CURRENT_SOURCE_DIR}/proto/task.proto TARGET
)
add_library(hcom_graph_adaptor  SHARED
                ${hcom_graph_adaptor_proto_h}
)
add_library(hcom_executor  SHARED
                ${hcom_graph_adaptor_proto_h}
)
add_library(hcom_opskernel_builder  SHARED
                ${hcom_graph_adaptor_proto_h}
)
add_library(hcom_gradtune_opskernel_builder  SHARED
                ${hcom_graph_adaptor_proto_h}
)
add_dependencies(hcom_graph_adaptor hcom_graph_adaptor_proto)
add_dependencies(hcom_executor hcom_graph_adaptor_proto)
add_dependencies(hcom_opskernel_builder hcom_graph_adaptor_proto)
add_dependencies(hcom_gradtune_opskernel_builder hcom_graph_adaptor_proto)

#hcom_graph_adaptor头文件和编译宏
target_compile_options(hcom_graph_adaptor  PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -fstack-protector-strong
    -std=c++14
    -Wno-error=comment
)
target_compile_definitions(hcom_graph_adaptor  PRIVATE
    google=ascend_private
    $<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:HCOM_DEBUG_MODE>
)
target_link_options(hcom_graph_adaptor PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)

#hcom_executor头文件和编译宏
target_compile_options(hcom_executor  PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -fstack-protector-strong
    -std=c++14
    -Wno-error=comment
)
target_compile_definitions(hcom_executor  PRIVATE
    google=ascend_private
    $<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:HCOM_DEBUG_MODE>
    HCOM_EXECUTOR
)
target_link_options(hcom_executor PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)

#hcom_opskernel_builder头文件和编译宏
target_compile_options(hcom_opskernel_builder  PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -fstack-protector-strong
    -std=c++14
    -Wno-error=comment
)
target_compile_definitions(hcom_opskernel_builder  PRIVATE
    google=ascend_private
)
target_link_options(hcom_opskernel_builder PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)


#hcom_gradtune_opskernel_builder头文件和编译宏
target_compile_options(hcom_gradtune_opskernel_builder  PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -fstack-protector-strong
    -std=c++14
    -Wno-error=comment
)
target_compile_definitions(hcom_gradtune_opskernel_builder  PRIVATE
    google=ascend_private
)
target_link_options(hcom_gradtune_opskernel_builder PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)

#头文件搜索路径
if(BUILD_OPEN_PROJECT)
    #hcom_graph_adaptor头文件
    target_include_directories(hcom_graph_adaptor PRIVATE
        ${CMAKE_BINARY_DIR}/proto/hcom_graph_adaptor_proto
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/common
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/hcom
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/auto_tuning
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/calculation_tool
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/include/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
        ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
        ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu
        ${ASCEND_INSTALL_PATH}/pkg_inc/dump
        ${ASCEND_INSTALL_PATH}/include/msprof
        ${ASCEND_INSTALL_PATH}/include/adump/external
        ${ASCEND_INSTALL_PATH}/include/slog/toolchain
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${METADEF_DIR}/inc
    )
 
    #hcom_executor头文件
    target_include_directories(hcom_executor PRIVATE
        ${CMAKE_BINARY_DIR}/proto/hcom_graph_adaptor_proto
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/common
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/hcom
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/auto_tuning
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/calculation_tool
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/include/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
        ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
        ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu
        ${ASCEND_INSTALL_PATH}/pkg_inc/dump
        ${ASCEND_INSTALL_PATH}/include/msprof
        ${ASCEND_INSTALL_PATH}/include/adump/external
        ${ASCEND_INSTALL_PATH}/include/slog/toolchain
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${METADEF_DIR}/inc
 
    )
 
    #hcom_opskernel_buildert头文件
    target_include_directories(hcom_opskernel_builder PRIVATE
        ${CMAKE_BINARY_DIR}/proto/hcom_graph_adaptor_proto
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/common
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/hcom
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/auto_tuning
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/calculation_tool
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/include/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
        ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
        ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu
        ${ASCEND_INSTALL_PATH}/pkg_inc/dump
        ${ASCEND_INSTALL_PATH}/include/msprof
        ${ASCEND_INSTALL_PATH}/include/adump/external
        ${ASCEND_INSTALL_PATH}/include/slog/toolchain
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${METADEF_DIR}/inc
    )
 
    #hcom_gradtune_opskernel_builder头文件
    target_include_directories(hcom_gradtune_opskernel_builder PRIVATE
        ${CMAKE_BINARY_DIR}/proto/hcom_graph_adaptor_proto
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/common
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/hcom
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/auto_tuning
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/calculation_tool
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/include/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
        ${ASCEND_INSTALL_PATH}/pkg_inc/profiling
        ${ASCEND_INSTALL_PATH}/pkg_inc/aicpu
        ${ASCEND_INSTALL_PATH}/pkg_inc/dump
        ${ASCEND_INSTALL_PATH}/include/msprof
        ${ASCEND_INSTALL_PATH}/include/adump/external
        ${ASCEND_INSTALL_PATH}/include/slog/toolchain
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/external
        ${METADEF_DIR}/inc
    )
 
    target_link_directories(hcom_graph_adaptor PRIVATE
        ${ASCEND_INSTALL_PATH}/lib
    )
 
    target_link_directories(hcom_executor PRIVATE
        ${ASCEND_INSTALL_PATH}/lib
        )
 
    target_link_directories(hcom_opskernel_builder PRIVATE
        ${ASCEND_INSTALL_PATH}/lib
        )
 
    target_link_directories(hcom_gradtune_opskernel_builder PRIVATE
        ${ASCEND_INSTALL_PATH}/lib
        )
 
    #hcom_graph_adaptor依赖的so,链接选项
    target_link_libraries(hcom_graph_adaptor PRIVATE
        -Wl,--no-as-needed
        c_sec
        runtime
        mmpa
        ascend_protobuf
        graph
        exe_graph
        lowering
        gert
        platform
        error_manager
        hcomm
        -Wl,--as-needed
        -lpthread
    )
 
    #hcom_executor依赖的so,链接选项
    target_link_libraries(hcom_executor PRIVATE
        -Wl,--no-as-needed
        c_sec
        runtime
        mmpa
        ascend_protobuf
        graph
        exe_graph
        lowering
        gert
        platform
        error_manager
        hcomm
        -Wl,--as-needed
        -lpthread
    )
 
    #hcom_opskernel_builder依赖的so,链接选项
    target_link_libraries(hcom_opskernel_builder PRIVATE
        -Wl,--no-as-needed
        c_sec
        runtime
        mmpa
        ascend_protobuf
        graph
        exe_graph
        lowering
        gert
        platform
        error_manager
        hcomm
        -Wl,--as-needed
        -lpthread
    )
 
    #hcom_gradtune_opskernel_builder依赖的so,链接选项
    target_link_libraries(hcom_gradtune_opskernel_builder PRIVATE
        -Wl,--no-as-needed
        c_sec
        runtime
        mmpa
        ascend_protobuf
        graph
        exe_graph
        lowering
        gert
        platform
        error_manager
        hcomm
        -Wl,--as-needed
        -lpthread
    )
else()
    #hcom_graph_adaptor头文件
    target_include_directories(hcom_graph_adaptor PRIVATE
        ${CMAKE_BINARY_DIR}/proto/hcom_graph_adaptor_proto
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/common
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/hcom
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/auto_tuning
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/calculation_tool
        ${CMAKE_CURRENT_SOURCE_DIR}/hcom_executor
        # ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/ace/comop/hccl/open_source/inc
        ${TOP_DIR}/ace/comop/inc
        ${TOP_DIR}/ace/comop/inc/hccl
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/ace/npuruntime/acl/inc/external
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/ace/comop/hccl/open_source/pkg_inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/graphengine/inc
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/air/inc
        ${TOP_DIR}/abl/adump/inc/adump
        ${TOP_DIR}/abl/slog/inc
        ${TOP_DIR}/abl/slog/inc/toolchain
        ${TOP_DIR}/abl/mmpa/inc
        ${TOP_DIR}/abl/msprof/inc
    )
 
    #hcom_executor头文件
    target_include_directories(hcom_executor PRIVATE
        ${CMAKE_BINARY_DIR}/proto/hcom_graph_adaptor_proto
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/common
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/hcom
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/auto_tuning
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/calculation_tool
        ${CMAKE_CURRENT_SOURCE_DIR}/hcom_executor
        # ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/ace/comop/hccl/open_source/inc
        ${TOP_DIR}/ace/comop/inc
        ${TOP_DIR}/ace/comop/inc/hccl
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/ace/npuruntime/acl/inc/external
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/ace/comop/hccl/open_source/pkg_inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/graphengine/inc
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/air/inc
        ${TOP_DIR}/abl/adump/inc/adump
        ${TOP_DIR}/abl/slog/inc
        ${TOP_DIR}/abl/slog/inc/toolchain
        ${TOP_DIR}/abl/mmpa/inc
        ${TOP_DIR}/abl/msprof/inc
    )
 
    #hcom_opskernel_buildert头文件
    target_include_directories(hcom_opskernel_builder PRIVATE
        ${CMAKE_BINARY_DIR}/proto/hcom_graph_adaptor_proto
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/common
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/hcom
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/auto_tuning
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/calculation_tool
        ${CMAKE_CURRENT_SOURCE_DIR}/hcom_executor
        # ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/ace/comop/hccl/open_source/inc
        ${TOP_DIR}/ace/comop/inc
        ${TOP_DIR}/ace/comop/inc/hccl
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/ace/npuruntime/acl/inc/external
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/ace/comop/hccl/open_source/pkg_inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/graphengine/inc
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/air/inc
        ${TOP_DIR}/abl/adump/inc/adump
        ${TOP_DIR}/abl/slog/inc
        ${TOP_DIR}/abl/slog/inc/toolchain
        ${TOP_DIR}/abl/mmpa/inc
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/metadef/inc/register
    )
 
    #hcom_gradtune_opskernel_builder头文件
    target_include_directories(hcom_gradtune_opskernel_builder PRIVATE
        ${CMAKE_BINARY_DIR}/proto/hcom_graph_adaptor_proto
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/common
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/hcom
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/auto_tuning
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/ge_plugin/calculation_tool
        ${CMAKE_CURRENT_SOURCE_DIR}/hcom_executor
        # ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/ace/comop/hccl/open_source/inc
        ${TOP_DIR}/ace/comop/inc
        ${TOP_DIR}/ace/comop/inc/hccl
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/ace/npuruntime/acl/inc/external
        ${TOP_DIR}/inc
        ${TOP_DIR}/ace/comop/hccl/open_source/pkg_inc
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/graphengine/inc
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/air/inc
        ${TOP_DIR}/abl/adump/inc/adump
        ${TOP_DIR}/abl/slog/inc
        ${TOP_DIR}/abl/slog/inc/toolchain
        ${TOP_DIR}/abl/mmpa/inc
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/metadef/inc/register
    )
 
    #hcom_graph_adaptor依赖的so,链接选项
    target_link_libraries(hcom_graph_adaptor PRIVATE
        $<BUILD_INTERFACE:intf_pub_cxx14>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:hccl_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:cce_headers>
        $<BUILD_INTERFACE:network_headers>
        -Wl,--no-as-needed
        c_sec
        slog
        runtime
        mmpa
        ascend_protobuf
        graph
        exe_graph
        lowering
        gert
        platform
        hccl
        error_manager
        -Wl,--as-needed
        -lpthread
        ofed_headers
    )
 
    #hcom_executor依赖的so,链接选项
    target_link_libraries(hcom_executor PRIVATE
        $<BUILD_INTERFACE:intf_pub_cxx14>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:hccl_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:cce_headers>
        $<BUILD_INTERFACE:network_headers>
        -Wl,--no-as-needed
        c_sec
        slog
        ascendcl
        runtime
        mmpa
        ascend_protobuf
        graph
        hccl
        error_manager
        -Wl,--as-needed
        -lpthread
        ofed_headers
        ascend_dump
    )
 
    #hcom_opskernel_builder依赖的so,链接选项
    target_link_libraries(hcom_opskernel_builder PRIVATE
        $<BUILD_INTERFACE:intf_pub_cxx14>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:hccl_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:cce_headers>
        $<BUILD_INTERFACE:network_headers>
        -Wl,--no-as-needed
        c_sec
        slog
        runtime
        mmpa
        ascend_protobuf
        graph
        hccl
        register
        error_manager
        -Wl,--as-needed
        -lpthread
        ofed_headers
    )
 
    #hcom_gradtune_opskernel_builder依赖的so,链接选项
    target_link_libraries(hcom_gradtune_opskernel_builder PRIVATE
        $<BUILD_INTERFACE:intf_pub_cxx14>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:hccl_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:cce_headers>
        $<BUILD_INTERFACE:network_headers>
        -Wl,--no-as-needed
        c_sec
        slog
        runtime
        ascend_protobuf
        graph
        hccl
        register
        mmpa
        error_manager
        -Wl,--as-needed
        -lpthread
        ofed_headers
    )
endif()

add_subdirectory(ge_plugin)
add_subdirectory(hcom_executor)

install(TARGETS hcom_graph_adaptor  OPTIONAL
    LIBRARY DESTINATION lib
)
install(TARGETS hcom_executor  OPTIONAL
    LIBRARY DESTINATION lib
)
install(TARGETS hcom_opskernel_builder  OPTIONAL
    LIBRARY DESTINATION lib
)
install(TARGETS hcom_gradtune_opskernel_builder  OPTIONAL
    LIBRARY DESTINATION lib
)