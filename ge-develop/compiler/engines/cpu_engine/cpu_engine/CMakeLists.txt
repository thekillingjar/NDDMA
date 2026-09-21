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

add_subdirectory(constant_folding_stub)

protobuf_generate(aicpu PROTO_SRCS PROTO_HDRS ${proto_src_files})

set(local_aicpu_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_engine/engine/aicpu_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_engine/kernel_info/aicpu_kernel_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_engine/kernel_info/aicpu_cust_kernel_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_engine/optimizer/aicpu_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_engine/kernel_builder/aicpu_kernel_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_engine/kernel_builder/aicpu_ascend_ops_kernel_builder.cpp
    ${PROTO_HDRS}
)

set(local_hostcpu_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/hostcpu_engine/engine/hostcpu_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hostcpu_engine/kernel_info/hostcpu_kernel_info.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hostcpu_engine/optimizer/hostcpu_optimizer.cpp
    ${PROTO_HDRS}
)

set(local_hostcpu_builder_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/hostcpu_engine/kernel_builder/hostcpu_kernel_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hostcpu_engine/kernel_builder/hostcpu_ops_kernel_builder.cpp
    ${PROTO_HDRS}
)

set(local_aicpu_const_folding_src_files
    ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_const_folding/folding.cc
)

set(local_lib_inc_path
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine
    ${AIR_CODE_DIR}/compiler/engines/cpu_engine/common
    ${CMAKE_BINARY_DIR}/proto/aicpu
    ${CMAKE_BINARY_DIR}/proto/aicpu/proto
)

add_library(aicpu_ascend_engine SHARED
    ${local_aicpu_src_files}
)

add_library(host_cpu_engine SHARED
    ${local_hostcpu_src_files}
)

add_library(host_cpu_opskernel_builder SHARED
    ${local_hostcpu_builder_src_files}
)

add_library(aicpu_const_folding SHARED
    ${local_aicpu_const_folding_src_files}
)

target_include_directories(aicpu_ascend_engine PRIVATE
    ${local_lib_inc_path}
)

target_include_directories(host_cpu_engine PRIVATE
    ${local_lib_inc_path}
)

target_include_directories(host_cpu_opskernel_builder PRIVATE
    ${local_lib_inc_path}
)

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    target_include_directories(aicpu_const_folding PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine/common
        ${CMAKE_BINARY_DIR}/proto/aicpu
        ${CMAKE_BINARY_DIR}/proto/aicpu/proto
        ${ASCEND_INSTALL_PATH}/include/aicpu_common/context/common
        ${ASCEND_INSTALL_PATH}/include/aicpu_common/context/cpu_proto
    )
else ()
    target_include_directories(aicpu_const_folding PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine
        ${AIR_CODE_DIR}/compiler/engines/cpu_engine/common
        ${CMAKE_BINARY_DIR}/proto/aicpu
        ${CMAKE_BINARY_DIR}/proto/aicpu/proto
        ${TOP_DIR}/inc/aicpu/cpu_kernels
        ${TOP_DIR}/inc/external/aicpu
        ${ASCEND_INSTALL_PATH}/include/aicpu_common/context/common
        ${ASCEND_INSTALL_PATH}/include/aicpu_common/context/cpu_proto
    )
endif ()

target_compile_options(aicpu_ascend_engine PRIVATE
    -Wno-deprecated-declarations
    -fno-common
    -fno-strict-aliasing
    -Werror
)

target_compile_options(host_cpu_engine PRIVATE
    -Wno-deprecated-declarations
    -fno-common
    -fno-strict-aliasing
    -Werror
)

target_compile_options(host_cpu_opskernel_builder PRIVATE
    -Wno-deprecated-declarations
    -fno-common
    -fno-strict-aliasing
    -Werror
)

target_compile_options(aicpu_const_folding PRIVATE
    -O2
    -ftrapv
    -fstack-protector-all
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -Werror
    -Wno-deprecated-declarations
)

target_compile_definitions(aicpu_ascend_engine PRIVATE
    AICPU_PLUGIN
    google=ascend_private
)

target_compile_definitions(host_cpu_engine PRIVATE
    AICPU_PLUGIN
    google=ascend_private
)

target_compile_definitions(host_cpu_opskernel_builder PRIVATE
    AICPU_PLUGIN
    google=ascend_private
)

target_compile_definitions(aicpu_const_folding PRIVATE
    _FORTIFY_SOURCE=2
    google=ascend_private
    LOG_CPP
)

target_link_options(aicpu_const_folding PRIVATE
    -Wl,-z,relro,-z,now
    -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
    $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
)

target_link_libraries(aicpu_ascend_engine PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    air_headers
    c_sec_headers
    mmpa_headers
    -Wl,--no-as-needed
    json
    slog_headers
    error_manager
    graph
#    graph_base
    platform
    ascend_protobuf
    aicpu_engine_common
    json
    -Wl,--as-needed
    -lrt
    -ldl
)

target_link_libraries(host_cpu_engine PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    air_headers
    c_sec_headers
    mmpa_headers
    -Wl,--no-as-needed
    json
    slog_headers
    error_manager
    graph
#    graph_base
    platform
    ascend_protobuf
    aicpu_engine_common
    json
    -Wl,--as-needed
    -lrt
    -ldl
)

target_link_libraries(host_cpu_opskernel_builder PRIVATE
    intf_pub
    msprof_headers
    runtime_headers
    cce_headers
    air_headers
    c_sec_headers
    mmpa_headers
    -Wl,--no-as-needed
    json
    slog_headers
    error_manager
    graph
#    graph_base
    platform
    ascend_protobuf
    aicpu_engine_common
    register
    json
    -Wl,--as-needed
    -lrt
    -ldl
)

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    target_link_libraries(aicpu_const_folding PRIVATE
        intf_pub
        aicpu_headers
        air_headers
        -Wl,--no-as-needed
        constant_folding_ops_stub
        slog_headers
        error_manager
        ascend_protobuf
        aicpu_engine_common
        -Wl,--as-needed
    )
else ()
    target_link_libraries(aicpu_const_folding PRIVATE
        intf_pub
        air_headers
        -Wl,--no-as-needed
        constant_folding_ops_stub
        slog_headers
        error_manager
        ascend_protobuf
        aicpu_engine_common
        -Wl,--as-needed
    )
endif ()

#####改名称用的，因为一个cmake不能2个相同的名字的链接so 可执行程序 等

install(TARGETS aicpu_ascend_engine OPTIONAL
    EXPORT aicpu_ascend_engine-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS host_cpu_engine OPTIONAL
    EXPORT host_cpu_engine-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS host_cpu_opskernel_builder OPTIONAL
    EXPORT host_cpu_opskernel_builder-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS aicpu_const_folding OPTIONAL
    EXPORT aicpu_const_folding-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)