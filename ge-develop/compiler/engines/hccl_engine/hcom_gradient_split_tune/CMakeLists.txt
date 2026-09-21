# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_library(hcom_gradient_split_tune  SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gradient_split_tune.cc
)

target_compile_options(hcom_gradient_split_tune  PRIVATE
    -Werror
	-Wall
	-fno-common
    -fstack-protector-strong
	-fno-strict-aliasing
	-pipe
)
target_link_options(hcom_gradient_split_tune PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
)

#头文件搜索路径
if(BUILD_OPEN_PROJECT)
    target_include_directories(hcom_gradient_split_tune  PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${ASCEND_INSTALL_PATH}/include
        ${ASCEND_INSTALL_PATH}/include/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/hccl
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
        ${ASCEND_INSTALL_PATH}/include/msprof
        ${ASCEND_INSTALL_PATH}/include/slog/toolchain
        ${ASCEND_3RD_LIB_PATH}/json/include
        ${AIR_CODE_DIR}/inc
        ${AIR_CODE_DIR}/inc/graph_metadef
        ${METADEF_DIR}/inc
    )
 
    target_link_directories(hcom_gradient_split_tune PRIVATE
        ${ASCEND_INSTALL_PATH}/lib
    )
 
    target_link_libraries(hcom_gradient_split_tune   PRIVATE
        -Wl,--no-as-needed
        mmpa
        platform
        c_sec
        -Wl,--as-needed
    )
else()
    target_include_directories(hcom_gradient_split_tune  PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../inc
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/toolchain
        ${TOP_DIR}/libc_sec/include
        ${TOP_DIR}/metadef/inc/common/util
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/abl/mmpa/inc/mmpa
    )
 
    target_link_libraries(hcom_gradient_split_tune   PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:hccl_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        -Wl,--no-as-needed
        slog
        mmpa
        platform
        c_sec
        -Wl,--as-needed
    )
endif()
install(TARGETS hcom_gradient_split_tune
    LIBRARY DESTINATION lib  OPTIONAL
)

