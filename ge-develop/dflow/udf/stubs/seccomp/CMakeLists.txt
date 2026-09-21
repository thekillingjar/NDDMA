# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.5)
project(seccomp_stub)

set_directory_properties(PROPERTIES COMPILE_OPTIONS "-std=c99;-Wall;-Wextra;-Wfloat-equal;-fno-common")

# target libseccomp stub specific
set(SECCOMP_FILE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)
add_library(seccomp_stub SHARED
    ${SECCOMP_FILE_DIR}/seccomp_stub.c
)

set_target_properties(seccomp_stub PROPERTIES
    INTERFACE_LINK_LIBRARIES "seccomp_headers"
    OUTPUT_NAME "seccomp"
    SOVERSION 2)

target_link_libraries(seccomp_stub PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:seccomp_headers>
)