# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if(EXISTS "${TOP_DIR}/abl/slog")
    message(STATUS "dir ${TOP_DIR}/abl/slog exists, no need use udf stackcore stub.")
    return()
endif() 

if (TARGET stackcore)
    message(STATUS "Target stackcore already exists, no need use udf stub.")
    return()
endif()

set_directory_properties(PROPERTIES COMPILE_OPTIONS "-std=c99;-Wall;-Wextra;-Wfloat-equal;-fno-common")

# target stackcore specific
set(SECCOMP_FILE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/)

add_library(stackcore SHARED
    ${SECCOMP_FILE_DIR}/stackcore_stub.c
)

target_link_libraries(stackcore PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)