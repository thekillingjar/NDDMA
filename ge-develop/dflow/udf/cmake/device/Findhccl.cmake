# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET hccl_headers)
    message(STATUS "target hccl_headers has been found.")
    return()
endif()

find_path(hccl_INCLUDE_DIR
    NAMES hccl/base.h
    PATH_SUFFIXES
        pkg_inc
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

if (hccl_INCLUDE_DIR)
    include(CMakePrintHelpers)
    message(STATUS "Variables in hccl module:")
    cmake_print_variables(hccl_INCLUDE_DIR)

    add_library(hccl_headers INTERFACE IMPORTED)
    set_target_properties(hccl_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${hccl_INCLUDE_DIR};${hccl_INCLUDE_DIR}/hccl;"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS hccl_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
endif()

# Cleanup temporary variables.
set(hccl_INCLUDE_DIR)
