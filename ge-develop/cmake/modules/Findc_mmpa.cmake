# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (c_mmpa_FOUND)
    message(STATUS "Package c_mmpa has been found.")
    return()
endif()

find_path(_INCLUDE_DIR
    NAMES experiment/c_mmpa/mmpa_api.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(c_mmpa
    FOUND_VAR
        c_mmpa_FOUND
    REQUIRED_VARS
        _INCLUDE_DIR
)

if(c_mmpa_FOUND)
    set(c_mmpa_INCLUDE_DIR "${_INCLUDE_DIR}/experiment")
    include(CMakePrintHelpers)
    message(STATUS "Variables in c_mmpa module:")
    cmake_print_variables(c_mmpa_INCLUDE_DIR)

    add_library(c_mmpa_headers INTERFACE IMPORTED)
    set_target_properties(c_mmpa_headers PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "\$<IF:\$<STREQUAL:Linux,LiteOS>,NANO_OS_TYPE=1,NANO_OS_TYPE=0>"
        INTERFACE_INCLUDE_DIRECTORIES "${c_mmpa_INCLUDE_DIR};${c_mmpa_INCLUDE_DIR}/c_mmpa"
        INTERFACE_LINK_LIBRARIES "c_sec_headers"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS c_mmpa_headers
        PROPERTIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_INCLUDE_DIRECTORIES INTERFACE_LINK_LIBRARIES
    )
endif()

# Cleanup temporary variables.
set(_INCLUDE_DIR)
