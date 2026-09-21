# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (securec_FOUND)
    message(STATUS "Package securec has been found.")
    return()
endif()

find_path(C_SEC_INCLUDE NAMES securec.h)
find_library(C_SEC_SHARED_LIBRARY
    NAMES libc_sec.so
    PATHS ${DEV_FIND_LIB_PATHS}
    PATH_SUFFIXES devlib/device
    NO_DEFAULT_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(securec
    FOUND_VAR
        securec_FOUND
    REQUIRED_VARS
        C_SEC_INCLUDE
        C_SEC_SHARED_LIBRARY
)

if(securec_FOUND)
    set(C_SEC_INCLUDE_DIR ${C_SEC_INCLUDE})
    get_filename_component(C_SEC_LIBRARY_DIR ${C_SEC_SHARED_LIBRARY} DIRECTORY)

    include(CMakePrintHelpers)
    message(STATUS "Variables in securec module:")
    cmake_print_variables(C_SEC_INCLUDE)
    cmake_print_variables(C_SEC_LIBRARY_DIR)
    cmake_print_variables(C_SEC_SHARED_LIBRARY)

    add_library(c_sec_headers INTERFACE IMPORTED)
    target_include_directories(c_sec_headers INTERFACE ${C_SEC_INCLUDE})

    add_library(c_sec SHARED IMPORTED)
    set_target_properties(c_sec PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${C_SEC_INCLUDE}"
        IMPORTED_LOCATION             "${C_SEC_SHARED_LIBRARY}"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS c_sec_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
    cmake_print_properties(TARGETS c_sec
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES IMPORTED_LOCATION
    )
endif()
