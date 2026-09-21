# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (c_base_FOUND)
    message(STATUS "Package c_base has been found.")
    return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS c_base_static c_json_static)
    list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
    if(TARGET "${_cmake_expected_target}")
        list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
    else()
        list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
    endif()
endforeach()
unset(_cmake_expected_target)

if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
    unset(_cmake_targets_defined)
    unset(_cmake_targets_not_defined)
    unset(_cmake_expected_targets)
    unset(CMAKE_IMPORT_FILE_VERSION)
    cmake_policy(POP)
    return()
endif()

if(NOT _cmake_targets_defined STREQUAL "")
    string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
    string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
    message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)

find_library(c_base_STATIC_LIBRARY
    NAMES libc_base.a
    PATH_SUFFIXES lib64/c
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

find_library(c_json_STATIC_LIBRARY
    NAMES libc_json.a
    PATH_SUFFIXES lib64/c
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(c_base
    FOUND_VAR
        c_base_FOUND
    REQUIRED_VARS
        c_base_STATIC_LIBRARY
        c_json_STATIC_LIBRARY
)

if(c_base_FOUND)
    include(CMakePrintHelpers)
    message(STATUS "Variables in c_base module:")
    cmake_print_variables(c_base_STATIC_LIBRARY)
    cmake_print_variables(c_json_STATIC_LIBRARY)

    add_library(c_base_static STATIC IMPORTED)
    set_target_properties(c_base_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "\$<LINK_ONLY:c_sec_headers>;"
        IMPORTED_LOCATION "${c_base_STATIC_LIBRARY}"
    )

    add_library(c_json_static STATIC IMPORTED)
    set_target_properties(c_json_static PROPERTIES
        INTERFACE_LINK_LIBRARIES "\$<LINK_ONLY:c_sec_headers>;"
        IMPORTED_LOCATION "${c_json_STATIC_LIBRARY}"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS c_base_static
        PROPERTIES INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
    )
    cmake_print_properties(TARGETS c_json_static
        PROPERTIES INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
    )
endif()
