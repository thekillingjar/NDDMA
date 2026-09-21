# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (atrace_FOUND)
    message(STATUS "Package atrace has been found.")
    return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS atrace_share atrace atrace_headers)
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

find_path(_INCLUDE_DIR
    NAMES experiment/atrace/utrace/atrace_pub.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

message("Atrace cmake ASCEND_ROOT=" ${ASCEND_ROOT})
message("Atrace cmake ASCEND_INSTALL_PATH=" ${ASCEND_INSTALL_PATH})
find_library(atrace_SHARED_LIBRARY
    NAMES libascend_trace.so
    PATH_SUFFIXES lib64
    PATHS ${ASCEND_ROOT}
          ${ASCEND_INSTALL_PATH}
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
)
message("Atrace cmake atrace_SHARED_LIBRARY=" ${atrace_SHARED_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(atrace
    FOUND_VAR
        atrace_FOUND
    REQUIRED_VARS
        _INCLUDE_DIR
        atrace_SHARED_LIBRARY
)

if(atrace_FOUND)
    set(atrace_INCLUDE_DIR "${_INCLUDE_DIR}/experiment")
    include(CMakePrintHelpers)
    message(STATUS "Variables in atrace module:")
    cmake_print_variables(atrace_INCLUDE_DIR)
    cmake_print_variables(atrace_SHARED_LIBRARY)

    add_library(atrace_share SHARED IMPORTED)
    set_target_properties(atrace_share PROPERTIES
        INTERFACE_LINK_LIBRARIES "atrace_headers"
        INTERFACE_COMPILE_DEFINITIONS "HOST_ALOG;ADX_LIB_C"
        IMPORTED_LINK_DEPENDENT_LIBRARIES "mmpa;alog"
        IMPORTED_LOCATION "${atrace_SHARED_LIBRARY}"
    )

    add_library(atrace_headers INTERFACE IMPORTED)
    set_target_properties(atrace_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${atrace_INCLUDE_DIR};${atrace_INCLUDE_DIR}/atrace;${atrace_INCLUDE_DIR}/atrace/utrace"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS atrace_share
        PROPERTIES INTERFACE_LINK_LIBRARIES INTERFACE_COMPILE_DEFINITIONS IMPORTED_LINK_DEPENDENT_LIBRARIES IMPORTED_LOCATION
    )
    cmake_print_properties(TARGETS atrace_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
endif()

# Cleanup temporary variables.
set(_INCLUDE_DIR)
