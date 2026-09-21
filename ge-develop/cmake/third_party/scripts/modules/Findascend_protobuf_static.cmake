# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (ascend_protobuf_static_FOUND)
    message(STATUS "Package ascend_protobuf_static has been found.")
    return()
endif()

find_path(ASCEND_PROTOBUF_STATIC_INCLUDE
    NAMES google/protobuf/api.pb.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

if(WIN32)
    if (${CMAKE_CONFIGURATION_TYPES} STREQUAL "Debug")
        find_library(ASCEND_PROTOBUF_STATIC_LIBRARY
            NAMES libprotobufd.lib
            PATH_SUFFIXES lib lib64
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    else()
        find_library(ASCEND_PROTOBUF_STATIC_LIBRARY
            NAMES libprotobuf.lib
            PATH_SUFFIXES lib lib64
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    endif()
else()
    find_library(ASCEND_PROTOBUF_STATIC_LIBRARY
        NAMES libascend_protobuf.a
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ascend_protobuf_static
    FOUND_VAR
        ascend_protobuf_static_FOUND
    REQUIRED_VARS
        ASCEND_PROTOBUF_STATIC_INCLUDE
        ASCEND_PROTOBUF_STATIC_LIBRARY
    )

if(ascend_protobuf_static_FOUND)
    find_package(absl CONFIG REQUIRED)
    find_package(utf8_range CONFIG REQUIRED)

    set(ASCEND_PROTOBUF_STATIC_INCLUDE_DIR ${ASCEND_PROTOBUF_STATIC_INCLUDE})

    add_library(ascend_protobuf_static STATIC IMPORTED)
    set_target_properties(ascend_protobuf_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_PROTOBUF_STATIC_INCLUDE}"
        IMPORTED_LOCATION             "${ASCEND_PROTOBUF_STATIC_LIBRARY}"
    )
    target_link_libraries(ascend_protobuf_static INTERFACE
        absl::absl_check
        absl::absl_log
        absl::algorithm
        absl::base
        absl::bind_front
        absl::bits
        absl::btree
        absl::cleanup
        absl::cord
        absl::core_headers
        absl::debugging
        absl::die_if_null
        absl::dynamic_annotations
        absl::flags
        absl::flat_hash_map
        absl::flat_hash_set
        absl::function_ref
        absl::hash
        absl::layout
        absl::log_initialize
        absl::log_severity
        absl::memory
        absl::node_hash_map
        absl::node_hash_set
        absl::optional
        absl::span
        absl::status
        absl::statusor
        absl::strings
        absl::synchronization
        absl::time
        absl::type_traits
        absl::utility
        absl::variant
        utf8_range::utf8_validity
    )

    add_library(ascend_protobuf_static_headers INTERFACE IMPORTED)
    target_include_directories(ascend_protobuf_static_headers INTERFACE ${ASCEND_PROTOBUF_STATIC_INCLUDE})
endif()
