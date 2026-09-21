# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (protobuf_static_FOUND)
    message(STATUS "Package protobuf_static has been found.")
    return()
endif()

find_path(PROTOBUF_STATIC_INCLUDE
    NAMES google/protobuf/api.pb.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

if(WIN32)
    if (${CMAKE_CONFIGURATION_TYPES} STREQUAL "Debug")
        find_library(PROTOBUF_STATIC_LIBRARY
            NAMES libprotobufd.lib
            PATH_SUFFIXES lib lib64
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    else()
        find_library(PROTOBUF_STATIC_LIBRARY
            NAMES libprotobuf.lib
            PATH_SUFFIXES lib lib64
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    endif()
else()
    find_library(PROTOBUF_STATIC_LIBRARY
        NAMES libprotobuf.a
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(protobuf_static
    FOUND_VAR
        protobuf_static_FOUND
    REQUIRED_VARS
        PROTOBUF_STATIC_INCLUDE
        PROTOBUF_STATIC_LIBRARY
    )

if(protobuf_static_FOUND)
    find_package(absl CONFIG REQUIRED)
    find_package(utf8_range CONFIG REQUIRED)

    set(PROTOBUF_STATIC_INCLUDE_DIR ${PROTOBUF_STATIC_INCLUDE})

    add_library(protobuf_static STATIC IMPORTED)
    set_target_properties(protobuf_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_STATIC_INCLUDE}"
        IMPORTED_LOCATION             "${PROTOBUF_STATIC_LIBRARY}"
    )
    target_link_libraries(protobuf_static INTERFACE
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
endif()
