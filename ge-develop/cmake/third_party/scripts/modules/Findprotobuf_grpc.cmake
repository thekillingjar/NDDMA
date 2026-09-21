# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (protobuf_grpc_FOUND)
    message(STATUS "Package protobuf_grpc has been found.")
    return()
endif()

find_path(PROTOBUF_GRPC_INCLUDE
    NAMES google/protobuf/api.pb.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

if(WIN32)
    if (${CMAKE_CONFIGURATION_TYPES} STREQUAL "Debug")
        find_library(PROTOBUF_GRPC_LIBRARY
            NAMES libprotobufd.lib
            PATH_SUFFIXES lib lib64
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    else()
        find_library(PROTOBUF_GRPC_LIBRARY
            NAMES libprotobuf.lib
            PATH_SUFFIXES lib lib64
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    endif()
else()
    find_library(PROTOBUF_GRPC_LIBRARY
        NAMES libprotobuf.a
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(protobuf_grpc
    FOUND_VAR
        protobuf_grpc_FOUND
    REQUIRED_VARS
        PROTOBUF_GRPC_INCLUDE
        PROTOBUF_GRPC_LIBRARY
    )

if(protobuf_grpc_FOUND)
    set(PROTOBUF_GRPC_INCLUDE_DIR ${PROTOBUF_GRPC_INCLUDE})

    add_library(protobuf::libprotobuf STATIC IMPORTED)
    set_target_properties(protobuf::libprotobuf PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_GRPC_INCLUDE}"
        IMPORTED_LOCATION             "${PROTOBUF_GRPC_LIBRARY}"
        )
endif()
