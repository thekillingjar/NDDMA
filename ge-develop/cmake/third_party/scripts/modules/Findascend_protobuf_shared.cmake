# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (ascend_protobuf_shared_FOUND)
    message(STATUS "Package ascend_protobuf_shared has been found.")
    return()
endif()

include(FindPackageHandleStandardArgs)

find_path(ASCEND_PROTOBUF_SHARED_INCLUDE
    NAMES google/protobuf/api.pb.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

if(UNIX)
    find_library(ASCEND_PROTOBUF_SHARED_LIBRARY
        NAMES libascend_protobuf.so
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)
    find_package_handle_standard_args(ascend_protobuf_shared
        FOUND_VAR
            ascend_protobuf_shared_FOUND
        REQUIRED_VARS
            ASCEND_PROTOBUF_SHARED_INCLUDE
            ASCEND_PROTOBUF_SHARED_LIBRARY
    )

    if(ascend_protobuf_shared_FOUND)
        add_library(ascend_protobuf SHARED IMPORTED)
        set_target_properties(ascend_protobuf PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_PROTOBUF_SHARED_INCLUDE}"
            IMPORTED_LOCATION             "${ASCEND_PROTOBUF_SHARED_LIBRARY}"
        )
    endif()
endif()

if(WIN32)
    if (${CMAKE_CONFIGURATION_TYPES} STREQUAL "Debug")
        find_file(ASCEND_PROTOBUF_SHARED_DLL
            NAMES libprotobufd.dll
            PATH_SUFFIXES bin
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
        find_library(ASCEND_PROTOBUF_SHARED_IMPLIB
            NAMES libprotobufd.lib
            PATH_SUFFIXES lib lib64
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    else()
        find_file(ASCEND_PROTOBUF_SHARED_DLL
            NAMES libprotobuf.dll
            PATH_SUFFIXES bin
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
        find_library(ASCEND_PROTOBUF_SHARED_IMPLIB
            NAMES libprotobuf.lib
            PATH_SUFFIXES lib lib64
            NO_CMAKE_SYSTEM_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    endif()

    find_package_handle_standard_args(ascend_protobuf_shared
        FOUND_VAR
            ascend_protobuf_shared_FOUND
        REQUIRED_VARS
            ASCEND_PROTOBUF_SHARED_INCLUDE
            ASCEND_PROTOBUF_SHARED_DLL
            ASCEND_PROTOBUF_SHARED_IMPLIB
    )

    if(ascend_protobuf_shared_FOUND)
        add_library(ascend_protobuf SHARED IMPORTED)
        set_target_properties(ascend_protobuf PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_PROTOBUF_SHARED_INCLUDE}"
            IMPORTED_IMPLIB "${ASCEND_PROTOBUF_SHARED_IMPLIB}"
            IMPORTED_LOCATION "${ASCEND_PROTOBUF_SHARED_DLL}"
        )
    endif()
endif()

if(ascend_protobuf_shared_FOUND)
    set(ASCEND_PROTOBUF_SHARED_INCLUDE_DIR ${ASCEND_PROTOBUF_SHARED_INCLUDE})
    get_filename_component(ASCEND_PROTOBUF_SHARED_LIBRARY_DIR ${ASCEND_PROTOBUF_SHARED_INCLUDE_DIR}/../lib ABSOLUTE)
    add_library(ascend_protobuf_shared_headers INTERFACE IMPORTED)
    target_include_directories(ascend_protobuf_shared_headers INTERFACE ${ASCEND_PROTOBUF_SHARED_INCLUDE})
endif()
