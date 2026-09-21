# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET protobuf_static_build)
    return()
endif ()

include(ExternalProject)

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

if(PROTOBUF_STATIC_INCLUDE AND PROTOBUF_STATIC_LIBRARY)
    set(protobuf_static_FOUND TRUE)
else()
    set(protobuf_static_FOUND FALSE)
endif()

if(protobuf_static_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[protobuf static] protobuf_static found, skip compiling.")
else()
    message(STATUS "[protobuf static] protobuf_static_FOUND:${protobuf_static_FOUND}, FORCE_REBUILD_CANN_3RD:${FORCE_REBUILD_CANN_3RD}")

    set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/protobuf/protobuf-all-25.1.tar.gz")
    set(REQ_URL_BACK "${CMAKE_THIRD_PARTY_LIB_DIR}/protobuf/protobuf-25.1.tar.gz")
    # 初始化可选参数列表
    set(PROTOBUF_EXTRA_ARGS "")
    if(EXISTS ${REQ_URL})
        message(STATUS "[protobuf static] ${REQ_URL} found.")
    elseif(EXISTS ${REQ_URL_BACK})
        message(STATUS "[protobuf static] ${REQ_URL_BACK} found.")
        set(REQ_URL ${REQ_URL_BACK})
    else()
        message(STATUS "[protobuf static] ${REQ_URL} not found, need download.")
        set(REQ_URL "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz")
        list(APPEND PROTOBUF_EXTRA_ARGS
            DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/protobuf_static
        )
    endif()
    
    set(protobuf_CXXFLAGS "-Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI} -O2")

    ExternalProject_Add(protobuf_static_build
                        URL ${REQ_URL}
                        TLS_VERIFY OFF
                        ${PROTOBUF_EXTRA_ARGS}
                        PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/patch/protobuf_25.1_change_version.patch
                        CONFIGURE_COMMAND ${CMAKE_COMMAND}
                            -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                            -Dprotobuf_WITH_ZLIB=OFF
                            -DCMAKE_SKIP_RPATH=TRUE
                            -Dprotobuf_BUILD_TESTS=OFF
                            -DCMAKE_CXX_STANDARD=14
                            -DCMAKE_CXX_FLAGS=${protobuf_CXXFLAGS}
                            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                            -DCMAKE_INSTALL_PREFIX=${CMAKE_THIRD_PARTY_LIB_DIR}/protobuf_static
                            -DLLVM_PATH=${LLVM_PATH}
                            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                            -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
                            -Dprotobuf_ABSL_PROVIDER=module
                            -DABSL_ROOT_DIR=${CMAKE_BINARY_DIR}/abseil_build-prefix/src/abseil_build
                            <SOURCE_DIR>
                        BUILD_COMMAND $(MAKE)
                        INSTALL_COMMAND $(MAKE) install
                        EXCLUDE_FROM_ALL TRUE
    )
    add_dependencies(protobuf_static_build abseil_build)
endif()