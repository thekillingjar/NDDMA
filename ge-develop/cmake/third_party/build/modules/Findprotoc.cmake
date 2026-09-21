# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET protoc_build)
    return()
endif ()

include(ExternalProject)

find_program(PROTOC_PROGRAM
    NAMES protoc
    PATHS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
    )
if(PROTOC_PROGRAM)
    set(protoc_FOUND TRUE)
    message(STATUS "[protoc] PROTOC_PROGRAM found: ${PROTOC_PROGRAM}")
else()
    set(protoc_FOUND FALSE)
endif()

if(protoc_FOUND)
    message(STATUS "[protoc] protoc found, skip compiling.")
else()
    message(STATUS "[protoc] protoc not found, finding binary file.")

    set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/protobuf/protobuf-all-25.1.tar.gz")
    set(REQ_URL_BACK "${CMAKE_THIRD_PARTY_LIB_DIR}/protobuf/protobuf-25.1.tar.gz")
    # 初始化可选参数列表
    set(PROTOBUF_EXTRA_ARGS "")
    if(EXISTS ${REQ_URL})
        message(STATUS "[protoc] ${REQ_URL} found.")
    elseif(EXISTS ${REQ_URL_BACK})
        message(STATUS "[protoc] ${REQ_URL_BACK} found.")
        set(REQ_URL ${REQ_URL_BACK})
    else()
        message(STATUS "[protoc] ${REQ_URL} not found, need download.")
        set(REQ_URL "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz")
        list(APPEND PROTOBUF_EXTRA_ARGS
            DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/protoc
        )
    endif()
    
    set(protoc_CXXFLAGS "-Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI} -O2")
    set(protoc_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")

    ExternalProject_Add(protoc_build
                        URL ${REQ_URL}
                        TLS_VERIFY OFF
                        ${PROTOBUF_EXTRA_ARGS}
                        PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/patch/protobuf_25.1_change_version.patch
                        CONFIGURE_COMMAND ${CMAKE_COMMAND}
                            -Dprotobuf_WITH_ZLIB=OFF
                            -Dprotobuf_BUILD_TESTS=OFF
                            -DBUILD_SHARED_LIBS=OFF
                            -DCMAKE_CXX_STANDARD=14
                            -DCMAKE_CXX_FLAGS=${protoc_CXXFLAGS}
                            -DCMAKE_CXX_LDFLAGS=${protoc_LDFLAGS}
                            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                            -DCMAKE_INSTALL_PREFIX=${CMAKE_THIRD_PARTY_LIB_DIR}/protoc
                            -Dprotobuf_ABSL_PROVIDER=module
                            -DABSL_ROOT_DIR=${CMAKE_BINARY_DIR}/abseil_build-prefix/src/abseil_build
                            <SOURCE_DIR>
                        BUILD_COMMAND $(MAKE)
                        INSTALL_COMMAND $(MAKE) install
                        EXCLUDE_FROM_ALL TRUE
    )

    add_dependencies(protoc_build abseil_build)
endif()