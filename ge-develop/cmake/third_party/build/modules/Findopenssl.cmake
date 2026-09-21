# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET openssl_build)
    return()
endif()

include(ExternalProject)

if(CCACHE_PROGRAM)
    set(OPENSSL_CC "${CCACHE_PROGRAM} ${CMAKE_C_COMPILER}")
    set(OPENSSL_CXX "${CCACHE_PROGRAM} ${CMAKE_CXX_COMPILER}")
else()
    set(OPENSSL_CC "${CMAKE_C_COMPILER}")
    set(OPENSSL_CXX "${CMAKE_CXX_COMPILER}")
endif()


find_path(OPENSSL_INCLUDE
    NAMES openssl/ssl.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)
if(UNIX)
    find_library(CRYPTO_STATIC_LIBRARY
        NAMES libcrypto.a
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)
    find_library(SSL_STATIC_LIBRARY
        NAMES libssl.a
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)
else()
    find_library(CRYPTO_STATIC_LIBRARY
        NAMES libcrypto.lib
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)
    find_library(SSL_STATIC_LIBRARY
        NAMES libssl.lib
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)
endif()

if(CRYPTO_STATIC_LIBRARY AND SSL_STATIC_LIBRARY AND OPENSSL_INCLUDE)
    set(OPENSSL_FOUND TRUE)
else()
    set(OPENSSL_FOUND FALSE)
endif()

if(OPENSSL_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[openssl] openssl found.")
else()
    message(STATUS "[openssl] OPENSSL_FOUND:${OPENSSL_FOUND}, FORCE_REBUILD_CANN_3RD:${FORCE_REBUILD_CANN_3RD}")

    if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL x86_64)
        set(OPENSSL_HOST_LOCAL_ARCH linux-x86_64)
    elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL aarch64)
        set(OPENSSL_HOST_LOCAL_ARCH linux-aarch64)
    elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL arm)
        set(OPENSSL_HOST_LOCAL_ARCH linux-armv4)
    endif ()

    set(SAFE_SP_OPTIONS -fstack-protector-all)
    set(COMPILE_OPTIONE -D_FORFIFY_SOURCE=2 -fvisibility=hidden -O2 -Wl,-z,relro,-z,now,-z,noexecstack$<$<CONFIG:Release>:,--build-id=none> -s)

    # 初始化可选参数列表
    set(OPENSSL_EXTRA_ARGS "")
    set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/openssl/openssl-openssl-3.0.9.tar.gz")
    if(EXISTS "${CMAKE_THIRD_PARTY_LIB_DIR}/openssl/tools")
        message(STATUS "[openssl] ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl/tools found.")
        list(APPEND OPENSSL_EXTRA_ARGS
            SOURCE_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl
        )
    elseif(EXISTS ${REQ_URL})
        message(STATUS "[openssl] ${REQ_URL} found.")
        list(APPEND OPENSSL_EXTRA_ARGS
            URL ${REQ_URL}
        )
    else()
        message(STATUS "[openssl] ${REQ_URL} not found, need download.")
        list(APPEND OPENSSL_EXTRA_ARGS
            URL "https://gitcode.com/cann-src-third-party/openssl/releases/download/openssl-3.0.9/openssl-openssl-3.0.9.tar.gz"
            DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl
        )
    endif()
    
    ExternalProject_Add(openssl_build
                        ${OPENSSL_EXTRA_ARGS}
                        TLS_VERIFY OFF
                        CONFIGURE_COMMAND
                            <SOURCE_DIR>/Configure
                            ${OPENSSL_HOST_LOCAL_ARCH} no-afalgeng no-asm no-shared threads enable-ssl3-method no-tests
                            ${SAFE_SP_OPTIONS} ${COMPILE_OPTIONE}
                            --prefix=${CMAKE_THIRD_PARTY_LIB_DIR}/openssl-temp
                            --libdir=lib64
                            CC=${OPENSSL_CC} CXX=${OPENSSL_CXX}
                        BUILD_COMMAND $(MAKE)
                        INSTALL_COMMAND $(MAKE) install_dev
                        EXCLUDE_FROM_ALL TRUE
                        )
    ExternalProject_Add_Step(openssl_build extra_install
        COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include/crypto ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl/include/crypto
        COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include/internal ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl/include/internal
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl ${CMAKE_COMMAND} -E create_symlink lib64 ${CMAKE_INSTALL_LIBDIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl-temp ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_THIRD_PARTY_LIB_DIR}/openssl-temp
        DEPENDEES install
    )
endif()