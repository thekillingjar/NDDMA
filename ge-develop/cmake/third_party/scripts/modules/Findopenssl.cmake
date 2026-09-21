# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (OPENSSL_FOUND)
    message(STATUS "Package OPENSSL has been found.")
    return()
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

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(openssl
    FOUND_VAR
        OPENSSL_FOUND
    REQUIRED_VARS
        OPENSSL_INCLUDE
        CRYPTO_STATIC_LIBRARY
        SSL_STATIC_LIBRARY
    )

if(OPENSSL_FOUND)
    set(OPENSSL_INCLUDE_DIR ${OPENSSL_INCLUDE})

    add_library(crypto_static STATIC IMPORTED GLOBAL)
    set_target_properties(crypto_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OPENSSL_INCLUDE}"
        IMPORTED_LOCATION             "${CRYPTO_STATIC_LIBRARY}"
        )
    add_library(OpenSSL::Crypto ALIAS crypto_static)

    add_library(ssl_static STATIC IMPORTED GLOBAL)
    set_target_properties(ssl_static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OPENSSL_INCLUDE}"
        IMPORTED_LOCATION             "${SSL_STATIC_LIBRARY}"
        )
    add_library(OpenSSL::SSL ALIAS ssl_static)
endif()
