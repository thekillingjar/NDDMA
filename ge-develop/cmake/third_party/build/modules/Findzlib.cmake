# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET zlib_build)
    return()
endif()

include(ExternalProject)
include(FindPackageHandleStandardArgs)

find_path(ZLIB_INCLUDE
    NAMES zlib.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)
find_library(ZLIB_LIBRARY
    NAMES libz.a
    PATH_SUFFIXES lib lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

if(ZLIB_INCLUDE AND ZLIB_LIBRARY)
    set(zlib_FOUND TRUE)
else()
    set(zlib_FOUND FALSE)
endif()

if(zlib_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[zlib] zlib found.")
else()
    message(STATUS "[zlib] zlib_FOUND:${zlib_FOUND}, FORCE_REBUILD_CANN_3RD:${FORCE_REBUILD_CANN_3RD}")
    set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/zlib/zlib-1.2.13.tar.xz")
    set(REQ_URL_BACK "${CMAKE_THIRD_PARTY_LIB_DIR}/zlib/zlib-1.2.13.tar.gz")
    if(EXISTS ${REQ_URL})
        message(STATUS "[zlib] ${REQ_URL} found.")
    elseif(EXISTS ${REQ_URL_BACK})
        message(STATUS "[zlib] ${REQ_URL_BACK} found.")
        set(REQ_URL ${REQ_URL_BACK})
    else()
        message(STATUS "[zlib] ${REQ_URL} not found, need download.")
        set(REQ_URL "https://gitcode.com/cann-src-third-party/zlib/releases/download/v1.2.13/zlib-1.2.13.tar.gz")
    endif()

    ExternalProject_Add(zlib_build
                        URL ${REQ_URL}
                        TLS_VERIFY OFF
                        CONFIGURE_COMMAND ""
                        BUILD_COMMAND ""
                        INSTALL_COMMAND ""
                        EXCLUDE_FROM_ALL TRUE
                        DWONLOAD_NO_PROGRESS TRUE
    )
endif()