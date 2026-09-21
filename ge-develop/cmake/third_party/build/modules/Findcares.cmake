# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET cares_build)
    return()
endif()

include(ExternalProject)

set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/c-ares/c-ares-1.19.1.tar.gz")

# 初始化可选参数列表
set(CARES_EXTRA_ARGS "")
if(EXISTS ${REQ_URL})
  message(STATUS "[c-ares] ${REQ_URL} found.")
else()
  message(STATUS "[c-ares] ${REQ_URL} not found, need download.")
  set(REQ_URL "https://gitcode.com/cann-src-third-party/c-ares/releases/download/v1.19.1/c-ares-1.19.1.tar.gz")
  list(APPEND CARES_EXTRA_ARGS
      DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/c-ares
  )
endif()

ExternalProject_Add(cares_build
                    URL ${REQ_URL}
                    TLS_VERIFY OFF
                    ${CARES_EXTRA_ARGS}
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    EXCLUDE_FROM_ALL TRUE
                    DWONLOAD_NO_PROGRESS TRUE
)