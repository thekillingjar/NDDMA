# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET re2_build)
    return()
endif()

include(ExternalProject)

set(REQ_URL "${CMAKE_THIRD_PARTY_LIB_DIR}/re2/2024-02-01.tar.gz")
set(REQ_URL_BACK "${CMAKE_THIRD_PARTY_LIB_DIR}/re2/re2-2024-02-01.tar.gz")
# 初始化可选参数列表
set(RE2_EXTRA_ARGS "")
if(EXISTS ${REQ_URL})
  message(STATUS "[re2] ${REQ_URL} found.")
elseif(EXISTS ${REQ_URL_BACK})
  message(STATUS "[re2] ${REQ_URL_BACK} found.")
  set(REQ_URL ${REQ_URL_BACK})
else()
  message(STATUS "[re2] ${REQ_URL} not found, need download.")
  set(REQ_URL "https://gitcode.com/cann-src-third-party/re2/releases/download/2024-02-01/re2-2024-02-01.tar.gz")
  list(APPEND RE2_EXTRA_ARGS
      DOWNLOAD_DIR ${CMAKE_THIRD_PARTY_LIB_DIR}/re2
  )
endif()

ExternalProject_Add(re2_build
                    URL ${REQ_URL}
                    TLS_VERIFY OFF
                    ${RE2_EXTRA_ARGS}
                    PATCH_COMMAND patch -N --batch --quiet -r - -p1 < ${CMAKE_CURRENT_LIST_DIR}/patch/re2-add_compatible_functions.patch
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND ""
                    EXCLUDE_FROM_ALL TRUE
                    DWONLOAD_NO_PROGRESS TRUE
)
