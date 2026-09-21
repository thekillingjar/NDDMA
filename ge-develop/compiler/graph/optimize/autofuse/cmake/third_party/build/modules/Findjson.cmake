# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (TARGET json_build)
    return()
endif ()

include(ExternalProject)

set(REQ_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz")

ExternalProject_Add(json_build
                    URL ${REQ_URL}
                    CONFIGURE_COMMAND ${CMAKE_COMMAND}
                        -DJSON_MultipleHeaders=ON
                        -DJSON_BuildTests=OFF
                        -DBUILD_SHARED_LIBS=OFF
                        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/json
                        <SOURCE_DIR>
                    EXCLUDE_FROM_ALL TRUE
)
