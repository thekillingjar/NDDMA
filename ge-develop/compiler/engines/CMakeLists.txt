# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(ENG_DIR ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(cpu_engine)
add_subdirectory(dvpp_engine)
add_subdirectory(local_engine)
add_subdirectory(manager/engine)
add_subdirectory(hccl_engine)
add_subdirectory(nn_engine)
add_subdirectory(ffts_engine)
if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    add_subdirectory(rts_engine)
else()
endif()