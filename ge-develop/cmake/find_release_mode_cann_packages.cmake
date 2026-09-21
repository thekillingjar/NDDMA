# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT OR ENABLE_OPEN_SRC)
    include(cmake/function.cmake)
    find_package_if_target_not_exists(slog MODULE REQUIRED)
    find_package_if_target_not_exists(runtime MODULE REQUIRED)
    find_package_if_target_not_exists(mmpa MODULE REQUIRED)
    find_package_if_target_not_exists(msprof MODULE REQUIRED)
    find_package_if_target_not_exists(adump MODULE REQUIRED)
    find_package_if_target_not_exists(hccl MODULE REQUIRED)
    find_package_if_target_not_exists(aicpu MODULE REQUIRED)
    find_package_if_target_not_exists(unified_dlog MODULE REQUIRED)
    find_package_if_target_not_exists(platform MODULE REQUIRED)
    find_package_if_target_not_exists(ascendcl MODULE REQUIRED)
    if(NOT MDC_COMPILE_RUNTIME) # MDC运行态不需要下列依赖
        find_package_if_target_not_exists(cce MODULE REQUIRED)
        find_package_if_target_not_exists(datagw MODULE REQUIRED)
        find_package_if_target_not_exists(ascend_hal MODULE REQUIRED)
        find_package_if_target_not_exists(atrace MODULE REQUIRED)
    endif()
    
    # 使用medadef发布包编译
    find_package(metadef MODULE REQUIRED)
    set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR}/../base/metadef)
    set(PARSER_DIR ${CMAKE_CURRENT_LIST_DIR}/parser)
endif ()
