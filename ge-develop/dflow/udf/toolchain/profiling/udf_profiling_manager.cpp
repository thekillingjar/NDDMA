/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "udf_profiling_manager.h"
#include "common/udf_log.h"

namespace FlowFunc {

UdfProfilingManager &UdfProfilingManager::GetInstance() {
    static UdfProfilingManager profiling_manager;
    return profiling_manager;
}

int32_t UdfProfilingManager::Report() const {
    UDF_LOG_ERROR("Udf profiling is not support now.");
    return FLOW_FUNC_ERR_NOT_SUPPORT;
}
};  // namespace FlowFunc
