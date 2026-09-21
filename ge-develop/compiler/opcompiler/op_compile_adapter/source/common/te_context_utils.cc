/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/te_context_utils.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_util_constants.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/tuning_utils.h"

namespace te {
namespace fusion {
std::string TeContextUtils::GetBuildMode()
{
    return GetContextValue(ge::BUILD_MODE);
}

std::string TeContextUtils::GetBuildStep()
{
    return GetContextValue(ge::BUILD_STEP);
}

std::string TeContextUtils::GetTuningPath()
{
    return GetContextValue(ge::TUNING_PATH);
}

std::string TeContextUtils::GetBuildInnerModel()
{
    return GetContextValue(ge::BUILD_INNER_MODEL);
}

std::string TeContextUtils::GetDeterministic()
{
    return (GetContextValue(ge::DETERMINISTIC) == "1") ? STR_TRUE : STR_FALSE;
}

std::string TeContextUtils::GetDeterministicLevel()
{
    return GetContextValue("ge.deterministicLevel");
}

std::string TeContextUtils::GetStatusCheck()
{
    return GetContextValue(ge::STATUS_CHECK);
}

std::string TeContextUtils::GetPerformanceMode()
{
    return GetContextValue(ge::PERFORMANCE_MODE);
}

JIT_MODE TeContextUtils::GetJitCompile()
{
    std::string jitCompileStr = GetContextValue(ge::JIT_COMPILE);
    // 0-binary; 1-compile; 2-compile for static and use binary for dynamic.
    if (jitCompileStr == "0") {
        return JIT_MODE::JIT_USE_BINARY;
    } else if (jitCompileStr == "2") {
        return JIT_MODE::JIT_AUTO;
    } else {
        return JIT_MODE::JIT_COMPILE_ONLINE;
    }
}

bool TeContextUtils::EnableShapeGeneralized()
{
    return GetContextValue("ge.shape_generalized") == "1";
}

bool TeContextUtils::EnableOpBankUpdate()
{
    return GetContextValue(ge::OP_BANK_UPDATE_FLAG, true) == "true";
}

bool TeContextUtils::GetContextValue(const std::string &key, std::string &value, bool isThreadLocal)
{
    if (isThreadLocal) {
        if (ge::GetThreadLocalContext().GetOption(key, value) == ge::GRAPH_SUCCESS && !value.empty()) {
            TE_DBGLOG("Get thread_local option of key [%s], result is %s.", key.c_str(), value.c_str());
            return true;
        }
    } else {
        if (ge::GetContext().GetOption(key, value) == ge::GRAPH_SUCCESS && !value.empty()) {
            TE_DBGLOG("Get option of key [%s], result is %s.", key.c_str(), value.c_str());
            return true;
        }
    }
    return false;
}

std::string TeContextUtils::GetContextValue(const std::string &key, bool isThreadLocal)
{
    std::string value;
    if (isThreadLocal) {
        (void)ge::GetThreadLocalContext().GetOption(key, value);
    } else {
        (void)ge::GetContext().GetOption(key, value);
    }
    TE_DBGLOG("The value of context [%s] is [%s].", key.c_str(), value.c_str());
    return value;
}
}
}