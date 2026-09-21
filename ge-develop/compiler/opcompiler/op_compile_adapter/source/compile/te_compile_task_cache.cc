/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compile/te_compile_task_cache.h"
#include "common/tbe_op_info_cache.h"
#include "inc/te_fusion_util_constants.h"

namespace te {
namespace fusion {
TeCompileTaskCache& TeCompileTaskCache::Instance()
{
    static TeCompileTaskCache teCompileTaskCacheInstance;
    return teCompileTaskCacheInstance;
}

std::string TeCompileTaskCache::GetOpPattern(const TbeOpInfoPtr &tbeOpInfo) const
{
    if (tbeOpInfo == nullptr) {
        return TE_OPAQUE_PATTERN;
    }
    std::lock_guard<std::mutex> lock_guard(opPatternMutex_);
    auto iter = opPatternMap_.find(tbeOpInfo);
    if (iter == opPatternMap_.end() || iter->second.first == nullptr ||
        iter->second.first->preCompileRetPtr == nullptr ||
        iter->second.first->preCompileRetPtr->opPattern.empty()) {
        return TE_OPAQUE_PATTERN;
    }
    return iter->second.first->preCompileRetPtr->opPattern;
}

std::string TeCompileTaskCache::GetOpPattern(const std::string &opName) const
{
    TbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().MutableSecondTbeOpInfo(opName);
    if (tbeOpInfo == nullptr) {
        tbeOpInfo = TbeOpInfoCache::Instance().MutableTbeOpInfo(opName);
    }
    return GetOpPattern(tbeOpInfo);
}

OpBuildTaskPair* TeCompileTaskCache::GetOpBuildTaskPair(const TbeOpInfoPtr &tbeOpInfo)
{
    if (tbeOpInfo == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock_guard(opPatternMutex_);
    auto iter = opPatternMap_.find(tbeOpInfo);
    if (iter == opPatternMap_.end()) {
        return nullptr;
    }
    return &iter->second;
}

void TeCompileTaskCache::SetOpPreBuildTask(const TbeOpInfoPtr &tbeOpInfo, const OpBuildTaskPtr &opTask)
{
    std::lock_guard<std::mutex> lock_guard(opPatternMutex_);
    (void)opPatternMap_.emplace(tbeOpInfo, std::make_pair(nullptr, std::vector<OpBuildTaskPtr>({opTask})));
}
}
}