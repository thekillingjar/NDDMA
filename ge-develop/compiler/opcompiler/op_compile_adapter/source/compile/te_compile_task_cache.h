/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMPILE_TE_COMPILE_TASK_CACHE_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMPILE_TE_COMPILE_TASK_CACHE_H_

#include <string>
#include <mutex>
#include <vector>
#include <unordered_map>
#include "tensor_engine/tbe_op_info.h"
#include "inc/te_fusion_types.h"
#include "common/tbe_op_info_hash.h"

namespace te {
namespace fusion {
using OpBuildTaskPair = std::pair<OpBuildTaskResultPtr, std::vector<OpBuildTaskPtr>>;
class TeCompileTaskCache {
public:
    TeCompileTaskCache(const TeCompileTaskCache &) = delete;
    TeCompileTaskCache &operator=(const TeCompileTaskCache &) = delete;

    static TeCompileTaskCache& Instance();
    std::string GetOpPattern(const std::string &opName) const;
    std::string GetOpPattern(const TbeOpInfoPtr &tbeOpInfo) const;
    OpBuildTaskPair* GetOpBuildTaskPair(const TbeOpInfoPtr &tbeOpInfo);
    void SetOpPreBuildTask(const TbeOpInfoPtr &tbeOpInfo, const OpBuildTaskPtr &opTask);
private:
    TeCompileTaskCache() {}
    ~TeCompileTaskCache() {}

    mutable std::mutex opPatternMutex_;
    // cache Op->pattern map
    std::unordered_map<TbeOpInfoPtr, OpBuildTaskPair, HashTbeOpInfo, EqualToTbeOpInfo> opPatternMap_;
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_COMPILE_TE_COMPILE_TASK_CACHE_H_