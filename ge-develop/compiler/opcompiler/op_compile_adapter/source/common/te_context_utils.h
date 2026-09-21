/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TE_CONTEXT_UTILS_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TE_CONTEXT_UTILS_H_

#include <string>
#include "inc/te_fusion_types.h"

namespace te {
namespace fusion {
class TeContextUtils {
public:
    static std::string GetBuildMode();
    static std::string GetBuildStep();
    static std::string GetTuningPath();
    static std::string GetBuildInnerModel();
    static std::string GetDeterministic();
    static std::string GetDeterministicLevel();
    static std::string GetStatusCheck();
    static std::string GetPerformanceMode();
    static JIT_MODE GetJitCompile();
    static bool EnableShapeGeneralized();
    static bool EnableOpBankUpdate();
private:
    static bool GetContextValue(const std::string &key, std::string &value, bool isThreadLocal = false);
    static std::string GetContextValue(const std::string &key, bool isThreadLocal = false);
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TE_CONTEXT_UTILS_H_