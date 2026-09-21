/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LLT_FUSION_ENGINE_UT_TESTCASE_FUSION_ENGINE_UB_FUSION_BUILTIN_BUFFER_FUSION_PASS_TEST_H_
#define LLT_FUSION_ENGINE_UT_TESTCASE_FUSION_ENGINE_UB_FUSION_BUILTIN_BUFFER_FUSION_PASS_TEST_H_
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"
namespace fe {
    class BuiltBufferFusionPassTest : public BufferFusionPassBase {
    public:
        std::vector<BufferFusionPattern *> DefinePatterns() override {
            return vector<BufferFusionPattern *>{};
        }
    };
}
#endif  // LLT_FUSION_ENGINE_UT_TESTCASE_FUSION_ENGINE_UB_FUSION_BUILTIN_BUFFER_FUSION_PASS_TEST_H_
