/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LLT_FUSION_ENGINE_UT_TESTCASE_FUSION_ENGINE_FUSION_STATISTIC_GRAPH_FUSION_BUILTIN_GRAPH_FUSION_ONE_PASS_TEST_H_
#define LLT_FUSION_ENGINE_UT_TESTCASE_FUSION_ENGINE_FUSION_STATISTIC_GRAPH_FUSION_BUILTIN_GRAPH_FUSION_ONE_PASS_TEST_H_
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "common/fe_log.h"

namespace fe {
    class BuiltGraphFusionOnePassTest : public PatternFusionBasePass {
    public:
        vector<FusionPattern *> DefinePatterns() override {
            vector<FusionPattern *> patterns;
            FusionPattern *pattern = new (std::nothrow) FusionPattern("MyBuiltPattern");
            FE_CHECK(pattern == nullptr, FE_LOGE("New a pattern object failed."),  return patterns);
            /* conv2_d_back_prop_filter(Fragz)
         *          |
         *        a.m.(NCHW)  */
            pattern->AddOpDesc("am1", {"am1"})
                    .AddOpDesc("conv2DBackPropFilter", {"conv2DBackPropFilter"})
                    .SetInputs("am1", {"conv2DBackPropFilter"})
                    .SetOutput("am1");
            patterns.push_back(pattern);
            return patterns;
        }

        Status Fusion(ge::ComputeGraph &graph,
                      Mapping &mapping,
                      vector<ge::NodePtr> &fusion_nodes) override {
            return SUCCESS;
        }
    };
}
#endif  // LLT_FUSION_ENGINE_UT_TESTCASE_FUSION_ENGINE_FUSION_STATISTIC_GRAPH_FUSION_BUILTIN_GRAPH_FUSION_ONE_PASS_TEST_H_
