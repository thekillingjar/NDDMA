/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LLT_FUSION_ENGINE_UT_TESTCASE_FUSION_ENGINE_FUSION_STATISTIC_GRAPH_FUSION_CUSTOM_GRAPH_FUSION_ONE_PASS_FAILED_TEST_H_
#define LLT_FUSION_ENGINE_UT_TESTCASE_FUSION_ENGINE_FUSION_STATISTIC_GRAPH_FUSION_CUSTOM_GRAPH_FUSION_ONE_PASS_FAILED_TEST_H_
#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"
#include "common/fe_log.h"

namespace fe {
class CustomGraphFusionOnePassTest2 : public GraphFusionPassBase {
public:
    vector<FusionPattern *> DefinePatterns() override {
        vector<FusionPattern *> patterns;
        FusionPattern *pattern1 = new (std::nothrow) FusionPattern("MyBuiltPattern1");
        FE_CHECK(pattern1 == nullptr, FE_LOGE("New a pattern1 object failed."),  return patterns);
        /* Conv2D(NC1HWC0)
     *          |
     *        L2Loss (NCHW)  */
        pattern1->AddOpDesc("L2Loss", {"L2Loss"})
                .AddOpDesc("Conv2D", {"Conv2D"})
                .SetInputs("L2Loss", {"Conv2D"})
                .SetOutput("L2Loss");
        patterns.push_back(pattern1);

        FusionPattern *pattern2 = new (std::nothrow) FusionPattern("MyBuiltPattern2");
        FE_CHECK(pattern2 == nullptr, FE_LOGE("New a pattern object failed."),  return patterns);
        /* L2Loss (NCHW)
     *          |
     *        AddN (NCHW)  */
        pattern2->AddOpDesc("L2Loss", {"L2Loss"})
                .AddOpDesc("AddN", {"AddN"})
                .SetInputs("AddN", {"L2Loss"})
                .SetOutput("AddN");
        patterns.push_back(pattern2);
        return patterns;
    }

    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusion_nodes) override {
        return NOT_CHANGED;
    }
};
}
#endif  // LLT_FUSION_ENGINE_UT_TESTCASE_FUSION_ENGINE_FUSION_STATISTIC_GRAPH_FUSION_CUSTOM_GRAPH_FUSION_ONE_PASS_FAILED_TEST_H_
