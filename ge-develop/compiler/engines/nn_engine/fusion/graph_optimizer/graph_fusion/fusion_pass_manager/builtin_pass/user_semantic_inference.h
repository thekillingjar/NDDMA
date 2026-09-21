/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_USER_SEMANTIC_INFERENCE_PASS_BASE_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_USER_SEMANTIC_INFERENCE_PASS_BASE_H_
 
#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
 
namespace fe {
class UserSemanticInferencePass : public PatternFusionBasePass {
  protected:
    vector<FusionPattern *> DefinePatterns() override;
    Status Fusion(ge::ComputeGraph &graph,
                  Mapping &mapping,
                  vector<ge::NodePtr> &fusionNodes) override;
  private:
    Status FusionForPyPTO(const ge::NodePtr &pyptoNode) const;
};
}  // namespace fe
 
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_USER_SEMANTIC_INFERENCE_PASS_BASE_H_
