/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_RULE_MANAGER_FUSION_CYCLE_DETECTOR_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_RULE_MANAGER_FUSION_CYCLE_DETECTOR_H_
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
namespace fe {
class FusionCycleDetector : public PatternFusionBasePass {
 public:
  FusionCycleDetector();

  ~FusionCycleDetector() override;

  std::vector<FusionPattern *> DefinePatterns() override;

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                std::vector<ge::NodePtr> &new_nodes) override;

  Status Initialize(const ge::ComputeGraph &graph);

  void BackupConnectionMatrix();

  void RestoreConnectionMatrix();

  Status UpdateConnectionMatrix(const ge::ComputeGraph &graph, const std::vector<ge::NodePtr> &fusion_nodes);

  bool IsConnected(const ge::NodePtr &a, const ge::NodePtr &b);

  bool IsDataFlowConnected(const ge::NodePtr &a, const ge::NodePtr &b);
};
using FusionCycleDetectorPtr = std::shared_ptr<FusionCycleDetector>;
}
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_RULE_MANAGER_FUSION_CYCLE_DETECTOR_H_
