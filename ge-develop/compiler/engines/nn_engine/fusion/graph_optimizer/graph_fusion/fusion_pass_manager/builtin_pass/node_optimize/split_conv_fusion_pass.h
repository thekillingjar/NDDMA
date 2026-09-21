/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_SPLIT_CONV_FUSION_PASS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_SPLIT_CONV_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/checker/split_optimize_checker.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/node_optimize_pass_base.h"

namespace fe {
class SplitConvFusionPass : public NodeOptimizePassBase {
 public:
  Status DoFusion(ge::ComputeGraph &graph, ge::NodePtr &split_node, vector<ge::NodePtr> &fusion_nodes) override;
  vector<string> GetNodeTypes() override;
  string GetPatternName() override;

 private:
  bool CheckInOutNodesAttr(const ge::NodePtr &node) const;
  Status FusionSplit(ge::ComputeGraph &graph, ge::NodePtr &split_node);
  SplitOptimizeChecker split_optimize_checker_;
};

}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_SPLIT_CONV_FUSION_PASS_H_
