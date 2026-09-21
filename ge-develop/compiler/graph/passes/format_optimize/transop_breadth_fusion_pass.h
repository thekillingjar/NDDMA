/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_TRANSOP_BREADTH_FUSION_PASS_H_
#define GE_GRAPH_PASSES_TRANSOP_BREADTH_FUSION_PASS_H_

#include <map>
#include <string>
#include <vector>

#include "graph/passes/graph_pass.h"
#include "graph/utils/cycle_detector.h"

namespace ge {
///
/// Transform operators breadth fusion
///
class TransOpBreadthFusionPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph) final;

 private:
  std::string GetNodeId(const int32_t anchor_index, const NodePtr& node);

  std::map<std::string, std::vector<NodePtr>> GetOutputTransOpNodes(const NodePtr& node);

  graphStatus Fusion(const std::vector<NodePtr>& trans_nodes, ComputeGraphPtr& graph) const;

  std::string JoinDims(const std::string& sp, const std::vector<int64_t>& dims) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_TRANSOP_BREADTH_FUSION_PASS_H_
