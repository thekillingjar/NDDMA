/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BUFFER_FUSION_OPTIMIZER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BUFFER_FUSION_OPTIMIZER_H_

#include <string>
#include <vector>
#include <unordered_map>
#include "graph/compute_graph.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.h"

namespace fe {
class BufferFusionOptimizer {
 public:
  BufferFusionOptimizer() {}
  ~BufferFusionOptimizer() {}

  void Initialize(const ge::ComputeGraph &graph);
  void GetHeadNodesByFusionPattern(const BufferFusionPattern &pattern, std::vector<ge::NodePtr> &nodes) const;

 private:
  std::unordered_map<std::string, std::vector<ge::NodePtr>> op_type_nodes_map_;
  std::unordered_map<std::string, std::vector<ge::NodePtr>> op_pattern_nodes_map_;
};
using BufferFusionOptimizerPtr = std::shared_ptr<BufferFusionOptimizer>;
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BUFFER_FUSION_OPTIMIZER_H_
