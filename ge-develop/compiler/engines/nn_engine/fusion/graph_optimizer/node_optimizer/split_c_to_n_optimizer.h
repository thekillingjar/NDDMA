/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPH_COMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_NODE_OPTIMIZER_SPLIT_C_TO_N_OPTIMIZER_H_
#define AIR_COMPILER_GRAPH_COMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_NODE_OPTIMIZER_SPLIT_C_TO_N_OPTIMIZER_H_
#include "graph_optimizer/node_optimizer/split_n_optimizer.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/node_optimize_pass_base.h"

namespace fe {
class SplitCToNOptimizer {
 public:
  Status SetFusionVirtualOp(const ge::ComputeGraph &graph) const;
 private:
  bool NeedSkip(const ge::ComputeGraph &graph, const ge::NodePtr &node, const ge::OpDescPtr &op_desc) const;
  bool CheckSplitDim(const ge::OpDescPtr &op_desc) const;
  bool CheckCommonCondition(const ge::ComputeGraph &graph, const ge::NodePtr &node,
                            const ge::OpDescPtr &op_desc) const;
  bool CheckAxis(const ge::OpDescPtr &op_desc) const;
  bool MeetAlignmentConditionFromNCHWTo5HD(const ge::OpDescPtr &op_desc) const;
  bool MeetDimNumConditionFromNDToNZ(const ge::OpDescPtr &op_desc) const;
  SplitNOptimizer checker_helper;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPH_COMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_NODE_OPTIMIZER_SPLIT_C_TO_N_OPTIMIZER_H_
