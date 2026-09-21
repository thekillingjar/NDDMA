/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_DATA_FLOW_PREPARE_PASS_H_
#define GE_GRAPH_PASSES_DATA_FLOW_PREPARE_PASS_H_

#include <unordered_set>
#include <map>
#include "graph/passes/graph_pass.h"

namespace ge {
class DataFlowPreparePass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph) override;
 private:
  static void SetMaxSize(const NodePtr &node);
  static Status SetHandle(const std::map<std::string, std::unordered_set<NodePtr>> &data_flow_ops_groups);
  static std::pair<NodePtr, int32_t> GetPeerOutNodeAndIndexByInputIndex(const NodePtr &node, const int32_t input_index);
  static Status GetResourceInputNode(const NodePtr &node, const int32_t in_idx, NodePtr &src_node);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_DATA_FLOW_PREPARE_PASS_H_
