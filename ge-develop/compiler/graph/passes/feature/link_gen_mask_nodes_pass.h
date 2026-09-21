/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_LINK_GEN_MASK_NODES_PASS_H_
#define GE_GRAPH_PASSES_LINK_GEN_MASK_NODES_PASS_H_

#include <map>
#include <string>
#include <vector>

#include "graph/graph.h"
#include "graph/passes/graph_pass.h"

namespace ge {
// Link all GenMask nodes using control edges.
class LinkGenMaskNodesPass : public GraphPass {
 public:
  LinkGenMaskNodesPass(const std::map<std::string, int32_t> &stream_max_parallel_num);
  ~LinkGenMaskNodesPass() override = default;
  LinkGenMaskNodesPass(const LinkGenMaskNodesPass &) = delete;
  LinkGenMaskNodesPass &operator=(const LinkGenMaskNodesPass &) = delete;

  Status Run(ComputeGraphPtr graph) override;

 private:
  bool AreAllInputsConst(const NodePtr &node) const;
  void GetAllGenMaskNodes(ComputeGraphPtr graph, std::vector<NodePtr> &gen_mask_nodes) const;
  Status GetGenMaskGroupSize(std::vector<NodePtr> &gen_mask_nodes, size_t &gen_mask_group_size) const;
  bool IsExpectedType(const NodePtr &node, const std::set<std::string> &expected_types) const;

  const std::map<std::string, int32_t> stream_max_parallel_num_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_LINK_GEN_MASK_NODES_PASS_H_
