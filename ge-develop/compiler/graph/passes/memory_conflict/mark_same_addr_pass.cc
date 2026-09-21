/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/memory_conflict/mark_same_addr_pass.h"

namespace ge {
bool MarkSameAddrPass::IsNextNodeExpected(const ge::NodePtr &cur_node, const std::vector<std::string> &next_nodes,
                                          int32_t &out_anchor_idx) const {
  for (auto out_anchor : cur_node->GetAllOutDataAnchors()) {
    for (auto in_anchor : out_anchor->GetPeerInDataAnchors()) {
      auto dst_node = in_anchor->GetOwnerNode();
      if (std::count(next_nodes.begin(), next_nodes.end(), dst_node->GetType()) > 0) {
        out_anchor_idx = out_anchor->GetIdx();
        GELOGD("Current node is %s, next node is %s.", cur_node->GetName().c_str(), dst_node->GetName().c_str());
        return true;
      }
    }
  }
  return false;
}

Status MarkSameAddrPass::Run(ComputeGraphPtr graph) {
  GELOGD("MarkSameAddrPass begin.");
  GE_CHECK_NOTNULL(graph);
  if (graph->GetGraphUnknownFlag()) {
    GELOGD("Graph[%s] is unknown shape, do not need to set fixed addr attr.", graph->GetName().c_str());
    return SUCCESS;
  }

  int32_t out_anchor_idx = 0;
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    static const std::vector<std::string> next_nodes({STREAMSWITCH, LABELSWITCHBYINDEX});
    if (IsNextNodeExpected(node, next_nodes, out_anchor_idx)) {
      std::string tensor_name = op_desc->GetOutputNameByIndex(out_anchor_idx);
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ATTR_DYNAMIC_SHAPE_FIXED_ADDR, tensor_name);
      (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ATTR_DYNAMIC_SHAPE_FIXED_ADDR_INDEX, out_anchor_idx);
    }
  }
  GELOGD("MarkSameAddrPass end.");
  return SUCCESS;
}
}  // namespace ge
