/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/ub_fusion/buffer_fusion_optimizer.h"
#include "common/fe_log.h"
#include "common/op_info_common.h"

namespace fe {
namespace {
bool ComparePriority(const ge::NodePtr &left_node, const ge::NodePtr &right_node) {
  return left_node->GetOpDesc()->GetId() < right_node->GetOpDesc()->GetId();
}
}

void BufferFusionOptimizer::Initialize(const ge::ComputeGraph &graph) {
  op_type_nodes_map_.clear();
  op_pattern_nodes_map_.clear();
  int64_t num = 0;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    node->GetOpDesc()->SetId(num++);
    if (NeedIgnoreOp(node, true)) {
      continue;
    }
    const std::string &op_type = node->GetType();
    if (op_type_nodes_map_.find(op_type) == op_type_nodes_map_.end()) {
      op_type_nodes_map_[op_type] = {};
    }
    op_type_nodes_map_[op_type].emplace_back(node);
    FE_LOGD("Added node [%s] for key type [%s].", node->GetName().c_str(), op_type.c_str());
    std::string op_pattern;
    if (!ge::AttrUtils::GetStr(node->GetOpDesc(), kOpPattern, op_pattern)) {
      FE_LOGD("Node[%s] get pattern unsuccessful.", node->GetName().c_str());
      continue;
    }
    if (op_pattern_nodes_map_.find(op_pattern) == op_pattern_nodes_map_.end()) {
      op_pattern_nodes_map_[op_pattern] = {};
    }
    op_pattern_nodes_map_[op_pattern].emplace_back(node);
    FE_LOGD("Add node[%s] for key pattern[%s].", node->GetName().c_str(), op_pattern.c_str());
  }
}

void BufferFusionOptimizer::GetHeadNodesByFusionPattern(const BufferFusionPattern &pattern,
                                                        std::vector<ge::NodePtr> &nodes) const {
  std::unordered_set<ge::NodePtr> node_set;
  int32_t head_type_cnt = 0;
  for (auto &head : pattern.GetHead()) {
    for (auto &head_type : head->types) {
      ++head_type_cnt;
      if (head->not_pattern && op_type_nodes_map_.find(head_type) != op_type_nodes_map_.end()) {
        for (const ge::NodePtr &node : op_type_nodes_map_.at(head_type)) {
          if (node_set.count(node) <= 0) {
            node_set.emplace(node);
            nodes.emplace_back(node);
          }
        }
      } else if (!head->not_pattern && op_pattern_nodes_map_.find(head_type) != op_pattern_nodes_map_.end()) {
        for (const ge::NodePtr &node : op_pattern_nodes_map_.at(head_type)) {
          if (node_set.count(node) <= 0) {
            node_set.emplace(node);
            nodes.emplace_back(node);
          }
        }
      }
    }
  }
  if (!nodes.empty() && (head_type_cnt > 1)) {
    std::sort(nodes.begin(), nodes.end(), ComparePriority);
  }
}
}

