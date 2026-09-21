/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cascade_reshape_remove_pass.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils_ex.h"
#include "utils/auto_fuse_config.h"
#include "utils/graph_utils.h"

namespace {
constexpr const char *kReshapeOpType = "Reshape";
bool HasSingleReshapeSuccessor(const ge::NodePtr &node) {
  if (node->GetOutDataNodesSize() != 1U) {
    return false;
  }
  ge::NodePtr next_node = node->GetOutNodes().at(0);
  GE_ASSERT_NOTNULL(next_node);
  GE_ASSERT_NOTNULL(next_node->GetOpDescBarePtr());
  return next_node->GetOpDescBarePtr()->GetType() == kReshapeOpType;
}

ge::NodePtr GetCascadedReshapeTail(const ge::NodePtr &start_node) {
  if (start_node->GetOpDescBarePtr()->GetType() != kReshapeOpType) {
    return nullptr;
  }

  ge::NodePtr current_node = start_node;
  while (HasSingleReshapeSuccessor(current_node)) {
    ge::NodePtr next_node = current_node->GetOutNodes().at(0);
    current_node = next_node;
  }
  return current_node;
}
}  // namespace
namespace ge {
graphStatus CascadeReshapeRemovePass::Run(const ComputeGraphPtr &graph, bool &changed) const {
  auto direct_nodes = graph->GetDirectNode();
  std::unordered_set<ge::NodePtr> redundant_reshape_nodes;
  std::unordered_set<ge::NodePtr> processed_nodes;
  for (const auto &current_node : direct_nodes) {
    GE_ASSERT_NOTNULL(current_node);
    if ((current_node->GetOpDescBarePtr()->GetType() != kReshapeOpType) ||
        (processed_nodes.count(current_node)) > 0UL) {
      continue;
    }

    ge::NodePtr tail_node = GetCascadedReshapeTail(current_node);
    // 不是单输出级联reshape
    if (tail_node == nullptr || tail_node == current_node) {
      processed_nodes.emplace(current_node);
      continue;
    }
    GELOGD("Find cascade reshape chain: [%s, %s]", current_node->GetNamePtr(), tail_node->GetNamePtr());

    // 遍历从current_node到tail_node的前一个节点
    ge::NodePtr temp_node = current_node;
    while (temp_node != tail_node) {
      redundant_reshape_nodes.insert(temp_node);
      processed_nodes.insert(temp_node);
      temp_node = temp_node->GetOutNodes().at(0);  // 已校验过size
    }
    processed_nodes.insert(tail_node);
  }

  // 删除冗余Reshape
  for (const auto &node : redundant_reshape_nodes) {
    GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(node, {0}), "Failed to delete single useless Reshape node: %s",
                      node->GetNamePtr());
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, node), "[Remove][JustNode] failed, graph:%s, node:%s.",
                            graph->GetName().c_str(), node->GetNamePtr());
    GELOGD("Success to delete redundant cascaded Reshape node: %s", node->GetNamePtr());
    changed = true;
  }
  return SUCCESS;
}
}  // namespace ge
