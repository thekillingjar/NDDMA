/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/stage_partitioner.h"

#include <stack>
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "graph/utils/op_type_utils.h"
#include "base/err_msg.h"

namespace ge {
Status StagePartitioner::Partition() {
  GE_CHECK_NOTNULL(root_graph_);
  if (root_graph_->GetParentGraph() != nullptr) {
    return SUCCESS;
  }

  for (const auto &node : root_graph_->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    uint32_t level = 0;
    if (!AttrUtils::GetInt(op_desc, ATTR_STAGE_LEVEL, level)) {
      continue;
    }
    if (OpTypeUtils::IsDataNode(op_desc->GetType()) && node->GetInAllNodes().empty()) {
      continue;
    }
    GELOGD("original node %s for stage %u", node->GetName().c_str(), level);
    stage_nodes_[level].insert(node);
  }
  if (stage_nodes_.empty()) {
    GELOGI("Graph %s does not set stage_level, it is not_changed.", root_graph_->GetName().c_str());
    return SUCCESS;
  }

  GE_DUMP(root_graph_, "BeforeStagePartition");
  if (SplitStageLevel() != SUCCESS) {
    GELOGE(FAILED, "[Split][GraphStage] for graph %s failed.", root_graph_->GetName().c_str());
    return FAILED;
  }

  GE_ASSERT_SUCCESS(SplitTailStage());

  if (StagePartition() != SUCCESS) {
    GELOGE(FAILED, "[Stage][Partition] for graph %s failed.", root_graph_->GetName().c_str());
    return FAILED;
  }

  root_graph_->TopologicalSorting([](const NodePtr &a, const NodePtr &b) -> bool {
    uint32_t a_level = UINT32_MAX;
    (void)AttrUtils::GetInt(a->GetOpDesc(), ATTR_STAGE_LEVEL, a_level);
    uint32_t b_level = UINT32_MAX;
    (void)AttrUtils::GetInt(b->GetOpDesc(), ATTR_STAGE_LEVEL, b_level);
    return a_level < b_level;
  });
  if (root_graph_->TopologicalSorting() != GRAPH_SUCCESS) {
    GELOGE(FAILED, "[Call][TopologicalSorting] for graph %s after stage partition failed, "
           "maybe stage_level was not set correctly.", root_graph_->GetName().c_str());
    return FAILED;
  }
  GE_DUMP(root_graph_, "AfterStagePartition");
  return SUCCESS;
}

Status StagePartitioner::SplitStageLevel() {
  std::stack<NodePtr> nodes;
  std::unordered_set<NodePtr> visited_stage_nodes;
  for (auto &stage : stage_nodes_) {
    uint32_t cur_stage_level = stage.first;
    const auto &cur_stage_nodes = stage.second;
    for (const auto &marked_node : cur_stage_nodes) {
      nodes.push(marked_node);
    }
    visited_stage_nodes.clear();
    while (!nodes.empty()) {
      auto node = nodes.top();
      nodes.pop();
      GE_CHECK_NOTNULL(node->GetOpDesc());
      for (const auto &in_node : node->GetInAllNodes()) {
        if (visited_stage_nodes.count(in_node) != 0) {
          continue;
        }
        uint32_t tmp_level = cur_stage_level;
        (void)AttrUtils::GetInt(in_node->GetOpDesc(), ATTR_STAGE_LEVEL, tmp_level);
        if (tmp_level != cur_stage_level) {
          continue;
        }
        if (!AttrUtils::SetInt(in_node->GetOpDesc(), ATTR_STAGE_LEVEL, cur_stage_level)) {
          REPORT_INNER_ERR_MSG("E19999", "Set Attr %s on node %s failed.",
                            ATTR_STAGE_LEVEL.c_str(), in_node->GetName().c_str());
          GELOGE(INTERNAL_ERROR, "[Set][Attr] %s on node %s failed.",
                 ATTR_STAGE_LEVEL.c_str(), in_node->GetName().c_str());
          return INTERNAL_ERROR;
        }
        GELOGD("Mark stage_level node %s, stage_level=%u", in_node->GetName().c_str(), cur_stage_level);
        if (OpTypeUtils::IsDataNode(in_node->GetType()) && in_node->GetInAllNodes().empty()) {
          GELOGD("skip data node %s for stage %u", in_node->GetName().c_str(), cur_stage_level);
          continue;
        }
        nodes.push(in_node);
      }
      visited_stage_nodes.emplace(node);
    }
    for (const auto &node : visited_stage_nodes) {
      stage.second.insert(node);
    }
  }

  return SUCCESS;
}

Status StagePartitioner::SplitTailStage() {
  auto tail_stage = stage_nodes_.rbegin()->first + 1U;
  for (const auto &node : root_graph_->GetDirectNode()) {
    if ((OpTypeUtils::IsDataNode(node->GetType()) && node->GetInAllNodes().empty()) ||
        (node->GetType() == ge::NETOUTPUT)) {
      continue;
    }
    if (!ge::AttrUtils::HasAttr(node->GetOpDesc(), ge::ATTR_STAGE_LEVEL)) {
      GELOGI("Set node %s as tail stage %u node", node->GetName().c_str(), tail_stage);
      GE_ASSERT(AttrUtils::SetInt(node->GetOpDesc(), ATTR_STAGE_LEVEL, tail_stage));
      (void)stage_nodes_[tail_stage].insert(node);
    }
  }
  return ge::SUCCESS;
}

Status StagePartitioner::StagePartition() {
  for (const auto &stage : stage_nodes_) {
    const std::string &subgraph_name = "Subgraph_Level_" + std::to_string(stage.first);
    const auto &stage_subgraph = GraphUtils::BuildSubgraphWithNodes(root_graph_, stage.second, subgraph_name);
    if (stage_subgraph == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Build subgraph %s failed.", subgraph_name.c_str());
      GELOGE(FAILED, "[Build][Subgraph] %s failed.", subgraph_name.c_str());
      return FAILED;
    }
    if (!AttrUtils::SetInt(stage_subgraph, ATTR_STAGE_LEVEL, stage.first)) {
      REPORT_INNER_ERR_MSG("E19999", "Set attr %s on graph %s failed.", ATTR_STAGE_LEVEL.c_str(),
                        stage_subgraph->GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] %s on graph %s failed.", ATTR_STAGE_LEVEL.c_str(), stage_subgraph->GetName().c_str());
      return FAILED;
    }
    const auto &parent_node = stage_subgraph->GetParentNode();
    GE_CHECK_NOTNULL(parent_node);
    if (!AttrUtils::SetInt(parent_node->GetOpDesc(), ATTR_STAGE_LEVEL, stage.first)) {
      REPORT_INNER_ERR_MSG("E19999", "Set attr %s on node %s failed", ATTR_STAGE_LEVEL.c_str(),
                        parent_node->GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] %s on node %s failed", ATTR_STAGE_LEVEL.c_str(), parent_node->GetName().c_str());
      return FAILED;
    }
    GE_ASSERT(AttrUtils::SetBool(parent_node->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true));
  }

  return SUCCESS;
}
}  // namespace ge
