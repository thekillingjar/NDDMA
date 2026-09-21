/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/end_of_sequence_add_control_pass.h"

#include <string>
#include <vector>

#include "api/gelib/gelib.h"

namespace ge {

Status EndOfSequenceAddControlPass::Run(ComputeGraphPtr graph) {
  if (graph->GetParentGraph() != nullptr) {
    return SUCCESS;
  }
  const auto &end_of_sequence = graph->FindFirstNodeMatchType(ENDOFSEQUENCE);
  if (end_of_sequence == nullptr) {
    return SUCCESS;
  }

  GELOGI("EndOfSequenceAddControlPass begin.");
  std::vector<NodePtr> target_nodes;
  for (NodePtr &node : graph->GetDirectNode()) {
    // op_desc of node should not be null
    if (node->GetOpDesc()->HasAttr(ATTR_NAME_STREAM_LABEL) ||
        DNNEngineManager::GetInstance().IsStreamAssignSkip(node)) {
      continue;
    }
    // Save the nodes whose pre-nodes are all data-like node
    bool flag = false;
    for (const auto &in_node : node->GetInDataNodes()) {
      if (!DNNEngineManager::GetInstance().IsStreamAssignSkip(in_node)) {
        flag = true;
        break;
      }
    }
    if (flag) {
      continue;
    }
    target_nodes.push_back(node);
  }

  // Insert control edge
  for (const auto &node : target_nodes) {
    GELOGI("Add ctrl edge between %s and %s", end_of_sequence->GetName().c_str(), node->GetName().c_str());
    if (GraphUtils::AddEdge(end_of_sequence->GetOutControlAnchor(), node->GetInControlAnchor()) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add ctrl edge between %s and %s failed", end_of_sequence->GetName().c_str(),
                        node->GetName().c_str());
      GELOGE(FAILED, "[Add][CtrlEdge] between %s and %s failed", end_of_sequence->GetName().c_str(),
             node->GetName().c_str());
      return FAILED;
    }
  }

  GELOGI("EndOfSequenceAddControlPass end.");
  return SUCCESS;
}

REG_PASS_OPTION("EndOfSequenceAddControlPass").LEVELS(OoLevel::kO0);
}  // namespace ge
