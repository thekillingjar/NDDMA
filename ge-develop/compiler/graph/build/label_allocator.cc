/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/label_allocator.h"

#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/label/label_maker.h"
#include "graph/ge_context.h"

namespace ge {
LabelAllocator::LabelAllocator(const ComputeGraphPtr &graph) : compute_graph_(graph) {}

Status LabelAllocator::AssignFunctionalLabels() {
  if (compute_graph_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "check param compute_graph nullptr");
    GELOGE(INTERNAL_ERROR, "[Check][Param] ComputeGraph not set, Assign labels failed.");
    return INTERNAL_ERROR;
  }
  if (GetContext().GetHostExecFlag()) {
    GELOGD("graph %s is on host, do not need to assign functional labels", compute_graph_->GetName().c_str());
    return SUCCESS;
  }

  // Add label task for sub graph.
  GELOGD("AssignFunctionalLabels start: %s.", compute_graph_->GetName().c_str());
  std::set<NodePtr> functional_nodes;
  for (auto graph : compute_graph_->GetAllSubgraphs()) {
    if (!CollectFunctionalNode(graph, functional_nodes)) {
      return INTERNAL_ERROR;
    }
  }

  // Add label for functional op.
  uint32_t label_index = 0;
  for (auto node : functional_nodes) {
    LabelMakerPtr maker = LabelMakerFactory::Instance().Create(node->GetType(), compute_graph_, node);
    if (maker == nullptr) {
      REPORT_INNER_ERR_MSG("E19999", "Check Node:%s(%s) label maker not registed",
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Create][LabelMaker] Node: %s label maker not registed.", node->GetType().c_str());
      return INTERNAL_ERROR;
    }

    if (maker->Run(label_index) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) run label maker failed",
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Call][Run] Node: %s run label maker failed.", node->GetType().c_str());
      return INTERNAL_ERROR;
    }
  }

  (void)AttrUtils::SetInt(*compute_graph_, ATTR_MODEL_LABEL_NUM, label_index);
  GELOGI("AssignFunctionalLabels success, Num: %u.", label_index);
  return SUCCESS;
}

bool LabelAllocator::CollectFunctionalNode(ComputeGraphPtr &graph, std::set<NodePtr> &functional_nodes) const {
  if (graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "check param compute_graph nullptr");
    GELOGE(INTERNAL_ERROR, "[Check][Param] Sub ComputeGraph is null.");
    return false;
  }

  if (graph->GetGraphUnknownFlag()) {
    GELOGD("Graph[%s] is unknown graph, skip label allocator.", graph->GetName().c_str());
    return true;
  }

  NodePtr func_node = graph->GetParentNode();
  if (func_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Parent node not set, graph:%s", graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Node] Parent functional node not set: %s.", graph->GetName().c_str());
    return false;
  }

  if (func_node->GetOpDesc() != nullptr && (func_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH) ||
                                            func_node->GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH))) {
    GELOGD("Graph[%s] is ffts/ffts+ subgraph, skip label allocator.", graph->GetName().c_str());
    return true;
  }

  ComputeGraphPtr owner_graph = func_node->GetOwnerComputeGraph();
  if (owner_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "ComputeGraph owner not set in node:%s(%s), graph:%s",
                       func_node->GetName().c_str(), func_node->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Graph] ComputeGraph owner not set: %s.", func_node->GetName().c_str());
    return false;
  }

  if (owner_graph->GetGraphUnknownFlag()) {
    GELOGD("Graph[%s] is unknown graph, skip label allocator.", owner_graph->GetName().c_str());
    return true;
  }

  (void)functional_nodes.insert(func_node);  // unique functional node.
  return true;
}
}  // namespace ge
