/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/global_step_insert_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/passes/pass_utils.h"
#include "graph/ge_context.h"

namespace ge {
NodePtr GlobalStepInsertPass::InsertOp(ComputeGraphPtr &compute_graph,
                                       const std::string &node_type,
                                       const std::string &node_name,
                                       const std::vector<GeTensorDesc> &input_list,
                                       const std::vector<GeTensorDesc> &output_list) const {
  OpDescPtr op_desc = MakeShared<OpDesc>(node_name, node_type);
  GE_IF_BOOL_EXEC(op_desc == nullptr,
                  REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
                  GELOGE(FAILED, "[New][OpDesc] failed");
                  return nullptr);

  for (auto &input_desc : input_list) {
    graphStatus graph_status = op_desc->AddInputDesc(input_desc);
    if (graph_status != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add input desc to op:%s(%s) failed",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Add][InputDesc] to op:%s(%s) failed",
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return nullptr;
    }
  }

  for (auto &output_desc : output_list) {
    graphStatus graph_status = op_desc->AddOutputDesc(output_desc);
    if (graph_status != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Add output desc to op:%s(%s) failed",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Add][OutputDesc] to op:%s(%s) failed, ret:%u.",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), graph_status);
      return nullptr;
    }
  }

  GE_IF_BOOL_EXEC(compute_graph == nullptr, GELOGE(FAILED, "[Check][Param] compute_graph is nullptr"); return nullptr);
  NodePtr node = compute_graph->AddNode(op_desc);
  GE_IF_BOOL_EXEC(node == nullptr,
    REPORT_INNER_ERR_MSG("E19999", "Add node:%s(%s) to graph:%s failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), compute_graph->GetName().c_str());
    GELOGE(FAILED, "[Add][Node] %s(%s) to graph:%s failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), compute_graph->GetName().c_str());
    return nullptr);

  GELOGI("Insert op success, name:%s, type:%s.", node_name.c_str(), node_type.c_str());
  return node;
}

Status GlobalStepInsertPass::Run(ComputeGraphPtr compute_graph) {
  if (ge::GetContext().GetHostExecFlag()) {
    GELOGD("No need to insert global_step when exec on host");
    return SUCCESS;
  }

  NodePtr output_node = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  if (output_node == nullptr) {
    GELOGD("Node type %s can't be found in graph %u", NETOUTPUT, compute_graph->GetGraphID());
    return SUCCESS;
  }

  if (compute_graph->GetParentGraph() != nullptr) {
    GELOGD("Subgraph %s no need global step variable.", compute_graph->GetName().c_str());
    return SUCCESS;
  }

  NodePtr exist_node = compute_graph->FindNode(NODE_NAME_GLOBAL_STEP);
  if (exist_node != nullptr) {
    GELOGD("Node %s already exist, no need add.", NODE_NAME_GLOBAL_STEP.c_str());
    return SUCCESS;
  }
  // set global step tensor desc
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_UINT64);
  std::vector<GeTensorDesc> input_desc_list = {};
  std::vector<GeTensorDesc> output_desc_list = {tensor_desc};
  NodePtr global_step = InsertOp(compute_graph, VARIABLE, NODE_NAME_GLOBAL_STEP,
                                 input_desc_list, output_desc_list);
  if (global_step == nullptr) {
    GELOGE(FAILED, "[Insert][Op] to graph:%s failed.", compute_graph->GetName().c_str());
    return FAILED;
  }

  // add ctrl edges
  graphStatus add_ret = GraphUtils::AddEdge(global_step->GetOutControlAnchor(), output_node->GetInControlAnchor());
  if (add_ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                      global_step->GetName().c_str(), global_step->GetType().c_str(),
                      output_node->GetName().c_str(), output_node->GetType().c_str());
    GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
           global_step->GetName().c_str(), global_step->GetType().c_str(),
           output_node->GetName().c_str(), output_node->GetType().c_str());
    return FAILED;
  }
  GELOGD("Add global_step to netoutput edge in graph %u success", compute_graph->GetGraphID());
  return SUCCESS;
}

REG_PASS_OPTION("GlobalStepInsertPass").LEVELS(OoLevel::kO0);
}  // namespace ge
