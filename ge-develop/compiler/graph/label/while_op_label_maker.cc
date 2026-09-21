/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/label/while_op_label_maker.h"

#include "framework/common/util.h"
#include "framework/common/types.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/utils/graph_utils.h"

namespace ge {
constexpr uint8_t kCondOutputNum = 1U;
constexpr uint8_t kCondOutputIndex = 0U;
constexpr uint8_t kCondBranchIndex = 0U;
constexpr uint8_t kBodyBranchIndex = 1U;

/**
 * @ingroup ge
 * @brief Make label node to functional call.
 * @param [in/out] label_index: serial id for whole graph.
 * @return: 0 for success / others for fail
 */
Status WhileOpLabelMaker::Run(uint32_t &label_index) {
  GE_CHECK_NOTNULL(parent_node_);
  GE_CHECK_NOTNULL(parent_graph_);

  OpDescPtr while_desc = parent_node_->GetOpDesc();
  GE_CHECK_NOTNULL(while_desc);

  std::string cond_name = while_desc->GetSubgraphInstanceName(kCondBranchIndex);
  std::string body_name = while_desc->GetSubgraphInstanceName(kBodyBranchIndex);
  if (cond_name.empty() || body_name.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) cond subgraph index:%d or body subgraph index:%d name is empty, "
                       "check invalid", while_desc->GetName().c_str(), while_desc->GetType().c_str(),
                       kCondBranchIndex, kBodyBranchIndex);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Node: %s has invalid subgraph, cond branch: %s, body branch: %s.",
           while_desc->GetName().c_str(), cond_name.c_str(), body_name.c_str());
    return FAILED;
  }

  ComputeGraphPtr cond_graph = parent_graph_->GetSubgraph(cond_name);
  ComputeGraphPtr body_graph = parent_graph_->GetSubgraph(body_name);
  GE_CHECK_NOTNULL(cond_graph);
  GE_CHECK_NOTNULL(body_graph);

  const uint32_t cond_enter_index = label_index++;
  const uint32_t body_enter_index = label_index++;
  const uint32_t body_leave_index = label_index++;
  const std::string cond_enter_name = parent_node_->GetName() + "/CondLabelSet";        // rtLabelSet
  const std::string cond_active_name = parent_node_->GetName() + "/CondStreamActive";   // rtStreamActive
  const std::string cond_leave_name = parent_node_->GetName() + "/LabelSwitch";         // rtLabelSwitchByIndex
  const std::string body_enter_name = parent_node_->GetName() + "/EnterLabelSet";       // rtLabelSet
  const std::string body_active_name = parent_node_->GetName() + "/EnterStreamActive";  // rtStreamActive
  const std::string goto_leave_name = parent_node_->GetName() + "/LabelGoto";           // rtLabelGoto
  const std::string body_leave_name = parent_node_->GetName() + "/LeaveLabelSet";       // rtLabelSet

  NodePtr cond_stream_active = AddStreamActive(cond_graph, cond_active_name);
  if (cond_stream_active == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add StreamActive node in graph:%s fail",
                      cond_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][StreamActive] in Subgraph:%s failed.", cond_graph->GetName().c_str());
    return FAILED;
  }

  if (AddLabelSetEnter(cond_graph, cond_enter_name, cond_enter_index, cond_stream_active) == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add LabelSetEnter node in graph:%s fail",
                      cond_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][LabelSetEnter] in Subgraph:%s failed.", cond_graph->GetName().c_str());
    return FAILED;
  }

  NodePtr body_stream_active = AddStreamActive(body_graph, body_active_name);
  if (body_stream_active == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add StreamActive node in graph:%s fail",
                      body_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][StreamActive] in Subgraph:%s failed.", body_graph->GetName().c_str());
    return FAILED;
  }

  if (AddLabelSetEnter(body_graph, body_enter_name, body_enter_index, body_stream_active) == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add LabelSetEnter node in graph:%s fail",
                      body_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][LabelSetEnter] in Subgraph:%s failed.", body_graph->GetName().c_str());
    return FAILED;
  }

  if (AddLabelGotoLeave(body_graph, goto_leave_name, cond_enter_index) == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add LabelGotoLeave node in graph:%s fail",
                      body_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][LabelGotoLeave] in Subgraph:%s failed.", body_graph->GetName().c_str());
    return FAILED;
  }

  if (AddLabelSetLeave(body_graph, body_leave_name, body_leave_index) == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add LabelSetLeave node in graph:%s fail",
                      body_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][LabelSetLeave] in Subgraph:%s failed.", body_graph->GetName().c_str());
    return FAILED;
  }

  NodePtr cond_out_node = cond_graph->FindFirstNodeMatchType(NETOUTPUT);
  GE_CHECK_NOTNULL(cond_out_node);
  OpDescPtr cond_out_desc = cond_out_node->GetOpDesc();
  GE_CHECK_NOTNULL(cond_out_desc);

  GeTensorDesc pred_desc = cond_out_desc->GetInputDesc(kCondOutputIndex);

  // false ==> 0 ==> switch_labels[0] ==> body_leave_index
  // true  ==> 1 ==> switch_labels[1] ==> body_enter_name
  const std::vector<uint32_t> switch_labels = {body_leave_index, body_enter_index};
  NodePtr switch_node = AddLabelSwitchLeave(cond_graph, cond_leave_name, pred_desc, switch_labels);
  if (switch_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Add LabelSwitchLeave node in graph:%s fail",
                      cond_graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][LabelSwitchLeave] in Subgraph:%s failed.", cond_graph->GetName().c_str());
    return FAILED;
  }

  // link Data input.
  const auto &all_in_data = cond_out_node->GetAllInDataAnchors();
  if (all_in_data.size() != kCondOutputNum) {
    GELOGE(FAILED, "[Check][Param] Node: %s Cond sbugraph output size:%zu should equal size:%u.",
           switch_node->GetName().c_str(), all_in_data.size(), kCondOutputNum);
    return FAILED;
  }

  InDataAnchorPtr in_anchor = all_in_data.at(kCondOutputIndex);
  GE_CHECK_NOTNULL(in_anchor);
  GE_CHECK_NOTNULL(in_anchor->GetPeerOutAnchor());
  if (GraphUtils::AddEdge(in_anchor->GetPeerOutAnchor(), switch_node->GetInDataAnchor(kCondOutputIndex)) != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add ctrl edge from %s to %s in graph:%s fail",
                      in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                      switch_node->GetName().c_str(), cond_graph->GetName().c_str());
    GELOGE(FAILED, "[Add][PredDataInput] to Node:%s failed.", switch_node->GetName().c_str());
    return FAILED;
  }

  GELOGI("Node: %s assign label success.", while_desc->GetName().c_str());
  return SUCCESS;
}

REGISTER_LABEL_MAKER(WHILE, WhileOpLabelMaker);
REGISTER_LABEL_MAKER(_WHILE, WhileOpLabelMaker);
REGISTER_LABEL_MAKER(STATELESSWHILE, WhileOpLabelMaker);
}  // namespace ge
