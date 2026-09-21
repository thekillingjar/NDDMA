/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/variable_optimize/variable_ref_delete_op_pass.h"
#include <string>
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"
#include "common/checker.h"

namespace ge {
Status VariableRefDeleteOpPass::Run(ge::ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  std::unordered_set<std::string> all_var_names;
  auto root_graph = GraphUtils::FindRootGraph(graph);
  GE_CHECK_NOTNULL(root_graph);
  for (const auto &n : root_graph->GetAllNodes()) {
    if (OpTypeUtils::IsVarLikeNode(n->GetType())) {
      all_var_names.insert(n->GetName());
    }
  }
  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::string ref_var_src_var_name;
    bool is_variable_ref = (OpTypeUtils::IsVarLikeNode(node->GetOpDesc()->GetType())) &&
                           (ge::AttrUtils::GetStr(node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name));
    if (!is_variable_ref) {
      continue;
    }
    if (all_var_names.count(ref_var_src_var_name) == 0) {
      REPORT_INNER_ERR_MSG("E19999", "Can not find source variable[%s] of variable ref[%s], check invalid",
                         ref_var_src_var_name.c_str(), node->GetName().c_str());
      GELOGE(FAILED, "[Check][Param] Can not find source variable[%s] of variable ref[%s]",
             ref_var_src_var_name.c_str(), node->GetName().c_str());
      return FAILED;
    }
    Status ret = DealVariableRef(graph, node, ref_var_src_var_name);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Deal][VariableRef] [%s] in graph:%s failed", node->GetName().c_str(), graph->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status VariableRefDeleteOpPass::DealVariableRef(ge::ComputeGraphPtr &graph, ge::NodePtr &variable_ref,
                                                const std::string &ref_var_src_var_name) const {
  GE_CHECK_NOTNULL(variable_ref);
  auto inAnchor0 = variable_ref->GetInDataAnchor(0);
  if (inAnchor0 == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Node:%s(%s) has no input anchor, check invalid",
                       variable_ref->GetName().c_str(), variable_ref->GetType().c_str());
    GELOGE(FAILED, "[Get][InDataAnchor] failed, variable_ref [%s] no input", variable_ref->GetName().c_str());
    return FAILED;
  }
  GE_CHECK_NOTNULL(inAnchor0->GetPeerOutAnchor());
  // get the output index of the previous node connected to the variable_ref
  // prepare for refreshing address in build phase
  int32_t index = inAnchor0->GetPeerOutAnchor()->GetIdx();

  // get previous node of variable_ref
  NodePtr peer_node = inAnchor0->GetPeerOutAnchor()->GetOwnerNode();

  // add attr [REF_VAR_SRC_VAR_NAME] to the previous op output desc of the variable_ref
  auto op_desc = peer_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto out_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
  if (peer_node->GetType() == PARTITIONEDCALL) {
    for (const auto &netoutput_node : ge::NodeUtils::GetSubgraphOutputNodes(*peer_node)) {
      GE_CHECK_NOTNULL(netoutput_node->GetOpDesc());
      for (size_t i = 0U; i < netoutput_node->GetOpDesc()->GetAllInputsSize(); ++i) {
        if (netoutput_node->GetOpDesc()->MutableInputDesc(i) == nullptr) {
          continue;
        }
        int32_t idx = -1;
        (void) AttrUtils::GetInt(netoutput_node->GetOpDesc()->MutableInputDesc(i), ATTR_NAME_PARENT_NODE_INDEX, idx);
        if (idx == index) {
          // get previous node out_anchor of netoutput
          GE_ASSERT_NOTNULL(netoutput_node->GetInDataAnchor(i));
          const auto out_anchor = netoutput_node->GetInDataAnchor(i)->GetPeerOutAnchor();
          GE_CHECK_NOTNULL(out_anchor);
          GE_CHECK_NOTNULL(out_anchor->GetOwnerNode()->GetOpDesc());
          out_desc = out_anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(out_anchor->GetIdx());
          break;
        }
      }
    }
  }
  // 避免自引用
  if (ref_var_src_var_name != op_desc->GetName()) {
    bool is_set_str = ge::AttrUtils::SetStr(out_desc, REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);
    GE_ASSERT_TRUE(is_set_str, "[Set][Attr] %s to output:%d desc of op:%s(%s) failed", REF_VAR_SRC_VAR_NAME.c_str(),
                   index, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGI("[%s-%d]: add attr [REF_VAR_SRC_VAR_NAME: %s ] ", peer_node->GetName().c_str(), index,
           ref_var_src_var_name.c_str());
  }

  // remove variable_ref
  if (GraphUtils::IsolateNode(variable_ref, {0}) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Isolate node:%s(%s) failed",
                      variable_ref->GetName().c_str(), variable_ref->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Isolate][Node] name:%s, type:%s failed", variable_ref->GetName().c_str(),
           variable_ref->GetType().c_str());
    return FAILED;
  }
  if (GraphUtils::RemoveNodeWithoutRelink(graph, variable_ref) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                      variable_ref->GetName().c_str(), variable_ref->GetType().c_str(), graph->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Remove][Node] %s(%s) without relink in graph:%s failed",
           variable_ref->GetName().c_str(), variable_ref->GetType().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

REG_PASS_OPTION("VariableRefDeleteOpPass").LEVELS(OoLevel::kO3);
}  // namespace ge
