/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_variable_into_subgraph_pass.h"

#include "graph/utils/op_type_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
namespace ge {
namespace {
bool IsNodeWithoutSubgraph(const NodePtr &node) {
  return node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty();
}
bool IsUnknownShapePartitionedCall(const NodePtr &node, const ComputeGraphPtr &subgraph) {
  if (node->GetType() != PARTITIONEDCALL) {
    return false;
  }
  return GraphUtils::IsUnknownShapeGraph(subgraph);
}

bool IsSubgraphInput(const NodePtr &node, int32_t input_index) {
  int32_t parent_node_index = -1;
  return (node->GetType() == DATA) &&
         AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_node_index) &&
         (parent_node_index == input_index);
}

void CollectVarsInGraph(const NodePtr &var, const ComputeGraphPtr &subgraph,
                        std::unordered_map<std::string, NodePtr> &var_names_2_nodes) {
  const auto &node = subgraph->FindNode(var->GetName());
  if (node == nullptr) {
    return;
  }
  var_names_2_nodes[var->GetName()] = node;
}

NodePtr CreateVariableFrom(const NodePtr &var, const ComputeGraphPtr &subgraph,
                           const NodePtr &inner_data,
                           std::unordered_map<std::string, NodePtr> &var_names_2_nodes) {
  if (var_names_2_nodes[var->GetName()] != nullptr) {
    GELOGD("Found Var %s exists in graph %s", var->GetNamePtr(), subgraph->GetName().c_str());
    return var_names_2_nodes[var->GetName()];
  }

  const auto var_op_desc = var->GetOpDescBarePtr();
  auto op_desc = MakeShared<OpDesc>(*var_op_desc);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Failed to create var op desc from var node %s.", var->GetNamePtr());
    return nullptr;
  }

  NodePtr variable_ref_node = subgraph->InsertNode(inner_data, op_desc);
  GE_ASSERT_NOTNULL(variable_ref_node);
  var_names_2_nodes[var->GetName()] = variable_ref_node;
  GELOGI("Create %s name:[%s] in subgraph %s", var_op_desc->GetTypePtr(), var_op_desc->GetNamePtr(),
         subgraph->GetName().c_str());
  return variable_ref_node;
}

NodePtr FindSubgraphDataByIndex(int32_t parent_input_index, const ComputeGraphPtr &subgraph) {
  for (const auto &inner_data : subgraph->GetDirectNode()) {
    if (IsSubgraphInput(inner_data, parent_input_index)) {
      return inner_data;
    }
  }
  return nullptr;
}
}  // namespace
Status SplitVariableIntoSubgraphPass::Run(NodePtr &node) {
  if (!OpTypeUtils::IsVarLikeNode(node->GetType())) {
    return SUCCESS;
  }
  auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  const auto &peer_in_anchor_2_nodes = NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, 0);
  const auto &var_out_anchor = node->GetOutDataAnchor(0);
  for (const auto &peer_in_anchor_2_node : peer_in_anchor_2_nodes) {
    const auto &peer_in_node = peer_in_anchor_2_node.second;
    GE_ASSERT_NOTNULL(peer_in_node->GetOpDescBarePtr());
    if (IsNodeWithoutSubgraph(peer_in_node)) {
      continue;
    }
    auto parent_node_input_idx = peer_in_anchor_2_node.first->GetIdx();
    // handle variable/refdata connect to a node with subgraph(if/case/partitioncall)
    Status ret_of_copy_var = NOT_CHANGED;
    for (const auto &name : peer_in_node->GetOpDescBarePtr()->GetSubgraphInstanceNames()) {
      const auto &subgraph = root_graph->GetSubgraph(name);
      GE_ASSERT_NOTNULL(subgraph, "Failed to get subgraph %s from root graph %s.", name.c_str(),
                        root_graph->GetName().c_str());
      // UnknownShapePartitionedCall will expand after build, will cause same vars in root graph, here to skip
      if (IsUnknownShapePartitionedCall(peer_in_node, subgraph)) {
        GELOGD("Var node %s(%s), peer in node %s(%s) is unknown shape parititonedcall ,skip split into.",
               node->GetNamePtr(), node->GetTypePtr(), peer_in_node->GetNamePtr(), peer_in_node->GetTypePtr());
        break;
      }
      ret_of_copy_var = CopyVarToSubgraph(node, parent_node_input_idx, subgraph);
      if (ret_of_copy_var != NOT_CHANGED) {
        GE_ASSERT_SUCCESS(ret_of_copy_var, "Failed to copy var %s to subgraph %s", node->GetName().c_str(),
                          subgraph->GetName().c_str());
      }
    }

    if (ret_of_copy_var == NOT_CHANGED) {
      continue;
    }

    if (kWhileOpTypes.count(peer_in_node->GetType()) > 0U) {
      // keep input edge from var to while, because node input can not be empty
      auto while_out_anchor = peer_in_node->GetOutDataAnchor(parent_node_input_idx);
      GE_ASSERT_NOTNULL(while_out_anchor);
      for (auto peer_in_anchors : while_out_anchor->GetPeerInDataAnchors()) {
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(while_out_anchor, peer_in_anchors));
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(var_out_anchor, peer_in_anchors));
        GE_ASSERT_NOTNULL(peer_in_anchors->GetOwnerNodeBarePtr());
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(peer_in_node->GetOutControlAnchor(),
                                                    peer_in_anchors->GetOwnerNodeBarePtr()->GetInControlAnchor()));
        GELOGI("Move while %s %d output node %s input_idx: %d as output of var %s.", peer_in_node->GetNamePtr(),
               while_out_anchor->GetIdx(), peer_in_anchors->GetOwnerNodeBarePtr()->GetNamePtr(),
               peer_in_anchors->GetIdx(), node->GetNamePtr());
      }
    }
    AddRePassNodesWithInOut(node);
  }
  return SUCCESS;
}

Status SplitVariableIntoSubgraphPass::RefreshMultiDataRefDataNodeShape(const NodePtr &data_node,
                                                                       NodePtr &ref_data_node) const {
  const auto data_op_desc = data_node->GetOpDesc();
  GE_ASSERT_NOTNULL(data_op_desc);
  auto ref_data_op_desc = ref_data_node->GetOpDesc();
  GE_ASSERT_NOTNULL(ref_data_op_desc);
  GE_ASSERT_TRUE(data_op_desc->GetAllOutputsDescSize() ==
      ref_data_op_desc->GetAllOutputsDescSize());
  GE_ASSERT_TRUE(data_op_desc->GetAllOutputsDescSize() > 0UL);
  auto ref_output_tensor = ref_data_op_desc->MutableOutputDesc(0U);
  GE_ASSERT_NOTNULL(ref_output_tensor);
  ref_output_tensor->SetShape(data_op_desc->GetOutputDesc(0U).GetShape());
  ref_output_tensor->SetOriginShape(data_op_desc->GetOutputDesc(0U).GetOriginShape());
  GE_ASSERT_TRUE(data_op_desc->GetAllInputsSize() ==
      ref_data_op_desc->GetAllInputsSize());
  GE_ASSERT_TRUE(data_op_desc->GetAllInputsSize() > 0UL);
  auto ref_input_tensor = ref_data_op_desc->MutableInputDesc(0U);
  GE_ASSERT_NOTNULL(ref_input_tensor);
  ref_input_tensor->SetShape(data_op_desc->GetInputDesc(0U).GetShape());
  ref_input_tensor->SetOriginShape(data_op_desc->GetInputDesc(0U).GetOriginShape());
  return SUCCESS;
}

bool SplitVariableIntoSubgraphPass::IsMultiBatchSubgraph(const ComputeGraphPtr &subgraph) const {
  const auto parent_node = subgraph->GetParentNode();
  if (parent_node == nullptr) {
    return false;
  }
  if (parent_node->GetType() != CASE) {
    return false;
  }
  bool is_insert = false;
  (void)AttrUtils::GetBool(parent_node->GetOpDesc(), ATTR_INSERT_BY_MBATCH, is_insert);
  return is_insert;
}
/**
 *    data                        var_ref
 *     |      ========>              |
 *     A                             A
 *
 *
 * Create a var_ref has same name of var in if/case subgraph. To replace inner_data which match parent_input_index
 * @param var
 * @param parent_input_index
 * @param subgraph
 * @return
 */
Status SplitVariableIntoSubgraphPass::CopyVarToSubgraph(const NodePtr &var, int32_t parent_input_index,
                                                        const ComputeGraphPtr &subgraph) const {
  auto inner_data = FindSubgraphDataByIndex(parent_input_index, subgraph);
  if ((inner_data == nullptr) || (inner_data->GetOutDataNodesSize() == 0U)) {
    GELOGI("Can not find inner data %d of subgraph %s.", parent_input_index, subgraph->GetName().c_str());
    return NOT_CHANGED;
  }
  // 1.create or get var in subgraph
  std::unordered_map<std::string, NodePtr> var_names_2_nodes;
  CollectVarsInGraph(var, subgraph, var_names_2_nodes);
  auto var_ref_node = CreateVariableFrom(var, subgraph, inner_data, var_names_2_nodes);
  GE_ASSERT_NOTNULL(var_ref_node);

  // 2. mv all inner_data outputs data nodes as var outputs
  auto var_ref_out_anchor = var_ref_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(var_ref_out_anchor);
  auto inner_data_out_anchor = inner_data->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(inner_data_out_anchor);

  auto in_anchors_2_out_nodes = NodeUtils::GetOutDataNodesWithAnchorByIndex(*inner_data, 0);
  for (const auto &in_anchor_2_out_node : in_anchors_2_out_nodes) {
    // make out node as new var out
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(inner_data_out_anchor, in_anchor_2_out_node.first));
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(var_ref_out_anchor, in_anchor_2_out_node.first));

    GELOGD("Create var ref [%s][%s] in subgraph [%s], make node [%s] input_[%d] as its output.",
           var_ref_node->GetNamePtr(), var_ref_node->GetTypePtr(), subgraph->GetName().c_str(),
           in_anchor_2_out_node.second->GetNamePtr(), in_anchor_2_out_node.first->GetIdx());
  }
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(inner_data->GetOutControlAnchor(), var_ref_node->GetInControlAnchor()));
  // 只有refdata需要刷新shape，因为分档场景子图的Data的shape不同
  if ((var_ref_node->GetType() == REFDATA) && (IsMultiBatchSubgraph(subgraph))) {
    GE_ASSERT_SUCCESS(RefreshMultiDataRefDataNodeShape(inner_data, var_ref_node));
  }
  return SUCCESS;
}

REG_PASS_OPTION("SplitVariableIntoSubgraphPass").LEVELS(OoLevel::kO0);
}  // namespace ge
