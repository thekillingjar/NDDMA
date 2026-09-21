/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/fusion/graph_rewriter.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ir_definitions_recover.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "common/types.h"
#include <queue>
#include "fusion_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
namespace fusion {
namespace {
constexpr int32_t kDataOutAnchorIdx = 0U;
std::vector<OutDataAnchorPtr> GetAllOutDataAnchors(const ComputeGraphPtr &compute_graph) {
  std::vector<OutDataAnchorPtr> output_anchors;
  for (const auto &node : compute_graph->GetDirectNode()) {
    if (strcmp(node->GetTypePtr(), NETOUTPUT) != 0) {
      continue;
    }
    for (const auto &in_data_anchor : node->GetAllInDataAnchorsPtr()) {
      if (in_data_anchor == nullptr) {
        continue;
      }
      output_anchors.emplace_back(in_data_anchor->GetPeerOutAnchor());
    }
  }
  return output_anchors;
}

std::vector<NodePtr> CollectNodes(const ComputeGraphPtr &replacement) {
  std::vector<NodePtr> all_nodes_except_io;
  for (const auto &node : replacement->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetTypePtr()) || (strcmp(node->GetTypePtr(), NETOUTPUT) == 0)) {
      continue;
    }
    all_nodes_except_io.emplace_back(node);
  }
  return all_nodes_except_io;
}

Status CollectIO(const ComputeGraphPtr &replacement, std::map<int32_t, NodePtr> &index_2_data_node,
                 std::map<int32_t, OutDataAnchorPtr> &idx_2_out_anchor) {
  for (const auto &node : replacement->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetTypePtr())) {
      int32_t index = 0;
      AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, index);
      GE_ASSERT_TRUE(index >= 0);
      index_2_data_node[index] = node;
    } else if (strcmp(node->GetTypePtr(), NETOUTPUT) == 0) {
      for(const auto in_anchor :node->GetAllInDataAnchorsPtr()) {
        if (in_anchor == nullptr) {
          continue;
        }
        idx_2_out_anchor[in_anchor->GetIdx()] = in_anchor->GetPeerOutAnchor();
      }
    }
  }
  return SUCCESS;
}

Status RecordOriginOpName(const std::vector<NodePtr> &nodes_before_fusion, const std::vector<NodePtr> &replacement_nodes) {
  for (const auto &node : replacement_nodes) {
    fe::GraphPassUtil::RecordOriginalNames(nodes_before_fusion, node);
  }
  return SUCCESS;
}

Status InheritedOriginAttrAndOpName(const InnerSubgraphBoundary &subgraph, const std::vector<NodePtr> &replacement_nodes) {
  const auto &nodes_before_fusion = subgraph.GetNodes();
  RecordOriginOpName(nodes_before_fusion, replacement_nodes);

  // todo need to replace with attr register mechanism
  fe::GraphPassUtil::InheritAttrFromOriNodes(nodes_before_fusion, replacement_nodes);

  auto tmp_opdesc = MakeShared<OpDesc>();
  GE_ASSERT_NOTNULL(tmp_opdesc);
  auto target_graph = subgraph.GetOwnerGraph();
  auto tmp_nodes = target_graph->FuseNodeKeepTopo(nodes_before_fusion, {tmp_opdesc});
  GE_ASSERT_TRUE(tmp_nodes.size() == 1U);

  const auto all_attrs = tmp_opdesc->GetAllAttrs();
  for (const auto &node : replacement_nodes) {
    for (const auto &attr : all_attrs) {
      GE_ASSERT_SUCCESS(node->GetOpDesc()->TrySetAttr(attr.first, attr.second));
    }
  }
  GE_ASSERT_SUCCESS(target_graph->RemoveNode(tmp_nodes[0]));
  return SUCCESS;
}

Status MakeSureReplacementNodeNameUnique(const ComputeGraphPtr &replacement) {
  thread_local int64_t new_node_count = 0;
  for (const auto &node : replacement->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetTypePtr()) || (strcmp(node->GetTypePtr(), NETOUTPUT) == 0)) {
      continue;
    }
    auto name = node->GetName();
    node->GetOpDesc()->SetName(name + to_string(new_node_count++));
  }
  return SUCCESS;
}

bool IsReplacementValid(const ComputeGraphPtr &replace_compute_graph, const SubgraphBoundary &boundary,
                        std::stringstream &invalid_reason) {
  size_t replace_input_size = 0U;
  for (const auto &node : replace_compute_graph->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetTypePtr())) {
      replace_input_size++;
    }
  }
  std::vector<SubgraphInput> all_inputs_of_subgraph;
  GE_ASSERT_SUCCESS(boundary.GetAllInputs(all_inputs_of_subgraph));
  if (replace_input_size != all_inputs_of_subgraph.size()) {
    invalid_reason << "Replacement input size: " << replace_input_size
                   << " not equal with Boundary input size: " << all_inputs_of_subgraph.size();
    return false;
  }

  std::vector<SubgraphOutput> all_outputs_of_subgraph;
  GE_ASSERT_SUCCESS(boundary.GetAllOutputs(all_outputs_of_subgraph));
  auto output_anchors = GetAllOutDataAnchors(replace_compute_graph);
  if (output_anchors.size() != all_outputs_of_subgraph.size()) {
    invalid_reason << "Replacement output size: " << output_anchors.size()
                   << " not equal with Boundary output size: " << all_outputs_of_subgraph.size();
    return false;
  }

  if (replace_compute_graph->TopologicalSorting() != SUCCESS) {
    invalid_reason << "Failed to topo sorting, There may exist cycle on replacement graph.";
    return false;
  }
  return true;
};

Status PruneUnusedNodes(const ComputeGraphPtr &target_graph, const std::vector<NodePtr> &subgraph_nodes) {
  Status remove_status = SUCCESS;
  std::unordered_set<NodePtr> removed_nodes;
  while (remove_status != NOT_CHANGED) {
    remove_status = NOT_CHANGED;
    for (const auto &node : subgraph_nodes) {
      if (node->GetOutDataNodesSize() == 0U && removed_nodes.insert(node).second) {
        GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(node, {}));
        GE_ASSERT_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(target_graph, node));
        GE_ASSERT_SUCCESS(node->ClearOwnerGraph(nullptr));
        GELOGI("[REPLACE]Remove node [%s][%s] from graph [%s] success", node->GetNamePtr(), node->GetTypePtr(), target_graph->GetName().c_str());
        remove_status = SUCCESS;
      }
    }
  }
  return SUCCESS;
}

Status RelinkSubgraphIO(const InnerSubgraphBoundary &subgraph, const std::map<int32_t, NodePtr> &r_index_2_data_node,
                        const std::map<int32_t, OutDataAnchorPtr> &r_index_2_output_data_anchor) {
  GE_ASSERT_TRUE(r_index_2_data_node.size() == subgraph.GetInputNum());
  for (const auto &idx_2_data : r_index_2_data_node) {
    const auto input_index = idx_2_data.first;
    const auto replace_data_node = idx_2_data.second;
    const auto &producer_out_data_anchor = subgraph.GetInputProducerOutAnchor(input_index);
    GE_ASSERT_NOTNULL(producer_out_data_anchor);
    const auto producer_owner_node = producer_out_data_anchor->GetOwnerNode();
    std::vector<int32_t> producer_node_output_map(producer_owner_node->GetAllOutDataAnchorsSize(), -1);
    producer_node_output_map[producer_out_data_anchor->GetIdx()] = kDataOutAnchorIdx;
    GE_ASSERT_SUCCESS(
        GraphUtils::ReplaceNodeDataAnchors(producer_owner_node, replace_data_node, {}, producer_node_output_map),
        "Failed to replace node data anchors from [%s][%s] to [%s][%s]", replace_data_node->GetNamePtr(),
        replace_data_node->GetTypePtr(), producer_owner_node->GetNamePtr(), producer_owner_node->GetTypePtr());
  }
  // replacement outputs re connection
  GE_ASSERT_TRUE(r_index_2_output_data_anchor.size() == subgraph.GetOutputNum());
  for (const auto &idx_2_out_anchor : r_index_2_output_data_anchor) {
    const auto output_index = idx_2_out_anchor.first;
    const auto r_output_data_anchor = idx_2_out_anchor.second;
    const auto t_output_data_anchor = subgraph.GetOutputOutAnchor(output_index);
    const auto r_output_owner_node = r_output_data_anchor->GetOwnerNode();
    if (OpTypeUtils::IsDataNode(r_output_owner_node->GetTypePtr())) {
      int64_t input_index = -1;
      (void)AttrUtils::GetInt(r_output_owner_node->GetOpDesc(), ATTR_NAME_INDEX, input_index);
      GE_ASSERT_TRUE(input_index != -1, "Invalid input node of replacement graph which without index attr");
      auto input_producer_out_anchor = subgraph.GetInputProducerOutAnchor(input_index);
      GE_ASSERT_NOTNULL(input_producer_out_anchor, "Failed to get input producer by input index: %ld", input_index);
      auto peer_in_anchors = t_output_data_anchor->GetPeerInDataAnchors();
      for (const auto &peer_in_anchor : peer_in_anchors) {
        GE_ASSERT_SUCCESS(GraphUtils::RemoveEdge(t_output_data_anchor, peer_in_anchor));
        GE_ASSERT_SUCCESS(GraphUtils::AddEdge(input_producer_out_anchor, peer_in_anchor));
      }
    } else {
      std::vector<int32_t> r_node_output_map(r_output_owner_node->GetAllOutDataAnchorsSize(), -1);
      r_node_output_map[r_output_data_anchor->GetIdx()] = t_output_data_anchor->GetIdx();
      GE_ASSERT_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(r_output_owner_node, t_output_data_anchor->GetOwnerNode(),
                                                           {}, r_node_output_map),
                        "Failed to replace node data anchors from [%s][%s] to [%s][%s]",
                        t_output_data_anchor->GetOwnerNode()->GetNamePtr(),
                        t_output_data_anchor->GetOwnerNode()->GetTypePtr(), r_output_owner_node->GetNamePtr(),
                        r_output_owner_node->GetTypePtr());
    }
  }
  return SUCCESS;
}

Status ReplaceSubgraph(const ComputeGraphPtr &target_graph, const InnerSubgraphBoundary &subgraph,
                       const ComputeGraphPtr &replacement) {
  std::string unsupport_fuse_reason;
  // todo check subgraph is valid
  GE_ASSERT_TRUE(target_graph->IsSupportFuse(subgraph.GetNodes(), unsupport_fuse_reason),
                 unsupport_fuse_reason.c_str());

  const std::vector<NodePtr> r_nodes_except_io = CollectNodes(replacement);
  GE_ASSERT_SUCCESS(InheritedOriginAttrAndOpName(subgraph, r_nodes_except_io));
  GE_ASSERT_SUCCESS(MakeSureReplacementNodeNameUnique(replacement));

  std::map<int32_t, NodePtr> r_index_2_data_node;
  std::map<int32_t, OutDataAnchorPtr> r_index_2_output_data_anchor;
  GE_ASSERT_SUCCESS(CollectIO(replacement, r_index_2_data_node, r_index_2_output_data_anchor));

  const auto &subgraph_nodes = subgraph.GetNodes();
  GE_ASSERT_SUCCESS(GraphUtils::MoveNodesToGraphAfterTargetNode(
      target_graph, subgraph_nodes.front(), replacement));

  // todo cycle search on replacement
  GE_ASSERT_SUCCESS(RelinkSubgraphIO(subgraph, r_index_2_data_node, r_index_2_output_data_anchor));

  // mv all ctrl from match to replacement io boundary, after that still need cycle search?
  if (!r_nodes_except_io.empty()) {
    GE_ASSERT_SUCCESS(GraphUtils::InheritExecutionOrder(r_nodes_except_io, subgraph_nodes, target_graph));
  }
  // isolate and remove node in match
  GE_ASSERT_SUCCESS(PruneUnusedNodes(target_graph, subgraph_nodes));
  return SUCCESS;
};
} // namespace

Status SubgraphRewriter::Replace(const SubgraphBoundary &subgraph, const Graph &replacement) {
  InnerSubgraphBoundary inner_boundary;
  std::string boundary_invalid_reason;
  GE_ASSERT_SUCCESS(inner_boundary.Init(subgraph, boundary_invalid_reason), boundary_invalid_reason.c_str());

  const auto replacement_compute_graph = GraphUtilsEx::GetComputeGraph(replacement);
  GE_ASSERT_NOTNULL(replacement_compute_graph);
  GE_ASSERT_GRAPH_SUCCESS(ge::RecoverIrDefinitions(replacement_compute_graph), "Failed to recover ir definitions");
  std::stringstream invalid_reason;
  if (!IsReplacementValid(replacement_compute_graph, subgraph, invalid_reason)) {
    GELOGE(FAILED, "[REPLACE][INVALID] replacement[%s], Reason: %s", replacement_compute_graph->GetName().c_str(),
           invalid_reason.str().c_str());
    return FAILED;
  }

  // copy replacement
  ComputeGraphPtr replacement_backup = MakeShared<ComputeGraph>("replacment");
  GE_ASSERT_NOTNULL(replacement_backup);
  GE_ASSERT_SUCCESS(GraphUtils::CopyComputeGraph(replacement_compute_graph, replacement_backup));

  return ReplaceSubgraph(inner_boundary.GetOwnerGraph(), inner_boundary, replacement_backup);
}

Status SubgraphRewriter::Replace(const SubgraphBoundary &subgraph, Graph &&replacement) {
  return Replace(subgraph, replacement);
}
} // namespace fusion
}  // namespace ge
