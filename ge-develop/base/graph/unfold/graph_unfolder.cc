/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_unfolder.h"
#include <cinttypes>
#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/util.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "base/err_msg.h"
#include "graph/utils/graph_utils.h"

namespace gert {
namespace {
constexpr uint32_t kHybridSubgraphIndex = 0U;
constexpr uint32_t kHybridSubgraphRecursion = 32U;
const std::set<std::string> kHybridMergeInputSkipTypes{ge::STREAMACTIVE, ge::STREAMSWITCH, ge::CONSTANT, ge::CONSTANTOP};
bool IsFftsGraphNode(const ge::OpDesc &op_desc) {
  return op_desc.HasAttr(ge::ATTR_NAME_FFTS_SUB_GRAPH) || op_desc.HasAttr(ge::ATTR_NAME_FFTS_PLUS_SUB_GRAPH);
}

std::vector<ge::NodePtr> GetAllPartitioncallNodes(const ge::ComputeGraphPtr &root_graph) {
  std::vector<ge::NodePtr> partiticall_nodes;
  for (const auto& node : root_graph->GetDirectNode()) {
    if (node->GetType() == ge::PARTITIONEDCALL) {
      partiticall_nodes.emplace_back(node);
    }
  }
  return partiticall_nodes;
}

void SetStageLevel4SubgraphNode(const ge::NodePtr& parent_node, const ge::ComputeGraphPtr& sub_graph) {
  if (!parent_node->GetOpDesc()->HasAttr(ge::ATTR_STAGE_LEVEL)) {
    return;
  }
  int64_t stage_level = std::numeric_limits<int64_t>::max();
  if (ge::AttrUtils::GetInt(parent_node->GetOpDesc(), ge::ATTR_STAGE_LEVEL, stage_level)) {
    for (const auto &stage_node : sub_graph->GetAllNodes()) {
      GELOGD("Set ATTR_STAGE_LEVEL on node %s, stage_level="
             "%" PRId64 "", stage_node->GetName().c_str(), stage_level);
      (void)ge::AttrUtils::SetInt(stage_node->GetOpDesc(), ge::ATTR_STAGE_LEVEL, stage_level);
    }
  }
}

ge::Status AddSubgraphNode2Rootgraph(const ge::ComputeGraphPtr& sub_graph, const ge::ComputeGraphPtr& root_graph){
  for (auto &sub_node : sub_graph->GetDirectNode()) {
    auto sub_node_type = sub_node->GetType();
    if (sub_node_type == ge::DATA_TYPE || sub_node_type == ge::NETOUTPUT) {
      continue;
    }
    (void)root_graph->AddNode(sub_node);
    GE_ASSERT_SUCCESS(sub_node->SetOwnerComputeGraph(root_graph));
  }
  return ge::SUCCESS;
}

ge::Status GetAllDirNodeSubGraphs(const ge::ComputeGraphPtr graph, std::vector<ge::ComputeGraphPtr> &subgraphs) {
  for (const auto &node : graph->GetDirectNode()) {
    std::vector<ge::ComputeGraphPtr> node_subgraphs;
    GE_CHK_STATUS_RET(ge::NodeUtils::GetDirectSubgraphs(node, node_subgraphs),
                      "Get Subgraphs failed for node %s", node->GetName().c_str());
    for (auto &subgraph : node_subgraphs) {
      if (subgraph != nullptr) {
        subgraphs.push_back(subgraph);
      }
    }
  }
  return ge::SUCCESS;
}
}  // namespace

ge::Status GraphUnfolder::DoUnlinkDataAnchors(const ge::OutDataAnchorPtr &out_data_anchor,
                                              const ge::InDataAnchorPtr &in_data_anchor) {
  GE_CHK_GRAPH_STATUS_RET(
      out_data_anchor->Unlink(in_data_anchor), "[Invoke][Unlink] failed to unlink %s(%s):%d from %s(%s):%d",
      out_data_anchor->GetOwnerNode()->GetName().c_str(), out_data_anchor->GetOwnerNode()->GetType().c_str(),
      out_data_anchor->GetIdx(), in_data_anchor->GetOwnerNode()->GetName().c_str(),
      in_data_anchor->GetOwnerNode()->GetType().c_str(), in_data_anchor->GetIdx());

  GELOGD("Succeeded in unlinking %s:%d from %s:%d", out_data_anchor->GetOwnerNode()->GetName().c_str(),
         out_data_anchor->GetIdx(), in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetIdx());
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::DoLinkDataAnchors(const ge::OutDataAnchorPtr &out_data_anchor,
                                            const ge::InDataAnchorPtr &in_data_anchor) {
  GE_CHK_GRAPH_STATUS_RET(
      out_data_anchor->LinkTo(in_data_anchor), "[Invoke][LinkTo]Failed to link %s(%s):%d to %s(%s):%d",
      out_data_anchor->GetOwnerNode()->GetName().c_str(), out_data_anchor->GetOwnerNode()->GetType().c_str(),
      out_data_anchor->GetIdx(), in_data_anchor->GetOwnerNode()->GetName().c_str(),
      in_data_anchor->GetOwnerNode()->GetType().c_str(), in_data_anchor->GetIdx());

  GELOGD("Succeeded in linking %s:%d to %s:%d", out_data_anchor->GetOwnerNode()->GetName().c_str(),
         out_data_anchor->GetIdx(), in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetIdx());
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::UnfoldPartitionedCallSubgraph(const ge::ComputeGraphPtr &sub_graph,
                                                        ge::ComputeGraphPtr &merged_graph,
                                                        const ge::ComputeGraphPtr &root_graph,
                                                        const ge::NodePtr &node,
                                                        const uint32_t depth) {
  GE_ASSERT_NOTNULL(sub_graph);
  if (!sub_graph->GetGraphUnknownFlag()) {
    (void)merged_graph->AddNode(node);
    GE_ASSERT_SUCCESS(node->SetOwnerComputeGraph(merged_graph));
    GELOGI("[%s] Known shape partitioned call added to merged graph.", node->GetName().c_str());
    return ge::SUCCESS;
  }
  if (node->GetOpDesc()->HasAttr(ge::ATTR_STAGE_LEVEL)) {
    int64_t stage_level = std::numeric_limits<int64_t>::max();
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_STAGE_LEVEL, stage_level)) {
      for (const auto &stage_node : sub_graph->GetAllNodes()) {
        GELOGD("Set ATTR_STAGE_LEVEL on node %s, stage_level="
          "%" PRId64 "", stage_node->GetName().c_str(), stage_level);
        (void)ge::AttrUtils::SetInt(stage_node->GetOpDesc(), ge::ATTR_STAGE_LEVEL, stage_level);
      }
    }
  }
  GE_CHK_STATUS_RET(MergeInputNodes(*sub_graph),
      "[Invoke][MergeInputNodes][%s] Failed to merge data nodes for subgraph",
                    sub_graph->GetName().c_str());
  GE_CHK_STATUS_RET(MergeNetOutputNode(*sub_graph),
                    "[Invoke][MergeNetOutputNode][%s] Failed to merge net output nodes for subgraph",
                    sub_graph->GetName().c_str());
  GELOGD("[%s] Done merging subgraph inputs and outputs successfully", sub_graph->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(UnfoldSubgraph(root_graph, sub_graph, merged_graph, depth + 1U));
  GELOGD("[%s] Done merging subgraph. remove it from root graph", sub_graph->GetName().c_str());
  auto anchors = node->GetAllInDataAnchors();
  (void)std::for_each(anchors.begin(), anchors.end(), [](ge::InDataAnchorPtr &anchor)->void {
    return anchor->UnlinkAll();
  });
  root_graph->RemoveSubgraph(sub_graph->GetName());
  GELOGD("[%s] Done merging subgraph", sub_graph->GetName().c_str());
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::UnfoldControlNodeSubgraph(const std::vector<ge::ComputeGraphPtr> &subgraphs,
                                                    const ge::ComputeGraphPtr &root_graph,
                                                    const ge::NodePtr &node,
                                                    const uint32_t depth) {
  for (size_t i = 0UL; i < subgraphs.size(); i++) {
    GE_CHECK_NOTNULL(subgraphs[i]);
    if (!subgraphs[i]->GetGraphUnknownFlag()) {
      GELOGI("subgraph is known: %s, should skip unfold", subgraphs[i]->GetName().c_str());
      continue;
    }
    GELOGI("Start unfold subgraph graph: %s of node: %s",
        subgraphs[i]->GetName().c_str(), node->GetName().c_str());
    std::string merged_graph_name = subgraphs[i]->GetName() + "_merged_graph";
    ge::ComputeGraphPtr temp_graph = ge::MakeShared<ge::ComputeGraph>(merged_graph_name);
    GE_CHECK_NOTNULL(temp_graph);
    GE_ASSERT_SUCCESS(UnfoldSubgraph(root_graph, subgraphs[i], temp_graph, depth + 1U));
    GE_ASSERT_SUCCESS(MarkGraphNodeIndex(temp_graph));
    GE_ASSERT_SUCCESS(ge::NodeUtils::SetSubgraph(*node, static_cast<uint32_t>(i), temp_graph));
    root_graph->RemoveSubgraph(subgraphs[i]->GetName());
  }
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::UnfoldSubgraph(const ge::ComputeGraphPtr &root_graph, const ge::ComputeGraphPtr &origin_sub_graph,
                                         ge::ComputeGraphPtr &merged_graph,
                                         const uint32_t depth) {
  if (depth >= kHybridSubgraphRecursion) {
    GELOGE(ge::FAILED, "[Invoke][Unfold]There are too much recursion:%u > max:%u", depth, kHybridSubgraphRecursion);
    REPORT_INNER_ERR_MSG("E19999", "[Unfold]There are too much recursion:%u > max:%u", depth, kHybridSubgraphRecursion);
    return ge::FAILED;
  }
  GE_ASSERT_NOTNULL(root_graph);
  GE_ASSERT_NOTNULL(origin_sub_graph);
  const bool is_need_merge_subgraph =
      ((origin_sub_graph->GetParentNode() != nullptr) &&
      (origin_sub_graph->GetParentNode()->GetType() == ge::PARTITIONEDCALL));
  if (!is_need_merge_subgraph) {
    merged_graph->SetGraphUnknownFlag(root_graph->GetGraphUnknownFlag());
    merged_graph->SetGraphID(root_graph->GetGraphID());
    merged_graph->SetSessionID(root_graph->GetSessionID());
    merged_graph->SetNeedIteration(root_graph->GetNeedIteration());
    ge::GraphUtils::InheritOriginalAttr(origin_sub_graph, merged_graph);
  }
  for (const auto &node : origin_sub_graph->GetDirectNode()) {
    if (((node->GetType() == ge::DATA_TYPE) || (node->GetType() == ge::NETOUTPUT)) &&
        (is_need_merge_subgraph)) {
      continue;
    }
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::vector<ge::ComputeGraphPtr> subgraphs;
    GE_ASSERT_SUCCESS(ge::NodeUtils::GetDirectSubgraphs(node, subgraphs));
    if (op_desc->GetType() != ge::PARTITIONEDCALL) {
      if ((!subgraphs.empty()) && (!IsFftsGraphNode(*op_desc))) {
        GE_ASSERT_SUCCESS(UnfoldControlNodeSubgraph(subgraphs, root_graph, node, depth));
      }
      (void)merged_graph->AddNode(node);
      GE_ASSERT_SUCCESS(node->SetOwnerComputeGraph(merged_graph));
      GELOGI("[%s] Node added to merged graph.", op_desc->GetName().c_str());
      continue;
    }
    GE_ASSERT_TRUE(!subgraphs.empty());
    GE_ASSERT_SUCCESS(UnfoldPartitionedCallSubgraph(subgraphs[kHybridSubgraphIndex],
        merged_graph, root_graph, node, depth));
  }
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::MarkGraphNodeIndex(const ge::ComputeGraphPtr &merged_graph) {
  int64_t index = 0;
  int64_t pre_node_id = -1;
  for (const auto &node : merged_graph->GetDirectNode()) {
    const int64_t current_node_id = node->GetOpDesc()->GetId();
    GE_CHECK_LE(pre_node_id, current_node_id);
    pre_node_id = current_node_id;
    node->GetOpDesc()->SetId(index++);
  }
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::UnfoldSubgraphs(const ge::ComputeGraphPtr &root_graph, ge::ComputeGraphPtr &merged_graph) {
  merged_graph = ge::MakeShared<ge::ComputeGraph>(root_graph->GetName());
  GE_CHECK_NOTNULL(merged_graph);
  GE_ASSERT_SUCCESS(UnfoldSubgraph(root_graph, root_graph, merged_graph));
  GE_ASSERT_SUCCESS(MarkGraphNodeIndex(merged_graph));
  for (auto &remained_subgraph : root_graph->GetAllSubgraphs()) {
    const auto &parent_node = remained_subgraph->GetParentNode();
    GE_CHECK_NOTNULL(parent_node);
    remained_subgraph->SetParentGraph(parent_node->GetOwnerComputeGraph());
    GE_CHK_GRAPH_STATUS_RET(merged_graph->AddSubgraph(remained_subgraph),
                            "[Invoke][AddSubgraph]Failed to add subgraph [%s]", remained_subgraph->GetName().c_str());
    GELOGD("Adding subgraph [%s], parent node: %s to merged-graph.", remained_subgraph->GetName().c_str(),
           parent_node->GetName().c_str());
  }
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::UnfoldAllPartitioncallInPlace(const ge::ComputeGraphPtr &root_graph) {
  GELOGD("Start unfloder partitioncall node, graph[%s]", root_graph->GetName().c_str());
  uint32_t depth = 0U;
  GE_ASSERT_SUCCESS(UnfoldPartitioncallInPlace(root_graph, root_graph, depth));
  root_graph->TopologicalSorting();
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::UnfoldSubGraphPartitioncall(const ge::ComputeGraphPtr &root_graph,
  const ge::ComputeGraphPtr &sub_graph) {
  std::vector<ge::NodePtr> partiticall_nodes = GetAllPartitioncallNodes(sub_graph);
  if (partiticall_nodes.empty()) {
    GELOGI("No partitioncall node exists, sub_graph[%s]", sub_graph->GetName().c_str());
    return ge::SUCCESS;
  }
  for (auto &node : partiticall_nodes) {
    ge::ComputeGraphPtr partiticall_sub_graph = ge::NodeUtils::GetSubgraph(*node, kHybridSubgraphIndex);
    if (partiticall_sub_graph == nullptr) {
      GELOGW("Node[%s][%s] subgraph not exists.", node->GetNamePtr(), node->GetTypePtr());
      continue;
    }

    SetStageLevel4SubgraphNode(node, partiticall_sub_graph);
    GE_ASSERT_SUCCESS(AddSubgraphNode2Rootgraph(partiticall_sub_graph, sub_graph));
    GE_CHK_STATUS_RET(MergeInputNodes(*partiticall_sub_graph),
                      "[Invoke][MergeInputNodes][%s] Failed to merge data nodes for subgraph",
                      partiticall_sub_graph->GetName().c_str());
    GE_CHK_STATUS_RET(MergeNetOutputNode(*partiticall_sub_graph),
                      "[Invoke][MergeNetOutputNode][%s] Failed to merge net output nodes for subgraph",
                      partiticall_sub_graph->GetName().c_str());
    auto anchors = node->GetAllInDataAnchors();
    (void)std::for_each(anchors.begin(), anchors.end(), [](ge::InDataAnchorPtr &anchor)->void {
      return anchor->UnlinkAll();
    });
    root_graph->RemoveSubgraph(partiticall_sub_graph->GetName());
    GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveJustNode(sub_graph, node));
    GELOGD("[%s] Done merging partitioncall_subgraph", partiticall_sub_graph->GetName().c_str());
  }
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::UnfoldPartitioncallInPlace(const ge::ComputeGraphPtr &root_graph,
  const ge::ComputeGraphPtr &sub_graph, uint32_t depth) {
  GE_ASSERT_NOTNULL(sub_graph);
  // 超过最大深度，不报错仅跳过
  if (depth >= kHybridSubgraphRecursion) {
    GELOGW("Recursion depth %u exceeds max %u, skip unfolding graph: %s",
           depth, kHybridSubgraphRecursion, sub_graph->GetName().c_str());
    return ge::SUCCESS;
  }

  // 在递归处理其它子图
  std::vector<ge::ComputeGraphPtr> subgraphs_to_process;
  GE_ASSERT_SUCCESS(GetAllDirNodeSubGraphs(sub_graph, subgraphs_to_process));

  // 子图上的节点都没有子图，无需处理
  if (subgraphs_to_process.empty()) {
    GELOGI("Subgraph[%s] not has subgraphs.", sub_graph->GetName().c_str());
    return ge::SUCCESS;
  }

  for (const auto &subgraph : subgraphs_to_process) {
    GE_CHK_STATUS_RET(UnfoldPartitioncallInPlace(root_graph, subgraph, depth + 1U),
                      "Recurse Unfold part failed for subgraph %s", subgraph->GetName().c_str());
  }

  GE_ASSERT_SUCCESS(UnfoldSubGraphPartitioncall(root_graph, sub_graph));

  return ge::SUCCESS;
}

ge::Status GraphUnfolder::MergeInputNodes(ge::ComputeGraph &compute_graph) {
  const auto &wrapped_node = compute_graph.GetParentNode();
  std::set<ge::NodePtr> root_nodes;
  for (const auto &node : compute_graph.GetDirectNode()) {
    GE_CHK_STATUS_RET_NOLOG(MergeInputInData(node, wrapped_node, root_nodes));
  }

  // transfer in control edges to all root nodes
  for (const auto &root_node : root_nodes) {
    const auto &in_nodes = root_node->GetInAllNodes();
    const std::set<ge::NodePtr> in_node_set(in_nodes.begin(), in_nodes.end());
    for (const auto &in_control_node : wrapped_node->GetInControlNodes()) {
      if ((in_node_set.count(in_control_node) == 0U) &&
          (kHybridMergeInputSkipTypes.count(root_node->GetType()) == 0U)) {
        GELOGD("[%s] Restore control edge to [%s]", in_control_node->GetName().c_str(), root_node->GetName().c_str());
        GE_CHECK_NOTNULL(in_control_node->GetOutControlAnchor());
        (void) in_control_node->GetOutControlAnchor()->LinkTo(root_node->GetInControlAnchor());
      }
    }
  }

  wrapped_node->GetInControlAnchor()->UnlinkAll();
  return ge::SUCCESS;
}
ge::Status GraphUnfolder::CheckInputInData(const ge::NodePtr &node, std::set<ge::NodePtr> &root_nodes) {
  GE_CHECK_NOTNULL(node);
  if (node->GetType() != ge::DATA_TYPE) {
    if (node->GetInAllNodes().empty()) {
      (void)root_nodes.emplace(node);
    }
    return ge::SUCCESS;
  }
  return ge::FAILED;
}
ge::Status GraphUnfolder::MergeInputInData(const ge::NodePtr &node, const ge::NodePtr &wrapped_node,
                                           std::set<ge::NodePtr> &root_nodes) {
  if (CheckInputInData(node, root_nodes) == ge::SUCCESS) {
    return ge::SUCCESS;
  }
  const auto &data_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(data_op_desc);

  int32_t parent_index = 0;
  if (!ge::AttrUtils::GetInt(data_op_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(ge::FAILED, "[Invoke][GetInt] failed, node:[%s(%s)] attr:[%s]", data_op_desc->GetName().c_str(),
           data_op_desc->GetType().c_str(), ge::ATTR_NAME_PARENT_NODE_INDEX.c_str());
    REPORT_INNER_ERR_MSG("E19999", "GetInt failed, node:[%s(%s)] attr:[%s]", data_op_desc->GetName().c_str(),
                      data_op_desc->GetType().c_str(), ge::ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return ge::FAILED;
  }

  const auto &wrapped_node_in_anchor = wrapped_node->GetInDataAnchor(parent_index);
  GE_CHECK_NOTNULL(wrapped_node_in_anchor);
  const auto &src_out_anchor = wrapped_node_in_anchor->GetPeerOutAnchor();
  if ((src_out_anchor == nullptr) || (src_out_anchor->GetOwnerNode() == nullptr)) {
    return ge::SUCCESS;
  }
  wrapped_node_in_anchor->UnlinkAll();

  // link src to outputs of DataNode
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    for (auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_data_anchor);
      const auto &dst_node = peer_in_data_anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(dst_node);
      const auto &in_nodes = dst_node->GetInAllNodes();
      const bool is_data = std::all_of(in_nodes.begin(), in_nodes.end(), [](const ge::NodePtr &n) {
        return n->GetType() == ge::DATA;
      });
      if (is_data) {
        (void)root_nodes.emplace(dst_node);
      }
      GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(out_data_anchor, peer_in_data_anchor));
      GE_CHK_STATUS_RET_NOLOG(DoLinkDataAnchors(src_out_anchor, peer_in_data_anchor));
    }
  }
  auto control_anchor = node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(control_anchor);
  auto parent_control_anchor = src_out_anchor->GetOwnerNodeBarePtr()->GetOutControlAnchor();
  GE_CHECK_NOTNULL(parent_control_anchor);
  for (const auto &dst_anchor : control_anchor->GetPeerInControlAnchors()) {
    GE_CHECK_NOTNULL(dst_anchor);
    GE_ASSERT_SUCCESS(parent_control_anchor->LinkTo(dst_anchor));
  }
  // Data may have in control edges which should also be unlinked.
  ge::NodeUtils::UnlinkAll(*node);
  return ge::SUCCESS;
}

ge::Status GraphUnfolder::MergeNetOutputNode(ge::ComputeGraph &compute_graph) {
  const auto &parent_node = compute_graph.GetParentNode();
  const ge::NodePtr &net_output_node = compute_graph.FindFirstNodeMatchType(ge::NETOUTPUT);
  if (net_output_node == nullptr) {
    GELOGD("Graph has no netoutput no need to merge");
    return ge::SUCCESS;
  }
  const auto &net_output_desc = net_output_node->GetOpDesc();
  GE_CHECK_NOTNULL(net_output_desc);

  const auto all_in_nodes = net_output_node->GetInAllNodes();
  const auto all_out_nodes = parent_node->GetOutAllNodes();
  auto in_anchor = net_output_node->GetInControlAnchor();
  in_anchor->UnlinkAll();
  parent_node->GetOutControlAnchor()->UnlinkAll();

  for (const auto &in_data_anchor : net_output_node->GetAllInDataAnchors()) {
    GE_CHK_STATUS_RET_NOLOG(MergeNetOutputInData(parent_node, net_output_desc, in_data_anchor));
  }

  // transfer out control edges
  const std::set<ge::NodePtr> in_node_set(all_in_nodes.begin(), all_in_nodes.end());
  const std::set<ge::NodePtr> out_node_set(all_out_nodes.begin(), all_out_nodes.end());
  for (const auto &src_node : in_node_set) {
    GELOGD("[%s] process in node.", src_node->GetName().c_str());
    const auto &out_nodes = src_node->GetOutAllNodes();
    const std::set<ge::NodePtr> node_set(out_nodes.begin(), out_nodes.end());
    for (auto &dst_node : out_node_set) {
      if (node_set.count(dst_node) == 0U) {
        (void) src_node->GetOutControlAnchor()->LinkTo(dst_node->GetInControlAnchor());
        GELOGD("[%s] Restore control edge to [%s]", src_node->GetName().c_str(), dst_node->GetName().c_str());
      }
    }
  }

  return ge::SUCCESS;
}

ge::Status GraphUnfolder::MergeNetOutputInData(const ge::NodePtr &parent_node, const ge::OpDescPtr &net_output_desc,
                                               const ge::InDataAnchorPtr &in_data_anchor) {
  const auto &src_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(src_out_anchor);
  GE_CHECK_NOTNULL(src_out_anchor->GetOwnerNodeBarePtr());
  GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(src_out_anchor, in_data_anchor));

  const auto anchor_index = in_data_anchor->GetIdx();
  const auto &input_desc = net_output_desc->MutableInputDesc(static_cast<uint32_t>(anchor_index));
  if (input_desc == nullptr) {
    GELOGE(ge::INTERNAL_ERROR, "[Invoke][MutableInputDesc][%s(%s)] Failed to get input desc[%d]",
           net_output_desc->GetName().c_str(), net_output_desc->GetType().c_str(), anchor_index);
    REPORT_INNER_ERR_MSG("E19999", "[%s(%s)] Failed to get input desc[%d].", net_output_desc->GetName().c_str(),
                      net_output_desc->GetType().c_str(), anchor_index);
    return ge::INTERNAL_ERROR;
  }

  int32_t parent_index = 0;
  if (!ge::AttrUtils::GetInt(input_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGW("SubGraph: %s NetOutput input tensor %d, attr %s not found.", net_output_desc->GetName().c_str(),
           anchor_index, ge::ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return ge::SUCCESS;
  }

  const ge::OutDataAnchorPtr &parent_out_anchor = parent_node->GetOutDataAnchor(parent_index);
  GE_CHECK_NOTNULL(parent_out_anchor);
  for (ge::InDataAnchorPtr &dst_in_anchor : parent_out_anchor->GetPeerInDataAnchors()) {
    if (dst_in_anchor == nullptr) {
      continue;
    }

    GE_CHECK_NOTNULL(dst_in_anchor->GetOwnerNodeBarePtr());
    GE_CHK_STATUS_RET_NOLOG(DoUnlinkDataAnchors(parent_out_anchor, dst_in_anchor));
    GE_CHK_STATUS_RET_NOLOG(DoLinkDataAnchors(src_out_anchor, dst_in_anchor));
  }

  return ge::SUCCESS;
}

bool GraphUnfolder::IsGraphNeedUnfold(const ge::ComputeGraphPtr &root_graph) {
  for (const auto &node : root_graph->GetDirectNode()) {
    if (node->GetType() != ge::PARTITIONEDCALL) {
      continue;
    }
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);

    const auto &subgraph = ge::NodeUtils::GetSubgraph(*node, 0U);
    if ((subgraph != nullptr) && (!IsFftsGraphNode(*op_desc)) && IsGraphDynamicCompiled(subgraph)) {
      GELOGI("Graph: %s need to be unfolded.", root_graph->GetName().c_str());
      return true;
    }
  }
  return false;
}

bool GraphUnfolder::IsGraphDynamicCompiled(const ge::ComputeGraphPtr &graph) {
  if (graph->GetGraphUnknownFlag()) {  // We can believe non-default value 'true'
    return true;
  }
  const std::string kIsOwnerGraphKnown = "OwnerGraphIsUnknown";
  bool is_unknown = false;
  for (auto &node : graph->GetDirectNode()) {
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), kIsOwnerGraphKnown, is_unknown)) {
      return is_unknown;
    }
  }
  return ge::GraphUtils::IsUnknownShapeGraph(graph);
}

bool GraphUnfolder::IsDataNotNeedRefConst(const ge::NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  return IsDataNotNeedRefConst(node.get());
}

bool GraphUnfolder::IsDataNotNeedRefConst(const ge::Node *node) {
  GE_ASSERT_NOTNULL(node);
  if (!ge::OpTypeUtils::IsDataNode(node->GetType())) {
    GELOGD("Node %s is not DATA type.", node->GetName().c_str());
    return false;
  }
  const auto &parent_input_node = ge::NodeUtils::GetParentInput(*node);
  std::string const_type;
  if (!ge::NodeUtils::GetConstOpType(parent_input_node, const_type)) {
    GE_ASSERT_NOTNULL(parent_input_node);
    GELOGD("Parent input node %s is not const type.", parent_input_node->GetName().c_str());
    return false;
  }
  const auto &owner = parent_input_node->GetOwnerComputeGraph();
  bool dynamic_shape_partition = false;
  (void)ge::AttrUtils::GetBool(owner, ge::ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, dynamic_shape_partition);
  const bool dynamic_flag = (owner->GetGraphUnknownFlag()) || dynamic_shape_partition;
  if (dynamic_flag) {
    GELOGD("This is model input node: %s.", node->GetName().c_str());
    return true;
  }
  return false;
}
}  // namespace gert
