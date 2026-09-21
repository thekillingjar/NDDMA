/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary_graph_builder.h"

#include "graph/utils/graph_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace ge {

ComputeGraphPtr BinaryGraphBuilder::BuildGraph(const std::vector<NodePtr> &nodes, const std::string &name) const {
  if (nodes.empty()) {
    GELOGE(ge::FAILED, "nodes is empty, no need to build graph:%s", name.c_str());
    return nullptr;
  }
  std::unordered_set<NodePtr> nodes_set(nodes.begin(), nodes.end());
  auto graph = GraphUtils::BuildGraphFromNodes(nodes_set, name);
  if (graph != nullptr) {
    RefreshNodeName(graph, name);
    GE_ASSERT_SUCCESS(graph->TopologicalSorting());
    auto netout_node = graph->GetOrUpdateNetOutputNode();
    GE_ASSERT_NOTNULL(netout_node);
    graph->SetOutputSize(static_cast<uint32_t>(netout_node->GetInDataNodesSize()));
    GE_ASSERT_SUCCESS(CopySubgraph(graph, nodes));
  }
  return graph;
}

Status BinaryGraphBuilder::CopySubgraph(const ComputeGraphPtr &graph, const std::vector<NodePtr> &nodes) const {
  for (const auto &node : nodes) {
    const auto &subgraph_names = node->GetOpDesc()->GetSubgraphInstanceNames();
    for (const auto &subgraph_name : subgraph_names) {
      const auto &src_subgraph = node->GetOwnerComputeGraph()->GetSubgraph(subgraph_name);
      GE_ASSERT_NOTNULL(src_subgraph, "node:%s subgraph is null, subgraph name:%s", node->GetName().c_str(), subgraph_name.c_str());
      auto dst_subgraph = ComGraphMakeShared<ComputeGraph>(src_subgraph->GetName());
      GE_ASSERT_NOTNULL(dst_subgraph);
      dst_subgraph->SetParentGraph(graph);
      std::map<ConstNodePtr, NodePtr> old_2_new_node;
      std::map<ConstOpDescPtr, OpDescPtr> old_2_new_op_desc;
      GE_ASSERT_SUCCESS(GraphUtils::CopyComputeGraph(src_subgraph, dst_subgraph, old_2_new_node, old_2_new_op_desc, 0), "copy %s of node:%s fail",
                        src_subgraph->GetName().c_str(), node->GetName().c_str());
      (void)graph->AddSubGraph(dst_subgraph);
      const auto &new_node = graph->FindNode(node->GetName());
      GE_ASSERT_NOTNULL(new_node, "node:%s does not exist", node->GetName().c_str());
      dst_subgraph->SetParentNode(new_node);
    }
  }
  return GRAPH_SUCCESS;
}

void BinaryGraphBuilder::RefreshNodeName(const ComputeGraphPtr &graph, const std::string &name) const {
  for (const NodePtr &node : graph->GetDirectNode()) {
    auto node_name = node->GetName();
    std::string prefix_name = name + "/";
    const size_t pos = node_name.find(prefix_name);
    if (pos != std::string::npos) {
      (void)node_name.erase(0, pos + prefix_name.size());
      node->GetOpDesc()->SetName(node_name);
    }
  }
}

Status BinaryGraphBuilder::GetIOMapping(BinaryGraphIOLinkage &io_link) const {
  GE_ASSERT_SUCCESS(GetIONodeMapping(io_link), "GetIOMapping failed! sliced graph:%s, remaining graph:%s",
                    io_link.sliced_graph->GetName().c_str(), io_link.remaining_graph->GetName().c_str());

  GE_ASSERT_SUCCESS(GetIOIdxMapping(io_link), "GetIOMapping failed! sliced graph:%s, remaining graph:%s",
                    io_link.sliced_graph->GetName().c_str(), io_link.remaining_graph->GetName().c_str());
  return GRAPH_SUCCESS;
}

Status BinaryGraphBuilder::GetIONodeMapping(BinaryGraphIOLinkage &io_link) const {
  for (const auto &node : io_link.infered_nodes) {
    std::list<std::pair<std::string, uint32_t>> peer_data;
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      peer_data.clear();
      const auto &peer_in_anchors = out_data_anchor->GetPeerInDataAnchorsPtr();
      (void)std::for_each(peer_in_anchors.begin(), peer_in_anchors.end(),
                          [io_link, &peer_data](const InDataAnchor *peer_in_anchor) {
                            if (std::count(io_link.infered_nodes.begin(), io_link.infered_nodes.end(),
                                peer_in_anchor->GetOwnerNode()) == 0) {
                              peer_data.emplace_back(peer_in_anchor->GetOwnerNode()->GetName(), peer_in_anchor->GetIdx());
                            }
                          });
      GE_ASSERT_TRUE(CheckPeerNodeIsValid(peer_data, io_link.uninfer_nodes),
                     "GetIONodeMapping failed! peer node is not in remaining graph");
      if (!peer_data.empty()) {
        auto it = io_link.binary_graph_mapping.find(node->GetName());
        if (it != io_link.binary_graph_mapping.end()) {
          (void)it->second.emplace(out_data_anchor->GetIdx(), peer_data);
        } else {
          OutIdxToInput out_to_in_idx;
          (void)out_to_in_idx.emplace(out_data_anchor->GetIdx(), peer_data);
          (void)io_link.binary_graph_mapping.emplace(node->GetName(), out_to_in_idx);
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

Status BinaryGraphBuilder::GetIOIdxMapping(BinaryGraphIOLinkage &io_link) const {
  auto netout_node = io_link.sliced_graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netout_node);
  for (const auto &in_data_anchor : netout_node->GetAllInDataAnchorsPtr()) {
    const auto out_idx = in_data_anchor->GetIdx();
    auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(out_data_anchor, 
                      "GetIOIdxMapping failed! netout's idx:%d out_data_anchor is null", out_idx);
    auto out_node_name = out_data_anchor->GetOwnerNode()->GetName();
    const auto out_node_idx = out_data_anchor->GetIdx();
    GE_ASSERT_SUCCESS(FindIOIdxMappingAndSet(io_link, out_node_name, out_node_idx, out_idx));
  }
  (void)DebugIOMapping(io_link);
  return GRAPH_SUCCESS;
}

Status BinaryGraphBuilder::FindIOIdxMappingAndSet(BinaryGraphIOLinkage &io_link, const std::string &out_node_name,
                                                  const int32_t out_node_idx, const int32_t out_idx) const {
  auto out_to_in_idx = io_link.binary_graph_mapping.find(out_node_name);
  GE_ASSERT_TRUE((out_to_in_idx != io_link.binary_graph_mapping.end()),
                 "FindIOIdxMappingAndSet failed! out_node_name:%s does not exist", out_node_name.c_str());
  auto peer_data = out_to_in_idx->second.find(static_cast<uint32_t>(out_node_idx));
  GE_ASSERT_TRUE((peer_data != out_to_in_idx->second.end()),
                 "FindIOIdxMappingAndSet failed! out_node_name:%s idx:%d does not exist", out_node_name.c_str(), out_node_idx);
  for (const auto &in_data_pair : peer_data->second) {
    auto node = io_link.remaining_graph->FindNode(in_data_pair.first);
    GE_ASSERT_NOTNULL(node, "FindIOIdxMappingAndSet failed! remaining garph node:%s does not exist", in_data_pair.first.c_str());
    GE_ASSERT_NOTNULL(node->GetInDataAnchor(static_cast<int32_t>(in_data_pair.second)), "node:%s in data anchor null", in_data_pair.first.c_str());
    auto in_data_anchor = node->GetInDataAnchor(static_cast<int32_t>(in_data_pair.second))->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(in_data_anchor, "FindIOIdxMappingAndSet failed! remaining garph node:%s has no in data",
                      in_data_pair.first.c_str());

    auto in_data_node = in_data_anchor->GetOwnerNode();
    if (IsReplaceNode(in_data_node)) {
      continue;
    }
    auto in_data_nodes = io_link.remaining_graph->GetInputNodes();
    auto it = std::find(in_data_nodes.begin(), in_data_nodes.end(), in_data_node);
    GE_ASSERT_TRUE((it != in_data_nodes.end()), "FindIOIdxMappingAndSet failed! in_data_node:%s does not exist",
                   in_data_node->GetName().c_str());

    auto in_idx = std::distance(in_data_nodes.begin(), it);
    // 临时将outidx设置到idx，保证output和input一一对应，io加入后需要改回in_idx
    GE_ASSERT_TRUE(AttrUtils::SetInt(in_data_node->GetOpDesc(), ATTR_NAME_INDEX, out_idx),
                   "set attr %s failed for node:%s", ATTR_NAME_INDEX.c_str(), in_data_node->GetName().c_str());
    io_link.out_idx_2_in_idxs.emplace_back(out_idx, in_idx);
  }
  return GRAPH_SUCCESS;
}

bool BinaryGraphBuilder::CheckPeerNodeIsValid(const std::list<std::pair<std::string, uint32_t>> &peer_data,
                                              const std::vector<NodePtr> &peer_nodes) const {
  for (const auto &pair_node : peer_data) {
    auto it = std::find_if(peer_nodes.begin(), peer_nodes.end(), [pair_node](const NodePtr &node) {
      return pair_node.first == node->GetName();
    });
    GE_ASSERT_TRUE((it != peer_nodes.end()), "Invalid peer node:%s is not in remaining graph",
                   pair_node.first.c_str());
  }
  return true;
}

Status BinaryGraphBuilder::ReplaceInputNode(BinaryGraphIOLinkage &io_link) const {
  auto netout_node = io_link.sliced_graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netout_node);
  auto in_data_nodes = io_link.remaining_graph->GetInputNodes();
  bool has_replace_node = false;
  for (const auto &io_pair : io_link.out_idx_2_in_idxs) {
    GE_ASSERT_NOTNULL(netout_node->GetInDataAnchor(static_cast<int32_t>(io_pair.first)));
    GE_ASSERT_NOTNULL(netout_node->GetInDataAnchor(static_cast<int32_t>(io_pair.first))->GetPeerOutAnchor());
    auto out_node = netout_node->GetInDataAnchor(static_cast<int32_t>(io_pair.first))->GetPeerOutAnchor()->GetOwnerNode();
    if (IsReplaceNode(out_node)) {
      auto dst_node = io_link.remaining_graph->FindNode(out_node->GetName());
      if (dst_node == nullptr) {
        dst_node = io_link.remaining_graph->AddNode(GraphUtils::CopyOpDesc(out_node->GetOpDesc()));
        GE_ASSERT_NOTNULL(dst_node);
      }
      auto replaced_node = in_data_nodes.at(static_cast<size_t>(io_pair.second));
      GE_ASSERT_SUCCESS(ReplaceNode(replaced_node, dst_node, io_link.remaining_graph));
      has_replace_node = true;
    }
  }
  if (!has_replace_node) {
    return GRAPH_SUCCESS;
  }
  GE_ASSERT_SUCCESS(UpdateNetOutNode(io_link));
  GE_ASSERT_SUCCESS(io_link.remaining_graph->TopologicalSorting());
  io_link.out_idx_2_in_idxs.clear();
  return GetIOIdxMapping(io_link);
}

Status BinaryGraphBuilder::UpdateNetOutNode(const BinaryGraphIOLinkage &io_link) const {
  std::vector<OutDataAnchorPtr> peer_out_data_anchors;
  std::vector<OutControlAnchorPtr> peer_out_ctrl_anchors;
  auto node_desc = MakeNetOutputDesc(io_link, peer_out_data_anchors, peer_out_ctrl_anchors);
  GE_ASSERT_NOTNULL(node_desc);
  GE_ASSERT_SUCCESS(RemoveOutputNode(io_link, peer_out_ctrl_anchors));
  GE_ASSERT_SUCCESS(AddNetOutputNodeWithLink(io_link, node_desc, peer_out_data_anchors, peer_out_ctrl_anchors));
  return GRAPH_SUCCESS;
}

OpDescPtr BinaryGraphBuilder::MakeNetOutputDesc(const BinaryGraphIOLinkage &io_link,
                                                std::vector<OutDataAnchorPtr> &peer_out_data_anchors,
                                                std::vector<OutControlAnchorPtr> &peer_out_ctrl_anchors) const {
  const std::string node_name = "Node_Output";
  OpDescPtr net_output_desc = ComGraphMakeShared<OpDesc>(node_name, NETOUTPUT);
  GE_ASSERT_NOTNULL(net_output_desc);
  auto netout_node = io_link.sliced_graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netout_node);
  for (const auto &in_data_anchor : netout_node->GetAllInDataAnchorsPtr()) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor, "GetPeerOutAnchor failed! out_node_idx:%d does not exist", in_data_anchor->GetIdx());
    auto out_node = peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(out_node, "GetOwnerNode failed! out_node_idx:%d does not exist", in_data_anchor->GetIdx());
    GE_ASSERT_NOTNULL(out_node->GetOpDesc());
    if (IsReplaceNode(out_node)) {
      peer_out_ctrl_anchors.push_back(out_node->GetOutControlAnchor());
      continue;
    }
    ge::GeTensorDesc tensor = out_node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(peer_out_anchor->GetIdx()));
    GE_ASSERT_SUCCESS(net_output_desc->AddInputDesc(tensor));
    peer_out_data_anchors.push_back(peer_out_anchor);
  }
  return net_output_desc;
}

Status BinaryGraphBuilder::AddNetOutputNodeWithLink(const BinaryGraphIOLinkage &io_link,
                                                    const OpDescPtr &net_output_desc,
                                                    const std::vector<OutDataAnchorPtr> &peer_out_data_anchors,
                                                    const std::vector<OutControlAnchorPtr> &peer_out_ctrl_anchors) const {
  const NodePtr net_output = io_link.sliced_graph->AddNode(net_output_desc);
  GE_ASSERT_NOTNULL(net_output);
  io_link.sliced_graph->SetNetOutputNode(net_output);
  for (size_t i = 0U; i < peer_out_data_anchors.size(); i++) {
    GE_ASSERT_SUCCESS(GraphUtils::AddEdge(peer_out_data_anchors[i],
                      net_output->GetInDataAnchor(static_cast<int32_t>(i))));
  }
  for (size_t i = 0U; i < peer_out_ctrl_anchors.size(); i++) {
    GE_ASSERT_SUCCESS(GraphUtils::AddEdge(peer_out_ctrl_anchors[i], net_output->GetInControlAnchor()));
  }
  io_link.sliced_graph->SetOutputSize(static_cast<uint32_t>(net_output->GetInDataNodesSize()));
  return GRAPH_SUCCESS;
}

Status BinaryGraphBuilder::RemoveOutputNode(const BinaryGraphIOLinkage &io_link,
                                            std::vector<OutControlAnchorPtr> &peer_out_ctrl_anchors) const {
  auto netout_node = io_link.sliced_graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netout_node);
  GE_ASSERT_NOTNULL(netout_node->GetInControlAnchor());
  for (auto it : netout_node->GetInControlAnchor()->GetPeerOutControlAnchors()) {
    peer_out_ctrl_anchors.push_back(it);
  }
  ge::NodeUtils::UnlinkAll(*netout_node);
  GE_ASSERT_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(io_link.sliced_graph, netout_node));
  return GRAPH_SUCCESS;
}

Status BinaryGraphBuilder::MergeSameInputNode(BinaryGraphIOLinkage &io_link) const {
  std::unordered_map<int64_t, std::vector<int64_t>> out_2_in_map;
  (void)std::for_each(io_link.out_idx_2_in_idxs.begin(), io_link.out_idx_2_in_idxs.end(),
                      [&out_2_in_map](const std::pair<int64_t, int64_t> &idx_pair) {
                        out_2_in_map[idx_pair.first].push_back(idx_pair.second);
                      });
  auto in_data_nodes = io_link.remaining_graph->GetInputNodes();
  for (const auto &it : out_2_in_map) {
    if (it.second.size() > 1UL) {
      auto in_data_node = in_data_nodes.at(static_cast<size_t>(it.second.at(0)));
      for (size_t i = 1; i < it.second.size(); ++i) {
        auto replaced_node = in_data_nodes.at(static_cast<size_t>(it.second.at(i)));
        GE_ASSERT_SUCCESS(ReplaceNode(replaced_node, in_data_node, io_link.remaining_graph));
      }
    }
  }
  io_link.out_idx_2_in_idxs.clear();
  return GetIOIdxMapping(io_link);
}

Status BinaryGraphBuilder::ReplaceNode(const NodePtr &src_node, const NodePtr &dst_node, ComputeGraphPtr graph) const {
  GE_ASSERT_NOTNULL(src_node->GetOutDataAnchor(0));
  auto peer_node_anchors = src_node->GetOutDataAnchor(0)->GetPeerInDataAnchors();
  GE_ASSERT_SUCCESS(GraphUtils::ReplaceEdgeSrc(src_node->GetOutDataAnchor(0), peer_node_anchors.at(0), dst_node->GetOutDataAnchor(0)),
                    "ReplaceNode failed! ReplaceEdgeSrc failed, src:%s dst:%s", src_node->GetName().c_str(), dst_node->GetName().c_str());
  GE_ASSERT_SUCCESS(GraphUtils::RemoveNodeWithoutRelink(graph, src_node),
                    "ReplaceNode failed! RemoveNodeWithoutRelink failed graph:%s, node:%s", graph->GetName().c_str(), src_node->GetName().c_str());
  return GRAPH_SUCCESS;
}

Status BinaryGraphBuilder::SetInputNodeDesc(const BinaryGraphIOLinkage &io_link) const {
  auto in_data_nodes = io_link.remaining_graph->GetInputNodes();
  for (const auto &in_data_node : in_data_nodes) {
    GE_ASSERT_NOTNULL(in_data_node->GetOutDataAnchor(0));
    auto peer_in_anchors = in_data_node->GetOutDataAnchor(0)->GetPeerInDataAnchorsPtr();
    auto in_node = peer_in_anchors.at(0)->GetOwnerNodeBarePtr();
    auto in_node_desc = in_node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(peer_in_anchors.at(0)->GetIdx()));
    auto op_desc = in_data_node->GetOpDesc();
    GE_ASSERT_SUCCESS(op_desc->UpdateInputDesc(0U, in_node_desc),
                      "SetInputNodeDesc failed: update tensor_desc for %s failed.", in_data_node->GetName().c_str());
    GE_ASSERT_SUCCESS(op_desc->UpdateOutputDesc(0U, in_node_desc),
                      "SetInputNodeDesc failed: update tensor_desc for %s failed.", in_data_node->GetName().c_str());
  }
  return GRAPH_SUCCESS;
}

bool BinaryGraphBuilder::IsReplaceNode(const NodePtr &node) const {
  if (node->GetType() == VARIABLE || node->GetType() == VARIABLEV2 || node->GetType() == CONSTANT ||
      node->GetType() == CONSTANTOP) {
    return true;
  }
  return false;
}

Status BinaryGraphBuilder::DebugIOMapping(const BinaryGraphIOLinkage &io_link) const {
  auto netout_node = io_link.sliced_graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netout_node);
  auto in_data_nodes = io_link.remaining_graph->GetInputNodes();
  GELOGI("io map size:%zu", io_link.out_idx_2_in_idxs.size());
  for (const auto &io_idx_pair : io_link.out_idx_2_in_idxs) {
    GE_ASSERT_NOTNULL(netout_node->GetInDataAnchor(static_cast<int32_t>(io_idx_pair.first)));
    GE_ASSERT_NOTNULL(netout_node->GetInDataAnchor(static_cast<int32_t>(io_idx_pair.first))->GetPeerOutAnchor());
    auto out_name = netout_node->GetInDataAnchor(static_cast<int32_t>(io_idx_pair.first))->GetPeerOutAnchor()->GetOwnerNode()->GetName();
    const auto out_idx = netout_node->GetInDataAnchor(static_cast<int32_t>(io_idx_pair.first))->GetPeerOutAnchor()->GetIdx();
    GE_ASSERT_NOTNULL(in_data_nodes.at(static_cast<size_t>(io_idx_pair.second))->GetOutDataAnchor(0));
    auto in_name = in_data_nodes.at(static_cast<size_t>(io_idx_pair.second))->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName();
    const auto in_idx = in_data_nodes.at(static_cast<size_t>(io_idx_pair.second))->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx();
    int64_t in_idx_attr;
    (void)AttrUtils::GetInt(in_data_nodes.at(static_cast<size_t>(io_idx_pair.second))->GetOpDesc(), ATTR_NAME_INDEX, in_idx_attr);

    std::stringstream info;
    info << "out_name:" << out_name 
         << ", out_idx:" << out_idx 
         << ", netout_idx:" << io_idx_pair.first 
         << ", in_name:" << in_name 
         << ", in_idx:" << in_idx 
         << ", data_idx:" << io_idx_pair.second
         << ", data_idx_attr:" << in_idx_attr;
  GELOGI("GetIOIdxMapping:%s", info.str().c_str());
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge