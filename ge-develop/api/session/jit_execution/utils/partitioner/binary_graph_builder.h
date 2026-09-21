/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

#include "graph/compute_graph.h"
#include "ge/ge_api_types.h"

namespace ge {
// <output_idx, <input_node_name, input_idx>>
using OutIdxToInput = std::unordered_map<uint32_t, std::list<std::pair<std::string, uint32_t>>>;
// <output_node_name, <output_idx, <input_node_name, input_idx>>>
using GraphMapping = std::unordered_map<std::string, OutIdxToInput>;
struct BinaryGraphIOLinkage {
  ComputeGraphPtr sliced_graph{nullptr};
  ComputeGraphPtr remaining_graph{nullptr};
  std::vector<NodePtr> infered_nodes;
  std::vector<NodePtr> uninfer_nodes;
  GraphMapping binary_graph_mapping;
  std::vector<std::pair<int64_t, int64_t>> out_idx_2_in_idxs;
};


class BinaryGraphBuilder {
 public:
  BinaryGraphBuilder() noexcept = default;
  ~BinaryGraphBuilder() = default;
  BinaryGraphBuilder(const BinaryGraphBuilder&) = delete;
  BinaryGraphBuilder& operator=(const BinaryGraphBuilder&) = delete;
  BinaryGraphBuilder(BinaryGraphBuilder&&) = delete;
  BinaryGraphBuilder& operator=(BinaryGraphBuilder&&) = delete;

  ComputeGraphPtr BuildGraph(const std::vector<NodePtr> &nodes, const std::string &name) const;
  Status GetIOMapping(BinaryGraphIOLinkage &io_link) const;
  // preserve the variable and const nodes in the original graph and repalce data node in remaining graph
  Status ReplaceInputNode(BinaryGraphIOLinkage &io_link) const;
  // merge the output nodes corresponding to nodes with single output and multiple references in remaining graph
  Status MergeSameInputNode(BinaryGraphIOLinkage &io_link) const;
  Status SetInputNodeDesc(const BinaryGraphIOLinkage &io_link) const;
 private:
  void RefreshNodeName(const ComputeGraphPtr &graph, const std::string &name) const;
  Status GetIONodeMapping(BinaryGraphIOLinkage &io_link) const;
  Status GetIOIdxMapping(BinaryGraphIOLinkage &io_link) const;
  Status FindIOIdxMappingAndSet(BinaryGraphIOLinkage &io_link, const std::string &out_node_name,
                                const int32_t out_node_idx, const int32_t out_idx) const;
  Status DebugIOMapping(const BinaryGraphIOLinkage &io_link) const;
  bool CheckPeerNodeIsValid(const std::list<std::pair<std::string, uint32_t>> &peer_data,
                            const std::vector<NodePtr> &peer_nodes) const;
  bool IsReplaceNode(const NodePtr &node) const;
  Status ReplaceNode(const NodePtr &src_node, const NodePtr &dst_node, ComputeGraphPtr graph) const;
  Status UpdateNetOutNode(const BinaryGraphIOLinkage &io_link) const;
  OpDescPtr MakeNetOutputDesc(const BinaryGraphIOLinkage &io_link,
                              std::vector<OutDataAnchorPtr> &peer_out_data_anchors,
                              std::vector<OutControlAnchorPtr> &peer_out_ctrl_anchors) const;
  Status AddNetOutputNodeWithLink(const BinaryGraphIOLinkage &io_link,
                                  const OpDescPtr &net_output_desc,
                                  const std::vector<OutDataAnchorPtr> &peer_out_data_anchors,
                                  const std::vector<OutControlAnchorPtr> &peer_out_ctrl_anchors) const;
  Status RemoveOutputNode(const BinaryGraphIOLinkage &io_link, std::vector<OutControlAnchorPtr> &peer_out_ctrl_anchors) const;
  Status CopySubgraph(const ComputeGraphPtr &graph, const std::vector<NodePtr> &nodes) const;
 private:
  GraphMapping binary_graph_mapping_;
};

} // namspace ge