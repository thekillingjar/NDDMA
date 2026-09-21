/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_SAME_TRANSDATA_BREADTH_FUSION_PASS_H_
#define GE_GRAPH_PASSES_SAME_TRANSDATA_BREADTH_FUSION_PASS_H_

#include <utility>
#include <vector>
#include <queue>
#include <stack>
#include "graph/passes/graph_pass.h"

namespace ge {
namespace {
enum class NodeType { kData, kNetOutput, kWrapperNode, kCast, kTransdata, kOthers};
struct LinkNode {
  InDataAnchorPtr in_anchor;
  OutDataAnchorPtr real_peer_out_anchor;
  NodeType node_type;
};
struct CompareInfo {
  std::string stream_label;
  std::set<std::string> in_ctrl_nodes;
  ConstGeTensorDescPtr input_tensor_desc;
  ConstGeTensorDescPtr output_tensor_desc;
};

using Path = std::vector<LinkNode>;
using Paths = std::vector<Path>;
using OrderedGraphToNodes = std::map<ComputeGraphPtr, std::map<uint32_t, NodePtr>, ComputeGraphCompareKey>;
using AnchorPairStack = std::stack<std::pair<OutDataAnchorPtr, OutDataAnchorPtr>>;
}

class SameTransdataBreadthFusionPass : public GraphPass {
 public:
  SameTransdataBreadthFusionPass() {}
  virtual ~SameTransdataBreadthFusionPass() {}
  graphStatus Run(ComputeGraphPtr graph) override;
 private:
  graphStatus CollectAllSubgraphDataNodesMap();
  graphStatus DoRun(ComputeGraphPtr graph);
  graphStatus RunForNode(OutDataAnchorPtr &head_out_anchor);

  graphStatus GetPathsToTransdata(const OutDataAnchorPtr &head_out_anchor, Paths &paths) const;
  graphStatus GetRealInAnchors(const OutDataAnchorPtr &real_out_anchor,
                               const OutDataAnchorPtr &out_anchor,
                               std::queue<Path> &path_queue,
                               const Path &path) const;
  graphStatus GetRealInAnchorsForWrapperNode(
      const InDataAnchorPtr &in_anchor, std::stack<OutDataAnchorPtr> &out_anchor_stack) const;
  graphStatus GetSubgraphDataOutAnchor(const ComputeGraphPtr &sub_graph, const int32_t wrapper_node_input_index,
                                       OutDataAnchorPtr &data_out_anchor) const;
  graphStatus GetRealInAnchorsForNetOutput(
      const OutDataAnchorPtr &real_out_anchor, const InDataAnchorPtr &in_anchor, const Path &path,
      std::stack<OutDataAnchorPtr> &out_anchor_stack) const;
  graphStatus FuseTransdata(Paths &paths);
  graphStatus GetSameTransdataPath(Paths &paths, std::vector<Paths> &same_transdata_paths_groups);
  graphStatus RemoveUnSupportedPath(Paths &paths_with_same_transdata) const;
  graphStatus GetCompareInfo(const Path &path, const LinkNode &link_node, CompareInfo &info);
  graphStatus UpdateTensorDesc(const Paths &paths_group, size_t keep_transdata_path_index);
  graphStatus UpdateTensorDescForConnectData(const GeTensorDesc &trans_out_tensor_desc,
                                             const LinkNode &link_node, std::stack<LinkNode> &link_node_stack) const;
  graphStatus UpdateTensorDescForConnectWrapper(const GeTensorDesc &trans_out_tensor_desc,
                                                const LinkNode &link_node, std::stack<LinkNode> &link_node_stack);
  graphStatus UpdateTensorDescForDiffGraph(const GeTensorDesc &trans_out_tensor_desc,
                                           const LinkNode &link_node);
  graphStatus ExtractTransdata(const Paths &paths_group, size_t keep_transdata_path_index) const;

  graphStatus CollectFusedInAnchors(const InDataAnchorPtr &in_anchor,
                                    const std::set<InDataAnchorPtr> &allowed_in_anchors,
                                    const NodeType head_next_type,
                                    std::vector<InDataAnchorPtr> &fused_anchors,
                                    std::vector<InDataAnchorPtr> &not_fused_anchors) const;
  graphStatus LinkHeadToTransdata(const Paths &paths_group,
                                  size_t keep_transdata_path_index) const;
  graphStatus DeleteTransdata(const Path &path) const;
  graphStatus AddNewPath(OutDataAnchorPtr &out_anchor,
                         OutDataAnchorPtr &new_out_anchor,
                         const std::set<InDataAnchorPtr> &allowed_in_anchors);
  graphStatus AddNewInputForWrapper(InDataAnchorPtr &wrapper_in_anchor,
                                    std::vector<InDataAnchorPtr> &fused_anchors,
                                    AnchorPairStack &out_anchor_pair_stack);
  graphStatus AddNewInputForNetOutput(InDataAnchorPtr &netout_in_anchor,
                                      std::vector<InDataAnchorPtr> &fused_anchors,
                                      AnchorPairStack &out_anchor_pair_stack) const;
  graphStatus AddNewPathToTransdataForDiffGraph(Paths &paths_group);

  void UpdateGraphNode(const ComputeGraphPtr &sub_graph, const uint32_t parent_index, NodePtr &node);

  ComputeGraphPtr root_graph_;
  OrderedGraphToNodes graph_nodes_;
  std::map<NodePtr, CompareInfo> node_to_info_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_SAME_TRANSDATA_BREADTH_FUSION_PASS_H_
