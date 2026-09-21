/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
#define GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "omg/omg_inner_types.h"

namespace ge {
class ParserGraphOptimizer {
 public:
  explicit ParserGraphOptimizer(ge::ComputeGraphPtr graph, domi::FrameworkType type = domi::TENSORFLOW)
      : graph_(graph), fmktype_(type) {}

  ~ParserGraphOptimizer() {}

  domi::Status FusionFmkop();

 private:
  ge::ComputeGraphPtr graph_;
  domi::FrameworkType fmktype_;

  domi::Status FindFmkNodeCluser(std::unordered_map<std::string, std::vector<ge::NodePtr>> &node_cluser_Map) const;

  domi::Status MarkForFusion(std::unordered_map<std::string, std::vector<ge::NodePtr>> &node_cluster_map);

  domi::Status GetFusionCluster(const bool has_get_next, const bool has_dyn_get_next,
                                unordered_map<string, vector<NodePtr>> &node_cluster_map);

  domi::Status UpdateGraph(std::vector<ge::NodePtr> &nodes);

  static domi::Status InsertNode(ge::ComputeGraphPtr sub_graph, std::vector<ge::NodePtr> &nodes,
                                 std::vector<ge::InDataAnchorPtr> &input_anchors,
                                 std::vector<ge::OutDataAnchorPtr> &output_anchors,
                                 std::map<ge::OutDataAnchorPtr, std::vector<ge::InDataAnchorPtr>> &output_in_map,
                                 std::vector<ge::InControlAnchorPtr> &input_control_anchors,
                                 std::vector<ge::OutControlAnchorPtr> &output_control_anchors,
                                 std::unordered_map<std::string, ge::NodePtr> &node_map);

  domi::Status LinkInnerAnchor(std::unordered_map<std::string, ge::NodePtr> &node_map) const;

  static domi::Status RebuildOutputAnchors(std::vector<ge::OutDataAnchorPtr> &output_anchors,
                                           ge::OpDescPtr fusion_op_desc);

  static domi::Status RebuildInputAnchors(std::vector<ge::InDataAnchorPtr> &input_anchors,
                                          ge::OpDescPtr fusion_op_desc);

  static domi::Status RebuildFusionNode(std::vector<ge::InDataAnchorPtr> &input_anchors,
                                        std::vector<ge::OutDataAnchorPtr> &output_anchors,
                                        std::map<ge::OutDataAnchorPtr, std::vector<ge::InDataAnchorPtr>> &output_in_map,
                                        std::vector<ge::InControlAnchorPtr> &input_control_anchors,
                                        std::vector<ge::OutControlAnchorPtr> &output_control_anchors,
                                        ge::NodePtr fusion_node);
};
}  // namespace ge
#endif  // GE_GRAPH_OPTIMIZE_GRAPH_OPTIMIZER_H_
