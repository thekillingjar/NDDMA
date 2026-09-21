/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OPTIMIZE_FUSED_GRAPH_FUSED_GRAPH_UNFOLDER_H__
#define __OPTIMIZE_FUSED_GRAPH_FUSED_GRAPH_UNFOLDER_H__

#include "ascendc_ir.h"
#include "ascgen_log.h"
#include "node.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize {
const char *const kAscGraphNodeType = "AscGraph";
const char *const kAscBackendType = "AscBackend";
using AscGraphPtr = ge::AscGraph;

class FusedGraphUnfolder {
 public:
  static Status UnfoldFusedGraph(const ge::ComputeGraphPtr &fused_graph,
                                 std::map<ge::Node *, ge::AscGraph> &asc_backend_to_asc_graph,
                                 ge::AscGraph &unfolded_asc_graph);

 private:
  static Status SelectCommonLoopAxis(std::map<ge::Node *, ge::AscGraph> &asc_backend_to_asc_graph,
                                     std::vector<ge::AxisPtr> &new_loop_axes);
  static Status MarkAllOutputAxisId(ge::Node *concat_ascbc_node,
                                    std::map<ge::Node *, ge::AscGraph> &asc_backend_to_asc_graph,
                                    const ge::AxisId &axis_id,
                                    std::map<const ge::AscGraph *, ge::AxisId> &seen_graph_to_changed_axis_id,
                                    std::set<ge::Node *> &seen_node);
  // 需要考虑向前以及向后的场景
  static Status MarkAllInputAxisId(ge::Node *concat_input_node,
                                   std::map<ge::Node *, ge::AscGraph> &asc_backend_to_asc_graph,
                                   const ge::AxisId &axis_id,
                                   std::map<const ge::AscGraph *, ge::AxisId> &seen_graph_to_changed_axis_id,
                                   std::set<ge::Node *> &seen_node);

  static Status ApplyMergedLoopAxis(const ge::AscGraph &graph, const std::vector<ge::AxisPtr> &new_loop_axes,
                                    const std::vector<ge::AxisId> &loop_axis_ids, const size_t concat_dim);
  static Status UnfoldAscbcNode(ge::Node *const &ascbc_node, const ge::AscGraph &asc_graph,
                                const ge::ComputeGraphPtr &target_computer_graph);
  static Status ReAssembleDataIrAttr(const ge::ComputeGraphPtr &fused_graph,
                                     const std::map<ge::Node *, ge::AscGraph> &asc_backend_to_asc_graph);
  static Status ReAssembleOutputIndex(const ge::ComputeGraphPtr &fused_graph);

  static Status TransferInControlEdges(const std::set<ge::NodePtr> &src_nodes, ge::Node *const &asc_backend);
  static Status MergeInputNodes(const ge::ComputeGraphPtr &graph, ge::Node *const &asc_backend);
  static Status MergeOutputNodes(const ge::ComputeGraphPtr &graph, ge::Node *const &asc_backend);
  static Status DoSameLoadCse(const ge::ComputeGraphPtr &fused_graph);
  static bool IsSameLoadNode(const ge::AscNodePtr &lhs, const ge::AscNodePtr &rhs);
  static Status RemoveRedundantLoads(const ge::ComputeGraphPtr &graph);
  static Status RemoveUnusedNode(const ge::ComputeGraphPtr &graph, const ge::NodePtr &node, const bool force = false);
  static Status CollectPostConcatAscGraphs(ge::Node *concat_ascbc_node,
                                           const std::map<ge::Node *, ge::AscGraph> &asc_backend_to_asc_graph,
                                           const std::vector<ge::AxisPtr> &new_loop_axes,
                                           const std::vector<ge::AxisId> &loop_axis_ids,
                                           std::map<ge::Node *, ge::AscGraph> &post_concat_node_to_asc_graph);
  static Status DoAxisMappingForConstPostAscGraph(const ge::AscGraph &graph,
                                                  const std::vector<ge::AxisPtr> &new_loop_axes,
                                                  const std::vector<ge::AxisId> &loop_axis_ids);
};
}  // namespace optimize

#endif  // __OPTIMIZE_FUSED_GRAPH_FUSED_GRAPH_UNFOLDER_H__
