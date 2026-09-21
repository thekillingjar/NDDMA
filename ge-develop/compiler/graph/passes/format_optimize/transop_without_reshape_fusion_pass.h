/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_TRANSOP_WITHOUT_RESHAPE_FUSION_PASS_H_
#define GE_GRAPH_PASSES_TRANSOP_WITHOUT_RESHAPE_FUSION_PASS_H_

#include <vector>
#include <utility>
#include <unordered_set>
#include "graph/passes/graph_pass.h"

namespace ge {
/// Transform operators depth fusion
class TransOpWithoutReshapeFusionPass : public GraphPass {
 public:
  TransOpWithoutReshapeFusionPass() {}
  virtual ~TransOpWithoutReshapeFusionPass() {}

  graphStatus Run(ge::ComputeGraphPtr graph) override;

 private:
  graphStatus AddControlEdgeForNewTransNode(const int32_t index,
                                            const std::vector<NodePtr> &new_trans_nodes);
  bool CheckIfHasSameOutControlEdge(const NodePtr node, const NodePtr out_node);
  void SetRemainNode(const std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &nodes_anchor);
  bool IsFormatContinuous(const OutDataAnchorPtr &out_anchor, const InDataAnchorPtr &in_anchor) const;
  bool HasPrecisionLoss(const OutDataAnchorPtr &out_anchor, const InDataAnchorPtr &in_anchor) const;
  void RemoveNousedNodes(const ComputeGraphPtr &graph);
  void GetBeginOutDescAndEndInDesc(const int32_t index, GeTensorDesc &out_desc, GeTensorDesc &in_desc);

  void GetFormatTransferDesc(const GeTensorDesc &out_desc,
                             const GeTensorDesc &in_desc,
                             GeTensorDesc &format_transfer_input,
                             GeTensorDesc &format_transfer_output) const;

  void GetCastOpDesc(const GeTensorDesc &out_desc,
                     const GeTensorDesc &in_desc,
                     GeTensorDesc &cast_input,
                     GeTensorDesc &cast_output) const;

  graphStatus FormatFusion(const int32_t index,
                           OpDescPtr &format_transfer_op,
                           int32_t &fusion_op_count,
                           bool &fusion_continue);

  graphStatus DataTypeFusion(const int32_t index, OpDescPtr &cast_op, int32_t &fusion_op_count);

  void GetOutDataPeerInControlAnchors(const size_t index,
                                      std::vector<vector<InControlAnchorPtr>> &out_data_peer_in_control_anchors);

  void GetInControlPeerOutControlAnchors(
      const size_t index,
      std::vector<vector<OutControlAnchorPtr>> &in_control_peer_out_control_anchors);

  void GetOutControlPeerAnchors(
      const size_t index,
      std::vector<vector<InControlAnchorPtr>> &out_control_peer_in_control_anchors,
      std::vector<vector<InDataAnchorPtr>> &out_control_peer_in_data_anchors);

  graphStatus TransOpFuse(const ComputeGraphPtr &graph);

  graphStatus GetSubGraphsBetweenNormalNode(
      const OutDataAnchorPtr &out_anchor,
      std::vector<vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>>
  >& sub_graphs_out,
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &nodes_list
  );

  graphStatus GetSubGraphNodesInfo();

  void GetControlAnchors();

  graphStatus InsertNewTransOp(const ComputeGraphPtr &graph, const OpDescPtr &cast_op,
                               const OpDescPtr &format_transfer_op, const int32_t index,
                               const bool insert_cast_first);

  void EraseInvalidAnchorsPair();

  graphStatus RelinkNodesWhenDescNotChanged(const std::pair<OutDataAnchorPtr, InDataAnchorPtr> &begin_anchors_pair,
                                            const std::pair<OutDataAnchorPtr, InDataAnchorPtr> &end_anchors_pair,
                                            const int32_t index);

  OpDescPtr GetFormatTransferOp(const GeTensorDesc &format_trans_input_desc,
                                const GeTensorDesc &format_trans_output_desc) const;

  OpDescPtr GetCastOp(const GeTensorDesc &cast_input_desc, const GeTensorDesc &cast_output_desc) const;

  graphStatus TransOpFuseHandle(const ge::ComputeGraphPtr &graph, const int32_t index);

  graphStatus AddTransNode(const ComputeGraphPtr &graph, const OpDescPtr &transop, NodePtr &trans_node) const;

  bool DescEqualCheck(ConstGeTensorDescPtr &desc_src, ConstGeTensorDescPtr &desc_dst) const;

  bool ShapeEqualCheck(const GeShape &src, const GeShape &dst) const;

  bool InsertCastFirstCheck(const GeTensorDesc &out_desc, const GeTensorDesc &in_desc) const;

  graphStatus RelinkControlEdge(const int32_t index, const OutDataAnchorPtr &out_anchor,
                                const std::vector<NodePtr> &new_trans_nodes);

  graphStatus GetTransNode(const ComputeGraphPtr &graph,
                           const OpDescPtr &cast_op,
                           const OpDescPtr &format_transfer_op,
                           const bool insert_cast_first,
                           std::vector<NodePtr> &new_trans_nodes) const;

  void UpdateOutputName(const OutDataAnchorPtr &out_anchor, const InDataAnchorPtr &old_peer_in_anchor,
                        const NodePtr &in_owner_node) const;
  void UpdateInputName(const OutDataAnchorPtr &old_peer_out_anchor, const InDataAnchorPtr &in_anchor,
                       const NodePtr &out_owner_node) const;

  graphStatus RelinkControlEdgesWhenDescNotChanged(
      const std::pair<OutDataAnchorPtr, InDataAnchorPtr> &begin_anchors_pair,
      const std::pair<OutDataAnchorPtr, InDataAnchorPtr> &end_anchors_pair, const int32_t index);

  graphStatus RelinkSubGraphControlEdges(const std::pair<OutDataAnchorPtr, InDataAnchorPtr> &begin_anchors_pair,
                                         const std::pair<OutDataAnchorPtr, InDataAnchorPtr> &end_anchors_pair,
                                         const int32_t index);

  /// judge whether an operator is a transform op or not
  /// @param node
  /// @return True or False
  static bool IsTransOp(const NodePtr &node);

  static bool FusionFormatSupport(Format format);

  std::vector<vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>>>
  sub_graph_anchors_;
  std::vector<vector<NodePtr>> sub_graph_nodes_;
  std::vector<int32_t> transop_num_count_;
  std::vector<bool> sub_graph_has_reshape_node_;
  std::vector<vector<OutControlAnchorPtr>> in_control_peer_out_control_anchors_;
  std::vector<vector<InControlAnchorPtr>> out_control_peer_in_control_anchors_;
  std::vector<vector<InDataAnchorPtr>> out_control_peer_in_data_anchors_;
  std::vector<vector<InControlAnchorPtr>> out_data_peer_in_control_anchors_;
  std::vector<bool> sub_graph_has_control_edge_;
  std::vector<bool> sub_graph_has_out_data_peer_in_control_edge_;
  // remain node's out control edges do not need to transfer
  std::unordered_set<InControlAnchorPtr> remain_in_control_anchors_;
  std::unordered_set<OutControlAnchorPtr> remain_out_control_anchors_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_TRANSOP_WITHOUT_RESHAPE_FUSION_PASS_H_

