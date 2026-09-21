/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_COND_PASS_H
#define GE_GRAPH_PASSES_COND_PASS_H

#include "graph/passes/base_pass.h"

namespace ge {
class CondPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  /// @brief Get cond info for if / while
  /// @param [in] node: If / While op
  /// @param [out] graph: owner_graph of if node / while_cond subgraph
  /// @param [out] peer_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: cond_input
  /// @return Status
  Status GetCondInfo(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &peer_out_anchor,
                     InDataAnchorPtr &cond_in_anchor) const;

  /// @brief Get cond info for if node
  /// @param [in] node: If op
  /// @param [out] graph: owner_graph of if node
  /// @param [out] peer_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: cond_input of if
  /// @return Status
  Status GetCondInfoForIf(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &peer_out_anchor,
                          InDataAnchorPtr &cond_in_anchor) const;

  /// @brief Get cond info for while node
  /// @param [in] node: While op
  /// @param [out] graph: while_cond subgraph
  /// @param [out] peer_out_anchor: peer_cond_anchor
  /// @param [out] cond_in_anchor: input of NetOutput in cond_graph
  /// @return Status
  Status GetCondInfoForWhile(const NodePtr &node, ComputeGraphPtr &graph, OutDataAnchorPtr &peer_out_anchor,
                             InDataAnchorPtr &cond_in_anchor) const;

  /// @brief Process Cond Op with non-scalar cond_input
  /// @param [in] graph
  /// @param [in] peer_out_anchor: peer_cond_anchor
  /// @param [in] cond_in_anchor: cond_input
  /// @return Status
  Status HandleNonScalarCond(const OutDataAnchorPtr &peer_out_anchor,
                             const InDataAnchorPtr &cond_in_anchor);

  /// @brief Process Cond Op with scalar-string cond_input
  /// @param [in] graph
  /// @param [in] peer_out_anchor: peer_cond_anchor
  /// @param [in] cond_in_anchor: cond_input
  /// @return Status
  Status HandleStringCond(const OutDataAnchorPtr &peer_out_anchor,
                          const InDataAnchorPtr &cond_in_anchor);

  /// @brief Process Cond Op with scalar cond_input
  /// @param [in] graph
  /// @param [in] peer_out_anchor: peer_cond_anchor
  /// @param [in] cond_in_anchor: cond_input
  /// @param [in] src_type
  /// @return Status
  Status HandleScalarCond(const OutDataAnchorPtr &peer_out_anchor,
                          const InDataAnchorPtr &cond_in_anchor, DataType src_type);

  /// @brief Insert node
  /// @param [in] graph
  /// @param [in] peer_out_anchor
  /// @param [in] in_data_anchor
  /// @param [in] type
  /// @return Status
  Status InsertNode(const OutDataAnchorPtr &peer_out_anchor,
                    const InDataAnchorPtr &in_data_anchor, const std::string &type);

  /// @brief Add cast node
  /// @param [in] graph
  /// @param [in] name
  /// @param [in] tensor
  /// @param [in] src
  /// @param [in] dst
  /// @return NodePtr
  OpDescPtr AddCastOpDesc(const std::string &name, const GeTensorDesc &tensor,
                          DataType src, DataType dst) const;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_COND_PASS_H
