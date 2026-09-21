/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_INPUT_OUTPUT_CONNECTION_IDENTIFY_PASS_H_
#define GE_GRAPH_PASSES_INPUT_OUTPUT_CONNECTION_IDENTIFY_PASS_H_

#include <map>
#include <vector>
#include "graph/graph.h"
#include "graph/passes/graph_pass.h"

namespace ge {
using Node2Indexs = std::unordered_map<const Node *, std::vector<uint32_t>>;
class InputOutputConnectionIdentifyPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  ///
  /// Find all nodes that connect to input node.
  /// @param [in] input node
  /// @param [out] map of nodes and anchor index that connect to input
  /// @param [out] map of nodes and anchor index that connect to output
  /// @return Status
  ///
  Status ProcessInputNode(const NodePtr &node, Node2Indexs &connect_input_node_idx,
                          Node2Indexs &connect_output_node_idx);

  ///
  /// Find all nodes that connect to output node.
  /// @param [in] output node
  /// @param [out] map of nodes and anchor index that connect to input
  /// @param [out] map of nodes and anchor index that connect to output
  /// @return Status
  ///
  Status ProcessOutputNode(const NodePtr &node, Node2Indexs &connect_input_node_idx,
                           Node2Indexs &connect_output_node_idx);

  ///
  /// Update all nodes that have shared memory.
  /// @param [in] symbol string
  /// @param [out] map of nodes and anchor index that connect to input
  /// @param [out] map of nodes and anchor index that connect to output
  /// @return Status
  ///
  Status UpdateNodeIdxMap(const std::string &symbol_string, Node2Indexs &connect_input_node_idx,
                          Node2Indexs &connect_output_node_idx);

  ///
  /// Set attr for all nodes that connect to input and output.
  /// @param [in] map of nodes and anchor index that connect to input
  /// @param [in] map of nodes and anchor index that connect to output
  /// @return Status
  ///
  Status SetNodeAttrOfConnectingInputOutput(const Node2Indexs &connect_input_node_idx,
                                            const Node2Indexs &connect_output_node_idx) const;

  // Members for ref mapping
  SymbolToAnchors symbol_to_anchors_;
  AnchorToSymbol anchor_to_symbol_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_INPUT_OUTPUT_CONNECTION_IDENTIFY_PASS_H_