/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_CONNECTION_MATRIX_IMPL_H_
#define GRAPH_CONNECTION_MATRIX_IMPL_H_

#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "common/large_bm.h"

namespace ge {
class ConnectionMatrixImpl {
public:
  explicit ConnectionMatrixImpl(const ComputeGraphPtr &graph);

  ~ConnectionMatrixImpl();

  bool IsConnected(const NodePtr &a, const NodePtr &b) const;

  // inputs are all input nodes of parameter node.
  // if there is a path between A->B, then B will own A's
  // connectivity. The reason is ---
  // If some node can reach A, than it can also reach B.
  void SetConnectivity(const Node::Vistor<NodePtr> &inputs, const NodePtr &node);

  /* Computes the connectivity between two nodes in the
   * computation. The returned ConnectivityMatrix is constructed such that
   * ConnectivityMatrix::IsConnected(a, b) returns true iff there exists a
   * directed path (from producer to consumer) from 'a' to 'b'. Both data
   * connection and control connection are considered for connectivity.
   * A node is connected to itself. */
  graphStatus Generate(const ComputeGraphPtr &graph);

  // update reachablity map for fused nodes.
  void Update(const ComputeGraphPtr &graph, const std::vector<NodePtr> &fusion_nodes);

  uint64_t AddNode(const string &op_name);

  void ExpandAndUpdate(const vector<ge::NodePtr> &fusion_nodes, const std::string &node_name);

private:
  ConnectionMatrixImpl() = delete;
  uint64_t GetIndex(const NodePtr &node) const;

  uint64_t GetIndex(const std::string &op_name) const;

  const LargeBitmap &GetBitMap(const NodePtr &node) const;

  LargeBitmap &GetBitMap(const NodePtr &node);

  LargeBitmap &GetBitMap(uint64_t index);

   size_t size_ = 0;
   size_t used_ = 0;
   size_t expand_step_ = 64;

  std::vector<LargeBitmap> bit_maps_;

  std::unordered_map<std::string, uint64_t> name_to_index_;

  std::weak_ptr<ComputeGraph> graph_;
};
}
#endif  // GRAPH_CONNECTION_MATRIX_H_
