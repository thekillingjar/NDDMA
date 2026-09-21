/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_BINARY_PARTITIONER_H
#define CANN_GRAPH_ENGINE_BINARY_PARTITIONER_H

#include <cstdint>
#include <vector>
#include <utility>

#include "binary_graph_builder.h"
#include "ge/ge_api_types.h"

namespace ge {
const std::string SLICED_GRAPH_NAME = "Sliced_Graph";
const std::string REMAINING_GRAPH_NAME = "Remaining_Graph";
struct PartionResult {
  ComputeGraphPtr sliced_graph{nullptr};
  ComputeGraphPtr remaining_graph{nullptr};
  // io:<out_idx, in_idx>
  std::vector<std::pair<int64_t, int64_t>> out_idx_2_in_idxs;
};
class BinaryPartitioner {
 public:
  /*
   * Splits the original graph into two separate graphs based on node sets.
   * @param[in] graph : The original graph before slicing.
   * @param[in] infered_nodes : The set of nodes that have completed symbol inference.
   * @param[out] p_ret : Contains two graphs, where the first graph includes nodes that have been inferred, and the second graph includes nodes that have not been inferred.
   * @return Status : Return GRAPH_SUCCESS on success, and return GRAPH_FAILED on failure.
   */
  static Status Partition(const ComputeGraphPtr &graph, const std::vector<NodePtr> &infered_nodes, PartionResult &p_ret);
 private:
  static std::vector<NodePtr> GetRemainingNodes(const ComputeGraphPtr &graph,
                                                const std::vector<NodePtr> &infered_nodes);
  static bool CheckNodesContainsCycle(const std::vector<NodePtr> &infered_nodes,
                                      const std::vector<NodePtr> &uninfer_nodes);
 private:
  static BinaryGraphBuilder graph_builder_;
};
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_BINARY_PARTITIONER_H
