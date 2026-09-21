/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_PASS_UTILS_H_
#define GE_GRAPH_PASSES_PASS_UTILS_H_

#include <vector>
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "base/err_msg.h"

namespace ge {
class PassUtils {
 public:
  PassUtils() = delete;
  ~PassUtils() = delete;

  static NodePtr GetInDataNode(const ConstNodePtr &node, int32_t index);

  static NodePtr GetInNodeCrossSubgraphByIndex(const ConstNodePtr &node, int32_t index);

  static bool IsConstant(const ConstNodePtr &node);

  static Status SetOutNodeWeight(const OutDataAnchorPtr &out_data_anchor, const NodePtr &src_node);

  static Status RemoveBranch(const NodePtr &node, std::vector<NodePtr> &delete_nodes, std::vector<NodePtr> &end_nodes);

  static Status RemoveInactiveBranchToMerge(const OutDataAnchorPtr &inactive_output_anchor,
      std::vector<NodePtr> &delete_nodes, std::vector<NodePtr> &end_nodes);

  /// check is need iter flow ctrl.
  /// @param compute_graph graph
  /// @return true:need iter flow ctrl.
  ///         false:no need
  static bool IsNeedTrainIteFlowCtrl(const ComputeGraphPtr &compute_graph);

  /// find in data anchor index with a valid peer out node existed
  /// @param node_ptr
  /// @return index
  static int32_t GetUniqueInDataAnchorIndex(const NodePtr &node_ptr);

  /// unlink node's in data anchors[index]'s father node with node itself
  /// then link father node's all in control nodes to node
  /// if any and not connected yet
  /// @param node
  /// @param index: in data anchor index
  /// @return
  static Status UnlinkNodeWithControlCopy(NodePtr &node, int32_t index);

  static std::set<std::string> GetDisabledOptimizations();

  static bool IsOptimizationDisabled(const std::set<std::string> &disabled_optimizations,
                                     const std::string &optimization_name);
  static bool GetPerm(const NodePtr &node, const uint32_t kTransposeInputPerm, std::vector<int64_t> &perm);

  static Status UpdateRefAttr(const ComputeGraphPtr &graph);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_PASS_UTILS_H_
