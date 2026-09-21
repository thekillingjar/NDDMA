/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_CHECKER_REUSE_CHECKER_H_
#define GE_GRAPH_BUILD_MEMORY_CHECKER_REUSE_CHECKER_H_

#include <limits>
#include <map>
#include <set>
#include <string>
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "ge/ge_api_types.h"
#include "dependency_analyzer.h"

namespace ge {
struct ReuseNodeMem {
  const Node *node;
  IOType type;
  size_t out_index;
  size_t symbol_index;
  int64_t mem_size;
  int64_t topo_id;
  int64_t offset;
};

struct ReuseNodeMemCompare {
  bool operator()(const ReuseNodeMem &lh, const ReuseNodeMem &rh) const {
    return lh.topo_id < rh.topo_id;
  }
};

using ReuseNodes = std::set<ReuseNodeMem, ReuseNodeMemCompare>;
using OffsetReuseNodes = std::map<int64_t, ReuseNodes>;
class ReuseChecker {
 public:
  /// @ingroup ge
  /// @brief constructor of ReuseChecker, will copy anchor_to_symbol and symbol_to_anchors
  /// @param [in] graph
  /// @param [in] anchor_to_symbol
  /// @param [in] symbol_to_anchors
  ReuseChecker(ComputeGraphPtr &graph, AnchorToSymbol anchor_to_symbol, SymbolToAnchors symbol_to_anchors);
  ~ReuseChecker() = default;

  /// @ingroup ge
  /// @brief node dependency analyzer initialize
  /// @return error code of Init
  Status Init();

  /// @ingroup ge
  /// @brief Analyze node reuse based on node output offset and check if reuse satisfies dependency relationships.
  Status Check();

  /// @ingroup ge
  /// @brief set feature map max offset, used to filter nodes that need to check memory reuse
  /// @param [in] feature map max offset
  void SetMaxOffset(const size_t offset);

 private:
  size_t GetSymbolIndex(const Node *node_ptr, const uint32_t out_index);
  void ConstructReuseNodesMap(OffsetReuseNodes &offset_reuse_nodes_map) const;
  Status CollectNodeOffset();
  Status CheckReuseNodes(OffsetReuseNodes &offset_reuse_nodes_map);

 private:
  ComputeGraphPtr graph_;
  AnchorToSymbol anchor_to_symbol_;
  SymbolToAnchors symbol_to_anchors_;
  DependencyAnalyzer deps_analyzer_;
  size_t max_offset_ = std::numeric_limits<size_t>::max();
  size_t unique_index_ = 0U;

  // batch_lable, memory_type, offset
  std::map<std::string, std::map<int64_t, OffsetReuseNodes>> reuse_nodes_;
  std::map<std::string, size_t> symbol_str_to_index_;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_CHECKER_REUSE_CHECKER_H_
