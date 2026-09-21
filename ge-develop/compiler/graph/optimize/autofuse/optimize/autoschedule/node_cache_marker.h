/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_AUTOSCHEDULE_NODE_CACHE_MARKER_H
#define OPTIMIZE_AUTOSCHEDULE_NODE_CACHE_MARKER_H

#include "ascir.h"

namespace optimize::autoschedule {
class NodeCacheMarker {
 public:
  NodeCacheMarker() = delete;
  ~NodeCacheMarker() = default;

  explicit NodeCacheMarker(ascir::ImplGraph &graph) : graph_(graph) {};

  bool IsNodeVisited(const ge::NodePtr &node) const;
  void VisitNode(const ge::NodePtr &node);
  void AddToCacheStartSet(const ge::NodePtr &node);

  ge::ExecuteCondition DoesNodeNeedCache(const std::vector<int64_t> &in_axis, const std::vector<int64_t> &out_axis,
                                         const std::vector<ge::Expression> &in_repeats,
                                         const std::vector<ge::Expression> &out_repeats) const;
  ge::ExecuteCondition DoesNodeNeedCache(const ge::AscNodePtr &node) const;
  ge::ExecuteCondition DoesNodeNeedCache(const ge::NodePtr &node) const;
  ge::ExecuteCondition DoesInlineNodeNeedCache(const ge::NodePtr &node, int32_t brc_idx) const;

  static void MarkNodeCacheable(const ge::NodePtr &node);

  void MarkNodesCacheableBottomUp(const ge::AscNodePtr &node, const ge::ExecuteCondition condition);
  void MarkNodesCacheableBottomUp(const ge::NodePtr &node, const ge::ExecuteCondition condition);
  static void MarkNodesCacheableUpBottom(const ge::NodePtr &node);

  ge::Status ReverseDfsCacheNode(const ge::NodePtr &ge_node);
  ge::Status MarkIfNodeNeedsCache();

  static ge::Status GetAscNodeInputAttr(const ge::NodePtr &node, int32_t idx, ge::AscTensorAttr &attr);

 private:
  ascir::ImplGraph &graph_;
  std::set<ge::NodePtr> cache_start_nodes_;
  std::set<ge::NodePtr> visited_nodes_;
};
}  // namespace optimize::autoschedule

#endif  // OPTIMIZE_AUTOSCHEDULE_NODE_CACHE_MARKER_H
