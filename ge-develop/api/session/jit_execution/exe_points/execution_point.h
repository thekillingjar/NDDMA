/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_EXECUTION_POINT_H_
#define CANN_GRAPH_ENGINE_EXECUTION_POINT_H_
#include <cstdint>
#include <memory>
#include "common/plugin/ge_make_unique_util.h"
#include "graph/compute_graph.h"
#include "guarded_execution_point.h"
#include "guard_cache.h"

namespace ge {
static const int64_t MAX_GUARD_CACHE_COUNT = 10;
/**
 * 该类描述了一张切出来的小图，以及对应的guards
 */
class ExecutionPoint {
 public:
  ExecutionPoint() = delete;
  ExecutionPoint(int64_t id, const ComputeGraphPtr &sliced_graph, const ComputeGraphPtr &remaining_graph)
      : sliced_graph_id_(id), sliced_graph_(sliced_graph), remaining_graph_(remaining_graph) {
  }

  bool IsLast() const {
    return remaining_graph_ == nullptr;
  }
  ComputeGraphPtr GetSlicedGraph() const {
    return sliced_graph_;
  }
  ComputeGraphPtr GetRemainingGraph() const {
    return remaining_graph_;
  }
  GuardedExecutionPoint *FindGuarded(const std::vector<gert::Tensor> &inputs) {
    return models_.FindGuardedExecutionPoint(inputs);
  }

  GuardedExecutionPoint *FindOrCreateGuarded(const std::vector<gert::Tensor> &inputs) {
    return models_.FindOrCreateGuarded(inputs);
  }

  size_t GetEpOutNum() const {
    return sliced_graph_->GetOutputSize();
  }
  int64_t GetId() const {
    return sliced_graph_id_;
  }

  uint32_t GetSavedCacheNum() const {
    return models_.GetSavedCacheNum();
  }

 private:
  int64_t sliced_graph_id_ = 0;
  ComputeGraphPtr sliced_graph_;
  ComputeGraphPtr remaining_graph_;
  // todo io relation between sliced_graph and remaining graph

  GuardCheckCache models_{MAX_GUARD_CACHE_COUNT, this};
  friend class ExecutionPointUtil;
};
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_EXECUTION_POINT_H_
