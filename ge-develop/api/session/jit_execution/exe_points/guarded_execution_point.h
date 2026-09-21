/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_GUARDED_EXECUTION_POINT_H
#define AIR_GUARDED_EXECUTION_POINT_H
#include <cstdint>
#include <iostream>
#include <array>

#include "exe_graph/runtime/tensor.h"
#include "ge/ge_api_types.h"

#include "graph/compute_graph.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"

namespace ge {

class ExecutionPoint;

class GuardCheckFuncCaller {
 public:
  GuardCheckFuncCaller() = default;
  bool Match(const std::vector<gert::Tensor> &inputs) const;
  Status LoadGuardCheckFunc(ComputeGraphPtr computeGraphPtr);
  Status UnloadGraphCheckFunc() const;
 private:
  GuardCheckFunc func_{nullptr};
  int32_t file_handle_{-1};
  void* so_handle_{nullptr};
};

class ExecutionPoint;
class GuardedExecutionPoint {
 public:
  GuardedExecutionPoint() = delete;
  GuardedExecutionPoint(ExecutionPoint *owner_point) : owner_point_(owner_point){};

  bool Match(const std::vector<gert::Tensor> &inputs) const;

  const ExecutionPoint *GetOwnerEp() const {
    return owner_point_;
  };

  ComputeGraphPtr GetGraph() const;
  ComputeGraphPtr GetSlicedGraph() const;
  Status CopySlicedGraph();

  bool Compiled() const {
    return compiled_;
  }
  uint32_t GetCompiledGraphId() const {
    return compiled_graph_id_;
  }
  bool SetCompiled(uint32_t compiled_graph_id, ComputeGraphPtr graph);
  void SetForked(uint32_t forked_graph_id) {
    forked_graph_ids_.push_back(forked_graph_id);
  }

  Status RemoveItem();
  uint32_t GetPriority() const;
  void SetPriority(uint32_t userPriority);

 private:
  uint32_t priority_{0U};

  ExecutionPoint *owner_point_;

  GuardCheckFuncCaller matcher_;
  ComputeGraphPtr compiled_graph_;

  bool compiled_{false};
  uint32_t compiled_graph_id_{0U};
  std::vector<uint32_t> forked_graph_ids_;
  friend class GuardedExecutionPointUtil;
};
}  // namespace ge

#endif  // AIR_GUARDED_EXECUTION_POINT_H
