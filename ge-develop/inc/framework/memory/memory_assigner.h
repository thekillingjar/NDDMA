/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_MEMORY_MEMORY_ASSIGNER_H_
#define INC_FRAMEWORK_MEMORY_MEMORY_ASSIGNER_H_

#include <utility>

#include "graph/build/memory/graph_mem_assigner.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"


namespace ge {
class GE_FUNC_VISIBILITY MemoryAssigner {
 public:
  explicit MemoryAssigner(ge::ComputeGraphPtr compute_graph) : compute_graph_(std::move(compute_graph)) {}
  virtual ~MemoryAssigner() = default;

  MemoryAssigner(const MemoryAssigner &) = delete;

  MemoryAssigner &operator=(const MemoryAssigner &) = delete;

  Status AssignMemory(std::map<uint64_t, size_t> &mem_offsets, size_t &zero_copy_mem_size,
                      const bool has_assigned_var_mem = false);

  const std::vector<std::vector<int64_t>> &GetSubMemOffsets() const {
    return sub_mem_offsets_;
  }

  const std::shared_ptr<GraphMemoryAssigner> GetGraphMemoryAssigner() const {
    return graph_mem_assigner_;
  }

 private:
  ge::ComputeGraphPtr compute_graph_;
  std::vector<std::vector<int64_t>> sub_mem_offsets_;
  std::shared_ptr<GraphMemoryAssigner> graph_mem_assigner_;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_MEMORY_MEMORY_ASSIGNER_H_
