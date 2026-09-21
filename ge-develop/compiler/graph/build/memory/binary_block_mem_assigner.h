/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_BINARY_BLOCK_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_BINARY_BLOCK_MEM_ASSIGNER_H_

#include <utility>
#include <vector>
#include "graph/build/memory/block_mem_assigner.h"

namespace ge {
class BinaryBlockMemAssigner : public BlockMemAssigner {
 public:
  BinaryBlockMemAssigner(MemAssistInfo &mem_assist_info)
      : BlockMemAssigner(mem_assist_info) {}

  BinaryBlockMemAssigner(const BinaryBlockMemAssigner &) = delete;

  BinaryBlockMemAssigner &operator=(const BinaryBlockMemAssigner &) = delete;

  ~BinaryBlockMemAssigner() override = default;

  Status GetMemoryRanges(std::vector<int64_t> &range_ceils) override;

 private:
  void PlanRanges(size_t range_number_limit, std::vector<std::vector<int64_t>> &ranges) const;

 protected:
  bool NeedLevel2Reuse() override { return true; }
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_BINARY_BLOCK_MEM_ASSIGNER_H_
