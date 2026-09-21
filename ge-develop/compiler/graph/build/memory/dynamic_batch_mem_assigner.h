/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_DYNAMIC_BATCH_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_DYNAMIC_BATCH_MEM_ASSIGNER_H_

#include <map>
#include <vector>

#include "runtime/rt.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/build/memory/block_mem_assigner.h"

namespace ge {
constexpr size_t kMaxSplitSizeForDynamicBatch = 400UL * 1024UL * 1024UL;

class DynamicBatchMemAssigner {
 public:
  explicit DynamicBatchMemAssigner(const ReuseStrategy &reuse_strategy, std::vector<MemoryBlock *> &memory_blocks,
                                   std::vector<MemoryBlock *> &blocks_store)
      : memory_blocks_(memory_blocks),
        blocks_store_(blocks_store),
        use_extend_size_memory_(false),
        reuse_strategy_(reuse_strategy) {}

  ~DynamicBatchMemAssigner() {}

  DynamicBatchMemAssigner(const MemoryBlock &) = delete;

  DynamicBatchMemAssigner &operator=(const MemoryBlock &) = delete;

  void ResizeDynamicBatchBlocks();
 private:
  MemoryBlock *CreateContinuousBlock(std::vector<MemoryBlock *> &continuous_blocks);
  void DoResizeDynamicBatchBlocks(std::map<std::string, std::vector<MemoryBlock *>> &dynamic_batch_blocks,
                                  bool continuous = false);

 private:
  std::vector<MemoryBlock *> &memory_blocks_;
  std::vector<MemoryBlock *> &blocks_store_;
  bool use_extend_size_memory_;
  ReuseStrategy reuse_strategy_{};
};

}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_DYNAMIC_BATCH_MEM_ASSIGNER_H_
