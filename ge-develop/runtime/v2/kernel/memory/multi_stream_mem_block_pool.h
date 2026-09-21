/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_GERT_MEM_BLOCK_POOL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_GERT_MEM_BLOCK_POOL_H_
#include "graph/utils/object_pool.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "multi_stream_mem_block.h"
#include "ref_object_pool.h"
#include "multi_stream_mem_block_helper.h"
namespace gert {
namespace memory {
constexpr size_t kPoolSize = 1000U;
class MultiStreamMemBlockPool {
 public:
  template<typename... Args>
  MultiStreamMemBlock *Acquire(Args &&...args) {
    return pool_.Acquire(args...);
  }
  void Release(MultiStreamMemBlock *block) {
    block->GetMemBlock()->Free();
    MultiStreamMemBlockHelper::PlusVersion(block);
    MultiStreamMemBlockHelper::ClearMemBlock(block);
    pool_.Release(block);
  }

  std::set<MultiStreamMemBlock *> GetFreeSet() {
    return pool_.GetFreeElements();
  }
  std::vector<MultiStreamMemBlock *> GetAll() {
    return pool_.GetAllElements();
  }

 private:
  NoDestructRefObjectPool<MultiStreamMemBlock, kPoolSize> pool_;
};
}  // namespace memory
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_GERT_MEM_BLOCK_POOL_H_
