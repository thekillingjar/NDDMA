/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_TI_BLOCK_ALLOCATOR_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_TI_BLOCK_ALLOCATOR_H_
#include "ge/ge_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/tensor_data.h"
#include "ref_object_pool.h"
#include "multi_stream_mem_block.h"
#include "multi_stream_mem_block_pool.h"
namespace gert {
namespace memory {
/**
 * Tensor-data In Block
 */
class TiBlock : public ge::MemBlock {
 public:
  explicit TiBlock(ge::Allocator &allocator, const TensorData &td)
      : ge::MemBlock(allocator, td.GetAddr(), td.GetSize()) {
    td_.ShareFrom(td);
  }

  TensorData &GetTd() {
    return td_;
  }

 private:
  TensorData td_;
};

class TiBlockAllocator : public ge::Allocator {
 public:
  ge::MemBlock *ShareFromTensorData(const TensorData &td) {
    return pool_.Acquire(*this, td);
  }

  void Free(ge::MemBlock *block) override {
    pool_.Release(reinterpret_cast<TiBlock *>(block));
  }

 private:
  ge::MemBlock *Malloc(size_t size) override {
    (void)size;
    return nullptr;
  }

 private:
  RefObjectPool<TiBlock> pool_;
};

/**
 * Tensor-data in GertTensorData Allocator
 */
class TiGtdAllocator {
 public:
  TiGtdAllocator(GertAllocator &allocator, MultiStreamMemBlockPool &msb_pool)
      : allocator_(allocator), msb_pool_(msb_pool) {}

  ge::graphStatus ShareFromTensorData(const TensorData &td, GertTensorData &gtd) {
    auto block = ti_allocator_.ShareFromTensorData(td);
    GE_ASSERT_NOTNULL(block);

    // ms_block is released to pool in l2 allocator
    auto msb = msb_pool_.Acquire(&allocator_, block, BlockAllocType{BlockAllocType::kTensorDataWrapped, 0U});
    GE_ASSERT_NOTNULL(msb);

    gtd = {td.GetSize(), td.GetPlacement(), allocator_.GetStreamId(), msb};

    return ge::GRAPH_SUCCESS;
  }

 private:
  GertAllocator &allocator_;
  MultiStreamMemBlockPool &msb_pool_;
  TiBlockAllocator ti_allocator_;
};
}  // namespace memory
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_TI_BLOCK_ALLOCATOR_H_
