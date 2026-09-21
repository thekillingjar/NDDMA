/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_stream_l2_allocator.h"
#include "core/utils/tensor_utils.h"
#include "core/debug/kernel_tracing.h"
#include "multi_stream_mem_block_helper.h"
#include "framework/runtime/device_memory_recorder.h"

namespace gert {
namespace memory {
SingleStreamL2Allocator::SingleStreamL2Allocator(TensorPlacement placement, ge::Allocator *caching_mem_allocator)
    : GertAllocator(0, placement),
      l1_allocator_(caching_mem_allocator),
      ms_block_pool_(),
      ti_allocator_(*this, ms_block_pool_) {
}

SingleStreamL2Allocator::~SingleStreamL2Allocator() noexcept {
  auto free_set = ms_block_pool_.GetFreeSet();
  auto all_blocks = ms_block_pool_.GetAll();
  for (auto block : all_blocks) {
    if (free_set.count(block) > 0U) {
      continue;
    }
    try {
      BirthRecycle(block);
    } catch (const std::exception &ex) {
      GELOGE(ge::FAILED, "BirthRecycle has exception %s", ex.what());
    }
  }
}

GertTensorData SingleStreamL2Allocator::MallocTensorData(size_t size) {
  return TensorUtils::ToGertTensorData(dynamic_cast<MultiStreamMemBlock *>(Malloc(size)), GetPlacement(),
                                       GetStreamId());
}

GertMemBlock *SingleStreamL2Allocator::Malloc(size_t size) {
  GE_ASSERT_NOTNULL(l1_allocator_);
  auto block = l1_allocator_->Malloc(size);
  if (block != nullptr) {
    if (GetPlacement() == kOnDeviceHbm) {
      // 记录开启的内存信息
      DeviceMemoryRecorder::SetRecorder(block->GetAddr(), static_cast<int64_t>(block->GetSize()));
    }
    return ms_block_pool_.Acquire(this, block, BlockAllocType(BlockAllocType::kNorm, 0U));
  }
  return nullptr;
}

void SingleStreamL2Allocator::Free(GertMemBlock *gert_mem_block) {
  if (gert_mem_block != nullptr) {
    auto multi_stream_mem_block = dynamic_cast<MultiStreamMemBlock *>(gert_mem_block);
    if (GetPlacement() == kOnDeviceHbm) {
      const int64_t free_size = static_cast<int64_t>(multi_stream_mem_block->GetSize()) * (-1);
      DeviceMemoryRecorder::SetRecorder(multi_stream_mem_block->GetAddr(), free_size);
    }
    if (multi_stream_mem_block->GetCount(GetStreamId()) > 0U &&
        multi_stream_mem_block->SubCount(GetStreamId()) == 0U) {
      BirthRecycle(multi_stream_mem_block);
    }
  }
}
void SingleStreamL2Allocator::BirthRecycle(MultiStreamMemBlock *block) {
  KERNEL_TRACE_NO_CTX("[MEM]Recycle memory at stream %" PRId64 ", address %p", GetStreamId(), block->GetAddr());
  ms_block_pool_.Release(block);
}

ge::graphStatus SingleStreamL2Allocator::SetL1Allocator(ge::Allocator *allocator) {
  GE_ASSERT_NOTNULL(allocator);
  l1_allocator_ = allocator;
  return ge::GRAPH_SUCCESS;
}

ge::Allocator *SingleStreamL2Allocator::GetL1Allocator() {
  return l1_allocator_;
}
ge::graphStatus SingleStreamL2Allocator::FreeAt(int64_t stream_id, GertMemBlock *block) {
  GE_ASSERT_TRUE(stream_id == GetStreamId());
  Free(block);
  return ge::GRAPH_SUCCESS;
}
int64_t SingleStreamL2Allocator::GetStreamNum() {
  return 1;
}

TensorData SingleStreamL2Allocator::MallocTensorDataFromL1(size_t size) {
  GE_ASSERT_NOTNULL(l1_allocator_);
  return TensorUtils::ToTensorData(l1_allocator_->Malloc(size), size, GetPlacement());
}

ge::graphStatus SingleStreamL2Allocator::ShareFromTensorData(const TensorData &td, GertTensorData &gtd) {
  return ti_allocator_.ShareFromTensorData(td, gtd);
}
}  // namespace memory
}  // namespace gert
