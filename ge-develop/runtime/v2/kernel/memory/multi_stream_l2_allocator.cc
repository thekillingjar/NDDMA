/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multi_stream_l2_allocator.h"
#include "caching_mem_allocator.h"
#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/utils/tensor_utils.h"
#include "utils/utils.h"
#include "core/debug/kernel_tracing.h"
#include "multi_stream_mem_block_helper.h"
#include "framework/runtime/device_memory_recorder.h"

namespace gert {
namespace memory {
MultiStreamL2Allocator::MultiStreamL2Allocator(
    int64_t stream_id, TensorPlacement placement,
    TypedContinuousVector<memory::MultiStreamL2Allocator *> *stream_ids_to_allocator,
    TypedContinuousVector<L2MemPool *> *all_l2_mem_pool)
    : GertAllocator(stream_id, placement),
      l1_allocator_{nullptr},
      // 不加nothrow，理由：由于构造函数无法返回失败，且这是关键资源申请，如果申请失败允许进程退出。
      own_allocator_(std::unique_ptr<L2MemPool>(new L2MemPool(nullptr, nullptr, all_l2_mem_pool))),
      stream_ids_to_allocator_{stream_ids_to_allocator},
      ms_block_pool_(),
      ti_allocator_(*this, ms_block_pool_),
      stream_(nullptr) {}

MultiStreamL2Allocator::MultiStreamL2Allocator(
    ge::Allocator *allocator, TensorPlacement placement, int64_t stream_id, rtStream_t stream,
    TypedContinuousVector<memory::MultiStreamL2Allocator *> *stream_ids_to_allocator,
    TypedContinuousVector<L2MemPool *> *all_l2_mem_pool)
    : GertAllocator(stream_id, placement),
      l1_allocator_(allocator),
      // 不加nothrow，理由：由于构造函数无法返回失败，且这是关键资源申请，如果申请失败允许进程退出。
      own_allocator_(std::unique_ptr<L2MemPool>(new L2MemPool(allocator, stream, all_l2_mem_pool))),
      stream_ids_to_allocator_(stream_ids_to_allocator),
      ms_block_pool_(),
      ti_allocator_(*this, ms_block_pool_),
      stream_(stream) {}

GertMemBlock *MultiStreamL2Allocator::Malloc(size_t size) {
  auto block = own_allocator_->Malloc(size);
  if (block != nullptr) {
    if (GetPlacement() == kOnDeviceHbm) {
      DeviceMemoryRecorder::SetRecorder(block->GetAddr(), static_cast<int64_t>(block->GetSize()));
    }
    return ms_block_pool_.Acquire(this, block, BlockAllocType{BlockAllocType::kNorm, 0U});
  }
  GELOGI("malloc memory failed, try to use borrow block");
  // todo borrow allocator的Alloc接口最好带上自己的stream id，或者初始化borrow allocator的时候，把自己的stream id传进去
  auto gert_mem_block = borrow_allocator_.Alloc(size);
  if (gert_mem_block != nullptr) {
    if (GetPlacement() == kOnDeviceHbm) {
      DeviceMemoryRecorder::SetRecorder(gert_mem_block->GetAddr(),
          static_cast<int64_t>(gert_mem_block->GetSize()));
    }
    gert_mem_block->NewAccessStream(GetStreamId(), GetStreamId());
  }
  return gert_mem_block;
}

void MultiStreamL2Allocator::Free(GertMemBlock *block) {
  auto multi_stream_mem_block = reinterpret_cast<MultiStreamMemBlock *>(block);
  if (GetPlacement() == kOnDeviceHbm) {
    const int64_t free_size = static_cast<int64_t>(multi_stream_mem_block->GetSize()) * (-1);
    DeviceMemoryRecorder::SetRecorder(multi_stream_mem_block->GetAddr(), free_size);
  }
  bool need_recycle = false;
  if (multi_stream_mem_block->GetCount(GetStreamId()) > 0U &&
      multi_stream_mem_block->SubCount(GetStreamId()) == 0U) {
    need_recycle = true;
    MultiStreamMemBlockHelper::UpdateLocalRecycleMif(multi_stream_mem_block, GetStreamId());
  }

  // we do global recycle instead of local recycle if we can
  if (MultiStreamMemBlockHelper::CanGlobalRecycle(multi_stream_mem_block, GetStreamId())) {
    if (multi_stream_mem_block->GetBirthStreamId() == GetStreamId()) {
      BirthRecycle(multi_stream_mem_block);
    } else {
      BorrowRecycle(multi_stream_mem_block);
    }
    return;
  }

  if (need_recycle) {
    LocalRecycle(multi_stream_mem_block);
  }
}

GertTensorData MultiStreamL2Allocator::MallocTensorData(size_t size) {
  return TensorUtils::ToGertTensorData(reinterpret_cast<MultiStreamMemBlock *>(Malloc(size)), GetPlacement(),
                                       GetStreamId());
}
TensorData MultiStreamL2Allocator::MallocTensorDataFromL1(size_t size) {
  return TensorUtils::ToTensorData(l1_allocator_->Malloc(size), size, GetPlacement());
}

ge::graphStatus MultiStreamL2Allocator::BirthRecycle(MultiStreamMemBlock *block) {
  if (SECUREC_UNLIKELY((block == nullptr) || (block->GetMemBlock() == nullptr))) {
    GELOGE(ge::GRAPH_FAILED, "Failed to birth recycle block %p, the block maybe already recycled", block);
    return ge::GRAPH_FAILED;
  }
  GE_ASSERT_TRUE(GetStreamId() == block->GetBirthStreamId());
  KERNEL_TRACE_NO_CTX("[MEM]Recycle memory at stream %" PRId64 ", address %p", GetStreamId(), block->GetAddr());
  ms_block_pool_.Release(block);
  return ge::GRAPH_SUCCESS;
}

void MultiStreamL2Allocator::BorrowRecycle(MultiStreamMemBlock *block) {
  // todo ut check log
  KERNEL_TRACE_NO_CTX("[MEM]Borrow-recycle memory at stream %" PRId64 ", address %p, birth stream %" PRId64,
                      GetStreamId(), block->GetAddr(), block->GetBirthStreamId());
  borrow_allocator_.Free(block);
  // borrow
  // recycle后，block可能会再次从borrow池中被申请出来，而从borrow池中被申请出来的流程是不带初始化的，因此在这里做一次mifs的复位操作
  MultiStreamMemBlockHelper::PlusVersion(block);
  MultiStreamMemBlockHelper::ReInitMifs(block);
}

void MultiStreamL2Allocator::LocalRecycle(MultiStreamMemBlock *block) {
  // todo ut check log
  KERNEL_TRACE_NO_CTX("[MEM]Local-recycle memory at stream %" PRId64 ", address %p, birth stream %" PRId64,
                      GetStreamId(), block->GetAddr(), block->GetBirthStreamId());
  local_recycle_blocks_.push_back({{block->GetVersion(), block}, 0U});
}

std::list<MultiStreamMemBlock *> MultiStreamL2Allocator::GetAndClearBorrowBlocks(int64_t dst_stream_id) {
  return borrow_allocator_.GetAndClearBorrowBlocks(dst_stream_id);
}

ge::graphStatus MultiStreamL2Allocator::SetL1Allocator(ge::Allocator *allocator) {
  GE_ASSERT_NOTNULL(allocator);
  l1_allocator_ = allocator;
  GE_ASSERT_NOTNULL(own_allocator_);
  own_allocator_->SetL1Allocator(allocator);
  return ge::GRAPH_SUCCESS;
}

int64_t MultiStreamL2Allocator::GetStreamNum() {
  return static_cast<int64_t>(stream_ids_to_allocator_->GetSize());
}
MultiStreamL2Allocator::~MultiStreamL2Allocator() {
  auto free_set = ms_block_pool_.GetFreeSet();
  auto all_blocks = ms_block_pool_.GetAll();
  for (auto block : all_blocks) {
    if (free_set.count(block) > 0U) {
      continue;
    }
    GELOGW("The block %p address %p does not recycled when l2 allocator destructing, will be recycled force", block,
           block->GetAddr());
    BirthRecycle(block);
  }
}
ge::graphStatus MultiStreamL2Allocator::ShareFromTensorData(const TensorData &td, GertTensorData &gtd) {
  return ti_allocator_.ShareFromTensorData(td, gtd);
}

ge::graphStatus MultiStreamL2Allocator::MoveL2ToL1(GertMemBlock *block) {
  GE_ASSERT_NOTNULL(block);
  auto ms_block = reinterpret_cast<MultiStreamMemBlock *>(block);
  if ((ms_block->GetAllocType().GetType() == BlockAllocType::kTensorDataWrapped)) {
    return ge::GRAPH_SUCCESS;
  }
  auto l1_block = own_allocator_->MoveL2ToL1(ms_block->GetMemBlock());
  GE_ASSERT_NOTNULL(l1_block);
  ms_block->SetMemBlock(l1_block);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MultiStreamL2Allocator::RecycleFreeMem() {
  GELOGI("stream id %" PRId64 "'s l2 allocator will recycle free memory, stream addr %p", GetStreamId(), stream_);
  own_allocator_->Recycle();
  return ge::GRAPH_SUCCESS;
}
}  // namespace memory
}  // namespace gert
