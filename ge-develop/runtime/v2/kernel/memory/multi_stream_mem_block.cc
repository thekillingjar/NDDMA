/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multi_stream_mem_block.h"
#include "core/debug/kernel_tracing.h"
#include "common/checker.h"

namespace gert {
namespace memory {
MultiStreamMemBlock::MultiStreamMemBlock()
    : mem_block_{nullptr},
      version_{0},
      birth_allocator_{nullptr},
      alloc_type_{BlockAllocType::kEnd},
      occupy_mif_{},
      stream_ids_to_occupy_count_{} {
  stream_ids_to_occupy_count_.resize(kDefaultMaxStreamNum, 0U);
}
MultiStreamMemBlock::~MultiStreamMemBlock() {
  if (mem_block_ != nullptr) {
    mem_block_->Free();
    mem_block_ = nullptr;
  }
}
void MultiStreamMemBlock::Free(int64_t stream_id) {
  if (SECUREC_UNLIKELY(mem_block_ == nullptr)) {
    return;
  }
  if (SECUREC_UNLIKELY(!IsStreamIdValid(stream_id))) {
    return;
  }
  // 全局回收、本地回收、什么都没做都有可能
  birth_allocator_->FreeAt(stream_id, this);
}
ge::graphStatus MultiStreamMemBlock::ReInit(GertAllocator *birth_allocator, ge::MemBlock *block,
                                            BlockAllocType alloc_type) {
  if (SECUREC_UNLIKELY(birth_allocator == nullptr)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  birth_allocator_ = birth_allocator;
  if (SECUREC_UNLIKELY(mem_block_ != nullptr)) {
    GELOGE(ge::GRAPH_PARAM_INVALID,
           "Failed to ReInit GertMemBlock, origin mem_block is not cleared, this may cased memory leaks");
    return ge::GRAPH_PARAM_INVALID;
  }
  mem_block_ = block;
  alloc_type_ = std::move(alloc_type);
  ReInitMifs();
  return NewAccessStream(birth_allocator->GetStreamId(), birth_allocator->GetStreamId());
}
ge::graphStatus MultiStreamMemBlock::NewAccessStream(int64_t src_stream, int64_t dst_stream) {
  GE_ASSERT_TRUE(IsStreamIdValid(src_stream), "Invalid src stream id %" PRId64, src_stream);
  GE_ASSERT_TRUE(IsStreamIdValid(dst_stream), "Invalid dst stream id %" PRId64, dst_stream);
  ++stream_ids_to_occupy_count_[dst_stream];
  occupy_mif_.SetAll(dst_stream);
  // todo log access
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus MultiStreamMemBlock::SyncLocalRecycleStatus(int64_t src_stream, int64_t dst_stream) {
  GE_ASSERT_TRUE(IsStreamIdValid(src_stream), "Invalid src stream id %" PRId64, src_stream);
  GE_ASSERT_TRUE(IsStreamIdValid(dst_stream), "Invalid dst stream id %" PRId64, dst_stream);

  // 自己的clear状态一定是最新的，不可能出现别人先标记自己已经clear的场景。所以这里只是健壮行检查，理论上永远不会失败
  GE_ASSERT_TRUE(occupy_mif_.IsSet(src_stream, dst_stream) >= occupy_mif_.IsSet(dst_stream, dst_stream));
  GE_ASSERT_TRUE(occupy_mif_.IsSet(dst_stream, src_stream) >= occupy_mif_.IsSet(src_stream, src_stream));

  occupy_mif_.MergeClearedFrom(src_stream, dst_stream);
  if (!occupy_mif_.IsAnySet(dst_stream)) {
    // 全局回收(borrow or birth)
    Free(dst_stream);
  }

  return ge::GRAPH_SUCCESS;
}
ge::graphStatus MultiStreamMemBlock::SyncAllLocalRecycleStatus(int64_t dst_stream) {
  GE_ASSERT_TRUE(IsStreamIdValid(dst_stream), "Invalid dst stream id %" PRId64, dst_stream);
  for (int64_t stream_id = 0; static_cast<size_t>(stream_id) < stream_ids_to_occupy_count_.size(); ++stream_id) {
    if (stream_id == dst_stream) {
      continue;
    }
    occupy_mif_.MergeClearedFrom(stream_id, dst_stream);
    if (!occupy_mif_.IsAnySet(dst_stream)) {
      Free(dst_stream);
      break;
    }
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace memory
}  // namespace gert
