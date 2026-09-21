/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>

#include "graph/utils/math_util.h"
#include "base/err_msg.h"
#include "common/plugin/ge_make_unique_util.h"
#include "ffts_mem_allocator.h"

#include "runtime/rt.h"

#include "framework/common/debug/ge_log.h"
#include "common/checker.h"

namespace gert {
namespace memory {
FftsMemAllocator::~FftsMemAllocator() {
  Recycle();
}

FftsMemBlock *FftsMemAllocator::Malloc(size_t size) {
  GELOGI("Malloc size:%zu.", size);
  return static_cast<FftsMemBlock *>(Alloc(*this, size));
}

BlockAddr FftsMemAllocator::DevAlloc(const MemSize size) {
  auto allocate_size = size * window_size_;
  auto block = device_allocator_.Alloc(allocate_size);
  GELOGI("Block alloc from level_1 allocator addr:%p size:%zu allocate_size:%lu window_size:%u.", block,
              size, allocate_size, window_size_);
  return block;
}

PageSpan *FftsMemAllocator::BlockAlloc(ge::Allocator &allocator, const BlockAddr block_addr, const MemAddr addr,
                                       const size_t size) {
  (void)allocator;
  GELOGI("FftsMemBlock mem_addr:%p block_size:%lu window_size:%u.", addr, size, window_size_);
  auto span = new (ffts_mem_block_allocator_.Alloc()) FftsMemBlock{*this, *this, block_addr, addr, size, window_size_};
  return span;
}

FftsMemAllocator::FftsMemAllocator(ge::Allocator &allocator, DeviceId device_id, uint32_t window_size)
    : ScalableAllocator(ffts_mem_block_allocator_, first_level_pool_, ScalableConfig()),
      first_level_pool_(allocator, device_id, "ffts memory pool") {
  allocator_id_with_type_ = "[ffts_allocator_" + std::to_string(allocator_id_) + "]";
  SetWidowSize(window_size);
}

std::unique_ptr<FftsMemAllocator> FftsMemAllocator::GetAllocator(ge::Allocator &mem_allocator, DeviceId device_id,
                                                                 uint32_t window_size) {
  return ge::MakeUnique<FftsMemAllocator>(mem_allocator, device_id, window_size);
}
}  // namespace memory
}  // namespace gert
