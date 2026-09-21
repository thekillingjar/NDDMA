/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_V2_KERNEL_MEMORY_FFTS_MEM_ALLOCATOR_H
#define RUNTIME_V2_KERNEL_MEMORY_FFTS_MEM_ALLOCATOR_H

#include <cstdint>
#include <array>
#include <memory>
#include <sstream>
#include <list>
#include "mem_block.h"
#include "ge/ge_api_types.h"
#include "ge/ge_allocator.h"
#include "kernel/memory/allocator/scalable_allocator.h"
#include "runtime/base.h"
#include "runtime/mem_allocator.h"
#include "caching_mem_allocator.h"

namespace gert {
namespace memory {
class FftsMemBlock : public PageSpan {
 public:
  FftsMemBlock(ge::Allocator &allocator, ScalableAllocator &scalabel_allocator, BlockAddr block_addr,
               MemAddr addr, size_t size, uint32_t window_size)
      : PageSpan(allocator, scalabel_allocator, block_addr, addr, size), window_size_(window_size), block_size_(size) {}
  ~FftsMemBlock() = default;

  // index begin from 0
  const void *Addr(uint32_t index) const {
    return static_cast<const void *>(static_cast<const uint8_t *>(GetAddr()) + (index % window_size_) * block_size_);
  }

  const std::string DebugString() const {
    std::stringstream ss;
    ss << "[ window_size:" << window_size_ << ", block_size:" << block_size_ << ", Addrs: ";
    for (uint32_t i = 0U; i < window_size_; ++i) {
      ss << std::hex << Addr(i) << " ";
    }
    ss << " ]";
    return ss.str();
  }

 private:
  uint32_t window_size_ = 1U;
  size_t block_size_;
};

class FftsMemBlockAllocator : public SpanAllocator {
 public:
  explicit FftsMemBlockAllocator(size_t capacity = ScalableConfig().span_prepared_count)
      : ffts_block_allocator_(capacity) {}
  ~FftsMemBlockAllocator() = default;
  PageSpan *Alloc() override {
    auto addr = ffts_block_allocator_.Alloc();
    GELOGD("FftsMemBlock addr:%p.", addr);
    return addr;
  }

  void Free(PageSpan &span) override {
    GELOGD("FftsMemBlock addr:%p.", &span);
    ffts_block_allocator_.Free((dynamic_cast<FftsMemBlock &>(span)));
  }

 private:
  ObjectAllocator<FftsMemBlock> ffts_block_allocator_;
};

struct FftsMemAllocator : public ge::Allocator, public ScalableAllocator {
 public:
  FftsMemAllocator(ge::Allocator &allocator, DeviceId device_id, uint32_t window_size);
  ~FftsMemAllocator() override;

  FftsMemBlock *Malloc(size_t size) override;
  void Free(ge::MemBlock *block) override {
    ScalableAllocator::Free(dynamic_cast<PageSpan *>(block));
  }

  static std::unique_ptr<FftsMemAllocator> GetAllocator(ge::Allocator &mem_allocator, DeviceId device_id,
                                                        uint32_t window_size);
  static std::unique_ptr<FftsMemAllocator> GetAllocator(CachingMemAllocator &caching_mem_allocator,
                                                        uint32_t window_size) {
    return GetAllocator(caching_mem_allocator, caching_mem_allocator.GetDeviceId(), window_size);
  }
  void SetWidowSize(uint32_t window_size) {
    if (window_size != 0U) {
      window_size_ = window_size;
    }
  }

 protected:
  BlockAddr DevAlloc(const MemSize size) override;
  PageSpan *BlockAlloc(ge::Allocator &allocator, const BlockAddr block_addr, const MemAddr addr,
                       const size_t size) override;

 private:
  uint32_t window_size_ = 1U;
  FftsMemBlockAllocator ffts_mem_block_allocator_;
  FirstLevelPool first_level_pool_;
};
}  // namespace memory
}  // namespace gert

#endif
